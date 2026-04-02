"""
crs/byte_strategies.py — Generate raw PoC input bytes for fuzz targets.

The core idea: the LLM's advantage over a fuzzer is that it can READ CODE
and REASON about what input structure reaches the vulnerable code path.
Every strategy forces the LLM through a reasoning chain:

  1. What does the harness do with input bytes?
  2. What function processes them first?
  3. What code path leads to the vulnerable function?
  4. What input structure/values trigger the specific bug?
  5. Write a Python script that crafts those exact bytes.

This is project-agnostic — the same reasoning works for libmagic, mosquitto,
openssl, ffmpeg, or any other project.
"""
from __future__ import annotations

import logging
import os
import re
import struct
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from crs.code_intelligence import CodeContext
from crs.config import cfg
from crs.harness_finder import HarnessInfo
from crs.llm_router import LLMRouter

logger = logging.getLogger(__name__)


@dataclass
class PoCBytes:
    strategy_name: str
    data: bytes
    poc_path: Path
    confidence: float
    notes: str
    generator_script: Optional[str] = None


# ── Prompts ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_CODE_TRACER = """\
You are an expert C/C++ vulnerability analyst. Your job is to trace how raw
input bytes flow through a program from the fuzz harness entry point to the
vulnerable code.

You will be given:
- A fuzz harness (LLVMFuzzerTestOneInput or similar)
- The vulnerability description
- Source code snippets from the project

You MUST answer these questions IN ORDER:

STEP 1 — FORMAT IDENTIFICATION (CRITICAL):
  What FILE FORMAT or PROTOCOL does the parser expect?
  Look for: magic byte checks, signature validation, format dispatch.
  Common patterns:
    - memcmp(header, "\\x89PNG", 4)  -> PNG format
    - memcmp(header, "\\x8aMNG", 4)  -> MNG format (NOT the same as PNG!)
    - "\\x7fELF" -> ELF, "II"/"MM" -> TIFF, "PK" -> ZIP, "GIF8" -> GIF
    - MQTT, HTTP, JSON, XML, protobuf, etc.
  The input MUST start with the correct magic bytes/signature or the
  parser will REJECT it before ever reaching the vulnerable code.
  WARNING: Similar formats (PNG vs MNG, TIFF vs BigTIFF) have DIFFERENT
  signatures. Check which one the VULNERABLE function actually parses.

STEP 2 — ENTRY: What does the harness do with the raw bytes?
  Which function does it call first? What arguments does it pass?
  Does it wrap the bytes in a struct, null-terminate them, copy them?

STEP 3 — PARSING: How does the parser validate and dispatch the input?
  What happens after the signature check?
  What chunk/record/packet structure does it expect?
  What are the LENGTH and TYPE fields? (byte offsets, endianness)
  What minimum structure must be present for the parser to proceed?

STEP 4 — PATH TO BUG: How do bytes reach the vulnerable code?
  Trace the call chain from the entry function to where the bug lives.
  What conditions must be true for the vulnerable code path to execute?
  What branches, checks, or format requirements must the input satisfy?
  Name the SPECIFIC chunk type / record type / message type that reaches
  the vulnerable function.

STEP 5 — TRIGGER: What specific input values trigger the bug?
  For a buffer overflow: what size/length field causes the overflow?
  For use-after-free: what sequence of operations triggers free-then-use?
  For uninitialized memory: what code path skips initialization?
  For null deref: what input causes a NULL return that gets dereferenced?
  For under-validated length: what LENGTH value passes the check but
    causes out-of-bounds access? (e.g., length > 0 but code reads 5 bytes)

Be SPECIFIC. Name functions, struct fields, byte offsets, values.
Reference the actual code you can see in the snippets.
"""

SYSTEM_PROMPT_BYTE_CRAFTER = """\
You are an expert at writing Python scripts that generate crafted input bytes
to trigger specific C/C++ vulnerabilities in fuzz targets.

You will receive a detailed code path analysis explaining exactly how input
bytes reach the vulnerable code. Use that analysis to write a precise script.

CRITICAL RULES:
1. Output ONLY a Python script inside ```python ... ```
2. The script writes raw bytes to stdout: sys.stdout.buffer.write(...)
3. The bytes will be piped to the fuzz target: ./fuzz_target < output
4. Use ONLY Python standard library (struct, sys, os, zlib, etc.)
5. MOST IMPORTANT — GET THE FORMAT RIGHT:
   - Start with the EXACT magic bytes/signature the parser expects.
   - If it's MNG, use \\x8a\\x4d\\x4e\\x47\\x0d\\x0a\\x1a\\x0a (NOT PNG!)
   - If it's PNG, use \\x89\\x50\\x4e\\x47\\x0d\\x0a\\x1a\\x0a
   - If it's ELF, TIFF, ZIP, etc. — use THAT format's signature.
   - Wrong signature = parser rejects input = never reaches the bug.
6. Build STRUCTURALLY VALID input for the format:
   - Include required header chunks/records (e.g., MNG needs MHDR first)
   - Use correct chunk structure: [4-byte length][4-byte type][data][4-byte CRC]
   - Calculate CRCs with zlib.crc32() for PNG/MNG chunks
7. Malform ONLY the specific field that triggers the bug:
   - E.g., for "length not validated >= 5", make the chunk data 1-4 bytes
   - E.g., for "buffer overflow", set length field to 0xFFFFFFFF
8. Include comments explaining what each part of the input does:
   # bytes 0-7: MNG signature
   # bytes 8-19: MHDR chunk (required first chunk)
   # bytes 20-35: LOOP chunk with 1-byte data (triggers OOB read)
   # etc.
"""

SYSTEM_PROMPT_MULTI_VARIANT = """\
You are an expert vulnerability researcher. Write a Python script that
generates MULTIPLE crafted input variants (at least 3) to trigger a
vulnerability, writing each to a separate file.

Each variant should try a DIFFERENT approach to reaching the same bug:
- Variant 0: Minimal input — shortest possible bytes to reach the vulnerable code
- Variant 1: Boundary values — use max/min integer values, zero-length fields
- Variant 2: Format-specific — valid format with one malformed field
- Variant 3: Oversized — trigger overflow via large fields
- Variant 4: Empty/null fields — trigger null derefs or uninit reads

RULES:
1. Script takes one argument: output directory
2. Writes files named variant_0.bin, variant_1.bin, etc.
3. Each file contains raw bytes
4. Use sys.argv[1] as the output directory
5. Use ONLY Python standard library
6. Output ONLY the script inside ```python ... ```
"""


# ── Helpers ────────────────────────────────────────────────────────────────

def _save_poc_bytes(data, task_id, strategy_name):
    safe_task = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(task_id))
    safe_strat = re.sub(r"[^a-zA-Z0-9_\-]", "_", strategy_name)
    work = cfg.task_work_dir(safe_task)
    path = work / f"poc_{safe_strat}.bin"
    path.write_bytes(data)
    return path


def _run_python_script(script, timeout=30, args=None):
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, dir="/tmp") as tmp:
        tmp.write(script)
        tmp_path = tmp.name
    try:
        cmd = ["python3", tmp_path] + (args or [])
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if r.returncode != 0:
            stderr = r.stderr.decode(errors="replace")
            print(f"  [byte_gen] Script failed (rc={r.returncode}): {stderr[:500]}")
            return None
        return r.stdout
    except subprocess.TimeoutExpired:
        print("  [byte_gen] Script timed out")
        return None
    except Exception as e:
        print(f"  [byte_gen] Script error: {e}")
        return None
    finally:
        try: os.unlink(tmp_path)
        except: pass


def _extract_python_block(response):
    m = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if m: return m.group(1).strip()
    m = re.search(r"```\w*\s*\n(.*?)```", response, re.DOTALL)
    if m: return m.group(1).strip()
    return None


def _build_context_prompt(context, harness):
    parts = []
    parts.append(f"## Vulnerability Description\n{context.description}\n")
    parts.append(f"## Vulnerability Type: {context.vuln_type}\n")
    if harness:
        parts.append(f"## Fuzz Harness ({harness.harness_path.name})\n```c\n{harness.harness_code[:4000]}\n```\n")
        if harness.called_functions:
            parts.append(f"## Functions Called by Harness\n{', '.join(harness.called_functions[:15])}\n")
    snippets = context.top_snippets[:6000] if isinstance(context.top_snippets, str) else ""
    if snippets:
        parts.append(f"## Relevant Source Code\n```c\n{snippets}\n```\n")
    return "\n".join(parts)


class ByteStrategy(ABC):
    name = "base"
    @abstractmethod
    def generate(self, context, router, harness=None): ...


class DeepTraceStrategy(ByteStrategy):
    """Flagship: trace code path then craft precise bytes. Works for any project."""
    name = "deep_trace"

    def generate(self, context, router, harness=None):
        print(f"[Strategy] Running '{self.name}' ...")
        results = []
        ctx = _build_context_prompt(context, harness)

        trace_prompt = (ctx +
            "\n## YOUR TASK\n"
            "Trace the EXACT path that input bytes must take to trigger this vulnerability.\n\n"
            "STEP 1 — FORMAT: What file format/protocol does the parser expect?\n"
            "  Look for magic bytes, signature checks, format dispatch. What EXACT\n"
            "  signature must the input start with? (e.g. MNG \\x8aMNG != PNG \\x89PNG)\n"
            "STEP 2 — ENTRY: How does the harness pass bytes to the project?\n"
            "STEP 3 — PARSING: What chunk/record structure does the format use?\n"
            "  What required chunks must appear first? What are the length/type fields?\n"
            "STEP 4 — PATH TO BUG: What specific chunk/record type reaches the vulnerable code?\n"
            "STEP 5 — TRIGGER: What specific field value/size triggers the bug?\n\n"
            "Be SPECIFIC. Name exact functions, struct fields, conditions, byte offsets.\n"
            "The #1 reason PoCs fail is WRONG FILE FORMAT SIGNATURE. Get this right first.\n")

        print(f"  [{self.name}] Step 1: Tracing code path...")
        analysis = router.chat(system_prompt=SYSTEM_PROMPT_CODE_TRACER, user_prompt=trace_prompt, max_tokens=3000, temperature=0.1)
        if not analysis:
            return results

        print(f"  [{self.name}] Step 2: Crafting bytes from trace...")
        craft_prompt = (
            f"## Code Path Analysis\n{analysis}\n\n"
            f"## Vulnerability Description\n{context.description}\n\n")
        if harness:
            craft_prompt += f"## Harness Code\n```c\n{harness.harness_code[:3000]}\n```\n\n"
        craft_prompt += (
            "Based on the code path analysis above, write a Python script that "
            "generates the EXACT bytes needed to trigger this vulnerability.\n"
            "Follow the analysis step by step.\n")

        raw = router.chat(system_prompt=SYSTEM_PROMPT_BYTE_CRAFTER, user_prompt=craft_prompt, max_tokens=cfg.MAX_TOKENS, temperature=0.2)
        if not raw: return results
        script = _extract_python_block(raw)
        if not script: return results
        poc_data = _run_python_script(script)
        if poc_data and len(poc_data) > 0:
            poc_path = _save_poc_bytes(poc_data, context.task.task_id, self.name)
            results.append(PoCBytes(self.name, poc_data, poc_path, 0.8,
                f"Deep trace: {len(poc_data)} bytes", script))
            print(f"  [{self.name}] Generated {len(poc_data)} bytes")
        return results


class MultiVariantStrategy(ByteStrategy):
    """Multiple attack angles in one shot."""
    name = "multi_variant"

    def generate(self, context, router, harness=None):
        print(f"[Strategy] Running '{self.name}' ...")
        results = []
        ctx = _build_context_prompt(context, harness)
        prompt = (ctx +
            "\n## YOUR TASK\n"
            "Write a Python script that generates MULTIPLE variant input files, "
            "each trying a DIFFERENT approach to trigger the vulnerability.\n\n"
            "Script takes output dir as sys.argv[1], writes variant_0.bin, variant_1.bin, etc.\n\n"
            "Each variant should try a different angle:\n"
            "- Variant 0: Minimal bytes to reach the bug\n"
            "- Variant 1: Oversized fields to trigger overflow\n"
            "- Variant 2: Boundary values (0, -1, MAX_INT) in key fields\n"
            "- Variant 3: Valid format with one corrupted field\n"
            "- Variant 4: Empty/null fields for null deref or uninit\n\n"
            "Read the harness and code carefully. Each variant MUST target the actual vulnerable function.\n")

        raw = router.chat(system_prompt=SYSTEM_PROMPT_MULTI_VARIANT, user_prompt=prompt, max_tokens=cfg.MAX_TOKENS, temperature=0.3)
        if not raw: return results
        script = _extract_python_block(raw)
        if not script: return results

        work = cfg.task_work_dir(context.task.task_id)
        variant_dir = work / "variants"
        variant_dir.mkdir(parents=True, exist_ok=True)
        _run_python_script(script, args=[str(variant_dir)])

        for vf in sorted(variant_dir.glob("variant_*.bin")):
            try:
                data = vf.read_bytes()
                if len(data) > 0:
                    poc_path = _save_poc_bytes(data, context.task.task_id, f"{self.name}_{vf.stem}")
                    results.append(PoCBytes(f"{self.name}_{vf.stem}", data, poc_path, 0.6,
                        f"Variant {vf.stem}: {len(data)} bytes", script))
            except: pass
        print(f"  [{self.name}] Generated {len(results)} variant(s)")
        return results


class DirectScriptStrategy(ByteStrategy):
    """Quick single-shot with step-by-step reasoning."""
    name = "direct_script"

    def generate(self, context, router, harness=None):
        print(f"[Strategy] Running '{self.name}' ...")
        results = []
        ctx = _build_context_prompt(context, harness)
        prompt = (ctx +
            "\n## YOUR TASK\n"
            "Write a Python script that generates raw input bytes to trigger this vulnerability.\n\n"
            "THINK STEP BY STEP before writing code:\n"
            "1. What FILE FORMAT does the parser expect? Look for magic bytes/signature checks.\n"
            "   (e.g. MNG uses \\x8aMNG, PNG uses \\x89PNG — they are NOT interchangeable!)\n"
            "2. What is the chunk/record structure? (length fields, type fields, CRC?)\n"
            "3. What specific chunk TYPE reaches the vulnerable function?\n"
            "4. What field value/size in that chunk triggers the bug?\n"
            "5. Build: correct signature + required headers + malformed target chunk.\n"
            "   Use struct.pack() for binary fields, zlib.crc32() for CRCs.\n\n"
            "Use sys.stdout.buffer.write() to output the bytes.\n"
            "WRONG FORMAT SIGNATURE = input rejected before reaching the bug. Get it right.\n")

        raw = router.chat(system_prompt=SYSTEM_PROMPT_BYTE_CRAFTER, user_prompt=prompt, max_tokens=cfg.MAX_TOKENS, temperature=0.3)
        if not raw: return results
        script = _extract_python_block(raw)
        if not script: return results
        poc_data = _run_python_script(script)
        if poc_data and len(poc_data) > 0:
            poc_path = _save_poc_bytes(poc_data, context.task.task_id, self.name)
            results.append(PoCBytes(self.name, poc_data, poc_path, 0.6,
                f"Direct: {len(poc_data)} bytes", script))
            print(f"  [{self.name}] Generated {len(poc_data)} bytes")
        return results


class CorpusMutationStrategy(ByteStrategy):
    """Mutate existing test/seed files from the repo."""
    name = "corpus_mutation"
    _SEED_DIRS = ["corpus","seed","seeds","testdata","test_data","test/corpus",
                  "tests/corpus","fuzz/corpus","fuzzing/corpora","test","tests",
                  "examples","samples"]
    _TEST_EXTS = {".testfile",".bin",".raw",".dat",".input",".txt",".json",
                  ".xml",".conf",".cfg"}

    def generate(self, context, router, harness=None):
        print(f"[Strategy] Running '{self.name}' ...")
        results = []
        repo = Path(context.task.repo_path).resolve()
        seeds = self._find_seeds(repo)
        if not seeds:
            print(f"  [{self.name}] No seed corpus found")
            return results
        print(f"  [{self.name}] Found {len(seeds)} seed(s)")
        for i, seed in enumerate(seeds[:5]):
            for j, (mutated, desc) in enumerate(self._mutate(seed)):
                name = f"{self.name}_{i}_{j}"
                poc_path = _save_poc_bytes(mutated, context.task.task_id, name)
                results.append(PoCBytes(name, mutated, poc_path, 0.45,
                    f"Mutated seed {i}: {desc} ({len(mutated)}B)"))
        print(f"  [{self.name}] Generated {len(results)} mutations")
        return results

    def _find_seeds(self, repo):
        seeds = []
        for d in self._SEED_DIRS:
            sd = repo / d
            if sd.is_dir():
                for f in sorted(sd.iterdir()):
                    if f.is_file() and f.stat().st_size < 1_000_000:
                        try: seeds.append(f.read_bytes())
                        except: pass
                    if len(seeds) >= 20: return seeds
        for ext in self._TEST_EXTS:
            for f in sorted(repo.rglob(f"*{ext}")):
                if f.is_file() and f.stat().st_size < 100_000:
                    if any(x in str(f).lower() for x in ["test","corpus","seed"]):
                        try: seeds.append(f.read_bytes())
                        except: pass
                if len(seeds) >= 20: return seeds
        return seeds

    def _mutate(self, data):
        m = []
        m.append((data + b"\x00" * 10000, "extend_nulls"))
        m.append((data + b"\xff" * 10000, "extend_ff"))
        if len(data) > 4:
            m.append((data[:len(data)//2], "truncate_half"))
            m.append((data[:1], "truncate_1"))
        if len(data) > 0:
            f = bytearray(data); f[0] ^= 0xFF
            m.append((bytes(f), "flip_first"))
        if len(data) >= 4:
            l = bytearray(data); l[0:4] = b"\xff\xff\xff\x7f"
            m.append((bytes(l), "large_len"))
        m.append((data * 3, "tripled"))
        return m


class IterativeRefineStrategy(ByteStrategy):
    """Generate → test → learn from output → regenerate."""
    name = "iterative_refine"
    _MAX_ITERS = 3

    def __init__(self, run_callback=None):
        self._run_cb = run_callback

    def generate(self, context, router, harness=None):
        print(f"[Strategy] Running '{self.name}' ...")
        results = []
        initial = DirectScriptStrategy().generate(context, router, harness)
        if not initial: return results
        best = initial[0]
        results.append(best)
        if not self._run_cb: return results

        for it in range(1, self._MAX_ITERS + 1):
            print(f"  [{self.name}] Iteration {it}/{self._MAX_ITERS}")
            triggered, output = self._run_cb(best)
            if triggered:
                best.confidence = 0.95
                print(f"  [{self.name}] TRIGGERED on iteration {it}!")
                return results

            ctx = _build_context_prompt(context, harness)
            prompt = (ctx +
                f"\n## Previous Attempt\n```python\n{best.generator_script or 'N/A'}\n```\n"
                f"\n## Execution Output\n```\n{output[:3000]}\n```\n\n"
                f"## YOUR TASK\nThe previous input did NOT trigger the vulnerability.\n"
                f"Analyze: was input parsed? What code path did it take? What needs to change?\n"
                f"Write an IMPROVED script with comments explaining what changed.\n")

            raw = router.chat(system_prompt=SYSTEM_PROMPT_BYTE_CRAFTER, user_prompt=prompt, max_tokens=cfg.MAX_TOKENS, temperature=0.2)
            if not raw: break
            script = _extract_python_block(raw)
            if not script: break
            poc_data = _run_python_script(script)
            if poc_data and len(poc_data) > 0:
                poc_path = _save_poc_bytes(poc_data, context.task.task_id, f"{self.name}_iter{it}")
                best = PoCBytes(f"{self.name}_iter{it}", poc_data, poc_path,
                    0.6 + it * 0.05, f"Refined iter {it}: {len(poc_data)}B", script)
                results.append(best)
        return results


class SimpleBytePatterns(ByteStrategy):
    """Dumb fast patterns. Baseline."""
    name = "simple_patterns"
    _PATTERNS = [
        (b"", "empty"), (b"\x00", "null"), (b"\x00"*4096, "null_4k"),
        (b"\xff"*4096, "ff_4k"), (b"A"*65536, "long_ascii"),
        (b"\xff\xff\xff\xff"*256, "max_ints"), (b"%s"*100, "fmtstr"),
        (struct.pack("<I", 0xFFFFFFFF) + b"A"*10000, "large_len"),
        (struct.pack("<I", 0) + b"\x00"*100, "zero_len"),
    ]

    def generate(self, context, router, harness=None):
        print(f"[Strategy] Running '{self.name}' ...")
        results = []
        for data, desc in self._PATTERNS:
            name = f"{self.name}_{desc}"
            path = _save_poc_bytes(data, context.task.task_id, name)
            results.append(PoCBytes(name, data, path, 0.1, f"Pattern: {desc}"))
        print(f"  [{self.name}] Generated {len(results)} patterns")
        return results


class ByteOrchestrator:
    def __init__(self, router, run_callback=None):
        self.router = router
        self.strategies = [
            DeepTraceStrategy(),
            MultiVariantStrategy(),
            DirectScriptStrategy(),
            CorpusMutationStrategy(),
            IterativeRefineStrategy(run_callback=run_callback),
            SimpleBytePatterns(),
        ]

    def run(self, context, harness=None):
        all_results = []
        for s in self.strategies:
            print(f"\n{'='*50}\n  ByteOrchestrator -> '{s.name}'\n{'='*50}")
            try:
                r = s.generate(context, self.router, harness)
                all_results.extend(r)
                print(f"  -> {len(r)} PoC(s)")
            except Exception:
                logger.exception("Strategy '%s' failed", s.name)

        all_results.sort(key=lambda r: r.confidence, reverse=True)
        print(f"\n[ByteOrchestrator] {len(all_results)} total PoC(s):")
        for r in all_results[:10]:
            print(f"  * {r.strategy_name:30s}  conf={r.confidence:.2f}  size={len(r.data):>8d}B")
        return all_results
