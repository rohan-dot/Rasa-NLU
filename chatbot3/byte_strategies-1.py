"""
crs/byte_strategies.py — Generate raw PoC input bytes for fuzz targets.

Instead of writing C programs, these strategies produce raw bytes that get
fed to the project's fuzz harness: ./fuzz_target < poc_bytes

The primary strategy asks the LLM to write a Python script that generates
the crafted bytes based on its understanding of the harness and vulnerability.
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
from crs.harness_finder import HarnessInfo, summarize_harness
from crs.llm_router import LLMRouter

logger = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class PoCBytes:
    strategy_name: str
    data: bytes
    poc_path: Path
    confidence: float
    notes: str
    generator_script: Optional[str] = None  # Python script that produced the bytes


# ── Prompts ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_BYTE_GENERATOR = """\
You are an expert at crafting malicious inputs to trigger C/C++ vulnerabilities
in fuzz targets. You deeply understand binary protocols, file formats, and how
parsers process raw bytes.

Given a vulnerability description, fuzz harness code, and relevant source
snippets, write a Python script that generates raw bytes to trigger the
vulnerability.

CRITICAL RULES:
1. The script must write raw bytes to stdout: sys.stdout.buffer.write(...)
2. The bytes will be fed directly to the fuzz harness: ./fuzz_target < output
3. Understand how the harness parses the input — what function is called first,
   what format it expects (binary, text, JSON, XML, etc.)
4. Craft bytes that navigate through the parser to reach the vulnerable code path
5. Use oversized fields, boundary values, or malformed structures to trigger
   the specific bug (overflow, UAF, null deref, etc.)
6. The script MUST be self-contained — only use Python standard library
7. Output ONLY the Python script inside a single ```python ... ``` block
8. No explanation, no comments outside the script

=== EXAMPLE ===
For a heap-buffer-overflow in an MQTT parser:
```python
import sys
# MQTT PUBLISH with oversized payload
packet_type = bytes([0x30])
remaining_len = bytes([0xFF, 0xFF, 0xFF, 0x7F])
topic_len = bytes([0x00, 0x04])
topic = b"test"
payload = b"A" * 0xFFFF
sys.stdout.buffer.write(packet_type + remaining_len + topic_len + topic + payload)
```
"""

SYSTEM_PROMPT_INPUT_ANALYZER = """\
You are an expert at reverse-engineering fuzz harnesses and understanding
how raw input bytes flow through C/C++ programs.

Given a fuzz harness (LLVMFuzzerTestOneInput) and project source code,
determine:
1. What input format does the harness expect? (binary protocol, text, JSON,
   XML, image, configuration file, custom binary format, etc.)
2. What parser/function processes the raw bytes first?
3. What is the byte-level structure of a valid input?
   (header bytes, length fields, data sections, etc.)
4. What byte offset or field, if malformed, would trigger the vulnerability?

Be precise. Reference actual function names. Describe the byte-level structure.
Output your analysis as a structured text description, not code.
"""


# ── Helpers ────────────────────────────────────────────────────────────────

def _save_poc_bytes(data: bytes, task_id: str, strategy_name: str) -> Path:
    """Save raw PoC bytes to a file."""
    safe_task = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(task_id))
    safe_strat = re.sub(r"[^a-zA-Z0-9_\-]", "_", strategy_name)
    work = cfg.task_work_dir(safe_task)
    path = work / f"poc_{safe_strat}.bin"
    path.write_bytes(data)
    return path


def _run_python_script(script: str, timeout: int = 30) -> Optional[bytes]:
    """
    Run a Python script and capture its stdout as raw bytes.
    Returns None if the script fails.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, dir="/tmp"
    ) as tmp:
        tmp.write(script)
        tmp_path = tmp.name

    try:
        r = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            timeout=timeout,
        )
        if r.returncode != 0:
            stderr = r.stderr.decode(errors="replace")
            print(f"  [byte_gen] Script failed (rc={r.returncode}): {stderr[:500]}")
            return None
        return r.stdout  # raw bytes
    except subprocess.TimeoutExpired:
        print("  [byte_gen] Script timed out")
        return None
    except Exception as e:
        print(f"  [byte_gen] Script error: {e}")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _extract_python_block(response: str) -> Optional[str]:
    """Extract Python code from a fenced block."""
    m = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Try any fenced block
    m = re.search(r"```\w*\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _build_user_prompt(
    context: CodeContext,
    harness: Optional[HarnessInfo],
    extra_instructions: str = "",
) -> str:
    """Build the standard user prompt for byte generation strategies."""
    parts = [f"## Vulnerability Description\n{context.description}\n"]

    if harness:
        parts.append(f"## Fuzz Harness Code\n```c\n{harness.harness_code[:4000]}\n```\n")
        parts.append(f"## Functions Called from Harness\n{harness.called_functions[:15]}\n")

    # Add relevant source snippets
    snippets = context.top_snippets[:5000] if isinstance(context.top_snippets, str) else ""
    if snippets:
        parts.append(f"## Relevant Source Snippets\n```c\n{snippets}\n```\n")

    parts.append(f"## Vulnerability Type\n{context.vuln_type}\n")

    if extra_instructions:
        parts.append(extra_instructions)

    return "\n".join(parts)


# ── Base strategy ──────────────────────────────────────────────────────────

class ByteStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate(
        self,
        context: CodeContext,
        router: LLMRouter,
        harness: Optional[HarnessInfo] = None,
    ) -> list[PoCBytes]:
        ...


# ── Strategy 1: LLM Script Generation ─────────────────────────────────────

class LLMScriptStrategy(ByteStrategy):
    """
    Ask the LLM to write a Python script that generates crafted input bytes.
    This is the most powerful strategy — the LLM can reason about protocol
    structure, field sizes, and how to reach the vulnerable code path.
    """

    name = "llm_script"

    def generate(
        self,
        context: CodeContext,
        router: LLMRouter,
        harness: Optional[HarnessInfo] = None,
    ) -> list[PoCBytes]:
        print(f"[Strategy] Running '{self.name}' ...")
        results: list[PoCBytes] = []

        user_prompt = _build_user_prompt(context, harness, extra_instructions=(
            "Write a Python script that generates raw bytes to trigger this vulnerability.\n"
            "The bytes will be fed directly to the fuzz harness shown above.\n"
            "Use sys.stdout.buffer.write() to output the raw bytes.\n"
            "The script must be self-contained (Python standard library only).\n"
        ))

        raw = router.chat(
            system_prompt=SYSTEM_PROMPT_BYTE_GENERATOR,
            user_prompt=user_prompt,
            max_tokens=cfg.MAX_TOKENS,
            temperature=0.3,
        )

        if not raw:
            print(f"  [{self.name}] LLM returned empty")
            return results

        script = _extract_python_block(raw)
        if not script:
            print(f"  [{self.name}] No Python code block found in response")
            return results

        # Run the script to produce bytes
        poc_data = _run_python_script(script)
        if poc_data and len(poc_data) > 0:
            poc_path = _save_poc_bytes(poc_data, context.task.task_id, self.name)
            results.append(PoCBytes(
                strategy_name=self.name,
                data=poc_data,
                poc_path=poc_path,
                confidence=0.7,
                notes=f"LLM-generated script produced {len(poc_data)} bytes",
                generator_script=script,
            ))
            print(f"  [{self.name}] Generated {len(poc_data)} bytes")
        else:
            print(f"  [{self.name}] Script produced no output")

        return results


# ── Strategy 2: Analyze-then-Generate ──────────────────────────────────────

class AnalyzeThenGenerateStrategy(ByteStrategy):
    """
    Two-step: first ask the LLM to analyze the input format,
    then ask it to craft bytes based on that analysis.
    """

    name = "analyze_then_generate"

    def generate(
        self,
        context: CodeContext,
        router: LLMRouter,
        harness: Optional[HarnessInfo] = None,
    ) -> list[PoCBytes]:
        print(f"[Strategy] Running '{self.name}' ...")
        results: list[PoCBytes] = []

        # Step 1: Analyze input format
        analysis_prompt = _build_user_prompt(context, harness, extra_instructions=(
            "Analyze the fuzz harness and determine:\n"
            "1. What input format does it expect?\n"
            "2. What is the byte-level structure of a valid input?\n"
            "3. What specific bytes/fields would trigger the vulnerability?\n"
            "4. What minimum input size is needed to reach the vulnerable code?\n"
        ))

        analysis = router.chat(
            system_prompt=SYSTEM_PROMPT_INPUT_ANALYZER,
            user_prompt=analysis_prompt,
            max_tokens=2048,
            temperature=0.1,
        )

        if not analysis:
            print(f"  [{self.name}] Analysis step returned empty")
            return results

        print(f"  [{self.name}] Analysis complete, generating bytes...")

        # Step 2: Generate bytes based on analysis
        gen_prompt = (
            f"## Vulnerability Description\n{context.description}\n\n"
            f"## Input Format Analysis\n{analysis}\n\n"
        )
        if harness:
            gen_prompt += f"## Fuzz Harness Code\n```c\n{harness.harness_code[:3000]}\n```\n\n"

        gen_prompt += (
            "Based on the analysis above, write a Python script that generates\n"
            "raw bytes to trigger this vulnerability.\n"
            "Use sys.stdout.buffer.write() to output the bytes.\n"
            "Follow the byte-level structure identified in the analysis.\n"
        )

        raw = router.chat(
            system_prompt=SYSTEM_PROMPT_BYTE_GENERATOR,
            user_prompt=gen_prompt,
            max_tokens=cfg.MAX_TOKENS,
            temperature=0.2,
        )

        if not raw:
            return results

        script = _extract_python_block(raw)
        if not script:
            return results

        poc_data = _run_python_script(script)
        if poc_data and len(poc_data) > 0:
            poc_path = _save_poc_bytes(poc_data, context.task.task_id, self.name)
            results.append(PoCBytes(
                strategy_name=self.name,
                data=poc_data,
                poc_path=poc_path,
                confidence=0.75,
                notes=f"Analyze-then-generate: {len(poc_data)} bytes. Analysis: {analysis[:200]}",
                generator_script=script,
            ))
            print(f"  [{self.name}] Generated {len(poc_data)} bytes")

        return results


# ── Strategy 3: Corpus Seed Mutation ───────────────────────────────────────

class CorpusMutationStrategy(ByteStrategy):
    """
    Find existing test inputs / seed corpus in the repo and mutate them.
    """

    name = "corpus_mutation"

    # Directories to search for seed inputs
    _SEED_DIRS = [
        "corpus", "seed", "seeds", "testdata", "test_data",
        "test/corpus", "tests/corpus", "fuzz/corpus",
        "test", "tests", "examples", "samples",
    ]

    def generate(
        self,
        context: CodeContext,
        router: LLMRouter,
        harness: Optional[HarnessInfo] = None,
    ) -> list[PoCBytes]:
        print(f"[Strategy] Running '{self.name}' ...")
        results: list[PoCBytes] = []

        repo = Path(context.task.repo_path).resolve()
        seeds = self._find_seeds(repo)

        if not seeds:
            print(f"  [{self.name}] No seed corpus found in repo")
            return results

        print(f"  [{self.name}] Found {len(seeds)} seed file(s)")

        # Apply mutations to each seed
        for i, seed_data in enumerate(seeds[:5]):  # cap at 5 seeds
            mutations = self._mutate(seed_data)
            for j, (mutated, desc) in enumerate(mutations):
                name = f"{self.name}_{i}_{j}"
                poc_path = _save_poc_bytes(mutated, context.task.task_id, name)
                results.append(PoCBytes(
                    strategy_name=name,
                    data=mutated,
                    poc_path=poc_path,
                    confidence=0.4,
                    notes=f"Mutated seed {i}: {desc} ({len(mutated)} bytes)",
                ))

        print(f"  [{self.name}] Generated {len(results)} mutated variants")
        return results

    def _find_seeds(self, repo: Path) -> list[bytes]:
        """Find seed/corpus files in the repo."""
        seeds: list[bytes] = []
        for dir_name in self._SEED_DIRS:
            seed_dir = repo / dir_name
            if seed_dir.is_dir():
                for f in sorted(seed_dir.iterdir()):
                    if f.is_file() and f.stat().st_size < 1_000_000:  # < 1MB
                        try:
                            seeds.append(f.read_bytes())
                        except Exception:
                            pass
                    if len(seeds) >= 20:
                        return seeds
        return seeds

    def _mutate(self, data: bytes) -> list[tuple[bytes, str]]:
        """Apply various mutations to a seed input."""
        mutations: list[tuple[bytes, str]] = []

        # 1. Extend — append oversized data
        mutations.append((data + b"A" * 10000, "extend_10k"))

        # 2. Truncate at various points
        if len(data) > 4:
            mutations.append((data[:len(data)//2], "truncate_half"))

        # 3. Bit flip at beginning
        if len(data) > 0:
            flipped = bytearray(data)
            flipped[0] ^= 0xFF
            mutations.append((bytes(flipped), "flip_first_byte"))

        # 4. Insert max-int values at beginning
        mutations.append((b"\xff\xff\xff\xff" + data, "prepend_maxint"))

        # 5. Replace length-like fields with large values
        if len(data) >= 4:
            large = bytearray(data)
            large[0:4] = b"\xff\xff\xff\x7f"
            mutations.append((bytes(large), "large_length_field"))

        # 6. All zeros of same length
        mutations.append((b"\x00" * len(data), "all_zeros"))

        # 7. Double the input
        mutations.append((data * 2, "doubled"))

        return mutations


# ── Strategy 4: Simple Byte Patterns ───────────────────────────────────────

class SimpleBytePatterns(ByteStrategy):
    """
    Generate common crash-inducing byte patterns without LLM involvement.
    These are dumb but fast — good as a baseline.
    """

    name = "simple_patterns"

    _PATTERNS: list[tuple[bytes, str]] = [
        (b"", "empty"),
        (b"\x00", "single_null"),
        (b"\x00" * 1024, "null_1k"),
        (b"\xff" * 1024, "ff_1k"),
        (b"A" * 65536, "long_ascii"),
        (b"\x00\x00\x00\x00" * 1024, "zero_ints"),
        (b"\xff\xff\xff\xff" * 256, "max_ints"),
        (b"%s%s%s%s%s%s%s%s%s%s", "format_string"),
        (b"{{{{{{{{{{{{{{{{{{", "nested_braces"),
        (b"[" * 1000 + b"]" * 1000, "nested_brackets"),
        (b'{"a":' * 500 + b"1" + b"}" * 500, "deep_json"),
        (b"<" + b"a>" * 5000, "long_xml_tags"),
        (struct.pack("<I", 0xFFFFFFFF) + b"A" * 10000, "large_len_header"),
        (struct.pack("<I", 0) + b"\x00" * 100, "zero_len_header"),
        (b"\x80" * 4096, "high_bytes"),
    ]

    def generate(
        self,
        context: CodeContext,
        router: LLMRouter,
        harness: Optional[HarnessInfo] = None,
    ) -> list[PoCBytes]:
        print(f"[Strategy] Running '{self.name}' ...")
        results: list[PoCBytes] = []

        for data, desc in self._PATTERNS:
            name = f"{self.name}_{desc}"
            poc_path = _save_poc_bytes(data, context.task.task_id, name)
            results.append(PoCBytes(
                strategy_name=name,
                data=data,
                poc_path=poc_path,
                confidence=0.15,
                notes=f"Simple pattern: {desc} ({len(data)} bytes)",
            ))

        print(f"  [{self.name}] Generated {len(results)} patterns")
        return results


# ── Strategy 5: Iterative Refinement ───────────────────────────────────────

class IterativeRefineStrategy(ByteStrategy):
    """
    Generate bytes with LLM, test them, feed the result back to the LLM
    for refinement. Requires a runner callback.
    """

    name = "iterative_refine"
    _MAX_ITERATIONS = 3

    def __init__(self, run_callback=None):
        """
        run_callback: function(PoCBytes) -> (bool triggered, str output)
        If None, just generates without testing.
        """
        self._run_callback = run_callback

    def generate(
        self,
        context: CodeContext,
        router: LLMRouter,
        harness: Optional[HarnessInfo] = None,
    ) -> list[PoCBytes]:
        print(f"[Strategy] Running '{self.name}' ...")
        results: list[PoCBytes] = []

        # Initial generation
        initial = LLMScriptStrategy()
        initial_results = initial.generate(context, router, harness)

        if not initial_results:
            return results

        best = initial_results[0]
        results.append(best)

        if not self._run_callback:
            return results

        for iteration in range(1, self._MAX_ITERATIONS + 1):
            print(f"  [{self.name}] Iteration {iteration}/{self._MAX_ITERATIONS}")

            triggered, output = self._run_callback(best)
            if triggered:
                print(f"  [{self.name}] Triggered on iteration {iteration}!")
                best.confidence = 0.9
                return results

            # Feed output back to LLM
            refine_prompt = (
                f"## Vulnerability Description\n{context.description}\n\n"
                f"## Previous PoC\nGenerated {len(best.data)} bytes using this script:\n"
                f"```python\n{best.generator_script or 'N/A'}\n```\n\n"
                f"## Execution Result\n```\n{output[:3000]}\n```\n\n"
            )
            if harness:
                refine_prompt += f"## Fuzz Harness\n```c\n{harness.harness_code[:3000]}\n```\n\n"

            refine_prompt += (
                "The previous PoC did NOT trigger the vulnerability.\n"
                "Based on the execution output, adjust the byte payload.\n"
                "Write an improved Python script that generates better crafted bytes.\n"
                "Use sys.stdout.buffer.write() to output the bytes.\n"
            )

            raw = router.chat(
                system_prompt=SYSTEM_PROMPT_BYTE_GENERATOR,
                user_prompt=refine_prompt,
                max_tokens=cfg.MAX_TOKENS,
                temperature=0.2,
            )

            if not raw:
                break

            script = _extract_python_block(raw)
            if not script:
                break

            poc_data = _run_python_script(script)
            if poc_data and len(poc_data) > 0:
                poc_path = _save_poc_bytes(poc_data, context.task.task_id,
                                           f"{self.name}_iter{iteration}")
                best = PoCBytes(
                    strategy_name=f"{self.name}_iter{iteration}",
                    data=poc_data,
                    poc_path=poc_path,
                    confidence=0.6 + iteration * 0.05,
                    notes=f"Refined iteration {iteration}: {len(poc_data)} bytes",
                    generator_script=script,
                )
                results.append(best)

        return results


# ── Orchestrator ───────────────────────────────────────────────────────────

class ByteOrchestrator:
    """Run byte generation strategies and collect results."""

    def __init__(
        self,
        router: LLMRouter,
        run_callback=None,
    ) -> None:
        self.router = router
        self.strategies: list[ByteStrategy] = [
            AnalyzeThenGenerateStrategy(),
            LLMScriptStrategy(),
            CorpusMutationStrategy(),
            IterativeRefineStrategy(run_callback=run_callback),
            SimpleBytePatterns(),
        ]

    def run(
        self,
        context: CodeContext,
        harness: Optional[HarnessInfo] = None,
    ) -> list[PoCBytes]:
        """Execute all strategies, return combined results sorted by confidence."""
        all_results: list[PoCBytes] = []

        for strategy in self.strategies:
            print(f"\n{'='*50}")
            print(f"  ByteOrchestrator → strategy '{strategy.name}'")
            print(f"{'='*50}")
            try:
                results = strategy.generate(context, self.router, harness)
                all_results.extend(results)
                print(f"  → {len(results)} PoC(s) from '{strategy.name}'")
            except Exception:
                logger.exception(
                    "Strategy '%s' raised an exception — skipping.", strategy.name
                )

        # Sort by confidence descending
        all_results.sort(key=lambda r: r.confidence, reverse=True)

        print(f"\n[ByteOrchestrator] Collected {len(all_results)} PoC(s) total:")
        for r in all_results[:10]:
            print(f"  • {r.strategy_name:30s}  conf={r.confidence:.2f}  "
                  f"size={len(r.data):>8d} bytes  → {r.poc_path}")

        return all_results
