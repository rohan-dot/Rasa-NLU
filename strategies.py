"""
strategies.py — LLM-driven bug-finding strategies.

Implements the core techniques used by AIxCC finalist CRS systems:

1. SOURCE CODE AUDIT — LLM reads source code like a security researcher,
   identifies vulnerability patterns (off-by-one, integer overflow, use-after-free,
   missing bounds checks, etc.) and generates PoC inputs directly.

2. HARNESS GENERATION — LLM reads the target library's API and generates
   new LibFuzzer harnesses targeting different attack surfaces.

3. COVERAGE-GUIDED SEED GENERATION — Reads LibFuzzer's coverage stats and
   asks the LLM to generate inputs targeting uncovered branches.

4. VARIANT ANALYSIS — When a crash is found, LLM searches for the same
   vulnerability pattern elsewhere in the codebase.

5. DIRECT POC GENERATION — LLM identifies a specific code path and crafts
   a byte-level input designed to reach and trigger it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from llm_client import VLLMClient

logger = logging.getLogger("gemma-fuzzer.strategies")


# ══════════════════════════════════════════════════════════════════
# Strategy 1: SOURCE CODE AUDIT
# ══════════════════════════════════════════════════════════════════

SOURCE_AUDIT_SYSTEM = """\
You are an elite security researcher performing a code audit.
Analyze the provided C/C++ source code for security vulnerabilities.

Look specifically for:
- Buffer overflows (stack and heap)
- Off-by-one errors in array/pointer access
- Integer overflows/underflows that affect allocation sizes or loop bounds
- Use-after-free / double-free
- Null pointer dereferences from unchecked returns
- Format string vulnerabilities
- Missing bounds checks on user-controlled input
- Type confusion or incorrect casts
- Race conditions in shared state
- Uninitialized variable usage

For each potential vulnerability found, respond with a JSON array:
[{
  "function": "function_name",
  "file": "filename.c",
  "line_hint": 42,
  "bug_type": "heap-buffer-overflow",
  "description": "The loop at line 42 uses <= instead of <, reading one byte past the buffer",
  "severity": "high",
  "trigger_hint": "provide input longer than 1024 bytes with a null byte at position 1023"
}]

If no vulnerabilities are found, return an empty array: []
Your response must start with [ and end with ] — nothing else."""


def strategy_source_audit(
    llm: VLLMClient,
    src_dir: str,
    output_dir: str,
    harness_name: str,
) -> list[dict]:
    """
    Read source files and ask the LLM to audit them for vulnerabilities.
    Returns list of potential vulnerability reports.
    """
    if not llm.is_available():
        return []

    src_path = Path(src_dir)
    bugs_dir = Path(output_dir) / "bugs"
    bugs_dir.mkdir(parents=True, exist_ok=True)

    # Collect relevant source files (prioritize files related to the harness)
    source_files = _collect_source_files(src_path, harness_name)
    if not source_files:
        logger.warning("No source files found for audit in %s", src_dir)
        return []

    all_findings: list[dict] = []

    for src_file, content in source_files[:8]:  # audit up to 8 files per round
        logger.info("[audit] Analyzing: %s (%d bytes)", src_file, len(content))

        response = llm.chat(
            system=SOURCE_AUDIT_SYSTEM,
            user=f"File: {src_file}\n\n```c\n{content[:8000]}\n```",
            max_tokens=2000,
            temperature=0.2,
        )
        if not response:
            continue

        findings = _parse_json_array(response)
        for finding in findings:
            finding["source_file"] = src_file
            finding["strategy"] = "source_audit"
            finding["timestamp"] = time.time()

            # Write as bug-candidate
            fhash = hashlib.sha256(
                json.dumps(finding, sort_keys=True).encode()
            ).hexdigest()[:12]
            bug_path = bugs_dir / f"audit-{fhash}.json"
            bug_path.write_text(json.dumps(finding, indent=2))
            logger.info(
                "[audit] FINDING: %s in %s — %s",
                finding.get("bug_type", "?"),
                finding.get("function", "?"),
                finding.get("description", "")[:80],
            )
            all_findings.append(finding)

    return all_findings


# ══════════════════════════════════════════════════════════════════
# Strategy 2: HARNESS GENERATION
# ══════════════════════════════════════════════════════════════════

HARNESS_GEN_SYSTEM = """\
You are a fuzzing engineer. Given C/C++ library header files and source code,
generate a NEW LibFuzzer harness that targets a different API function than
the one already being fuzzed.

The harness must:
1. Include the correct headers
2. Implement `int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)`
3. Use the fuzz input (data, size) to exercise the target function
4. Handle edge cases (null checks, minimum size checks)
5. Free any allocated resources
6. Return 0

Focus on functions that:
- Parse or process external input (files, network data, strings)
- Handle complex data structures
- Have multiple code paths based on input

Respond with ONLY the C source code. No markdown fences, no explanation.
Start directly with #include."""


def strategy_generate_harness(
    llm: VLLMClient,
    src_dir: str,
    build_dir: str,
    output_dir: str,
    harness_name: str,
    existing_harnesses: list[str],
) -> str | None:
    """
    Generate a new LibFuzzer harness targeting a different API surface.
    Compiles and returns the path to the binary, or None on failure.
    """
    if not llm.is_available():
        return None

    src_path = Path(src_dir)
    source_files = _collect_source_files(src_path, harness_name)
    headers = _collect_headers(src_path)

    if not headers and not source_files:
        return None

    # Build context for the LLM
    context = f"Already fuzzing: {harness_name}\n"
    if existing_harnesses:
        context += f"Other existing harnesses: {', '.join(existing_harnesses)}\n"
    context += "\nAvailable headers:\n"
    for hdr, content in headers[:3]:
        context += f"\n// {hdr}\n{content[:2000]}\n"

    if source_files:
        context += "\nKey source files:\n"
        for sf, content in source_files[:2]:
            context += f"\n// {sf}\n{content[:2000]}\n"

    response = llm.chat(
        system=HARNESS_GEN_SYSTEM,
        user=context,
        max_tokens=2000,
        temperature=0.4,
    )

    if not response:
        return None

    # Clean up response
    code = response.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]

    # Write the harness
    harness_dir = Path(output_dir) / "generated_harnesses"
    harness_dir.mkdir(parents=True, exist_ok=True)

    harness_hash = hashlib.sha256(code.encode()).hexdigest()[:8]
    harness_src = harness_dir / f"harness_{harness_hash}.c"
    harness_bin = harness_dir / f"harness_{harness_hash}"
    harness_src.write_text(code)

    # Try to compile it
    compile_cmd = [
        "clang", "-g", "-O1", "-fsanitize=address,fuzzer",
        "-I/usr/include/libxml2",
        "-I", str(src_path),
        str(harness_src),
        "-lxml2", "-lz", "-llzma",
        "-o", str(harness_bin),
    ]

    logger.info("[harness-gen] Compiling: %s", harness_src.name)
    result = subprocess.run(
        compile_cmd,
        capture_output=True, text=True, timeout=30,
    )

    if result.returncode != 0:
        logger.warning(
            "[harness-gen] Compilation failed: %s", result.stderr[:300]
        )
        # Try to fix with LLM
        fixed = _try_fix_compilation(llm, code, result.stderr)
        if fixed:
            harness_src.write_text(fixed)
            result = subprocess.run(
                compile_cmd, capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                logger.warning("[harness-gen] Fix attempt also failed.")
                return None
        else:
            return None

    logger.info("[harness-gen] SUCCESS: %s", harness_bin)
    return str(harness_bin)


def _try_fix_compilation(
    llm: VLLMClient, code: str, error: str
) -> str | None:
    """Ask LLM to fix a compilation error."""
    response = llm.chat(
        system=(
            "Fix the compilation error in this C fuzzer harness. "
            "Respond with ONLY the corrected C code, no markdown, "
            "no explanation. Start with #include."
        ),
        user=f"Code:\n{code}\n\nError:\n{error[:500]}",
        max_tokens=2000,
        temperature=0.1,
    )
    if not response:
        return None
    fixed = response.strip()
    if fixed.startswith("```"):
        fixed = fixed.split("\n", 1)[1]
    if fixed.endswith("```"):
        fixed = fixed.rsplit("```", 1)[0]
    return fixed


# ══════════════════════════════════════════════════════════════════
# Strategy 3: COVERAGE-GUIDED SEED GENERATION
# ══════════════════════════════════════════════════════════════════

COVERAGE_SEED_SYSTEM = """\
You are a fuzzing expert analyzing coverage data. Given:
1. Source code of the target
2. Coverage information showing which branches/lines are NOT yet covered
3. Any crashes found so far

Generate 5 raw input byte sequences (as hex strings) specifically designed
to reach the UNCOVERED code paths. Think about what input would cause
execution to take the untaken branches.

For XML parsers, consider:
- Different XML features (DTDs, namespaces, CDATA, processing instructions)
- Malformed XML that triggers error-handling paths
- Edge cases in encoding declarations
- Very deeply nested elements
- Entity expansion

Respond with ONLY a JSON array:
[{"hex": "3c3f786d6c...", "target": "uncovered function/branch", "rationale": "why this reaches it"}]

Your response must start with [ and end with ] — nothing else."""


def strategy_coverage_seeds(
    llm: VLLMClient,
    src_dir: str,
    output_dir: str,
    harness_name: str,
    corpus_dir: str,
    crash_summaries: list[str],
) -> int:
    """
    Analyze coverage gaps and generate targeted seeds.
    Returns count of seeds generated.
    """
    if not llm.is_available():
        return 0

    seeds_dir = Path(output_dir) / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    # Get corpus stats for context
    corpus_path = Path(corpus_dir)
    corpus_count = len(list(corpus_path.glob("*"))) if corpus_path.exists() else 0
    corpus_sizes = []
    if corpus_path.exists():
        for f in list(corpus_path.iterdir())[:10]:
            if f.is_file():
                corpus_sizes.append(f.stat().st_size)

    # Read some source code for context
    source_files = _collect_source_files(Path(src_dir), harness_name)

    user_msg = f"Target: {harness_name}\n"
    user_msg += f"Corpus size: {corpus_count} inputs\n"
    if corpus_sizes:
        user_msg += f"Input size range: {min(corpus_sizes)}-{max(corpus_sizes)} bytes\n"
    user_msg += "\n"

    if crash_summaries:
        user_msg += "Crashes found (already covered paths):\n"
        for s in crash_summaries[:5]:
            user_msg += f"- {s}\n"
        user_msg += "\nGenerate inputs targeting DIFFERENT code paths.\n\n"

    if source_files:
        user_msg += "Target source code (look for uncovered branches):\n"
        for sf, content in source_files[:2]:
            user_msg += f"\n// {sf}\n{content[:3000]}\n"

    response = llm.chat(
        system=COVERAGE_SEED_SYSTEM,
        user=user_msg,
        max_tokens=1000,
        temperature=0.6,
    )

    if not response:
        return 0

    seeds = _parse_json_array(response)
    count = 0
    for seed_info in seeds:
        hex_str = seed_info.get("hex", "")
        if not hex_str:
            continue
        try:
            data = bytes.fromhex(hex_str)
        except ValueError:
            continue

        seed_hash = hashlib.sha256(data).hexdigest()[:12]
        seed_path = seeds_dir / f"cov-seed-{seed_hash}"
        seed_path.write_bytes(data)
        count += 1
        logger.debug(
            "[cov-seed] %s (%d bytes) → %s",
            seed_path.name, len(data),
            seed_info.get("target", ""),
        )

    logger.info("[cov-seed] Generated %d coverage-targeted seeds.", count)
    return count


# ══════════════════════════════════════════════════════════════════
# Strategy 4: VARIANT ANALYSIS
# ══════════════════════════════════════════════════════════════════

VARIANT_ANALYSIS_SYSTEM = """\
You are a security researcher doing variant analysis. A vulnerability was
found in one function. Your job is to find the SAME pattern in other
functions in the codebase.

Given:
1. The original crash (type, function, root cause)
2. Source code from other files in the same project

Search for functions that have the same vulnerability pattern. For example:
- If the original bug is an off-by-one in a loop, find other loops with
  the same off-by-one pattern
- If the bug is a missing null check after malloc, find other malloc calls
  without null checks
- If the bug is an integer overflow in a size calculation, find similar
  size calculations

For each variant found, respond with a JSON array:
[{
  "function": "other_function_name",
  "file": "other_file.c",
  "original_bug": "the pattern you matched",
  "description": "why this is the same bug pattern",
  "trigger_hint": "how to trigger this variant"
}]

Your response must start with [ and end with ] — nothing else."""


def strategy_variant_analysis(
    llm: VLLMClient,
    src_dir: str,
    output_dir: str,
    crash_info: dict,
) -> list[dict]:
    """
    Given a crash, search for the same vulnerability pattern elsewhere.
    Returns list of variant findings.
    """
    if not llm.is_available():
        return []

    bugs_dir = Path(output_dir) / "bugs"
    bugs_dir.mkdir(parents=True, exist_ok=True)

    crash_desc = (
        f"Bug type: {crash_info.get('crash_type', 'unknown')}\n"
        f"Function: {crash_info.get('affected_function', 'unknown')}\n"
        f"Root cause: {crash_info.get('root_cause', 'unknown')}\n"
    )

    # Read source files to search for variants
    src_path = Path(src_dir)
    all_sources = _collect_source_files(src_path, "", max_files=10)

    all_variants: list[dict] = []

    for src_file, content in all_sources:
        response = llm.chat(
            system=VARIANT_ANALYSIS_SYSTEM,
            user=(
                f"## Original Vulnerability\n{crash_desc}\n"
                f"## Code to Search\nFile: {src_file}\n"
                f"```c\n{content[:5000]}\n```"
            ),
            max_tokens=1000,
            temperature=0.2,
        )
        if not response:
            continue

        variants = _parse_json_array(response)
        for variant in variants:
            variant["strategy"] = "variant_analysis"
            variant["original_crash"] = crash_info.get("crash_type", "unknown")
            variant["timestamp"] = time.time()

            vhash = hashlib.sha256(
                json.dumps(variant, sort_keys=True).encode()
            ).hexdigest()[:12]
            bug_path = bugs_dir / f"variant-{vhash}.json"
            bug_path.write_text(json.dumps(variant, indent=2))
            logger.info(
                "[variant] Found variant in %s:%s — %s",
                variant.get("file", "?"),
                variant.get("function", "?"),
                variant.get("description", "")[:80],
            )
            all_variants.append(variant)

    return all_variants


# ══════════════════════════════════════════════════════════════════
# Strategy 5: DIRECT POC GENERATION
# ══════════════════════════════════════════════════════════════════

POC_GEN_SYSTEM = """\
You are a security researcher crafting proof-of-concept inputs.
Given source code of a target program that parses input, write a Python
script that GENERATES a malicious input file.

CRITICAL: Write a Python script, NOT hex bytes. The script must write
the malicious input to "/tmp/poc_input".

Think step by step:
1. Identify the input format the program expects (e.g., XML)
2. Trace how input flows through the code
3. Find a path that leads to a crash
4. Write Python code that constructs the exact bytes needed

For XML parser targets, consider generating:
- XML with extremely long attribute values (b"A" * 100000)
- XML with recursive entity definitions causing exponential expansion
- XML with deeply nested elements (b"<a>" * 10000)
- XML with invalid UTF-8 sequences
- XML with very long namespace prefixes (b"<" + b"A"*65536 + b":x/>")
- Truncated XML at critical parse points
- XML with null bytes: b"<root>\\x00</root>"

Respond with ONLY the Python script. No markdown fences.
Start with a comment. The script must write to "/tmp/poc_input"."""


def strategy_direct_poc(
    llm: VLLMClient,
    src_dir: str,
    output_dir: str,
    harness_name: str,
) -> int:
    """
    LLM writes Python scripts that generate PoC inputs.
    Returns count of PoCs generated.
    """
    if not llm.is_available():
        return 0

    pov_dir = Path(output_dir) / "povs"
    scripts_dir = Path(output_dir) / "exploit_scripts"
    pov_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    source_files = _collect_source_files(Path(src_dir), harness_name)

    user_msg = f"Target harness: {harness_name}\n\n"
    if source_files:
        user_msg += "Source code:\n"
        for sf, content in source_files[:3]:
            user_msg += f"\n// {sf}\n{content[:4000]}\n"

    response = llm.chat(
        system=POC_GEN_SYSTEM,
        user=user_msg,
        max_tokens=2000,
        temperature=0.5,
    )

    if not response:
        return 0

    # Clean script
    script = response.strip()
    if script.startswith("```"):
        script = script.split("\n", 1)[1] if "\n" in script else script[3:]
    if script.endswith("```"):
        script = script.rsplit("```", 1)[0]

    # Save and run the script
    script_hash = hashlib.sha256(script.encode()).hexdigest()[:8]
    script_path = scripts_dir / f"poc_gen_{script_hash}.py"
    script_path.write_text(script)

    data = _run_poc_script(str(script_path))
    if data is None:
        logger.warning("[poc-gen] Script failed to generate input.")
        return 0

    poc_hash = hashlib.sha256(data).hexdigest()[:12]
    poc_path = pov_dir / f"script-poc-{poc_hash}"
    poc_path.write_bytes(data)
    logger.info("[poc-gen] Generated %d byte PoC via script.", len(data))
    return 1


def _run_poc_script(script_path: str) -> bytes | None:
    """Run a Python PoC script and return the generated input."""
    poc_path = "/tmp/poc_input"
    try:
        os.unlink(poc_path)
    except FileNotFoundError:
        pass

    try:
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True, timeout=30, text=True,
        )
        if result.returncode != 0:
            logger.warning("[poc-script] Error: %s", result.stderr[:200])
            return None
        if not os.path.exists(poc_path):
            return None
        data = Path(poc_path).read_bytes()
        return data if len(data) > 0 else None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
# STRATEGY ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

@dataclass
class StrategyResult:
    """Result from a strategy execution."""
    strategy_name: str
    findings: int
    details: list[dict]
    elapsed: float


class StrategyOrchestrator:
    """
    Rotates through LLM strategies on a schedule.
    Each strategy gets a turn, results feed into the next round.
    
    v0.3: Added codebase mapping, cross-file audit, PoC verification loop.
    """

    def __init__(
        self,
        llm: VLLMClient,
        src_dir: str,
        build_dir: str,
        output_dir: str,
        harness_name: str,
    ):
        self.llm = llm
        self.src_dir = src_dir
        self.build_dir = build_dir
        self.output_dir = output_dir
        self.harness_name = harness_name
        self.crash_summaries: list[str] = []
        self.crash_reports: list[dict] = []
        self.generated_harnesses: list[str] = []
        self.round_number = 0
        self.results_log: list[StrategyResult] = []
        # NEW: codebase intelligence (populated on first round)
        self.codebase_map: dict | None = None
        self.risky_files: list[tuple[str, str, int]] = []  # (filename, content, risk_score)
        self.binary_path: str | None = None  # for PoC verification

    def run_round(self, corpus_dir: str) -> list[StrategyResult]:
        """
        Run one round of strategies.
        Round 1 builds the codebase map. All subsequent rounds use it.
        """
        self.round_number += 1
        results: list[StrategyResult] = []
        logger.info(
            "═══ Strategy Round %d ═══════════════════════════════",
            self.round_number,
        )

        # ── ROUND 1 ONLY: Build codebase intelligence ──
        if self.round_number == 1:
            # Step 1: Fast regex pre-scan (no LLM needed)
            r = self._run_strategy(
                "prescan",
                lambda: self._do_prescan(),
            )
            results.append(r)

            # Step 2: LLM builds codebase map from pre-scan results
            r = self._run_strategy(
                "codebase_map",
                lambda: self._do_codebase_map(),
            )
            results.append(r)

        # ── EVERY ROUND: Cross-file audit (uses codebase map) ──
        if self.round_number <= 3 or self.round_number % 3 == 0:
            r = self._run_strategy(
                "cross_file_audit",
                lambda: self._do_cross_file_audit(),
            )
            results.append(r)

        # ── Harness Generation (rounds 1 and 5) ──
        if self.round_number in (1, 5):
            r = self._run_strategy(
                "harness_gen",
                lambda: self._do_harness_gen(),
            )
            results.append(r)

        # ── Coverage-Guided Seeds (every round) ──
        r = self._run_strategy(
            "coverage_seeds",
            lambda: strategy_coverage_seeds(
                self.llm, self.src_dir, self.output_dir,
                self.harness_name, corpus_dir, self.crash_summaries,
            ),
        )
        results.append(r)

        # ── Variant Analysis (when we have crashes) ──
        if self.crash_reports:
            latest_crash = self.crash_reports[-1]
            r = self._run_strategy(
                "variant_analysis",
                lambda: strategy_variant_analysis(
                    self.llm, self.src_dir, self.output_dir, latest_crash,
                ),
            )
            results.append(r)

        # ── PoC Generation + Verification Loop (every 2nd round) ──
        if self.round_number % 2 == 0:
            r = self._run_strategy(
                "poc_verify",
                lambda: self._do_poc_with_verification(),
            )
            results.append(r)

        self.results_log.extend(results)
        total = sum(r.findings for r in results)
        logger.info(
            "═══ Round %d complete: %d total findings ═══════════",
            self.round_number, total,
        )
        return results

    def add_crash(self, crash_type: str, report: dict | None) -> None:
        """Register a new crash for use by variant analysis."""
        self.crash_summaries.append(crash_type)
        if report:
            self.crash_reports.append(report)

    def _do_harness_gen(self) -> list[dict]:
        """Wrapper for harness generation that returns findings list."""
        result = strategy_generate_harness(
            self.llm, self.src_dir, self.build_dir, self.output_dir,
            self.harness_name, self.generated_harnesses,
        )
        if result:
            self.generated_harnesses.append(result)
            return [{"harness": result}]
        return []

    # ── NEW: Codebase Pre-scan (fast, no LLM) ────────────────────

    def _do_prescan(self) -> list[dict]:
        """Fast regex scan of ALL source files for dangerous patterns."""
        import re
        DANGER_PATTERNS = [
            (r'\bmemcpy\s*\(', "memcpy", 3),
            (r'\bstrcpy\s*\(', "strcpy", 4),
            (r'\bsprintf\s*\(', "sprintf", 4),
            (r'\bmalloc\s*\(', "malloc", 2),
            (r'\brealloc\s*\(', "realloc", 3),
            (r'\bfree\s*\(', "free", 2),
            (r'\bstrcat\s*\(', "strcat", 4),
            (r'\bstrncpy\s*\(', "strncpy", 2),
            (r'\bmemmove\s*\(', "memmove", 2),
            (r'\batoi\s*\(', "atoi", 3),
            (r'\bsscanf\s*\(', "sscanf", 3),
            (r'\bgets\s*\(', "gets", 5),
            (r'(\w+)\s*\+\+\s*.*<.*len', "loop_bound", 3),
            (r'if\s*\(\s*\w+\s*<\s*0', "signed_check", 2),
            (r'\(.*\)\s*\*\s*sizeof', "size_calc", 3),
        ]
        
        src_path = Path(self.src_dir)
        file_risks: list[tuple[str, str, int, list[str]]] = []
        
        for f in src_path.rglob("*.c"):
            fstr = str(f).lower()
            if any(skip in fstr for skip in ["test", ".git", "example", "python"]):
                continue
            try:
                content = f.read_text(errors="replace")
            except Exception:
                continue
            
            risk_score = 0
            found_patterns: list[str] = []
            for pattern, name, weight in DANGER_PATTERNS:
                matches = len(re.findall(pattern, content))
                if matches > 0:
                    risk_score += matches * weight
                    found_patterns.append(f"{name}({matches})")
            
            if risk_score > 0:
                rel_name = str(f.relative_to(src_path))
                file_risks.append((rel_name, content, risk_score, found_patterns))
        
        # Sort by risk score descending
        file_risks.sort(key=lambda x: -x[2])
        
        # Store top risky files for other strategies to use
        self.risky_files = [
            (name, content, score)
            for name, content, score, _ in file_risks[:15]
        ]
        
        # Log results
        findings = []
        for name, _, score, patterns in file_risks[:10]:
            logger.info(
                "[prescan] %s — risk=%d [%s]",
                name, score, ", ".join(patterns[:5]),
            )
            findings.append({"file": name, "risk_score": score, "patterns": patterns})
        
        logger.info("[prescan] Scanned %d files, %d with risk patterns.", 
                    len(file_risks) + sum(1 for _ in src_path.rglob("*.c")), len(file_risks))
        return findings

    # ── NEW: Codebase Map (LLM identifies attack surfaces) ────────

    def _do_codebase_map(self) -> list[dict]:
        """LLM reads the riskiest files' function signatures and identifies
        the most dangerous code paths to audit in depth."""
        if not self.llm.is_available() or not self.risky_files:
            return []

        # Build a summary of the riskiest files
        summary = "These are the riskiest source files based on dangerous pattern density:\n\n"
        for name, content, score in self.risky_files[:10]:
            # Extract function signatures
            funcs = _extract_function_sigs(content)
            summary += f"## {name} (risk_score={score})\n"
            summary += f"Functions: {', '.join(funcs[:15])}\n"
            # Show first 500 chars for context
            summary += f"Preview:\n{content[:500]}\n\n"

        response = self.llm.chat(
            system="""\
You are a security researcher planning a code audit. Given a summary of 
source files ranked by dangerous-pattern density, identify the TOP 5 most 
likely vulnerable functions and explain WHY, tracing the data flow.

For each, specify:
- Which file and function to audit deeply
- What type of bug is likely (overflow, use-after-free, integer overflow, etc.)
- How external input reaches this code path

Respond with ONLY a JSON array:
[{"file": "dict.c", "function": "xmlDictAddQString", "bug_type": "integer overflow",
  "data_flow": "XML input → parser → dict lookup → size calculation without overflow check",
  "audit_priority": "critical"}]

Your response must start with [ and end with ] — nothing else.""",
            user=summary,
            max_tokens=2000,
            temperature=0.2,
        )

        if not response:
            return []

        targets = _parse_json_array(response)
        self.codebase_map = {"targets": targets, "risky_files": [
            (n, s) for n, _, s in self.risky_files[:10]
        ]}

        for t in targets:
            logger.info(
                "[codebase-map] AUDIT TARGET: %s:%s — %s (%s)",
                t.get("file", "?"), t.get("function", "?"),
                t.get("bug_type", "?"), t.get("audit_priority", "?"),
            )

        # Write map to disk for debugging
        map_path = Path(self.output_dir) / "codebase_map.json"
        map_path.write_text(json.dumps(targets, indent=2))

        return targets

    # ── NEW: Cross-file Audit (traces data flow across files) ─────

    def _do_cross_file_audit(self) -> list[dict]:
        """Send pairs of related files to the LLM together, 
        tracing data flow across file boundaries."""
        if not self.llm.is_available():
            return []

        bugs_dir = Path(self.output_dir) / "bugs"
        bugs_dir.mkdir(parents=True, exist_ok=True)

        # Use codebase map targets if available, otherwise fall back
        if self.codebase_map and self.codebase_map.get("targets"):
            targets = self.codebase_map["targets"]
        else:
            # Fall back to top risky files
            targets = [{"file": name} for name, _, _ in self.risky_files[:5]]

        all_findings: list[dict] = []
        src_path = Path(self.src_dir)

        for target in targets[:5]:
            target_file = target.get("file", "")
            target_func = target.get("function", "")
            
            # Find the target file
            matches = list(src_path.rglob(target_file))
            if not matches:
                # Try partial match
                matches = [f for f in src_path.rglob("*.c") 
                           if target_file.lower() in f.name.lower()]
            if not matches:
                continue

            main_file = matches[0]
            try:
                main_content = main_file.read_text(errors="replace")
            except Exception:
                continue

            # Find files that call functions in the target file
            # or that the target file calls into
            related_content = ""
            func_names = _extract_function_sigs(main_content)
            for other_file in src_path.rglob("*.c"):
                if other_file == main_file:
                    continue
                try:
                    other_text = other_file.read_text(errors="replace")
                    # Check if this file calls any function defined in main_file
                    for fn in func_names[:10]:
                        if fn in other_text:
                            rel_name = str(other_file.relative_to(src_path))
                            # Extract just the calling context (200 chars around the call)
                            idx = other_text.find(fn)
                            start = max(0, idx - 200)
                            end = min(len(other_text), idx + 200)
                            related_content += f"\n// Caller in {rel_name}:\n{other_text[start:end]}\n"
                            break
                except Exception:
                    continue
                if len(related_content) > 3000:
                    break

            user_msg = f"AUDIT TARGET: {target_file}"
            if target_func:
                user_msg += f" function {target_func}"
            if target.get("bug_type"):
                user_msg += f"\nSuspected bug type: {target['bug_type']}"
            if target.get("data_flow"):
                user_msg += f"\nData flow: {target['data_flow']}"
            user_msg += f"\n\n## Main file: {target_file}\n```c\n{main_content[:6000]}\n```"
            if related_content:
                user_msg += f"\n\n## Callers (how external input reaches this code):\n```c\n{related_content[:3000]}\n```"

            logger.info("[cross-audit] Auditing %s with %d bytes of caller context",
                       target_file, len(related_content))

            response = self.llm.chat(
                system=SOURCE_AUDIT_SYSTEM,
                user=user_msg,
                max_tokens=2000,
                temperature=0.2,
            )
            if not response:
                continue

            findings = _parse_json_array(response)
            for finding in findings:
                finding["source_file"] = target_file
                finding["strategy"] = "cross_file_audit"
                finding["data_flow_context"] = bool(related_content)
                finding["timestamp"] = time.time()

                fhash = hashlib.sha256(
                    json.dumps(finding, sort_keys=True).encode()
                ).hexdigest()[:12]
                bug_path = bugs_dir / f"xaudit-{fhash}.json"
                bug_path.write_text(json.dumps(finding, indent=2))
                logger.info(
                    "[cross-audit] FINDING: %s in %s:%s — %s",
                    finding.get("bug_type", "?"),
                    target_file,
                    finding.get("function", "?"),
                    finding.get("description", "")[:80],
                )
                all_findings.append(finding)

        return all_findings

    # ── NEW: PoC Generation + Verification Loop ───────────────────

    def _do_poc_with_verification(self) -> list[dict]:
        """Generate PoC scripts, run them, test output, iterate with feedback."""
        if not self.llm.is_available():
            return []

        pov_dir = Path(self.output_dir) / "povs"
        scripts_dir = Path(self.output_dir) / "exploit_scripts"
        pov_dir.mkdir(parents=True, exist_ok=True)
        scripts_dir.mkdir(parents=True, exist_ok=True)

        binary = self._find_binary()
        if not binary:
            logger.warning("[poc-verify] No binary found, falling back to basic PoC gen")
            count = strategy_direct_poc(
                self.llm, self.src_dir, self.output_dir, self.harness_name
            )
            return [{"generated": count}] if count else []

        # Get source context
        source_context = ""
        if self.risky_files:
            for name, content, _ in self.risky_files[:3]:
                source_context += f"\n// {name}\n{content[:3000]}\n"
        else:
            source_files = _collect_source_files(Path(self.src_dir), self.harness_name)
            for sf, content in source_files[:3]:
                source_context += f"\n// {sf}\n{content[:3000]}\n"

        all_verified: list[dict] = []
        MAX_ATTEMPTS = 3
        feedback = ""

        for attempt in range(MAX_ATTEMPTS):
            user_msg = f"Target: {self.harness_name} (attempt {attempt + 1}/{MAX_ATTEMPTS})\n"
            if feedback:
                user_msg += f"\nPrevious attempt feedback:\n{feedback}\n"
            user_msg += f"\nSource:\n```c\n{source_context[:5000]}\n```"

            system = POC_GEN_SYSTEM
            if attempt > 0:
                system += (
                    "\n\nYour previous script did NOT crash the target. "
                    "Try a COMPLETELY DIFFERENT approach. "
                    "Consider larger sizes, different XML structures, or "
                    "targeting a different code path."
                )

            response = self.llm.chat(
                system=system,
                user=user_msg,
                max_tokens=2000,
                temperature=0.4 + (attempt * 0.2),
            )
            if not response:
                continue

            # Clean script
            script = response.strip()
            if script.startswith("```"):
                script = script.split("\n", 1)[1] if "\n" in script else script[3:]
            if script.endswith("```"):
                script = script.rsplit("```", 1)[0]

            script_hash = hashlib.sha256(script.encode()).hexdigest()[:8]
            script_path = scripts_dir / f"pocv_{script_hash}.py"
            script_path.write_text(script)

            # Run script to generate input
            data = _run_poc_script(str(script_path))
            if data is None:
                feedback = "Script failed to run or didn't create /tmp/poc_input"
                continue

            logger.info("[poc-verify] Script generated %d byte input.", len(data))

            # Test against binary
            crashed, output = _run_poc_against_binary(binary, data)

            if crashed:
                poc_hash = hashlib.sha256(data).hexdigest()[:12]
                poc_path = pov_dir / f"verified-poc-{poc_hash}"
                poc_path.write_bytes(data)
                logger.info(
                    "[poc-verify] *** VERIFIED CRASH *** %s (%d bytes)",
                    poc_path.name, len(data),
                )
                all_verified.append({
                    "poc_file": str(poc_path),
                    "crashed": True,
                    "size": len(data),
                    "output": output[:500],
                })
                break
            else:
                feedback = (
                    f"Script generated {len(data)} bytes but NO crash. "
                    f"Binary output: {output[:200]}"
                )

        logger.info("[poc-verify] %d verified crashes from %d attempts.",
                   len(all_verified), MAX_ATTEMPTS)
        return all_verified

    def _find_binary(self) -> str | None:
        """Find the fuzzer binary."""
        if self.binary_path:
            return self.binary_path
        build_path = Path(self.build_dir)
        for candidate in [
            build_path / self.harness_name,
            build_path / f"{self.harness_name}_fuzzer",
        ]:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                self.binary_path = str(candidate)
                return self.binary_path
        # Glob
        for g in build_path.glob(f"*{self.harness_name}*"):
            if g.is_file() and os.access(g, os.X_OK):
                self.binary_path = str(g)
                return self.binary_path
        return None

    def _run_strategy(
        self, name: str, func, 
    ) -> StrategyResult:
        """Run a strategy with timing and error handling."""
        logger.info("[strategy] Running: %s", name)
        t0 = time.monotonic()
        try:
            result = func()
            if isinstance(result, int):
                findings = result
                details = []
            elif isinstance(result, list):
                findings = len(result)
                details = result
            else:
                findings = 0
                details = []
        except Exception as exc:
            logger.error("[strategy] %s failed: %s", name, exc)
            findings = 0
            details = []

        elapsed = time.monotonic() - t0
        logger.info(
            "[strategy] %s → %d findings (%.1fs)",
            name, findings, elapsed,
        )
        return StrategyResult(
            strategy_name=name,
            findings=findings,
            details=details,
            elapsed=elapsed,
        )

    def get_generated_harnesses(self) -> list[str]:
        """Return paths to all successfully compiled harnesses."""
        return list(self.generated_harnesses)


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _collect_source_files(
    src_path: Path,
    harness_name: str,
    max_files: int = 8,
) -> list[tuple[str, str]]:
    """Collect and prioritize source files for security auditing.
    
    IMPORTANT: Prioritizes .c implementation files over headers.
    Headers have declarations, not bugs. Implementation has the bugs.
    Also boosts files containing security-relevant patterns.
    """
    files: list[tuple[str, str, int]] = []

    # Security-relevant keywords — files containing these are more
    # likely to have vulnerabilities
    SEC_KEYWORDS = [
        "malloc", "realloc", "calloc", "free",
        "memcpy", "memmove", "strcpy", "strncpy", "strcat",
        "sprintf", "snprintf", "sscanf",
        "strlen", "sizeof",
        "buffer", "overflow", "bounds",
        "size", "length", "count", "nargs",
    ]

    for ext in ("*.c", "*.cc", "*.cpp"):
        for f in src_path.rglob(ext):
            fstr = str(f).lower()
            if any(skip in fstr for skip in [
                "test", "bench", "example", "demo", ".git",
                "build/", "CMakeFiles", "python", "fuzz/",
            ]):
                continue

            try:
                content = f.read_text(errors="replace")
                
                # Base priority for .c files
                priority = 10
                fname = f.name.lower()
                hname = harness_name.lower() if harness_name else ""

                # Boost files matching harness name
                if hname and hname in fname:
                    priority += 20

                # Boost files with security-relevant names
                if any(kw in fname for kw in [
                    "parse", "read", "dict", "buf", "string",
                    "encoding", "uri", "xpath", "valid", "entity",
                    "catalog", "schema", "relaxng", "html",
                    "memory", "alloc",
                ]):
                    priority += 15

                # Boost files containing security-relevant code patterns
                content_lower = content.lower()
                sec_hits = sum(1 for kw in SEC_KEYWORDS if kw in content_lower)
                priority += min(sec_hits * 2, 20)  # cap at +20

                files.append((str(f.relative_to(src_path)), content, priority))
            except Exception:
                continue

    # Sort by priority descending
    files.sort(key=lambda x: -x[2])
    
    # Log what we're picking
    for name, _, prio in files[:max_files]:
        logger.debug("[file-select] %s (priority=%d)", name, prio)

    return [(name, content) for name, content, _ in files[:max_files]]


def _collect_headers(src_path: Path) -> list[tuple[str, str]]:
    """Collect header files for harness generation context."""
    headers: list[tuple[str, str]] = []
    for f in src_path.rglob("*.h"):
        fstr = str(f).lower()
        if "internal" in fstr or "private" in fstr or ".git" in fstr:
            continue
        try:
            content = f.read_text(errors="replace")
            headers.append((str(f.relative_to(src_path)), content))
        except Exception:
            continue
    # Sort by name, prefer shorter (public API) headers
    headers.sort(key=lambda x: len(x[1]))
    return headers[:5]


def _parse_json_array(response: str) -> list[dict]:
    """Robustly parse a JSON array from LLM output."""
    clean = response.strip()
    # Strip markdown fences
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
    if clean.endswith("```"):
        clean = clean.rsplit("```", 1)[0]
    clean = clean.strip()

    # Find JSON array anywhere in response
    start = clean.find("[")
    end = clean.rfind("]")
    if start != -1 and end != -1 and end > start:
        clean = clean[start:end + 1]
    else:
        return []

    try:
        result = json.loads(clean)
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
        return []
    except json.JSONDecodeError:
        return []


def _extract_function_sigs(content: str) -> list[str]:
    """Extract function names from C source code (quick regex)."""
    import re
    # Match C function definitions: return_type function_name(
    pattern = r'\b([a-zA-Z_]\w*)\s*\([^)]*\)\s*\{'
    matches = re.findall(pattern, content)
    # Filter out common non-function keywords
    skip = {"if", "while", "for", "switch", "return", "sizeof", "typeof"}
    return [m for m in matches if m not in skip]


def _run_poc_against_binary(binary: str, data: bytes) -> tuple[bool, str]:
    """Run a PoC input against the fuzzer binary and check for crash.
    Returns (crashed: bool, output: str)."""
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".poc") as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        env = os.environ.copy()
        env["ASAN_OPTIONS"] = "abort_on_error=1:detect_leaks=0"

        result = subprocess.run(
            [binary, tmp_path],
            capture_output=True,
            timeout=10,
            env=env,
        )

        output = result.stderr.decode("utf-8", errors="replace")
        # ASAN crashes return non-zero exit code
        crashed = result.returncode != 0 and (
            "AddressSanitizer" in output or
            "SUMMARY:" in output or
            result.returncode < 0  # killed by signal
        )

        return crashed, output[:1000]

    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:
        return False, str(exc)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
