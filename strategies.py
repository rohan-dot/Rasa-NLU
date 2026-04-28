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
Given source code of a target program that parses input, generate raw
byte sequences that would trigger specific bugs.

Think step by step:
1. Identify the input format the program expects
2. Trace how input flows through the code
3. Find a path that leads to a crash (buffer overflow, null deref, etc.)
4. Construct the exact bytes needed to reach that path

For an XML parser, consider crafting:
- XML with extremely long attribute values (buffer overflow)
- XML with recursive entity definitions (stack overflow)
- XML with invalid UTF-8 sequences (encoding bugs)
- XML with null bytes in unexpected places
- Truncated XML at critical parse points
- XML with mismatched tags causing state confusion

Respond with ONLY a JSON array of PoC inputs:
[{
  "hex": "3c3f786d6c...",
  "target_bug": "buffer overflow in attribute parsing",
  "code_path": "xmlParseAttribute -> xmlStringDecodeEntities",
  "rationale": "attribute value exceeds stack buffer at line 4521"
}]

Your response must start with [ and end with ] — nothing else."""


def strategy_direct_poc(
    llm: VLLMClient,
    src_dir: str,
    output_dir: str,
    harness_name: str,
) -> int:
    """
    LLM reads code and directly generates PoC inputs.
    Returns count of PoCs generated.
    """
    if not llm.is_available():
        return 0

    pov_dir = Path(output_dir) / "povs"
    pov_dir.mkdir(parents=True, exist_ok=True)

    source_files = _collect_source_files(Path(src_dir), harness_name)

    user_msg = f"Target harness: {harness_name}\n\n"
    if source_files:
        user_msg += "Source code:\n"
        for sf, content in source_files[:3]:
            user_msg += f"\n// {sf}\n{content[:4000]}\n"

    response = llm.chat(
        system=POC_GEN_SYSTEM,
        user=user_msg,
        max_tokens=1500,
        temperature=0.5,
    )

    if not response:
        return 0

    pocs = _parse_json_array(response)
    count = 0
    for poc_info in pocs:
        hex_str = poc_info.get("hex", "")
        if not hex_str:
            continue
        try:
            data = bytes.fromhex(hex_str)
        except ValueError:
            continue

        poc_hash = hashlib.sha256(data).hexdigest()[:12]
        poc_path = pov_dir / f"llm-poc-{poc_hash}"
        poc_path.write_bytes(data)
        count += 1
        logger.info(
            "[poc-gen] %s (%d bytes) targeting: %s",
            poc_path.name, len(data),
            poc_info.get("target_bug", "unknown"),
        )

    logger.info("[poc-gen] Generated %d direct PoC inputs.", count)
    return count


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

    def run_round(self, corpus_dir: str) -> list[StrategyResult]:
        """
        Run one round of all strategies.
        Called periodically by the main orchestrator.
        """
        self.round_number += 1
        results: list[StrategyResult] = []
        logger.info(
            "═══ Strategy Round %d ═══════════════════════════════",
            self.round_number,
        )

        # Strategy 1: Source Code Audit (run every 3rd round or first round)
        if self.round_number == 1 or self.round_number % 3 == 0:
            r = self._run_strategy(
                "source_audit",
                lambda: strategy_source_audit(
                    self.llm, self.src_dir, self.output_dir, self.harness_name,
                ),
            )
            results.append(r)

        # Strategy 2: Harness Generation (run on rounds 1 and 5)
        if self.round_number in (1, 5):
            r = self._run_strategy(
                "harness_gen",
                lambda: self._do_harness_gen(),
            )
            results.append(r)

        # Strategy 3: Coverage-Guided Seeds (every round)
        r = self._run_strategy(
            "coverage_seeds",
            lambda: strategy_coverage_seeds(
                self.llm, self.src_dir, self.output_dir,
                self.harness_name, corpus_dir, self.crash_summaries,
            ),
        )
        results.append(r)

        # Strategy 4: Variant Analysis (when we have crashes)
        if self.crash_reports:
            latest_crash = self.crash_reports[-1]
            r = self._run_strategy(
                "variant_analysis",
                lambda: strategy_variant_analysis(
                    self.llm, self.src_dir, self.output_dir, latest_crash,
                ),
            )
            results.append(r)

        # Strategy 5: Direct PoC Generation (every 2nd round)
        if self.round_number % 2 == 0:
            r = self._run_strategy(
                "direct_poc",
                lambda: strategy_direct_poc(
                    self.llm, self.src_dir, self.output_dir, self.harness_name,
                ),
            )
            results.append(r)

        self.results_log.extend(results)

        # Log summary
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
