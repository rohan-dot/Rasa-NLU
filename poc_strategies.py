"""
crs/poc_strategies.py — Step 4: PoC generation strategies.

Implements five strategies for generating proof-of-concept exploits,
ordered from cheapest/simplest to most expensive/complex.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crs.code_intelligence import CodeContext
from crs.data_loader import CyberGymTask
from crs.llm_router import (
    LLMRouter,
    SYSTEM_PROMPT_ANALYST,
    SYSTEM_PROMPT_POC_GENERATOR,
    SYSTEM_PROMPT_POC_REFINER,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PoCResult:
    """Container for a generated PoC attempt."""

    strategy_name: str
    poc_code: str           # raw C/C++ source code string
    poc_path: Path | None   # where it was written to disk (None if not yet written)
    confidence: float       # 0.0 – 1.0, heuristic estimate
    notes: str              # brief note on strategy / why it was chosen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(
    r"```(?:c|cpp|c\+\+|h)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)

# Common unsafe C/C++ APIs that often appear near vulnerabilities.
UNSAFE_APIS = frozenset({
    "strcpy", "strcat", "sprintf", "gets", "scanf", "sscanf",
    "memcpy", "memmove", "realloc", "malloc", "calloc", "free",
    "strncpy", "strncat", "snprintf", "vsprintf",
    "atoi", "atol", "atof",
})

# Regex to find C/C++ function definitions (simple heuristic).
_FUNC_DEF_RE = re.compile(
    r"(?:^|\n)"                       # start of line
    r"(?:static\s+|inline\s+|extern\s+|const\s+)*"  # optional qualifiers
    r"\w[\w\s\*&:<>,]*?"              # return type
    r"\s+(\w+)\s*\([^)]*\)"          # function name + params
    r"\s*\{",                         # opening brace
    re.MULTILINE,
)


def extract_code_from_response(response: str) -> str:
    """Pull the first fenced C/C++ code block from an LLM response.

    Falls back to the entire response (stripped) if no fence is found.
    """
    match = _CODE_FENCE_RE.search(response)
    if match:
        return match.group(1).strip()
    # If the response looks like raw C code already, return as-is.
    return response.strip()


def _format_snippets(context: CodeContext, max_snippets: int = 5) -> str:
    """Format the top ranked snippets from *context* for an LLM prompt."""
    parts: list[str] = []
    snippets = getattr(context, "ranked_snippets", None) or []
    for idx, snippet in enumerate(snippets[:max_snippets]):
        # Each snippet is expected to have .filepath and .content (or be a tuple).
        if hasattr(snippet, "filepath"):
            header = f"### File: {snippet.filepath}"
            body = snippet.content
        elif isinstance(snippet, tuple) and len(snippet) >= 2:
            header = f"### File: {snippet[0]}"
            body = snippet[1]
        elif isinstance(snippet, dict):
            header = f"### File: {snippet.get('filepath', f'snippet_{idx}')}"
            body = snippet.get("content", str(snippet))
        else:
            header = f"### Snippet {idx}"
            body = str(snippet)
        parts.append(f"{header}\n```\n{body}\n```")
    return "\n\n".join(parts) if parts else "(no snippets available)"


def _get_description(context: CodeContext) -> str:
    """Return the vulnerability description text from *context*."""
    return getattr(context, "description", "") or getattr(context, "vuln_description", "") or ""


def _get_vuln_type(context: CodeContext) -> str:
    return getattr(context, "vuln_type", "other") or "other"


def _get_build_type(context: CodeContext) -> str:
    return getattr(context, "build_type", "unknown") or "unknown"


# ---------------------------------------------------------------------------
# Vulnerability-class PoC skeleton templates (Strategy 4)
# ---------------------------------------------------------------------------

VULN_TEMPLATES: dict[str, str | None] = {
    "heap_overflow": """\
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    // Allocate a small heap buffer and write past its end.
    size_t sz = 16;
    char *buf = (char *)malloc(sz);
    if (!buf) return 1;
    // PLACEHOLDER: replace with actual triggering call.
    memset(buf, 'A', sz + 32);  // heap-buffer-overflow
    free(buf);
    return 0;
}
""",
    "use_after_free": """\
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    // PLACEHOLDER: replace with actual type/function.
    char *p = (char *)malloc(64);
    if (!p) return 1;
    free(p);
    // Use-after-free: access freed memory.
    p[0] = 'X';
    return 0;
}
""",
    "oob_read": """\
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int arr[10] = {0};
    // PLACEHOLDER: replace index with actual trigger.
    volatile int val = arr[42];  // out-of-bounds read
    (void)val;
    return 0;
}
""",
    "buffer_overflow": """\
#include <stdio.h>
#include <string.h>

int main(void) {
    char dst[16];
    // PLACEHOLDER: replace with actual oversized input.
    const char *oversized = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    strcpy(dst, oversized);  // stack-buffer-overflow
    return 0;
}
""",
    "integer_overflow": """\
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int main(void) {
    // PLACEHOLDER: replace with actual size computation.
    uint32_t count = 0xFFFFFFFF;
    uint32_t elem_size = 2;
    size_t total = (size_t)(count * elem_size);  // wraps to small value
    char *buf = (char *)malloc(total);
    if (!buf) return 1;
    memset(buf, 'B', 256);  // writes past actual allocation
    free(buf);
    return 0;
}
""",
    "null_deref": """\
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    // PLACEHOLDER: replace with actual failing allocation / null return.
    char *p = NULL;
    p[0] = 'A';  // null-pointer dereference
    return 0;
}
""",
    "other": None,
}


# ---------------------------------------------------------------------------
# Strategy base class
# ---------------------------------------------------------------------------

class PoCStrategy(ABC):
    """Abstract base for PoC generation strategies."""

    name: str = "base"

    @abstractmethod
    def generate(self, context: CodeContext, router: LLMRouter) -> PoCResult | None:
        """Try to produce a PoC.  Return *None* on failure."""
        ...


# ---------------------------------------------------------------------------
# Strategy 1 — DirectDescriptionPoC
# ---------------------------------------------------------------------------

class DirectDescriptionPoC(PoCStrategy):
    """Simplest strategy: one-shot LLM generation from the description and snippets."""

    name = "direct_description"

    def generate(self, context: CodeContext, router: LLMRouter) -> PoCResult | None:
        description = _get_description(context)
        vuln_type = _get_vuln_type(context)
        build_type = _get_build_type(context)
        top_snippets = _format_snippets(context)

        user_prompt = (
            f"## Vulnerability Description\n{description}\n\n"
            f"## Vulnerability Type (classified)\n{vuln_type}\n\n"
            f"## Build System\n{build_type}\n\n"
            f"## Relevant Code Snippets from the vulnerable codebase:\n{top_snippets}\n\n"
            "Write a complete C/C++ PoC program that triggers this vulnerability.\n"
            "The PoC will be compiled separately and linked against the project library.\n"
            "Make the PoC as simple as possible while reliably triggering the bug."
        )

        try:
            response = router.query(
                system_prompt=SYSTEM_PROMPT_POC_GENERATOR,
                user_prompt=user_prompt,
            )
        except Exception:
            logger.exception("[%s] LLM query failed", self.name)
            return None

        poc_code = extract_code_from_response(response)
        if not poc_code:
            logger.warning("[%s] LLM returned empty PoC code", self.name)
            return None

        return PoCResult(
            strategy_name=self.name,
            poc_code=poc_code,
            poc_path=None,
            confidence=0.6,
            notes="One-shot generation from description + snippets.",
        )


# ---------------------------------------------------------------------------
# Strategy 2 — AnalyzeFirstPoC
# ---------------------------------------------------------------------------

class AnalyzeFirstPoC(PoCStrategy):
    """Two-shot strategy: analyse the vulnerability first, then generate the PoC."""

    name = "analyze_then_generate"

    def generate(self, context: CodeContext, router: LLMRouter) -> PoCResult | None:
        description = _get_description(context)
        vuln_type = _get_vuln_type(context)
        top_snippets = _format_snippets(context)

        # ── SHOT 1: Analysis ──────────────────────────────────────────────
        analysis_prompt = (
            f"## Vulnerability Description\n{description}\n\n"
            f"## Vulnerability Type\n{vuln_type}\n\n"
            f"## Relevant Code Snippets\n{top_snippets}\n\n"
            "Analyze this vulnerability and respond with a JSON object containing:\n"
            '  "vulnerable_function": "<name of the function containing the bug>",\n'
            '  "trigger_inputs": "<description of argument values / input bytes / call sequence>",\n'
            '  "expected_crash": "<ASAN/UBSAN error type, e.g. heap-buffer-overflow>",\n'
            '  "root_cause": "<one-sentence explanation of why the bug occurs>",\n'
            '  "call_sequence": ["func_a()", "func_b(arg)", "..."]\n'
            "Return ONLY the JSON object, no other text."
        )

        analysis: dict[str, Any] | None = None
        try:
            analysis_resp = router.query(
                system_prompt=SYSTEM_PROMPT_ANALYST,
                user_prompt=analysis_prompt,
            )
            # Attempt to parse JSON — tolerate markdown fences.
            cleaned = re.sub(r"```(?:json)?\s*", "", analysis_resp).strip().rstrip("`")
            analysis = json.loads(cleaned)
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("[%s] Analysis shot failed (%s), falling back to direct", self.name, exc)

        if analysis is None:
            # Graceful degradation → Strategy 1.
            fallback = DirectDescriptionPoC()
            result = fallback.generate(context, router)
            if result:
                result.notes = "analyze_then_generate fell back to direct_description (analysis parse failed)."
                result.strategy_name = self.name
            return result

        # ── SHOT 2: Generation ────────────────────────────────────────────
        analysis_json = json.dumps(analysis, indent=2)
        gen_prompt = (
            f"Based on this analysis:\n```json\n{analysis_json}\n```\n\n"
            f"And these code snippets:\n{top_snippets}\n\n"
            "Write a complete C/C++ PoC program that triggers this vulnerability.\n"
            "The PoC will be compiled separately and linked against the project library.\n"
            "Include the necessary headers. Make the PoC as simple as possible."
        )

        try:
            gen_resp = router.query(
                system_prompt=SYSTEM_PROMPT_POC_GENERATOR,
                user_prompt=gen_prompt,
            )
        except Exception:
            logger.exception("[%s] Generation shot failed", self.name)
            return None

        poc_code = extract_code_from_response(gen_resp)
        if not poc_code:
            return None

        return PoCResult(
            strategy_name=self.name,
            poc_code=poc_code,
            poc_path=None,
            confidence=0.75,
            notes=f"Two-shot (analysis→generation). Analysis: {analysis.get('root_cause', 'n/a')}",
        )


# ---------------------------------------------------------------------------
# Strategy 3 — CallPathTargetedPoC
# ---------------------------------------------------------------------------

class CallPathTargetedPoC(PoCStrategy):
    """Targets specific call paths found in top-ranked files."""

    name = "call_path_targeted"

    # Upper limit on lines extracted per function body.
    MAX_FUNC_LINES = 200

    def generate(self, context: CodeContext, router: LLMRouter) -> PoCResult | None:
        description = _get_description(context)
        vuln_type = _get_vuln_type(context)
        top_snippets = _format_snippets(context, max_snippets=3)

        # ── Pre-processing: find target functions (no LLM) ────────────────
        ranked = getattr(context, "ranked_snippets", None) or []
        top_files: list[tuple[str, str]] = []
        for snippet in ranked[:3]:
            if hasattr(snippet, "filepath"):
                top_files.append((snippet.filepath, snippet.content))
            elif isinstance(snippet, tuple) and len(snippet) >= 2:
                top_files.append((snippet[0], snippet[1]))
            elif isinstance(snippet, dict):
                top_files.append((
                    snippet.get("filepath", "unknown"),
                    snippet.get("content", str(snippet)),
                ))

        # Collect all function names from top files.
        all_funcs: dict[str, str] = {}  # name → source text of enclosing file
        for fpath, content in top_files:
            for m in _FUNC_DEF_RE.finditer(content):
                fname = m.group(1)
                all_funcs[fname] = content

        if not all_funcs:
            logger.info("[%s] No function definitions found, falling back", self.name)
            return DirectDescriptionPoC().generate(context, router)

        # Find functions mentioned in the vulnerability description.
        desc_lower = description.lower()
        target_funcs = {fn for fn in all_funcs if fn.lower() in desc_lower}

        # If no direct match, look for functions that call unsafe APIs.
        if not target_funcs:
            for fn, src in all_funcs.items():
                if any(api in src for api in UNSAFE_APIS):
                    target_funcs.add(fn)
                    if len(target_funcs) >= 5:
                        break

        if not target_funcs:
            logger.info("[%s] No target functions identified, falling back", self.name)
            return DirectDescriptionPoC().generate(context, router)

        # Extract source for each target function (up to MAX_FUNC_LINES).
        func_source_parts: list[str] = []
        for fn in sorted(target_funcs):
            src = all_funcs.get(fn, "")
            # Attempt to extract just the function body.
            pattern = re.compile(
                rf"(?:^|\n)((?:[\w\s\*&:<>,])+\s+{re.escape(fn)}\s*\([^)]*\)\s*\{{)",
                re.MULTILINE,
            )
            m = pattern.search(src)
            if m:
                start = m.start()
                # Walk braces to find the end of the function body.
                brace_depth = 0
                end = start
                in_body = False
                for i in range(m.start(), len(src)):
                    if src[i] == "{":
                        brace_depth += 1
                        in_body = True
                    elif src[i] == "}":
                        brace_depth -= 1
                        if in_body and brace_depth == 0:
                            end = i + 1
                            break
                func_text = src[start:end]
                # Limit lines.
                lines = func_text.splitlines()
                if len(lines) > self.MAX_FUNC_LINES:
                    lines = lines[: self.MAX_FUNC_LINES] + ["// ... truncated ..."]
                func_source_parts.append(f"// Function: {fn}\n" + "\n".join(lines))
            else:
                func_source_parts.append(f"// Function: {fn}\n// (definition not fully extracted)")

        func_source = "\n\n".join(func_source_parts)

        user_prompt = (
            f"The vulnerability is in or near these functions: {', '.join(sorted(target_funcs))}.\n\n"
            f"## Vulnerability Description\n{description}\n\n"
            f"## Vulnerability Type\n{vuln_type}\n\n"
            f"## Target Function Source\n```\n{func_source}\n```\n\n"
            f"## Additional Context Snippets\n{top_snippets}\n\n"
            "Write a PoC that exercises these functions with edge-case inputs that "
            "trigger the described bug. Include all necessary headers."
        )

        try:
            response = router.query(
                system_prompt=SYSTEM_PROMPT_POC_GENERATOR,
                user_prompt=user_prompt,
            )
        except Exception:
            logger.exception("[%s] LLM query failed", self.name)
            return None

        poc_code = extract_code_from_response(response)
        if not poc_code:
            return None

        return PoCResult(
            strategy_name=self.name,
            poc_code=poc_code,
            poc_path=None,
            confidence=0.7,
            notes=f"Targeted functions: {', '.join(sorted(target_funcs))}",
        )


# ---------------------------------------------------------------------------
# Strategy 4 — PatternReplayPoC
# ---------------------------------------------------------------------------

class PatternReplayPoC(PoCStrategy):
    """Seeds the PoC with a known vulnerability-class skeleton template."""

    name = "pattern_replay"

    def generate(self, context: CodeContext, router: LLMRouter) -> PoCResult | None:
        description = _get_description(context)
        vuln_type = _get_vuln_type(context)
        top_snippets = _format_snippets(context)

        template = VULN_TEMPLATES.get(vuln_type)
        if template is None:
            logger.info("[%s] No template for vuln_type=%r, falling back", self.name, vuln_type)
            return DirectDescriptionPoC().generate(context, router)

        user_prompt = (
            f"Here is a skeleton template for **{vuln_type}** vulnerabilities:\n"
            f"```c\n{template}```\n\n"
            f"## Vulnerability Description\n{description}\n\n"
            f"## Relevant Code Snippets\n{top_snippets}\n\n"
            "Adapt this template to trigger the *specific* vulnerability in the "
            "provided codebase. Replace generic placeholders with actual types, "
            "function calls, and values from the code snippets above.\n"
            "Return a complete, compilable C/C++ program."
        )

        try:
            response = router.query(
                system_prompt=SYSTEM_PROMPT_POC_GENERATOR,
                user_prompt=user_prompt,
            )
        except Exception:
            logger.exception("[%s] LLM query failed", self.name)
            return None

        poc_code = extract_code_from_response(response)
        if not poc_code:
            return None

        return PoCResult(
            strategy_name=self.name,
            poc_code=poc_code,
            poc_path=None,
            confidence=0.65,
            notes=f"Pattern replay using {vuln_type} skeleton template.",
        )


# ---------------------------------------------------------------------------
# Strategy 5 — IterativeRefinePoC
# ---------------------------------------------------------------------------

class IterativeRefinePoC(PoCStrategy):
    """Generates a PoC via Strategy 1 then iteratively refines it using compile feedback."""

    name = "iterative_refine"
    MAX_ITERATIONS = 3

    def generate(self, context: CodeContext, router: LLMRouter) -> PoCResult | None:
        # Start with a direct-description PoC.
        base_result = DirectDescriptionPoC().generate(context, router)
        if base_result is None:
            logger.warning("[%s] Base strategy returned nothing", self.name)
            return None

        poc_code = base_result.poc_code

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            compile_ok, errors = self._try_compile(poc_code)
            if compile_ok:
                logger.info("[%s] PoC compiles on iteration %d", self.name, iteration)
                return PoCResult(
                    strategy_name=self.name,
                    poc_code=poc_code,
                    poc_path=None,
                    confidence=min(0.6 + 0.1 * iteration, 0.85),
                    notes=f"Compiled successfully after {iteration} iteration(s).",
                )

            # Ask the LLM to fix compile errors.
            logger.info("[%s] Compile failed (iter %d), refining…", self.name, iteration)
            refine_prompt = (
                "The following PoC code failed to compile.\n\n"
                f"## Current Code\n```c\n{poc_code}\n```\n\n"
                f"## Compiler Errors\n```\n{errors}\n```\n\n"
                "Fix the compilation errors. Return the corrected, complete C/C++ source.\n"
                "Do not change the overall logic — only fix syntax, missing headers, "
                "or type errors."
            )

            try:
                response = router.query(
                    system_prompt=SYSTEM_PROMPT_POC_REFINER,
                    user_prompt=refine_prompt,
                )
            except Exception:
                logger.exception("[%s] Refinement LLM query failed (iter %d)", self.name, iteration)
                break

            refined = extract_code_from_response(response)
            if not refined or refined == poc_code:
                logger.warning("[%s] Refinement produced no change (iter %d)", self.name, iteration)
                break
            poc_code = refined

        # Return whatever we have, even if it didn't compile.
        return PoCResult(
            strategy_name=self.name,
            poc_code=poc_code,
            poc_path=None,
            confidence=0.5,
            notes=f"Refinement exhausted {self.MAX_ITERATIONS} iterations (may not compile).",
        )

    @staticmethod
    def _try_compile(code: str) -> tuple[bool, str]:
        """Attempt a syntax-only compile (-c -fsyntax-only). Returns (success, error_text)."""
        suffix = ".cpp" if ("iostream" in code or "namespace" in code or "class " in code) else ".c"
        compiler = "g++" if suffix == ".cpp" else "gcc"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, prefix="poc_", delete=False
        ) as tmp:
            tmp.write(code)
            tmp.flush()
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [compiler, "-c", "-fsyntax-only", "-Wall", "-Wextra", tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return True, ""
            return False, (result.stderr or result.stdout or "unknown error").strip()
        except FileNotFoundError:
            # Compiler not found; try the other one.
            alt = "g++" if compiler == "gcc" else "gcc"
            try:
                result = subprocess.run(
                    [alt, "-c", "-fsyntax-only", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    return True, ""
                return False, (result.stderr or result.stdout or "unknown error").strip()
            except FileNotFoundError:
                return False, "No C/C++ compiler found in PATH."
        except subprocess.TimeoutExpired:
            return False, "Compilation timed out."
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Strategy orchestrator
# ---------------------------------------------------------------------------

# Default ordering: cheapest / fastest first.
DEFAULT_STRATEGY_ORDER: list[PoCStrategy] = [
    DirectDescriptionPoC(),      # 1 LLM call, confidence 0.60
    PatternReplayPoC(),          # 1 LLM call, confidence 0.65
    CallPathTargetedPoC(),       # 1 LLM call, confidence 0.70
    AnalyzeFirstPoC(),           # 2 LLM calls, confidence 0.75
    IterativeRefinePoC(),        # 1 + up to 3 LLM calls, confidence ≤ 0.85
]


def run_strategies(
    context: CodeContext,
    router: LLMRouter,
    strategies: list[PoCStrategy] | None = None,
    stop_on_first: bool = False,
) -> list[PoCResult]:
    """Execute strategies in order and collect results.

    Parameters
    ----------
    context:
        The analysed code context for the current task.
    router:
        LLM access layer.
    strategies:
        Custom list; defaults to :data:`DEFAULT_STRATEGY_ORDER`.
    stop_on_first:
        If *True*, return as soon as the first strategy succeeds.

    Returns
    -------
    list[PoCResult]
        All successful results, ordered by the strategy execution order.
    """
    if strategies is None:
        strategies = DEFAULT_STRATEGY_ORDER

    results: list[PoCResult] = []
    for strategy in strategies:
        logger.info("Running strategy: %s", strategy.name)
        try:
            result = strategy.generate(context, router)
        except Exception:
            logger.exception("Strategy %s raised an unexpected error", strategy.name)
            result = None

        if result is not None:
            logger.info(
                "Strategy %s succeeded (confidence=%.2f)", strategy.name, result.confidence
            )
            results.append(result)
            if stop_on_first:
                break
        else:
            logger.info("Strategy %s returned no result", strategy.name)

    return results


def best_result(results: list[PoCResult]) -> PoCResult | None:
    """Return the highest-confidence result, or *None* if the list is empty."""
    if not results:
        return None
    return max(results, key=lambda r: r.confidence)
