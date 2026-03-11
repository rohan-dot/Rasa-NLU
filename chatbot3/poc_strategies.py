"""
Step 4 – PoC Generation Strategies for CyberGym CRS (Task 1 / Level 1).

Five strategies of increasing sophistication attempt to produce a C/C++ PoC
that triggers the vulnerability described in the task.  An orchestrator runs
them in priority order and collects every result so the downstream executor
can try each one.
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

from crs.config import WORK_DIR
from crs.data_loader import CyberGymTask
from crs.code_intelligence import CodeContext
from crs.llm_router import (
    LLMRouter,
    SYSTEM_PROMPT_POC_GENERATOR,
    SYSTEM_PROMPT_POC_REFINER,
    SYSTEM_PROMPT_ANALYST,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PoCResult:
    """Container for a single PoC generation attempt."""

    strategy_name: str
    poc_code: str           # raw C/C++ source code string
    poc_path: Path | None   # where it was written to disk (None if not yet written)
    confidence: float       # 0.0 – 1.0, heuristic estimate
    notes: str              # brief note on strategy / why it was chosen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNSAFE_APIS = frozenset({
    "strcpy", "strcat", "sprintf", "gets", "scanf", "vsprintf",
    "memcpy", "memmove", "alloca", "realloc", "malloc", "calloc",
    "free", "strtok", "strncpy", "strncat",
})

_FUNC_DEF_RE = re.compile(
    r"(?:^|\n)"                         # start of line
    r"[\w\s\*]+?"                        # return type (e.g. "static int *")
    r"\b(\w+)\s*\("                      # function name + open paren
    r"[^)]*\)\s*\{",                     # params + open brace
    re.MULTILINE,
)


def _extract_code_block(text: str) -> str:
    """Pull the first ```c / ```cpp / ``` fenced block out of LLM output.

    Falls back to the entire text when no fenced block is found.
    """
    pattern = re.compile(r"```(?:c|cpp|c\+\+)?\s*\n(.*?)```", re.DOTALL)
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()


def _top_snippets_text(context: CodeContext, max_snippets: int = 8) -> str:
    """Format the highest-ranked code snippets for inclusion in a prompt."""
    snippets = context.ranked_snippets[:max_snippets]
    parts: list[str] = []
    for i, snip in enumerate(snippets, 1):
        header = f"### Snippet {i}"
        if hasattr(snip, "file_path") and snip.file_path:
            header += f" — {snip.file_path}"
        code = snip.code if hasattr(snip, "code") else str(snip)
        parts.append(f"{header}\n```\n{code}\n```")
    return "\n\n".join(parts) if parts else "(no snippets available)"


def _save_poc(code: str, task_id: str, strategy_name: str) -> Path:
    """Persist PoC source to WORK_DIR/<task_id>/poc_<strategy>.c and return the path."""
    out_dir = WORK_DIR / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    poc_path = out_dir / f"poc_{strategy_name}.c"
    poc_path.write_text(code, encoding="utf-8")
    logger.info("Saved PoC → %s", poc_path)
    return poc_path


def _try_compile(source_path: Path) -> tuple[bool, str]:
    """Attempt a syntax-only compile (-c) with gcc/g++.

    Returns (success, stderr_output).
    """
    suffix = source_path.suffix
    compiler = "g++" if suffix in (".cpp", ".cc", ".cxx") else "gcc"
    cmd = [
        compiler,
        "-c",
        "-fsyntax-only",
        "-Wall",
        "-Wextra",
        "-x", "c++" if compiler == "g++" else "c",
        str(source_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return proc.returncode == 0, proc.stderr
    except FileNotFoundError:
        # compiler not found — try the other one
        alt = "gcc" if compiler == "g++" else "g++"
        try:
            proc = subprocess.run(
                [alt, "-c", "-fsyntax-only", str(source_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return proc.returncode == 0, proc.stderr
        except FileNotFoundError:
            return False, "No C/C++ compiler found in PATH."
    except subprocess.TimeoutExpired:
        return False, "Compilation timed out."


# ---------------------------------------------------------------------------
# Vulnerability-pattern templates (Strategy 4)
# ---------------------------------------------------------------------------

VULN_TEMPLATES: dict[str, str | None] = {
    "heap_overflow": """\
#include <stdlib.h>
#include <string.h>

int main(void) {
    /* PLACEHOLDER: adjust size and write offset for actual target */
    size_t alloc_size = 64;
    char *buf = (char *)malloc(alloc_size);
    if (!buf) return 1;
    /* Overflow: write past the allocated region */
    memset(buf, 'A', alloc_size + 32);
    free(buf);
    return 0;
}
""",
    "use_after_free": """\
#include <stdlib.h>
#include <string.h>

int main(void) {
    /* PLACEHOLDER: replace with actual type / allocation */
    char *ptr = (char *)malloc(128);
    if (!ptr) return 1;
    strcpy(ptr, "data");
    free(ptr);
    /* Use-after-free: access freed memory */
    ptr[0] = 'X';
    return 0;
}
""",
    "oob_read": """\
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    /* PLACEHOLDER: adjust buffer and index for target */
    int arr[10] = {0};
    /* Out-of-bounds read */
    int val = arr[20];
    printf("%d\\n", val);
    return 0;
}
""",
    "buffer_overflow": """\
#include <string.h>
#include <stdio.h>

int main(void) {
    /* PLACEHOLDER: use actual buffer size and input from target */
    char dst[32];
    char src[256];
    memset(src, 'A', sizeof(src) - 1);
    src[sizeof(src) - 1] = '\\0';
    /* Stack buffer overflow via strcpy */
    strcpy(dst, src);
    printf("%s\\n", dst);
    return 0;
}
""",
    "integer_overflow": """\
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int main(void) {
    /* PLACEHOLDER: adapt multiplication to actual target code */
    size_t nmemb = (size_t)-1 / 2 + 2;
    size_t size  = 2;
    size_t total = nmemb * size;  /* wraps around */
    char *buf = (char *)malloc(total);
    if (!buf) return 1;
    memset(buf, 'B', nmemb * size);
    free(buf);
    return 0;
}
""",
    "null_deref": """\
#include <stdlib.h>
#include <string.h>

int main(void) {
    /* PLACEHOLDER: trigger a failed allocation or NULL return */
    char *ptr = NULL;
    /* NULL-pointer dereference */
    ptr[0] = 'Z';
    return 0;
}
""",
    "other": None,
}


# ---------------------------------------------------------------------------
# Strategy base class
# ---------------------------------------------------------------------------

class PoCStrategy(ABC):
    """Abstract base for a PoC-generation strategy."""

    name: str = "base"

    @abstractmethod
    def generate(self, context: CodeContext, router: LLMRouter) -> PoCResult | None:
        """Try to produce a PoC.  Return None on total failure."""
        ...


# ---------------------------------------------------------------------------
# Strategy 1 — DirectDescriptionPoC
# ---------------------------------------------------------------------------

class DirectDescriptionPoC(PoCStrategy):
    """Directly ask the LLM to write a PoC from the description + snippets."""

    name = "direct_description"

    def generate(self, context: CodeContext, router: LLMRouter) -> PoCResult | None:
        print(f"[Strategy] Running '{self.name}' …")

        snippets_text = _top_snippets_text(context)

        user_prompt = (
            f"## Vulnerability Description\n{context.description}\n\n"
            f"## Vulnerability Type (classified)\n{context.vuln_type}\n\n"
            f"## Build System\n{context.build_type}\n\n"
            f"## Relevant Code Snippets from the vulnerable codebase:\n{snippets_text}\n\n"
            "Write a complete C/C++ PoC program that triggers this vulnerability.\n"
            "The PoC will be compiled separately and linked against the project library.\n"
            "Make the PoC as simple as possible while reliably triggering the bug.\n"
        )

        raw = router.query(
            system_prompt=SYSTEM_PROMPT_POC_GENERATOR,
            user_prompt=user_prompt,
        )
        if not raw:
            logger.warning("%s: LLM returned empty response.", self.name)
            return None

        poc_code = _extract_code_block(raw)
        poc_path = _save_poc(poc_code, context.task_id, self.name)

        result = PoCResult(
            strategy_name=self.name,
            poc_code=poc_code,
            poc_path=poc_path,
            confidence=0.6,
            notes="Single-shot generation from description + snippets.",
        )
        print(f"[Strategy] '{self.name}' done — confidence {result.confidence:.2f}")
        return result


# ---------------------------------------------------------------------------
# Strategy 2 — AnalyzeFirstPoC
# ---------------------------------------------------------------------------

class AnalyzeFirstPoC(PoCStrategy):
    """Two-shot: analyse the vulnerability first, then generate the PoC."""

    name = "analyze_then_generate"

    def generate(self, context: CodeContext, router: LLMRouter) -> PoCResult | None:
        print(f"[Strategy] Running '{self.name}' …")

        snippets_text = _top_snippets_text(context)

        # -- Shot 1: Analysis --------------------------------------------------
        analysis_prompt = (
            f"## Vulnerability Description\n{context.description}\n\n"
            f"## Vulnerability Type\n{context.vuln_type}\n\n"
            f"## Relevant Code Snippets\n{snippets_text}\n\n"
            "Analyse the vulnerability and respond **only** with a JSON object "
            "(no markdown fences) containing these keys:\n"
            '  "vulnerable_function": "<name>",\n'
            '  "trigger_args": "<description of argument values / input that trigger the bug>",\n'
            '  "call_sequence": "<sequence of API / function calls needed>",\n'
            '  "expected_crash": "<ASAN/UBSAN error type expected>",\n'
            '  "reasoning": "<step-by-step reasoning>"\n'
        )

        analysis_raw = router.query(
            system_prompt=SYSTEM_PROMPT_ANALYST,
            user_prompt=analysis_prompt,
        )

        analysis: dict[str, Any] | None = None
        if analysis_raw:
            # Try to extract JSON from the response
            try:
                analysis = json.loads(analysis_raw)
            except json.JSONDecodeError:
                # Maybe wrapped in code fences
                m = re.search(r"```(?:json)?\s*\n?(.*?)```", analysis_raw, re.DOTALL)
                if m:
                    try:
                        analysis = json.loads(m.group(1))
                    except json.JSONDecodeError:
                        pass
                # Last resort: find first { … }
                if analysis is None:
                    m = re.search(r"\{.*\}", analysis_raw, re.DOTALL)
                    if m:
                        try:
                            analysis = json.loads(m.group(0))
                        except json.JSONDecodeError:
                            pass

        if analysis is None:
            logger.warning(
                "%s: analysis shot failed — falling back to direct_description.",
                self.name,
            )
            return DirectDescriptionPoC().generate(context, router)

        # -- Shot 2: Generation ------------------------------------------------
        analysis_json = json.dumps(analysis, indent=2)
        gen_prompt = (
            f"Based on this analysis:\n```json\n{analysis_json}\n```\n\n"
            f"And these code snippets:\n{snippets_text}\n\n"
            "Write a complete C/C++ PoC program that triggers this vulnerability.\n"
            "The PoC will be compiled separately and linked against the project library.\n"
            "Make it as simple as possible while reliably triggering the bug.\n"
        )

        raw = router.query(
            system_prompt=SYSTEM_PROMPT_POC_GENERATOR,
            user_prompt=gen_prompt,
        )
        if not raw:
            logger.warning("%s: generation shot returned empty.", self.name)
            return None

        poc_code = _extract_code_block(raw)
        poc_path = _save_poc(poc_code, context.task_id, self.name)

        result = PoCResult(
            strategy_name=self.name,
            poc_code=poc_code,
            poc_path=poc_path,
            confidence=0.75,
            notes=f"Two-shot (analyse → generate). Analysis keys: {list(analysis.keys())}",
        )
        print(f"[Strategy] '{self.name}' done — confidence {result.confidence:.2f}")
        return result


# ---------------------------------------------------------------------------
# Strategy 3 — CallPathTargetedPoC
# ---------------------------------------------------------------------------

class CallPathTargetedPoC(PoCStrategy):
    """Target specific call-paths found in top-ranked source files."""

    name = "call_path_targeted"

    # Maximum lines of source to include per function in the prompt
    _MAX_FUNC_LINES = 200

    def generate(self, context: CodeContext, router: LLMRouter) -> PoCResult | None:
        print(f"[Strategy] Running '{self.name}' …")

        # --- Pre-processing: identify target functions (no LLM) ---------------
        target_funcs, func_sources = self._find_target_functions(context)

        if not target_funcs:
            logger.info(
                "%s: no target functions found — falling back to direct_description.",
                self.name,
            )
            return DirectDescriptionPoC().generate(context, router)

        funcs_list = ", ".join(target_funcs)
        source_block = "\n\n".join(
            f"### {fname}\n```\n{src}\n```"
            for fname, src in func_sources.items()
        )

        user_prompt = (
            f"## Vulnerability Description\n{context.description}\n\n"
            f"## Vulnerability Type\n{context.vuln_type}\n\n"
            f"The vulnerability is in or near these functions: {funcs_list}.\n\n"
            f"Here is their source:\n{source_block}\n\n"
            "Write a complete C/C++ PoC that exercises these functions with "
            "edge-case inputs that trigger the described bug.\n"
            "The PoC will be compiled separately and linked against the project library.\n"
        )

        raw = router.query(
            system_prompt=SYSTEM_PROMPT_POC_GENERATOR,
            user_prompt=user_prompt,
        )
        if not raw:
            return None

        poc_code = _extract_code_block(raw)
        poc_path = _save_poc(poc_code, context.task_id, self.name)

        result = PoCResult(
            strategy_name=self.name,
            poc_code=poc_code,
            poc_path=poc_path,
            confidence=0.7,
            notes=f"Targeted functions: {funcs_list}",
        )
        print(f"[Strategy] '{self.name}' done — confidence {result.confidence:.2f}")
        return result

    # ---- helpers -------------------------------------------------------------

    def _find_target_functions(
        self, context: CodeContext
    ) -> tuple[list[str], dict[str, str]]:
        """Return (list_of_func_names, {name: source_excerpt})."""

        # Collect source text from top 3 ranked files
        ranked_files = context.ranked_snippets[:3]
        all_source = ""
        file_contents: list[tuple[str, str]] = []
        for snip in ranked_files:
            code = snip.code if hasattr(snip, "code") else str(snip)
            file_contents.append((getattr(snip, "file_path", "unknown"), code))
            all_source += code + "\n"

        # Extract function definitions
        func_names: list[str] = _FUNC_DEF_RE.findall(all_source)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_funcs: list[str] = []
        for fn in func_names:
            if fn not in seen:
                seen.add(fn)
                unique_funcs.append(fn)

        description_lower = context.description.lower()

        # 1) Functions mentioned in the vulnerability description
        mentioned = [f for f in unique_funcs if f.lower() in description_lower]

        # 2) Fallback: functions that use unsafe APIs
        if not mentioned:
            mentioned = [
                f for f in unique_funcs
                if any(api in all_source for api in _UNSAFE_APIS)
            ]
            # Keep only a handful
            mentioned = mentioned[:5]

        if not mentioned:
            return [], {}

        # Extract source excerpts per function
        func_sources: dict[str, str] = {}
        for fname in mentioned:
            excerpt = self._extract_function_body(fname, all_source)
            if excerpt:
                func_sources[fname] = excerpt

        return mentioned, func_sources

    @staticmethod
    def _extract_function_body(func_name: str, source: str) -> str | None:
        """Heuristically extract a function body (up to _MAX_FUNC_LINES lines)."""
        # Find the definition
        pattern = re.compile(
            rf"[\w\s\*]+?\b{re.escape(func_name)}\s*\([^)]*\)\s*\{{",
            re.MULTILINE,
        )
        m = pattern.search(source)
        if not m:
            return None

        start = m.start()
        # Walk forward counting braces
        depth = 0
        i = m.end() - 1  # points at the opening '{'
        lines_collected = 0
        max_lines = 200
        for j in range(i, len(source)):
            ch = source[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return source[start : j + 1]
            if ch == "\n":
                lines_collected += 1
                if lines_collected >= max_lines:
                    return source[start : j + 1] + "\n// … (truncated)"
        # Unbalanced — return what we have
        return source[start : start + 4000] + "\n// … (truncated / unbalanced)"


# ---------------------------------------------------------------------------
# Strategy 4 — PatternReplayPoC
# ---------------------------------------------------------------------------

class PatternReplayPoC(PoCStrategy):
    """Seed the PoC with a known vulnerability-class skeleton."""

    name = "pattern_replay"

    def generate(self, context: CodeContext, router: LLMRouter) -> PoCResult | None:
        print(f"[Strategy] Running '{self.name}' …")

        vuln_type = context.vuln_type or "other"
        template = VULN_TEMPLATES.get(vuln_type)

        if template is None:
            logger.info(
                "%s: no template for vuln_type '%s' — falling back to direct_description.",
                self.name,
                vuln_type,
            )
            return DirectDescriptionPoC().generate(context, router)

        snippets_text = _top_snippets_text(context)

        user_prompt = (
            f"## Vulnerability Description\n{context.description}\n\n"
            f"## Vulnerability Type\n{vuln_type}\n\n"
            f"Here is a template for **{vuln_type}** vulnerabilities:\n"
            f"```c\n{template}```\n\n"
            f"## Relevant Code Snippets from the vulnerable codebase:\n{snippets_text}\n\n"
            "Adapt this template to trigger the **specific** vulnerability in the "
            "provided codebase.  Replace generic placeholders with actual types, "
            "function calls, and values from the code snippets.\n"
            "Output a complete, compilable C/C++ PoC program.\n"
        )

        raw = router.query(
            system_prompt=SYSTEM_PROMPT_POC_GENERATOR,
            user_prompt=user_prompt,
        )
        if not raw:
            return None

        poc_code = _extract_code_block(raw)
        poc_path = _save_poc(poc_code, context.task_id, self.name)

        result = PoCResult(
            strategy_name=self.name,
            poc_code=poc_code,
            poc_path=poc_path,
            confidence=0.65,
            notes=f"Pattern-replay using '{vuln_type}' template.",
        )
        print(f"[Strategy] '{self.name}' done — confidence {result.confidence:.2f}")
        return result


# ---------------------------------------------------------------------------
# Strategy 5 — IterativeRefinePoC
# ---------------------------------------------------------------------------

class IterativeRefinePoC(PoCStrategy):
    """Generate with Strategy 1, then iteratively fix compilation errors."""

    name = "iterative_refine"

    _MAX_ITERATIONS = 3

    def generate(self, context: CodeContext, router: LLMRouter) -> PoCResult | None:
        print(f"[Strategy] Running '{self.name}' …")

        # Seed from DirectDescriptionPoC
        seed = DirectDescriptionPoC().generate(context, router)
        if seed is None:
            logger.warning("%s: seed generation failed.", self.name)
            return None

        poc_code = seed.poc_code

        for iteration in range(1, self._MAX_ITERATIONS + 1):
            print(f"  [Refine] iteration {iteration}/{self._MAX_ITERATIONS}")

            # Write to a temp file and try to compile
            with tempfile.NamedTemporaryFile(
                suffix=".c", mode="w", delete=False, dir="/tmp"
            ) as tmp:
                tmp.write(poc_code)
                tmp_path = Path(tmp.name)

            ok, stderr = _try_compile(tmp_path)
            tmp_path.unlink(missing_ok=True)

            if ok:
                print(f"  [Refine] compilation succeeded on iteration {iteration}.")
                poc_path = _save_poc(poc_code, context.task_id, self.name)
                result = PoCResult(
                    strategy_name=self.name,
                    poc_code=poc_code,
                    poc_path=poc_path,
                    confidence=0.8,
                    notes=f"Compiles successfully after {iteration} iteration(s).",
                )
                print(f"[Strategy] '{self.name}' done — confidence {result.confidence:.2f}")
                return result

            # Ask LLM to fix
            refine_prompt = (
                "The following PoC failed to compile.\n\n"
                f"## Compiler errors\n```\n{stderr[:3000]}\n```\n\n"
                f"## Current PoC code\n```c\n{poc_code}\n```\n\n"
                "Fix the compilation errors and return the corrected, complete "
                "C/C++ PoC program.  Do not remove functionality — only fix the "
                "syntax / type / include issues.\n"
            )

            raw = router.query(
                system_prompt=SYSTEM_PROMPT_POC_REFINER,
                user_prompt=refine_prompt,
            )
            if raw:
                poc_code = _extract_code_block(raw)
            else:
                logger.warning(
                    "%s: refiner returned empty on iteration %d.", self.name, iteration
                )
                break

        # Exhausted iterations — return best effort (may not compile)
        poc_path = _save_poc(poc_code, context.task_id, self.name)
        result = PoCResult(
            strategy_name=self.name,
            poc_code=poc_code,
            poc_path=poc_path,
            confidence=0.55,  # lower: did not successfully compile
            notes=f"Refinement exhausted {self._MAX_ITERATIONS} iterations without clean compile.",
        )
        print(f"[Strategy] '{self.name}' done — confidence {result.confidence:.2f}")
        return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PoCOrchestrator:
    """Run strategies in priority order and collect all results.

    The executor (Step 5) will later attempt to build + run each PoC.
    """

    DEFAULT_ORDER: list[type[PoCStrategy]] = [
        AnalyzeFirstPoC,
        CallPathTargetedPoC,
        PatternReplayPoC,
        IterativeRefinePoC,
        DirectDescriptionPoC,  # cheapest fallback last
    ]

    def __init__(
        self,
        router: LLMRouter,
        strategy_order: list[type[PoCStrategy]] | None = None,
    ) -> None:
        self.strategies: list[PoCStrategy] = [
            S() for S in (strategy_order or self.DEFAULT_ORDER)
        ]
        self.router = router

    def run(self, context: CodeContext) -> list[PoCResult]:
        """Execute every strategy, collecting all non-None results.

        Exceptions within a strategy are logged and swallowed so one failure
        never crashes the entire pipeline.
        """
        results: list[PoCResult] = []

        for strategy in self.strategies:
            print(f"\n{'='*60}")
            print(f"  Orchestrator → strategy '{strategy.name}'")
            print(f"{'='*60}")
            try:
                result = strategy.generate(context, self.router)
                if result is not None:
                    results.append(result)
                    logger.info(
                        "Strategy '%s' produced PoC (confidence %.2f).",
                        strategy.name,
                        result.confidence,
                    )
                else:
                    logger.info("Strategy '%s' returned None.", strategy.name)
            except Exception:
                logger.exception(
                    "Strategy '%s' raised an exception — skipping.",
                    strategy.name,
                )

        # Sort by confidence descending so the executor tries the best first
        results.sort(key=lambda r: r.confidence, reverse=True)

        print(f"\n[Orchestrator] Collected {len(results)} PoC(s):")
        for r in results:
            print(f"  • {r.strategy_name:25s}  confidence={r.confidence:.2f}  → {r.poc_path}")

        return results
