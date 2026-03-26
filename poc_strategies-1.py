"""
crs/poc_strategies.py — PoC generation strategies for the Cyber Reasoning System.

Changes from original:
  - Added _STANDARD_HEADERS whitelist and _sanitize_poc() to strip hallucinated includes
  - Added _strip_undefined_refs() to auto-remove calls that cause linker errors
  - _top_snippets_text() now prepends an inline warning for the LLM
  - Every strategy pipes output through _sanitize_poc() before saving
  - IterativeRefinePoC refine prompt explicitly forbids "fixing" by adding project headers
  - IterativeRefinePoC now runs _strip_undefined_refs() before asking the LLM to fix
"""
from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from crs.code_intelligence import CodeContext
from crs.config import cfg
from crs.llm_router import (
    LLMRouter,
    SYSTEM_PROMPT_ANALYST,
    SYSTEM_PROMPT_POC_GENERATOR,
    SYSTEM_PROMPT_POC_REFINER,
)

logger = logging.getLogger(__name__)

WORK_DIR = cfg.WORK_DIR


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class PoCResult:
    strategy_name: str
    poc_code: str
    poc_path: Optional[Path]
    confidence: float
    notes: str


# ── Vulnerability templates ────────────────────────────────────────────────

VULN_TEMPLATES: dict[str, str] = {
    "heap_overflow": """\
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    size_t size = 16;
    char *buf = (char *)malloc(size);
    if (!buf) return 1;
    // Trigger heap overflow by writing past allocation
    memcpy(buf, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", 34);
    printf("buf=%s\\n", buf);
    free(buf);
    return 0;
}
""",
    "stack_overflow": """\
#include <stdio.h>
#include <string.h>

void vulnerable(const char *input) {
    char buf[64];
    strcpy(buf, input);  // no bounds check
    printf("%s\\n", buf);
}

int main(void) {
    char payload[256];
    memset(payload, 'A', sizeof(payload) - 1);
    payload[sizeof(payload) - 1] = '\\0';
    vulnerable(payload);
    return 0;
}
""",
    "use_after_free": """\
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    char *p = (char *)malloc(64);
    if (!p) return 1;
    strcpy(p, "hello");
    free(p);
    // Use after free
    printf("%s\\n", p);
    return 0;
}
""",
    "null_deref": """\
#include <stdio.h>
#include <stdlib.h>

typedef struct { int value; } Node;

Node *get_node(int x) {
    if (x < 0) return NULL;
    Node *n = (Node *)malloc(sizeof(Node));
    if (n) n->value = x;
    return n;
}

int main(void) {
    Node *n = get_node(-1);
    printf("%d\\n", n->value);  // null deref
    return 0;
}
""",
}


# ── Standard headers whitelist ─────────────────────────────────────────────

_STANDARD_HEADERS = {
    # C standard
    "stdio.h", "stdlib.h", "string.h", "stdint.h", "stdbool.h",
    "stddef.h", "stdarg.h", "limits.h", "float.h", "math.h",
    "ctype.h", "errno.h", "assert.h", "signal.h", "setjmp.h",
    "time.h", "locale.h", "wchar.h", "wctype.h", "complex.h",
    "fenv.h", "inttypes.h", "iso646.h", "stdalign.h", "stdnoreturn.h",
    "tgmath.h", "uchar.h", "threads.h", "stdatomic.h",
    # POSIX (commonly needed for PoCs)
    "unistd.h", "fcntl.h", "pthread.h", "dlfcn.h", "dirent.h",
    "sys/types.h", "sys/stat.h", "sys/mman.h", "sys/wait.h",
    "sys/socket.h", "sys/un.h", "sys/ioctl.h", "sys/time.h",
    "sys/resource.h", "sys/select.h",
    "netinet/in.h", "arpa/inet.h", "netdb.h",
    "poll.h", "sched.h", "semaphore.h",
}


def _sanitize_poc(code: str) -> str:
    """
    Post-generation sanitizer: remove non-standard #include lines that the
    LLM hallucinated (e.g. #include "mosquitto.h", #include "db.h").
    Also removes lines with relative path includes to project dirs.
    """
    lines = code.split("\n")
    cleaned = []
    removed = []

    for line in lines:
        m = re.match(r'\s*#include\s*[<"]([^>"]+)[>"]', line)
        if m:
            header = m.group(1).strip()
            # Check against whitelist
            if header not in _STANDARD_HEADERS:
                removed.append(header)
                cleaned.append(f"// REMOVED non-standard include: {line.strip()}")
                continue
        cleaned.append(line)

    if removed:
        print(f"  [sanitize] Stripped {len(removed)} hallucinated include(s): {removed}")

    return "\n".join(cleaned)


def _strip_undefined_refs(code: str, stderr: str) -> str:
    """
    Parse 'undefined reference to `func_name`' from linker errors and comment
    out lines in the PoC that call those functions.  This gives the iterative
    refiner a cleaner starting point instead of letting the LLM "fix" the
    error by adding a project header.
    """
    # Collect undefined symbols
    undef = set(re.findall(r"undefined reference to [`'](\w+)'", stderr))
    if not undef:
        return code

    print(f"  [sanitize] Auto-stripping calls to undefined symbols: {undef}")

    lines = code.split("\n")
    cleaned = []
    for line in lines:
        # Check if line contains a call to an undefined symbol
        stripped = False
        for sym in undef:
            # Match function calls like  sym(  or  sym (  but not declarations/comments
            if re.search(rf'\b{re.escape(sym)}\s*\(', line) and not line.strip().startswith("//"):
                cleaned.append(f"// REMOVED (undefined): {line.strip()}")
                stripped = True
                break
        if not stripped:
            cleaned.append(line)

    return "\n".join(cleaned)


# ── Helpers ────────────────────────────────────────────────────────────────

def _save_poc(code: str, task_id: str, strategy_name: str) -> Path:
    """Save PoC code to a .c file and return the path."""
    safe_task = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(task_id))
    safe_strat = re.sub(r"[^a-zA-Z0-9_\-]", "_", strategy_name)
    work = cfg.task_work_dir(safe_task)
    path = work / f"poc_{safe_strat}.c"
    path.write_text(code, encoding="utf-8")
    return path


def _extract_code_block(text: str) -> str:
    """Extract C/C++ code from a fenced block, or return text as-is."""
    m = re.search(r"```(?:c\+\+|cpp|c)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def _extract_and_sanitize(raw_response: str) -> str:
    """Extract code block from LLM response, then sanitize it."""
    code = _extract_code_block(raw_response)
    code = _sanitize_poc(code)
    return code


def _try_compile(path: Path) -> Tuple[bool, str]:
    """
    Try to compile a standalone C file.
    Returns (success, stderr_output).
    """
    out = path.with_suffix("")
    cmd = ["gcc", "-fsanitize=address,undefined", "-g", "-O1",
           str(path), "-o", str(out), "-lm", "-lpthread"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        return r.returncode == 0, r.stderr.decode(errors="replace")
    except Exception as e:
        return False, str(e)


def _top_snippets_text(context: CodeContext) -> str:
    """
    Return top code snippets as a string, prefixed with an inline warning
    so the LLM is reminded these are read-only references.
    """
    prefix = (
        "// ======================================================================\n"
        "// NOTE: These snippets are READ-ONLY reference code from the project.\n"
        "// You MUST copy any needed logic INLINE into your PoC's main().\n"
        "// Do NOT #include any project headers (e.g. mosquitto.h, db.h, etc.).\n"
        "// Only use standard C headers: stdio.h, stdlib.h, string.h, stdint.h.\n"
        "// ======================================================================\n\n"
    )
    if isinstance(context.top_snippets, str):
        return prefix + context.top_snippets[:6000]
    return prefix


def _find_target_functions(context: CodeContext) -> str:
    """Extract function names from ranked files relevant to the vuln."""
    funcs = []
    for path, score in context.ranked_files[:5]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            matches = re.findall(r"\b(\w+)\s*\([^)]*\)\s*\{", text)
            funcs.extend(matches[:10])
        except Exception:
            pass
    return ", ".join(set(funcs[:20])) if funcs else "unknown"


# ── Base strategy ──────────────────────────────────────────────────────────

class PoCStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, context: CodeContext, router: LLMRouter) -> Optional[PoCResult]:
        ...


# ── Strategy 1 — AnalyzeFirstPoC ──────────────────────────────────────────

class AnalyzeFirstPoC(PoCStrategy):
    """Analyze the code before generating a PoC."""

    name = "analyze_then_generate"

    def generate(self, context: CodeContext, router: LLMRouter) -> Optional[PoCResult]:
        print(f"[Strategy] Running '{self.name}' ...")

        snippets_text = _top_snippets_text(context)

        # Step 1: Analyze
        analysis_prompt = (
            f"## Vulnerability Description\n{context.description}\n\n"
            f"## Code Snippets\n```c\n{snippets_text}\n```\n\n"
            "Analyze the vulnerability. Identify:\n"
            "1. The exact vulnerable function name (must exist in the snippets above)\n"
            "2. What input triggers the bug\n"
            "3. The memory corruption type\n"
            "4. Any structs needed (copy their definitions from the snippets)\n"
        )

        analysis_raw = router.chat(
            system_prompt=SYSTEM_PROMPT_ANALYST,
            user_prompt=analysis_prompt,
            max_tokens=1024,
            temperature=0.1,
        )
        if not analysis_raw:
            return None

        # Step 2: Generate PoC
        gen_prompt = (
            f"## Vulnerability Description\n{context.description}\n\n"
            f"## Analysis\n{analysis_raw}\n\n"
            f"## Code Snippets\n```c\n{snippets_text}\n```\n\n"
            "Write a complete, self-contained C PoC that triggers this vulnerability.\n"
            "ONLY use functions and structs visible in the code snippets above.\n"
            "Use only standard C headers. Copy struct definitions inline if needed.\n"
            "Do NOT include ANY project-specific headers (e.g. mosquitto.h, db.h, jq.h).\n"
        )

        raw = router.chat(
            system_prompt=SYSTEM_PROMPT_POC_GENERATOR,
            user_prompt=gen_prompt,
        )
        if not raw:
            return None

        poc_code = _extract_and_sanitize(raw)
        poc_path = _save_poc(poc_code, context.task.task_id, self.name)

        result = PoCResult(
            strategy_name=self.name,
            poc_code=poc_code,
            poc_path=poc_path,
            confidence=0.75,
            notes=f"Two-step: analyze then generate.",
        )
        print(f"[Strategy] '{self.name}' done — confidence {result.confidence:.2f}")
        return result


# ── Strategy 2 — CallPathTargetedPoC ──────────────────────────────────────

class CallPathTargetedPoC(PoCStrategy):
    """Target the specific call path to the vulnerability."""

    name = "call_path_targeted"

    def generate(self, context: CodeContext, router: LLMRouter) -> Optional[PoCResult]:
        print(f"[Strategy] Running '{self.name}' ...")

        snippets_text = _top_snippets_text(context)
        target_funcs  = _find_target_functions(context)

        user_prompt = (
            f"## Vulnerability Description\n{context.description}\n\n"
            f"## Potentially Vulnerable Functions\n{target_funcs}\n\n"
            f"## Code Snippets\n```c\n{snippets_text}\n```\n\n"
            "Write a self-contained C PoC that reproduces the vulnerable function's logic "
            "INLINE in main() and triggers the bug.\n"
            "- Copy any required struct definitions inline from the snippets.\n"
            "- Only use function names that appear verbatim in the snippets above.\n"
            "- Use only standard C headers (stdio.h, stdlib.h, string.h, stdint.h, stdbool.h).\n"
            "- Do NOT include any project-specific header files.\n"
            "- Do NOT call project functions — reproduce their logic inline.\n"
        )

        raw = router.chat(
            system_prompt=SYSTEM_PROMPT_POC_GENERATOR,
            user_prompt=user_prompt,
        )
        if not raw:
            return None

        poc_code = _extract_and_sanitize(raw)
        poc_path = _save_poc(poc_code, context.task.task_id, self.name)

        result = PoCResult(
            strategy_name=self.name,
            poc_code=poc_code,
            poc_path=poc_path,
            confidence=0.70,
            notes="Call-path targeted.",
        )
        print(f"[Strategy] '{self.name}' done — confidence {result.confidence:.2f}")
        return result


# ── Strategy 3 — PatternReplayPoC ─────────────────────────────────────────

class PatternReplayPoC(PoCStrategy):
    """Seed the PoC with a known vulnerability-class skeleton."""

    name = "pattern_replay"

    def generate(self, context: CodeContext, router: LLMRouter) -> Optional[PoCResult]:
        print(f"[Strategy] Running '{self.name}' ...")

        vuln_type = context.vuln_type or "other"
        template  = VULN_TEMPLATES.get(vuln_type)

        if template is None:
            logger.info("%s: no template for vuln_type '%s' — falling back to direct_description.",
                        self.name, vuln_type)
            return DirectDescriptionPoC().generate(context, router)

        snippets_text = _top_snippets_text(context)

        user_prompt = (
            f"## Vulnerability Description\n{context.description}\n\n"
            f"## Vulnerability Type\n{vuln_type}\n\n"
            f"Here is a template for **{vuln_type}** vulnerabilities:\n"
            f"```c\n{template}```\n\n"
            f"## Relevant Code Snippets\n```c\n{snippets_text}\n```\n\n"
            "Adapt this template to trigger the **specific** vulnerability in the provided codebase.\n"
            "Replace generic placeholders with actual types, function calls, and values from the snippets.\n"
            "ONLY use functions and structs visible in the snippets. No project headers.\n"
            "Reproduce any needed project logic INLINE — do NOT call project APIs.\n"
            "Output a complete, compilable C/C++ PoC program.\n"
        )

        raw = router.chat(
            system_prompt=SYSTEM_PROMPT_POC_GENERATOR,
            user_prompt=user_prompt,
        )
        if not raw:
            return None

        poc_code = _extract_and_sanitize(raw)
        poc_path = _save_poc(poc_code, context.task.task_id, self.name)

        result = PoCResult(
            strategy_name=self.name,
            poc_code=poc_code,
            poc_path=poc_path,
            confidence=0.65,
            notes=f"Pattern-replay using '{vuln_type}' template.",
        )
        print(f"[Strategy] '{self.name}' done — confidence {result.confidence:.2f}")
        return result


# ── Strategy 4 — IterativeRefinePoC ───────────────────────────────────────

class IterativeRefinePoC(PoCStrategy):
    """Generate with Strategy 1, then iteratively fix compilation errors."""

    name = "iterative_refine"
    _MAX_ITERATIONS = 3

    def generate(self, context: CodeContext, router: LLMRouter) -> Optional[PoCResult]:
        print(f"[Strategy] Running '{self.name}' ...")

        # Seed from DirectDescriptionPoC
        seed = DirectDescriptionPoC().generate(context, router)
        if seed is None:
            logger.warning("%s: seed generation failed.", self.name)
            return None

        poc_code = seed.poc_code

        for iteration in range(1, self._MAX_ITERATIONS + 1):
            print(f"  [Refine] iteration {iteration}/{self._MAX_ITERATIONS}")

            with tempfile.NamedTemporaryFile(
                suffix=".c", mode="w", delete=False, dir="/tmp"
            ) as tmp:
                tmp.write(poc_code)
                tmp_path = Path(tmp.name)

            ok, stderr = _try_compile(tmp_path)
            tmp_path.unlink(missing_ok=True)

            if ok:
                print(f"  [Refine] compilation succeeded on iteration {iteration}.")
                poc_path = _save_poc(poc_code, context.task.task_id, self.name)
                result = PoCResult(
                    strategy_name=self.name,
                    poc_code=poc_code,
                    poc_path=poc_path,
                    confidence=0.80,
                    notes=f"Compiles successfully after {iteration} iteration(s).",
                )
                print(f"[Strategy] '{self.name}' done — confidence {result.confidence:.2f}")
                return result

            # ── NEW: Auto-strip undefined references before asking LLM ─────
            poc_code = _strip_undefined_refs(poc_code, stderr)
            # Re-sanitize in case the LLM snuck in bad includes earlier
            poc_code = _sanitize_poc(poc_code)

            # Ask LLM to fix — with stronger anti-hallucination guidance
            refine_prompt = (
                f"The following PoC failed to compile.\n\n"
                f"## Compiler errors\n```\n{stderr[:3000]}\n```\n\n"
                f"## Current PoC code\n```c\n{poc_code}\n```\n\n"
                "Fix the compilation errors. CRITICAL RULES:\n"
                "- If errors say 'undefined reference to X', do NOT add a project header.\n"
                "  Instead REMOVE the call entirely, or reproduce X's logic inline using\n"
                "  only standard C (stdio.h, stdlib.h, string.h, stdint.h).\n"
                "- If you cannot reproduce a function, replace it with a simpler trigger\n"
                "  for the same bug class (e.g. direct buffer overflow via memcpy).\n"
                "- NEVER add project-specific headers like mosquitto.h, db.h, jq.h, etc.\n"
                "- NEVER add #include lines for headers that are not part of the C standard\n"
                "  library or POSIX.\n"
                "- Do NOT remove the core vulnerability trigger — only fix compile issues.\n"
                "- Output ONLY the corrected C/C++ code inside a single ```c ... ``` block.\n"
            )

            raw = router.chat(
                system_prompt=SYSTEM_PROMPT_POC_REFINER,
                user_prompt=refine_prompt,
                max_tokens=cfg.MAX_TOKENS,
                temperature=0.1,
            )
            if not raw:
                logger.warning("%s: refiner returned empty on iteration %d.", self.name, iteration)
                break

            poc_code = _extract_and_sanitize(raw)

        # Exhausted iterations — return best effort
        poc_path = _save_poc(poc_code, context.task.task_id, self.name)
        result = PoCResult(
            strategy_name=self.name,
            poc_code=poc_code,
            poc_path=poc_path,
            confidence=0.55,
            notes=f"Refinement exhausted {self._MAX_ITERATIONS} iterations without clean compile.",
        )
        print(f"[Strategy] '{self.name}' done — confidence {result.confidence:.2f}")
        return result


# ── Strategy 5 — DirectDescriptionPoC ─────────────────────────────────────

class DirectDescriptionPoC(PoCStrategy):
    """Generate a PoC directly from the vulnerability description."""

    name = "direct_description"

    def generate(self, context: CodeContext, router: LLMRouter) -> Optional[PoCResult]:
        print(f"[Strategy] Running '{self.name}' ...")

        snippets_text = _top_snippets_text(context)

        user_prompt = (
            f"## Vulnerability Description\n{context.description}\n\n"
            f"## Code Snippets\n```c\n{snippets_text}\n```\n\n"
            "Write a complete, self-contained C PoC that triggers this vulnerability.\n"
            "Requirements:\n"
            "- Only use standard C headers: stdio.h, stdlib.h, string.h, stdint.h, stdbool.h\n"
            "- Only call functions whose full definition you can see in the snippets above\n"
            "- Copy any required struct definitions inline\n"
            "- Do NOT include any project-specific header files (e.g. mosquitto.h, db.h)\n"
            "- Do NOT mock or stub any functions — use or reproduce their actual logic\n"
            "- Do NOT call project API functions — reproduce the vulnerable logic inline\n"
        )

        raw = router.chat(
            system_prompt=SYSTEM_PROMPT_POC_GENERATOR,
            user_prompt=user_prompt,
        )
        if not raw:
            return None

        poc_code = _extract_and_sanitize(raw)
        poc_path = _save_poc(poc_code, context.task.task_id, self.name)

        result = PoCResult(
            strategy_name=self.name,
            poc_code=poc_code,
            poc_path=poc_path,
            confidence=0.60,
            notes="Direct description PoC.",
        )
        print(f"[Strategy] '{self.name}' done — confidence {result.confidence:.2f}")
        return result


# ── Orchestrator ───────────────────────────────────────────────────────────

class PoCOrchestrator:
    """Run strategies in priority order and collect all results."""

    DEFAULT_ORDER: list[type[PoCStrategy]] = [
        AnalyzeFirstPoC,
        CallPathTargetedPoC,
        PatternReplayPoC,
        IterativeRefinePoC,
        DirectDescriptionPoC,
    ]

    def __init__(
        self,
        router: LLMRouter,
        strategy_order: Optional[list[type[PoCStrategy]]] = None,
    ) -> None:
        self.strategies: list[PoCStrategy] = [
            S() for S in (strategy_order or self.DEFAULT_ORDER)
        ]
        self.router = router

    def run(self, context: CodeContext) -> list[PoCResult]:
        """Execute every strategy, collecting all non-None results."""
        results: list[PoCResult] = []

        for strategy in self.strategies:
            print(f"\n{'='*50}")
            print(f"  Orchestrator → strategy '{strategy.name}'")
            print(f"{'='*50}")
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
                    "Strategy '%s' raised an exception — skipping.", strategy.name
                )

        # Sort by confidence descending
        results.sort(key=lambda r: r.confidence, reverse=True)

        print(f"\n[Orchestrator] Collected {len(results)} PoC(s):")
        for r in results:
            print(f"  • {r.strategy_name:25s}  confidence={r.confidence:.2f}  → {r.poc_path}")

        return results
