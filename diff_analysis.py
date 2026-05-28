"""
diff_analysis.py — Delta-aware vulnerability focusing.

AIxCC challenges inject a vulnerability in a specific commit. The diff
tells you exactly which functions changed. Instead of analyzing the
whole codebase, focus on the changed code — this is what every top
team does (Atlantis, FuzzingBrain, Theori).

Sources of the diff, in priority order:
1. A .diff/.patch file provided with the challenge (--diff-file)
2. git diff between the vulnerable commit and its parent (--base-commit)
3. git diff HEAD~1 (assume the last commit introduced the bug)

Output: a set of (file, function, line_range) that changed, used to
boost scanner priority and direct fuzzing.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("gemma-fuzzer.diff")


@dataclass
class ChangedRegion:
    file: str
    start_line: int
    end_line: int
    functions: list[str] = field(default_factory=list)


@dataclass
class DiffResult:
    changed_files: set[str] = field(default_factory=set)
    changed_functions: set[str] = field(default_factory=set)
    regions: list[ChangedRegion] = field(default_factory=list)
    has_diff: bool = False


def analyze_diff(src_dir: str, diff_file: str = None,
                 base_commit: str = None) -> DiffResult:
    """Extract changed files/functions from a diff or git history."""
    result = DiffResult()

    diff_text = ""

    # 1. Explicit diff file
    if diff_file and Path(diff_file).exists():
        diff_text = Path(diff_file).read_text(errors="replace")
        logger.info("[diff] Using diff file: %s", diff_file)

    # 2. git diff against base commit
    elif base_commit:
        try:
            result_proc = subprocess.run(
                ["git", "-C", src_dir, "diff", f"{base_commit}", "HEAD"],
                capture_output=True, text=True, timeout=30,
            )
            diff_text = result_proc.stdout
            logger.info("[diff] Using git diff %s..HEAD", base_commit)
        except Exception as exc:
            logger.warning("[diff] git diff failed: %s", exc)

    # 3. Last commit (assume it introduced the bug)
    if not diff_text:
        try:
            result_proc = subprocess.run(
                ["git", "-C", src_dir, "diff", "HEAD~1", "HEAD"],
                capture_output=True, text=True, timeout=30,
            )
            diff_text = result_proc.stdout
            if diff_text:
                logger.info("[diff] Using git diff HEAD~1..HEAD")
        except Exception:
            pass

    if not diff_text:
        logger.info("[diff] No diff available — analyzing full codebase.")
        return result

    result.has_diff = True
    _parse_unified_diff(diff_text, result)

    logger.info("[diff] %d changed files, %d changed functions.",
                len(result.changed_files), len(result.changed_functions))
    for fn in list(result.changed_functions)[:10]:
        logger.info("[diff]   changed function: %s", fn)

    return result


def _parse_unified_diff(diff_text: str, result: DiffResult):
    """Parse a unified diff to extract changed files and line ranges."""
    current_file = None

    # Match: +++ b/path/to/file.c
    file_re = re.compile(r'^\+\+\+ b/(.+)$')
    # Match: @@ -old,n +new,n @@ optional_function_context
    hunk_re = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@(.*)$')

    for line in diff_text.split("\n"):
        fm = file_re.match(line)
        if fm:
            current_file = fm.group(1).strip()
            # Only care about C/C++ source
            if any(current_file.endswith(ext) for ext in
                   (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp")):
                result.changed_files.add(current_file)
            else:
                current_file = None
            continue

        hm = hunk_re.match(line)
        if hm and current_file:
            start = int(hm.group(1))
            count = int(hm.group(2)) if hm.group(2) else 1
            context = hm.group(3).strip()

            region = ChangedRegion(
                file=current_file,
                start_line=start,
                end_line=start + count,
            )

            # The hunk context often names the enclosing function:
            # @@ -10,5 +10,7 @@ static int parse_value(const char *p)
            func_match = re.search(r'\b(\w+)\s*\(', context)
            if func_match:
                fname = func_match.group(1)
                if fname not in ("if", "while", "for", "switch", "return"):
                    region.functions.append(fname)
                    result.changed_functions.add(fname)

            result.regions.append(region)


def boost_findings(findings, diff_result: DiffResult):
    """Boost confidence of findings that touch changed code.
    
    A finding in a function that changed in the bug-injecting commit
    is far more likely to be the real target.
    """
    if not diff_result.has_diff:
        return findings

    for f in findings:
        in_changed_func = f.function in diff_result.changed_functions
        in_changed_file = any(
            cf.endswith(f.file) or f.file.endswith(cf)
            for cf in diff_result.changed_files
        )

        if in_changed_func:
            f.confidence = min(1.0, f.confidence + 0.3)
            logger.info("[diff] BOOST: %s is in changed function (+0.3 → %.2f)",
                       f.function, f.confidence)
        elif in_changed_file:
            f.confidence = min(1.0, f.confidence + 0.1)

    findings.sort(key=lambda f: -f.confidence)
    return findings


def get_diff_context(diff_result: DiffResult, max_funcs: int = 20) -> str:
    """Format changed code as context for the LLM scanner."""
    if not diff_result.has_diff:
        return ""

    lines = ["## CHANGED CODE (focus your analysis here)\n"]
    lines.append("This codebase has a diff. A vulnerability was likely "
                 "introduced in the following changed locations:\n")

    for cf in sorted(diff_result.changed_files):
        lines.append(f"  File: {cf}")

    if diff_result.changed_functions:
        lines.append("\nChanged functions (HIGHEST PRIORITY for audit):")
        for fn in sorted(diff_result.changed_functions)[:max_funcs]:
            lines.append(f"  - {fn}")

    return "\n".join(lines)
