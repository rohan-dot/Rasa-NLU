"""
crs/harness_finder.py — Locate the fuzz harness in a CyberGym task repo.

Every OSS-Fuzz project has a fuzz target with the entry point
LLVMFuzzerTestOneInput(const uint8_t *data, size_t size).
This module finds it, extracts the harness code, and identifies
what project functions the harness calls.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from crs.data_loader import CyberGymTask


@dataclass
class HarnessInfo:
    harness_path: Path                       # file containing the harness
    harness_code: str                        # full source of the harness file
    harness_function_code: str               # just the LLVMFuzzerTestOneInput function
    called_functions: List[str]              # functions called from the harness
    includes: List[str]                      # #include lines from the harness
    has_init: bool                           # whether LLVMFuzzerInitialize exists
    project_headers: List[str]               # project-specific headers included
    all_harness_files: List[Path]            # all files containing harness functions


def find_harness(task: CyberGymTask) -> Optional[HarnessInfo]:
    """
    Search the repo for LLVMFuzzerTestOneInput.
    Returns HarnessInfo if found, None otherwise.
    """
    repo = Path(task.repo_path).resolve()
    harness_files: list[Path] = []

    # Search all C/C++ files for the harness entry point
    for ext in ("*.c", "*.cpp", "*.cc", "*.cxx"):
        for f in sorted(repo.rglob(ext)):
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                if "LLVMFuzzerTestOneInput" in content:
                    harness_files.append(f)
            except Exception:
                continue

    if not harness_files:
        print(f"[harness] No LLVMFuzzerTestOneInput found in {repo}")
        # Try to find a main() that reads from stdin as fallback
        return _find_stdin_main(repo)

    # Pick the best harness — prefer files in fuzz/ or test/ directories,
    # or files with "fuzz" in the name
    best = harness_files[0]
    for f in harness_files:
        name_lower = f.name.lower()
        parts_lower = str(f).lower()
        if "fuzz" in name_lower or "fuzz" in parts_lower:
            best = f
            break

    print(f"[harness] Found fuzz harness: {best}")
    if len(harness_files) > 1:
        print(f"[harness] Also found {len(harness_files) - 1} other harness file(s)")

    content = best.read_text(encoding="utf-8", errors="replace")

    # Extract the harness function body
    harness_func = _extract_function(content, "LLVMFuzzerTestOneInput")

    # Extract called functions
    called = _extract_called_functions(harness_func or content)

    # Extract includes
    includes = re.findall(r'#include\s*[<"]([^>"]+)[>"]', content)

    # Identify project-specific headers (not standard C/C++)
    std_headers = {
        "stdio.h", "stdlib.h", "string.h", "stdint.h", "stdbool.h",
        "stddef.h", "unistd.h", "math.h", "assert.h", "errno.h",
        "limits.h", "ctype.h", "signal.h", "time.h", "fcntl.h",
        "sys/types.h", "sys/stat.h",
    }
    project_headers = [h for h in includes if h not in std_headers]

    # Check for LLVMFuzzerInitialize
    has_init = "LLVMFuzzerInitialize" in content

    return HarnessInfo(
        harness_path=best,
        harness_code=content,
        harness_function_code=harness_func or "",
        called_functions=called,
        includes=includes,
        has_init=has_init,
        project_headers=project_headers,
        all_harness_files=harness_files,
    )


def _find_stdin_main(repo: Path) -> Optional[HarnessInfo]:
    """
    Fallback: find a main() that reads from stdin or a file argument.
    Some older OSS-Fuzz targets use this pattern instead of LLVMFuzzerTestOneInput.
    """
    for ext in ("*.c", "*.cpp", "*.cc"):
        for f in sorted(repo.rglob(ext)):
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                if re.search(r"\bint\s+main\s*\(", content):
                    if any(kw in content for kw in ["stdin", "fread", "fgets", "argv[1]"]):
                        print(f"[harness] Found stdin-reading main() in {f}")
                        main_func = _extract_function(content, "main")
                        called = _extract_called_functions(main_func or content)
                        includes = re.findall(r'#include\s*[<"]([^>"]+)[>"]', content)
                        return HarnessInfo(
                            harness_path=f,
                            harness_code=content,
                            harness_function_code=main_func or "",
                            called_functions=called,
                            includes=includes,
                            has_init=False,
                            project_headers=[],
                            all_harness_files=[f],
                        )
            except Exception:
                continue
    return None


def _extract_function(source: str, func_name: str) -> Optional[str]:
    """
    Extract a function body from source code by matching braces.
    Returns the full function including signature and body.
    """
    # Find the function signature
    pattern = rf'(?:extern\s+"C"\s+)?(?:int|void)\s+{func_name}\s*\([^)]*\)'
    m = re.search(pattern, source)
    if not m:
        return None

    start = m.start()
    # Find the opening brace
    brace_pos = source.find("{", m.end())
    if brace_pos == -1:
        return None

    # Match braces to find the end
    depth = 0
    pos = brace_pos
    while pos < len(source):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[start:pos + 1]
        pos += 1

    return source[start:]  # unclosed — return what we have


def _extract_called_functions(code: str) -> list[str]:
    """
    Extract function calls from a code snippet.
    Returns a list of unique function names.
    """
    # Match identifier followed by (
    calls = re.findall(r"\b([a-zA-Z_]\w+)\s*\(", code)

    # Filter out C keywords and common macros
    skip = {
        "if", "for", "while", "switch", "return", "sizeof", "typeof",
        "malloc", "calloc", "realloc", "free", "memcpy", "memset",
        "memmove", "printf", "fprintf", "sprintf", "snprintf",
        "fopen", "fclose", "fread", "fwrite", "fseek", "ftell",
        "assert", "exit", "abort",
        "LLVMFuzzerTestOneInput", "LLVMFuzzerInitialize",
        "int", "void", "char", "unsigned",
    }
    return list(dict.fromkeys(c for c in calls if c not in skip))


def summarize_harness(harness: HarnessInfo) -> str:
    """
    Create a human-readable summary of the harness for LLM prompts.
    """
    lines = [
        f"Harness file: {harness.harness_path.name}",
        f"Project headers: {harness.project_headers}",
        f"Functions called from harness: {harness.called_functions[:15]}",
        f"Has LLVMFuzzerInitialize: {harness.has_init}",
        "",
        "--- Harness source code ---",
        harness.harness_code[:5000],
    ]
    return "\n".join(lines)
