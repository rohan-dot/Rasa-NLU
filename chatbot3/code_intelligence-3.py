"""
crs/code_intelligence.py — Source analysis for the Cyber Reasoning System.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from crs.data_loader import CyberGymTask, get_source_files

logger = logging.getLogger(__name__)

# ── Vulnerability classification keywords ─────────────────────────────────

_VULN_KEYWORDS: dict[str, list[str]] = {
    "heap_overflow":   ["heap", "heap-buffer-overflow", "heap overflow", "heap-overflow"],
    "stack_overflow":  ["stack", "stack-buffer-overflow", "stack overflow"],
    "buffer_overflow": ["buffer overflow", "buffer-overflow", "bufferoverflow", "out-of-bounds write"],
    "use_after_free":  ["use after free", "use-after-free", "uaf", "dangling pointer"],
    "null_deref":      ["null pointer", "null dereference", "null-dereference", "nullptr"],
    "integer_overflow":["integer overflow", "integer-overflow", "int overflow", "wrap"],
    "format_string":   ["format string", "printf", "sprintf", "fprintf"],
    "oob_read":        ["out-of-bounds read", "oob read", "read past", "invalid read"],
    "type_confusion":  ["type confusion", "cast", "downcast", "type mismatch"],
    "double_free":     ["double free", "double-free"],
    "memory_leak":     ["memory leak", "leak"],
    "uninit_memory":   ["uninitialized", "uninit", "msan", "memory sanitizer"],
}


def classify_vulnerability(description: str) -> str:
    """Classify vulnerability type from a text description."""
    desc_lower = description.lower()
    for vuln_type, keywords in _VULN_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            return vuln_type
    return "other"


# ── Build system detection ─────────────────────────────────────────────────

def detect_build_system(repo: Path) -> dict:
    """
    Detect the build system used by the project.
    Searches both the repo root and immediate subdirectories for build files,
    since tarballs sometimes have CMakeLists.txt/Makefile in a nested dir.
    """
    # Helper: check root, then first-level subdirs, then rglob
    def _find_build_file(name: str) -> Optional[Path]:
        # Check repo root first
        if (repo / name).exists():
            return repo / name
        # Check immediate subdirectories
        for d in sorted(repo.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                if (d / name).exists():
                    return d / name
        # Recursive search as last resort
        found = list(repo.rglob(name))
        return found[0] if found else None

    cmake_file = _find_build_file("CMakeLists.txt")
    if cmake_file:
        source_dir = cmake_file.parent
        return {
            "type": "cmake",
            "build_cmd": "cmake --build build -j4",
            "configure_cmd": f"cmake -B build -S {source_dir}",
            "source_dir": str(source_dir),
            "entry_points": [cmake_file],
        }

    if (repo / "configure.ac").exists() or (repo / "configure").exists():
        return {
            "type": "autotools",
            "build_cmd": "make -j4",
            "configure_cmd": "autoreconf -fi && ./configure",
            "source_dir": str(repo),
            "entry_points": list(repo.glob("configure*")),
        }

    makefile = _find_build_file("Makefile") or _find_build_file("makefile")
    if makefile:
        return {
            "type": "make",
            "build_cmd": "make -j4",
            "configure_cmd": "",
            "source_dir": str(makefile.parent),
            "entry_points": [makefile],
        }

    meson_file = _find_build_file("meson.build")
    if meson_file:
        return {
            "type": "meson",
            "build_cmd": "meson compile -C build",
            "configure_cmd": "meson setup build",
            "source_dir": str(meson_file.parent),
            "entry_points": [meson_file],
        }

    return {
        "type": "unknown",
        "build_cmd": "make -j4",
        "configure_cmd": "",
        "source_dir": str(repo),
        "entry_points": [],
    }


# ── File relevance ranking ─────────────────────────────────────────────────

_UNSAFE_APIS = {
    "strcpy", "strcat", "sprintf", "gets", "scanf",
    "memcpy", "memmove", "malloc", "realloc", "free",
    "strncpy", "strncat", "snprintf",
}


def rank_files_by_relevance(
    task: CyberGymTask,
    files: List[Path],
    description: str,
) -> List[Tuple[Path, float]]:
    """
    Rank source files by relevance to the vulnerability description.
    Returns list of (path, score) sorted descending.
    """
    desc_lower = description.lower()
    desc_words = set(re.findall(r"\b\w+\b", desc_lower))

    scored: list[tuple[Path, float]] = []
    for f in files:
        score = 0.0
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            text_lower = text.lower()
            text_words = set(re.findall(r"\b\w+\b", text_lower))

            # Word overlap with description
            overlap = len(desc_words & text_words)
            score += overlap * 0.1

            # Unsafe API usage
            for api in _UNSAFE_APIS:
                if api in text_lower:
                    score += 0.5

            # Description keyword hits in file
            for kw in desc_words:
                if len(kw) > 4 and kw in text_lower:
                    score += 0.3

            # File name hints
            fname_lower = f.name.lower()
            for kw in desc_words:
                if len(kw) > 3 and kw in fname_lower:
                    score += 2.0

        except Exception:
            pass

        scored.append((f, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ── Snippet extraction ─────────────────────────────────────────────────────

def extract_relevant_snippets(
    ranked_files: List[Tuple[Path, float]],
    max_chars: int = 8000,
) -> str:
    """
    Extract and concatenate the most relevant code snippets.
    Returns a single string with file headers.
    """
    parts: list[str] = []
    total = 0

    for path, score in ranked_files:
        if total >= max_chars:
            break
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            budget = max_chars - total
            snippet = content[:budget]
            header = f"\n// === {path.name} (score={score:.2f}) ===\n"
            parts.append(header + snippet)
            total += len(snippet) + len(header)
        except Exception:
            pass

    return "".join(parts)


# ── CodeContext ────────────────────────────────────────────────────────────

@dataclass
class CodeContext:
    task: CyberGymTask
    ranked_files: List[Tuple[Path, float]]
    top_snippets: str                      # always a str, never a list
    build_info: Dict
    vuln_type: str
    description: str
    include_dirs: List[Path] = field(default_factory=list)


def build_context(task: CyberGymTask) -> CodeContext:
    """Build a full CodeContext from a CyberGymTask."""
    repo = Path(task.repo_path)

    # Collect source files
    all_files = get_source_files(task)

    # Rank by relevance
    ranked = rank_files_by_relevance(task, all_files, task.vulnerability_description)

    # Top snippets as a string
    top_snippets = extract_relevant_snippets(ranked[:10])

    # Build system
    build_info = detect_build_system(repo)

    # Vuln type
    vuln_type = classify_vulnerability(task.vulnerability_description)

    # Include directories — all dirs containing .h files
    include_dirs: list[Path] = []
    seen: set[str] = set()
    for p in sorted(repo.rglob("*.h")):
        d = p.parent
        ds = str(d)
        if ds not in seen:
            include_dirs.append(d)
            seen.add(ds)

    # Report if many files
    if len(all_files) > 200:
        print(f"[data_loader] {len(all_files)} source files found; truncating to 200")

    print(f"  [Context] vuln_type={vuln_type}, build={build_info['type']}, "
          f"top_files={[f.name for f, _ in ranked[:3]]}")

    return CodeContext(
        task=task,
        ranked_files=ranked,
        top_snippets=top_snippets,
        build_info=build_info,
        vuln_type=vuln_type,
        description=task.vulnerability_description,
        include_dirs=include_dirs,
    )


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from crs.data_loader import load_task_from_local

    if len(sys.argv) < 2:
        print("Usage: python -m crs.code_intelligence <task_dir>")
        sys.exit(1)

    task = load_task_from_local(sys.argv[1])
    ctx = build_context(task)
    print(f"\nvuln_type : {ctx.vuln_type}")
    print(f"build     : {ctx.build_info['type']}")
    print(f"top files : {[str(p) for p, _ in ctx.ranked_files[:5]]}")
    print(f"\n--- Top snippets (first 500 chars) ---")
    print(ctx.top_snippets[:500])
