"""
crs/code_intelligence.py – Step 2
==================================
Static analysis of a vulnerable codebase.  Produces ranked, structured context
that the LLM will consume when generating a Proof-of-Concept.

Pure Python – no LLM calls, no external deps beyond stdlib + pathlib.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from crs.data_loader import CyberGymTask

# =====================================================================
# Constants
# =====================================================================

# Patterns whose presence in a source file increases the chance that
# the file is security-relevant.
UNSAFE_API_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bstrcpy\b"),
    re.compile(r"\bstrcat\b"),
    re.compile(r"\bsprintf\b"),
    re.compile(r"\bgets\b"),
    re.compile(r"\bmemcpy\b"),
    re.compile(r"\bmalloc\b"),
    re.compile(r"\brealloc\b"),
    re.compile(r"\bfree\b"),
    re.compile(r"\bscanf\b"),
    re.compile(r"\bstrlen\b"),
    re.compile(r"buffer\s*\["),
    re.compile(r"->\s*buf\b"),
    re.compile(r"->\s*data\b"),
]

VULN_TYPES: list[str] = [
    "buffer_overflow",
    "heap_overflow",
    "stack_overflow",
    "use_after_free",
    "double_free",
    "null_deref",
    "integer_overflow",
    "type_confusion",
    "oob_read",
    "oob_write",
    "format_string",
    "race_condition",
    "other",
]

# Keyword → vuln-type mapping.  Order matters: first match wins, so more
# specific patterns appear before general ones.
_VULN_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (["heap buffer overflow", "heap-buffer-overflow", "heap overflow"],
     "heap_overflow"),
    (["stack buffer overflow", "stack-buffer-overflow", "stack overflow",
      "stack-based buffer overflow"],
     "stack_overflow"),
    (["buffer overflow", "buffer overrun", "buffer over-read"],
     "buffer_overflow"),
    (["use after free", "use-after-free", "uaf"],
     "use_after_free"),
    (["double free", "double-free"],
     "double_free"),
    (["null pointer", "null dereference", "null deref", "nullptr",
      "null-pointer-dereference", "segmentation fault"],
     "null_deref"),
    (["integer overflow", "integer underflow", "int overflow",
      "integer wraparound", "integer truncation"],
     "integer_overflow"),
    (["type confusion", "type-confusion", "type cast", "bad cast",
      "bad-cast"],
     "type_confusion"),
    (["out-of-bounds read", "oob read", "out of bounds read",
      "heap-buffer-overflow read", "over-read"],
     "oob_read"),
    (["out-of-bounds write", "oob write", "out of bounds write",
      "heap-buffer-overflow write", "over-write"],
     "oob_write"),
    (["format string", "format-string", "printf format"],
     "format_string"),
    (["race condition", "toctou", "data race", "race-condition"],
     "race_condition"),
]


# =====================================================================
# Helper: lightweight tokeniser for natural-language descriptions
# =====================================================================

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STOPWORDS: frozenset[str] = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into about between through during and or but not "
    "this that these those it its he she they them their if then than "
    "so no all each any both few more most other some such only same "
    "too very just because when which what where who whom how".split()
)


def _tokenize(text: str) -> list[str]:
    """Return lower-cased, non-stopword tokens."""
    return [
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if t.lower() not in _STOPWORDS and len(t) > 2
    ]


def _read_text_safe(path: Path, max_bytes: int = 512_000) -> str:
    """Read a file as text, returning '' on any error."""
    try:
        raw = path.read_bytes()[:max_bytes]
        return raw.decode("utf-8", errors="replace")
    except (OSError, PermissionError):
        return ""


def _count_lines(text: str) -> int:
    return text.count("\n") + (1 if text and not text.endswith("\n") else 0)


# =====================================================================
# 1. File Ranker
# =====================================================================

def rank_files_by_relevance(
    task: CyberGymTask,
    source_files: list[Path],
    description: str,
) -> list[tuple[Path, float]]:
    """
    Rank source files by likely relevance to the vulnerability description.

    Scoring combines:
      a) Keyword overlap (description tokens vs filename + first 50 lines).
      b) Unsafe-API presence (weighted pattern count).
      c) Size heuristic (prefer medium files, penalise very large / tiny).

    Returns up to 20 (path, score) pairs sorted descending by score.
    """
    desc_tokens = set(_tokenize(description))
    if not desc_tokens:
        # Fallback: at least use the raw words so we don't score everything 0
        desc_tokens = {w.lower() for w in description.split() if len(w) > 2}

    scored: list[tuple[Path, float]] = []

    for fpath in source_files:
        content = _read_text_safe(fpath)
        if not content:
            continue

        lines = content.splitlines()
        num_lines = len(lines)

        # --- (a) keyword overlap ----------------------------------------
        # Check filename
        fname_lower = fpath.name.lower()
        fname_tokens = set(_tokenize(fname_lower))
        fname_hits = len(fname_tokens & desc_tokens)

        # Check first 50 lines
        head = "\n".join(lines[:50]).lower()
        head_tokens = set(_tokenize(head))
        head_hits = len(head_tokens & desc_tokens)

        # Also do a quick whole-file grep for exact substrings of multi-word
        # description phrases (catches function/struct names the tokeniser
        # would miss).
        content_lower = content.lower()
        exact_hits = sum(
            1 for tok in desc_tokens
            if len(tok) > 4 and tok in content_lower
        )

        # Normalise: divide by sqrt(num_lines) to avoid big-file bias but
        # not penalise too harshly.
        normaliser = max(math.sqrt(num_lines), 1.0)
        keyword_score = (fname_hits * 3.0 + head_hits * 2.0 + exact_hits) / normaliser

        # --- (b) unsafe API presence ------------------------------------
        unsafe_score = 0.0
        for pat in UNSAFE_API_PATTERNS:
            matches = pat.findall(content)
            if matches:
                unsafe_score += 0.3 * len(matches)

        # Cap the unsafe score so it doesn't overwhelm keyword relevance
        unsafe_score = min(unsafe_score, 15.0)

        # --- (c) size heuristic -----------------------------------------
        if num_lines < 20:
            size_score = -2.0
        elif num_lines > 5000:
            size_score = -1.0
        elif 200 <= num_lines <= 2000:
            size_score = 1.0
        else:
            size_score = 0.0

        total = keyword_score + unsafe_score + size_score
        scored.append((fpath, total))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:20]


# =====================================================================
# 2. Code Snippet Extractor
# =====================================================================

def extract_relevant_snippets(
    ranked_files: list[tuple[Path, float]],
    description: str,
    max_total_chars: int = 12_000,
) -> str:
    """
    From the top-ranked files extract the most relevant code regions.

    For each of the top-5 files:
      1. Score every line by keyword overlap with the description.
      2. Find the highest-scoring contiguous cluster (via a sliding window).
      3. Expand the cluster by ±40 lines.
      4. Emit the snippet with a header comment.
    Concatenate until *max_total_chars* is consumed.
    """
    desc_tokens = set(_tokenize(description))
    snippets: list[str] = []
    budget = max_total_chars

    for fpath, _score in ranked_files[:5]:
        if budget <= 0:
            break

        content = _read_text_safe(fpath)
        if not content:
            continue
        lines = content.splitlines()
        if not lines:
            continue

        # --- per-line relevance score ---
        line_scores = []
        for line in lines:
            tokens = set(_tokenize(line))
            line_scores.append(len(tokens & desc_tokens))

        # --- find best cluster via sliding window of size 10 ---
        window = min(10, len(lines))
        best_start = 0
        best_sum = sum(line_scores[:window])
        current_sum = best_sum
        for i in range(1, len(lines) - window + 1):
            current_sum += line_scores[i + window - 1] - line_scores[i - 1]
            if current_sum > best_sum:
                best_sum = current_sum
                best_start = i

        # --- expand ±40 around the cluster centre ---
        cluster_centre = best_start + window // 2
        start = max(0, cluster_centre - 40)
        end = min(len(lines), cluster_centre + 41)  # exclusive

        # Try to make the path relative for readability
        try:
            display_path = fpath.relative_to(fpath.parents[2])
        except (ValueError, IndexError):
            display_path = fpath.name

        header = f"// FILE: {display_path} (lines {start + 1}-{end})\n"
        snippet_body = "\n".join(lines[start:end])
        snippet = header + snippet_body + "\n\n"

        if len(snippet) > budget:
            # Truncate to fit remaining budget
            snippet = snippet[:budget]

        snippets.append(snippet)
        budget -= len(snippet)

    return "".join(snippets)


# =====================================================================
# 3. Build System Detector
# =====================================================================

def detect_build_system(repo_path: Path) -> dict:
    """
    Detect how to build the project and return a dict with:
      type, configure_cmd, build_cmd, entry_points
    """
    result: dict = {
        "type": "unknown",
        "configure_cmd": [],
        "build_cmd": [],
        "entry_points": [],
    }

    # Scan for build files (walk at most 2 levels to stay fast)
    has_cmake = False
    has_autotools_ac = False
    has_configure = False
    has_makefile = False
    has_meson = False

    for child in _shallow_glob(repo_path, max_depth=2):
        name = child.name
        if name == "CMakeLists.txt":
            has_cmake = True
        elif name == "configure.ac" or name == "configure.in":
            has_autotools_ac = True
        elif name == "configure" and child.is_file():
            has_configure = True
        elif name in ("Makefile", "makefile", "GNUmakefile"):
            has_makefile = True
        elif name == "meson.build":
            has_meson = True

    # Priority: cmake > autotools > meson > make > unknown
    if has_cmake:
        result["type"] = "cmake"
        result["configure_cmd"] = [
            "cmake", "-S", ".", "-B", "build",
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DCMAKE_C_FLAGS=-fsanitize=address -g",
            "-DCMAKE_CXX_FLAGS=-fsanitize=address -g",
        ]
        result["build_cmd"] = ["cmake", "--build", "build", "-j4"]
    elif has_autotools_ac or has_configure:
        result["type"] = "autotools"
        if has_autotools_ac and not has_configure:
            result["configure_cmd"] = ["autoreconf", "-fi", "&&", "./configure",
                                       'CFLAGS=-fsanitize=address -g',
                                       'CXXFLAGS=-fsanitize=address -g']
        else:
            result["configure_cmd"] = ["./configure",
                                       'CFLAGS=-fsanitize=address -g',
                                       'CXXFLAGS=-fsanitize=address -g']
        result["build_cmd"] = ["make", "-j4"]
    elif has_meson:
        result["type"] = "meson"
        result["configure_cmd"] = [
            "meson", "setup", "build",
            "-Db_sanitize=address",
            "--buildtype=debug",
        ]
        result["build_cmd"] = ["ninja", "-C", "build"]
    elif has_makefile:
        result["type"] = "make"
        result["configure_cmd"] = []
        result["build_cmd"] = [
            "make", "-j4",
            'CFLAGS=-fsanitize=address -g',
            'CXXFLAGS=-fsanitize=address -g',
        ]
    # else: stays "unknown"

    # --- entry points: look for int main( ---
    result["entry_points"] = _find_entry_points(repo_path)

    return result


def _shallow_glob(root: Path, max_depth: int = 2) -> list[Path]:
    """Yield paths up to *max_depth* levels below *root* (non-recursive for speed)."""
    results: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            children = list(current.iterdir())
        except (OSError, PermissionError):
            continue
        for child in children:
            results.append(child)
            if child.is_dir() and depth < max_depth and not child.name.startswith("."):
                stack.append((child, depth + 1))
    return results


_MAIN_RE = re.compile(r"\bint\s+main\s*\(")


def _find_entry_points(repo_path: Path) -> list[str]:
    """Find files containing `int main(` in common locations."""
    c_exts = {".c", ".cpp", ".cc", ".cxx"}
    search_dirs = [repo_path]
    src = repo_path / "src"
    if src.is_dir():
        search_dirs.append(src)

    entry_points: list[str] = []
    seen: set[Path] = set()

    for search_root in search_dirs:
        for ext in c_exts:
            for fpath in search_root.glob(f"*{ext}"):
                if fpath in seen:
                    continue
                seen.add(fpath)
                content = _read_text_safe(fpath, max_bytes=64_000)
                if _MAIN_RE.search(content):
                    try:
                        entry_points.append(str(fpath.relative_to(repo_path)))
                    except ValueError:
                        entry_points.append(fpath.name)

    # Also look one level into sub-dirs of repo root for main() files
    for child_dir in repo_path.iterdir():
        if child_dir.is_dir() and child_dir.name not in (".", "..", "test", "tests", "build"):
            for ext in c_exts:
                for fpath in child_dir.glob(f"*{ext}"):
                    if fpath in seen:
                        continue
                    seen.add(fpath)
                    content = _read_text_safe(fpath, max_bytes=64_000)
                    if _MAIN_RE.search(content):
                        try:
                            entry_points.append(str(fpath.relative_to(repo_path)))
                        except ValueError:
                            entry_points.append(fpath.name)

    return entry_points


# =====================================================================
# 4. Vulnerability Type Classifier
# =====================================================================

def classify_vulnerability(description: str) -> str:
    """
    Classify the vulnerability type via keyword matching on *description*.
    No LLM call – pure heuristic.  Returns one of VULN_TYPES.
    """
    text = description.lower()
    for keywords, vuln_type in _VULN_KEYWORD_MAP:
        for kw in keywords:
            if kw in text:
                return vuln_type
    return "other"


# =====================================================================
# 5. Context Bundle
# =====================================================================

@dataclass
class CodeContext:
    """Everything the LLM needs to generate a PoC."""
    task: CyberGymTask
    ranked_files: list[tuple[Path, float]]
    top_snippets: str           # formatted code string
    build_info: dict
    vuln_type: str
    description: str

    # ------------------------------------------------------------------
    # Pretty-print summary (useful for logging / debugging)
    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [
            f"=== CodeContext for task {self.task.task_id} ===",
            f"Vulnerability type : {self.vuln_type}",
            f"Build system       : {self.build_info.get('type', '?')}",
            f"Build cmd          : {' '.join(self.build_info.get('build_cmd', []))}",
            f"Entry points       : {self.build_info.get('entry_points', [])}",
            f"Ranked files (top 5):",
        ]
        for fpath, score in self.ranked_files[:5]:
            lines.append(f"  {score:6.2f}  {fpath.name}")
        snippet_preview = self.top_snippets[:500].rstrip()
        lines.append(f"Snippet preview ({len(self.top_snippets)} chars total):")
        lines.append(snippet_preview)
        lines.append("...")
        return "\n".join(lines)


def build_context(task: CyberGymTask) -> CodeContext:
    """
    Orchestrate static analysis for a single CyberGym task.

    1. Gather source files (if not already done).
    2. Rank files by relevance to the description.
    3. Extract the most relevant code snippets.
    4. Detect the build system.
    5. Classify the vulnerability type.
    6. Bundle everything into a CodeContext.
    """
    description = task.description

    # 1. ensure source files are collected
    if not task.source_files:
        task.gather_source_files()

    # 2. rank
    ranked = rank_files_by_relevance(task, task.source_files, description)

    # 3. snippets
    snippets = extract_relevant_snippets(ranked, description)

    # 4. build system
    build_info = detect_build_system(task.repo_path)

    # 5. vuln type
    vuln_type = classify_vulnerability(description)

    return CodeContext(
        task=task,
        ranked_files=ranked,
        top_snippets=snippets,
        build_info=build_info,
        vuln_type=vuln_type,
        description=description,
    )


# =====================================================================
# __main__ – quick self-test
# =====================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path as _P

    def _self_test() -> None:
        """
        Usage:
            python -m crs.code_intelligence [TASK_DIR]

        If TASK_DIR is omitted, a synthetic in-memory test is run.
        """
        if len(sys.argv) > 1:
            # Real task directory supplied
            from crs.data_loader import load_task
            task_dir = _P(sys.argv[1]).resolve()
            print(f"Loading task from {task_dir} ...")
            task = load_task(task_dir)
            ctx = build_context(task)
            print(ctx.summary())
            return

        # ---- synthetic self-test (no real repo needed) -----------------
        import tempfile, textwrap

        print("Running synthetic self-test …\n")

        with tempfile.TemporaryDirectory(prefix="crs_test_") as tmpdir:
            root = _P(tmpdir)

            # Fake source files
            (root / "CMakeLists.txt").write_text("project(vuln)\nadd_executable(vuln main.c)\n")

            main_c = root / "main.c"
            main_c.write_text(textwrap.dedent("""\
                #include <stdio.h>
                #include <string.h>
                #include <stdlib.h>

                void process_input(char *input) {
                    char buffer[64];
                    strcpy(buffer, input);   // vulnerable!
                    printf("Got: %s\\n", buffer);
                }

                int main(int argc, char **argv) {
                    if (argc < 2) return 1;
                    char *data = malloc(4096);
                    memcpy(data, argv[1], strlen(argv[1]));
                    process_input(data);
                    free(data);
                    return 0;
                }
            """))

            util_c = root / "util.c"
            util_c.write_text(textwrap.dedent("""\
                // utility helpers – not directly vulnerable
                int add(int a, int b) { return a + b; }
            """))

            desc = (
                "A heap-based buffer overflow exists in the process_input function "
                "of main.c. The strcpy call copies user-supplied input into a "
                "fixed-size stack buffer without bounds checking, allowing an "
                "attacker to overwrite the return address."
            )

            task = CyberGymTask(
                task_id="synth-test-001",
                repo_path=root,
                description=desc,
            )
            task.gather_source_files()

            ctx = build_context(task)

            print(ctx.summary())
            print()

            # Quick assertions
            assert ctx.vuln_type in ("heap_overflow", "buffer_overflow", "stack_overflow"), \
                f"Unexpected vuln_type: {ctx.vuln_type}"
            assert ctx.build_info["type"] == "cmake"
            assert any("main.c" in str(p) for p, _ in ctx.ranked_files), \
                "main.c should be top-ranked"
            assert "strcpy" in ctx.top_snippets, \
                "Snippet should contain the vulnerable strcpy call"

            print("All assertions passed.")

    _self_test()
