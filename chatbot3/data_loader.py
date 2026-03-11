"""
CRS Data Loader — Step 1
Handles loading CyberGym Level 1 tasks from HuggingFace or local disk,
extracting tarballs, detecting project language, and enumerating source files.
"""

from __future__ import annotations

import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from crs.config import cfg

# ── Extension sets ───────────────────────────────────────────────
_C_EXTS = {".c"}
_CPP_EXTS = {".cpp", ".cc", ".cxx"}
_HDR_EXTS = {".h", ".hpp", ".hxx"}
_ALL_SOURCE_EXTS = _C_EXTS | _CPP_EXTS | _HDR_EXTS


# ── Task dataclass ───────────────────────────────────────────────
@dataclass
class CyberGymTask:
    """Represents a single CyberGym Level 1 task ready for processing."""

    task_id: str
    project_name: str
    project_language: str            # "c" or "c++"
    vulnerability_description: str
    repo_path: Path                  # root of the extracted source tree
    raw_tarball: Path


# ── Tarball extraction ───────────────────────────────────────────

def _extract_tarball(tarball: Path, dest: Path) -> Path:
    """
    Extract *tarball* into *dest* and return the effective repo root.

    Handles two common layouts:
      1. Flat — files directly inside the tar.
      2. Nested — a single top-level directory wrapping everything.
    In case 2 we return the inner directory so callers always get the
    "real" source root.
    """
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[data_loader] Extracting {tarball.name} -> {dest}")

    with tarfile.open(tarball, "r:gz") as tf:
        # Security: reject paths that escape the destination
        for member in tf.getmembers():
            resolved = (dest / member.name).resolve()
            if not str(resolved).startswith(str(dest.resolve())):
                raise RuntimeError(
                    f"Tarball contains unsafe path: {member.name}"
                )
        tf.extractall(dest, filter="data")  # Python 3.12+ filter; falls back below
    
    # Determine effective root: if there's exactly one top-level dir and
    # no top-level files, step into it.
    top_level = [p for p in dest.iterdir() if not p.name.startswith(".")]
    if len(top_level) == 1 and top_level[0].is_dir():
        return top_level[0]
    return dest


def _safe_extract_tarball(tarball: Path, dest: Path) -> Path:
    """
    Wrapper around _extract_tarball that handles older Python versions
    where the `filter` kwarg to extractall is not supported.
    """
    try:
        return _extract_tarball(tarball, dest)
    except TypeError:
        # Python < 3.12: filter kwarg not accepted
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[data_loader] Extracting {tarball.name} -> {dest} (no filter)")
        with tarfile.open(tarball, "r:gz") as tf:
            for member in tf.getmembers():
                resolved = (dest / member.name).resolve()
                if not str(resolved).startswith(str(dest.resolve())):
                    raise RuntimeError(
                        f"Tarball contains unsafe path: {member.name}"
                    )
            tf.extractall(dest)

        top_level = [p for p in dest.iterdir() if not p.name.startswith(".")]
        if len(top_level) == 1 and top_level[0].is_dir():
            return top_level[0]
        return dest


# ── Language detection ───────────────────────────────────────────

def _detect_language(repo_path: Path) -> str:
    """
    Walk *repo_path* and count .c vs .cpp/.cc/.cxx files.
    Returns ``"c"`` or ``"c++"``; defaults to ``"c"`` on tie or empty repo.
    """
    counts: Counter[str] = Counter()
    for p in repo_path.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in _C_EXTS:
            counts["c"] += 1
        elif ext in _CPP_EXTS:
            counts["c++"] += 1

    if counts["c++"] > counts["c"]:
        return "c++"
    return "c"


# ── Infer project name ──────────────────────────────────────────

def _infer_project_name(repo_path: Path) -> str:
    """Best-effort project name from Makefile, CMakeLists, or dir name."""
    cmake = repo_path / "CMakeLists.txt"
    if cmake.exists():
        for line in cmake.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("project("):
                # project(FooBar ...)  or  project(FooBar)
                inner = stripped.split("(", 1)[1].rstrip(")")
                name = inner.split()[0].strip('"').strip("'")
                if name:
                    return name
    return repo_path.name


# ── HuggingFace loader ──────────────────────────────────────────

def load_task_from_hf(subset_ids: list[str]) -> list[CyberGymTask]:
    """
    Load tasks by ID from the HuggingFace ``sunblaze-ucb/cybergym`` dataset.

    Each row is expected to carry:
      - ``task_id``  (str)
      - ``task_difficulty`` dict with key ``"level1"`` whose value is a
        two-element list ``[<tarball_relative_path>, <description_relative_path>]``

    The actual files are fetched with ``huggingface_hub.hf_hub_download``.
    """
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace libraries required: pip install datasets huggingface_hub"
        ) from exc

    print(f"[data_loader] Loading CyberGym dataset from HuggingFace …")
    ds = load_dataset("sunblaze-ucb/cybergym", split="tasks")

    # Build a lookup by task_id
    id_set = set(subset_ids)
    matched_rows = [row for row in ds if row["task_id"] in id_set]

    if len(matched_rows) != len(id_set):
        found = {r["task_id"] for r in matched_rows}
        missing = id_set - found
        print(f"[data_loader] WARNING: tasks not found in dataset: {missing}")

    tasks: list[CyberGymTask] = []
    for row in matched_rows:
        tid = row["task_id"]
        level1_info = row["task_difficulty"]["level1"]
        tarball_relpath = level1_info[0]
        desc_relpath = level1_info[1]

        print(f"[data_loader] Downloading files for task {tid} …")

        tarball_local: Path = Path(
            hf_hub_download(
                repo_id="sunblaze-ucb/cybergym",
                filename=tarball_relpath,
                repo_type="dataset",
            )
        )
        desc_local: Path = Path(
            hf_hub_download(
                repo_id="sunblaze-ucb/cybergym",
                filename=desc_relpath,
                repo_type="dataset",
            )
        )

        # Extract repo
        task_dir = cfg.task_work_dir(tid)
        repo_dest = task_dir / "repo"
        repo_root = _safe_extract_tarball(tarball_local, repo_dest)

        # Read description
        vuln_desc = desc_local.read_text(errors="replace").strip()

        # Detect language and name
        lang = _detect_language(repo_root)
        proj_name = _infer_project_name(repo_root)

        task = CyberGymTask(
            task_id=tid,
            project_name=proj_name,
            project_language=lang,
            vulnerability_description=vuln_desc,
            repo_path=repo_root,
            raw_tarball=tarball_local,
        )
        tasks.append(task)
        print(f"[data_loader]   ✓ {tid}: {proj_name} ({lang}), repo @ {repo_root}")

    return tasks


# ── Local-directory loader ───────────────────────────────────────

def load_task_from_local(task_dir: Path) -> CyberGymTask:
    """
    Load a single task from a local directory containing:
        repo-vul.tar.gz   — the pre-patch codebase
        description.txt   — vulnerability description

    The tarball is extracted into ``WORK_DIR / <dir_name> / repo``.
    """
    task_dir = Path(task_dir).resolve()
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task directory does not exist: {task_dir}")

    # Locate tarball — accept a few common names
    tarball: Optional[Path] = None
    for candidate in ("repo-vul.tar.gz", "repo_vul.tar.gz", "repo.tar.gz"):
        p = task_dir / candidate
        if p.exists():
            tarball = p
            break
    if tarball is None:
        # Fall back: first .tar.gz in the directory
        tarballs = list(task_dir.glob("*.tar.gz"))
        if tarballs:
            tarball = tarballs[0]
            print(f"[data_loader] No standard tarball name found; using {tarball.name}")
        else:
            raise FileNotFoundError(
                f"No .tar.gz archive found in {task_dir}"
            )

    # Locate description
    desc_path: Optional[Path] = None
    for candidate in ("description.txt", "desc.txt", "vulnerability.txt"):
        p = task_dir / candidate
        if p.exists():
            desc_path = p
            break
    if desc_path is None:
        # Try any .txt
        txts = list(task_dir.glob("*.txt"))
        if txts:
            desc_path = txts[0]
            print(f"[data_loader] Using {desc_path.name} as description")
        else:
            print("[data_loader] WARNING: no description file found; using empty string")

    vuln_desc = desc_path.read_text(errors="replace").strip() if desc_path else ""

    # Extract
    tid = task_dir.name
    work = cfg.task_work_dir(tid)
    repo_dest = work / "repo"

    # Skip extraction if already done (idempotent)
    if repo_dest.exists() and any(repo_dest.iterdir()):
        print(f"[data_loader] Repo already extracted at {repo_dest}; reusing.")
        repo_root = repo_dest
        # Check for nested single-dir
        top = [p for p in repo_dest.iterdir() if not p.name.startswith(".")]
        if len(top) == 1 and top[0].is_dir():
            repo_root = top[0]
    else:
        repo_root = _safe_extract_tarball(tarball, repo_dest)

    lang = _detect_language(repo_root)
    proj_name = _infer_project_name(repo_root)

    task = CyberGymTask(
        task_id=tid,
        project_name=proj_name,
        project_language=lang,
        vulnerability_description=vuln_desc,
        repo_path=repo_root,
        raw_tarball=tarball,
    )
    print(f"[data_loader] Loaded task '{tid}': project={proj_name}, lang={lang}")
    print(f"[data_loader]   repo   : {repo_root}")
    print(f"[data_loader]   desc   : {vuln_desc[:120]}{'…' if len(vuln_desc) > 120 else ''}")
    return task


# ── Source-file enumerator ───────────────────────────────────────

# Directories that are considered higher-priority for analysis
_PRIORITY_DIRS = {"src", "lib", "source", "core"}
_LOW_PRIORITY_DIRS = {"test", "tests", "testing", "doc", "docs", "documentation",
                      "examples", "example", "bench", "benchmark", "benchmarks",
                      "third_party", "third-party", "vendor", "deps"}


def _priority_key(path: Path) -> tuple[int, str]:
    """
    Sort key: files in priority dirs come first (0), then "normal" (1),
    then low-priority dirs (2). Secondary sort is alphabetical by path.
    """
    parts_lower = {p.lower() for p in path.parts}
    if parts_lower & _PRIORITY_DIRS:
        bucket = 0
    elif parts_lower & _LOW_PRIORITY_DIRS:
        bucket = 2
    else:
        bucket = 1
    return (bucket, str(path))


def get_source_files(task: CyberGymTask, max_files: int = 200) -> list[Path]:
    """
    Collect C/C++ source and header files from the task repo.

    Files in ``src/``, ``lib/``, and repo root are prioritised.
    ``test/``, ``doc/``, and similar dirs are deprioritised.
    The result is truncated to *max_files*.
    """
    all_sources: list[Path] = []
    for p in task.repo_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in _ALL_SOURCE_EXTS:
            all_sources.append(p)

    all_sources.sort(key=_priority_key)

    if len(all_sources) > max_files:
        print(
            f"[data_loader] {len(all_sources)} source files found; "
            f"truncating to {max_files}"
        )

    return all_sources[:max_files]


# ── CLI quick-test ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    example_path = Path("data/arvo/example_task")
    if len(sys.argv) > 1:
        example_path = Path(sys.argv[1])

    if not example_path.exists():
        print(
            f"[data_loader] Test path '{example_path}' does not exist.\n"
            f"Usage: python -m crs.data_loader <path_to_task_dir>\n"
            f"The directory should contain repo-vul.tar.gz and description.txt."
        )
        sys.exit(1)

    task = load_task_from_local(example_path)
    sources = get_source_files(task)
    print(f"\n[data_loader] Source files ({len(sources)}):")
    for s in sources[:20]:
        print(f"  {s.relative_to(task.repo_path)}")
    if len(sources) > 20:
        print(f"  … and {len(sources) - 20} more")
