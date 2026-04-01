"""
crs/data_loader.py — Task loading for the Cyber Reasoning System.
"""
from __future__ import annotations

import os
import stat
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from crs.config import cfg


# ── Dataclass ──────────────────────────────────────────────────────────────

@dataclass
class CyberGymTask:
    task_id: str
    project_name: str
    project_language: str
    vulnerability_description: str
    repo_path: Path
    raw_tarball: Path


# ── Helpers ────────────────────────────────────────────────────────────────

_SOURCE_EXTENSIONS = {
    ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp",
    ".py", ".java", ".go", ".rs",
}


def get_source_files(task: CyberGymTask) -> List[Path]:
    """Return all source files found under the task repo path."""
    repo = Path(task.repo_path)
    if not repo.exists():
        return []
    files = []
    for p in sorted(repo.rglob("*")):
        if p.is_file() and p.suffix.lower() in _SOURCE_EXTENSIONS:
            files.append(p)
    return files


def _fix_permissions(path: Path) -> None:
    """Ensure all extracted files are readable by the current user."""
    for root, dirs, files in os.walk(str(path)):
        for d in dirs:
            try:
                os.chmod(os.path.join(root, d),
                         stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP |
                         stat.S_IROTH | stat.S_IXOTH)
            except OSError:
                pass
        for f in files:
            try:
                os.chmod(os.path.join(root, f),
                         stat.S_IRUSR | stat.S_IWUSR |
                         stat.S_IRGRP | stat.S_IROTH)
            except OSError:
                pass


def _extract_tarball(tarball: Path, dest: Path) -> Path:
    """Extract tarball into dest/repo, returning the repo path."""
    repo_dir = dest / "repo"
    if repo_dir.exists():
        print(f"[data_loader] Repo already extracted at {repo_dir}; reusing.")
        return repo_dir

    repo_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data_loader] Extracting {tarball.name} -> {repo_dir}")
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(path=str(repo_dir))

    _fix_permissions(repo_dir)
    return repo_dir


# ── Public loaders ─────────────────────────────────────────────────────────

def load_task_from_local(task_dir: str | Path) -> CyberGymTask:
    """
    Load a single CyberGym task from a local directory.

    Expected layout:
        <task_dir>/
            repo-vul.tar.gz
            description.txt
    """
    task_dir = Path(task_dir).resolve()

    tarball = task_dir / "repo-vul.tar.gz"
    desc_file = task_dir / "description.txt"

    if not tarball.exists():
        raise FileNotFoundError(f"repo-vul.tar.gz not found in {task_dir}")
    if not desc_file.exists():
        raise FileNotFoundError(f"description.txt not found in {task_dir}")

    # Derive task_id from directory name (e.g. "1065" -> "1065")
    task_id = task_dir.name

    # Extract
    work_dir = cfg.task_work_dir(task_id)
    repo_path = _extract_tarball(tarball, work_dir)

    # Find the actual source root (first subdirectory, often "src-vul")
    subdirs = [d for d in repo_path.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        repo_path = subdirs[0]

    description = desc_file.read_text(encoding="utf-8", errors="replace").strip()

    # Detect language
    lang = _detect_language(repo_path)

    task = CyberGymTask(
        task_id=task_id,
        project_name=repo_path.name,
        project_language=lang,
        vulnerability_description=description,
        repo_path=repo_path,
        raw_tarball=tarball,
    )

    print(f"[data_loader] Loaded task '{task_id}': project={task.project_name}, lang={lang}")
    print(f"[data_loader]   repo  : {repo_path}")
    print(f"[data_loader]   desc  : {description[:80]}{'...' if len(description) > 80 else ''}")

    return task


def load_task_from_hf(task_id: str, cache_dir: Optional[str] = None) -> CyberGymTask:
    """
    Download and load a task from the HuggingFace cybergym dataset.
    Requires: pip install huggingface_hub
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("pip install huggingface_hub to use load_task_from_hf")

    numeric_id = task_id.split(":")[-1]
    local_dir = Path(cache_dir or ".") / "data" / "arvo" / numeric_id
    local_dir.mkdir(parents=True, exist_ok=True)

    for fname in ["repo-vul.tar.gz", "description.txt"]:
        hf_hub_download(
            repo_id="sunblaze-ucb/cybergym",
            filename=f"data/arvo/{numeric_id}/{fname}",
            repo_type="dataset",
            local_dir=str(local_dir),
        )

    return load_task_from_local(local_dir)


def load_tasks_from_dir(root_dir: str | Path) -> List[CyberGymTask]:
    """Load all tasks found under root_dir (one subdirectory per task)."""
    root = Path(root_dir)
    tasks = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and (sub / "repo-vul.tar.gz").exists():
            try:
                tasks.append(load_task_from_local(sub))
            except Exception as e:
                print(f"[data_loader] Skipping {sub.name}: {e}")
    return tasks


# ── Language detection ─────────────────────────────────────────────────────

def _detect_language(repo: Path) -> str:
    counts: dict[str, int] = {}
    for p in repo.rglob("*"):
        if p.is_file():
            ext = p.suffix.lower()
            if ext in (".c",):
                counts["c"] = counts.get("c", 0) + 1
            elif ext in (".cpp", ".cc", ".cxx", ".hpp"):
                counts["cpp"] = counts.get("cpp", 0) + 1
            elif ext == ".py":
                counts["python"] = counts.get("python", 0) + 1
            elif ext == ".java":
                counts["java"] = counts.get("java", 0) + 1
            elif ext == ".go":
                counts["go"] = counts.get("go", 0) + 1
            elif ext == ".rs":
                counts["rust"] = counts.get("rust", 0) + 1
    if not counts:
        return "c"
    return max(counts, key=counts.__getitem__)


# ── CLI helper ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m crs.data_loader <task_dir>")
        sys.exit(1)
    t = load_task_from_local(sys.argv[1])
    files = get_source_files(t)
    print(f"\nSource files ({len(files)}):")
    for f in files[:20]:
        print(f"  {f}")
