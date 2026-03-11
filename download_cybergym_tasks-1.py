#!/usr/bin/env python3
"""
download_cybergym_tasks.py
==========================
Downloads CyberGym Level 3 tasks from HuggingFace into the directory
layout expected by crs_pipeline.py.

Usage:
    pip install huggingface_hub datasets
    python download_cybergym_tasks.py [--output-dir ./cybergym_tasks] [--num-tasks 10]

Output structure:
    cybergym_tasks/
    ├── arvo_64574/
    │   ├── repo-vul.tar.gz
    │   ├── repo-fix.tar.gz
    │   ├── description.txt
    │   ├── error.txt
    │   └── patch.diff
    ├── arvo_1461/
    │   └── ...
    └── ...
"""

import os
import sys
import json
import argparse

# ─────────────────────────────────────────────────────────────────────
# Hand-picked Level 3 tasks — chosen for DIVERSITY of:
#   - project (different codebases = different build systems)
#   - vuln class (overflow, UAF, null-deref, etc.)
#   - complexity (mix of easy single-file fixes and harder ones)
#
# These are all confirmed to have all 5 Level 3 artifacts.
# ─────────────────────────────────────────────────────────────────────
CURATED_TASKS = [
    # ── Easy tier (small repos, single-file patches, clear bugs) ──
    {
        "task_id": "arvo:64574",
        "project": "jq",
        "vuln": "heap-buffer-overflow in decNumberToString (off-by-one NUL)",
        "why": "Small C project, autotools build, classic off-by-one — good first test",
    },
    {
        "task_id": "arvo:62886",
        "project": "libxml2",
        "vuln": "null-deref in dict module with empty subdict",
        "why": "libxml2 is well-studied, null-deref is simple to patch",
    },
    {
        "task_id": "arvo:62911",
        "project": "libxml2",
        "vuln": "OOB read from memcmp vs strncmp in dict",
        "why": "Single function fix — memcmp→strncmp swap",
    },
    {
        "task_id": "arvo:54949",
        "project": "libplist",
        "vuln": "use-after-free in oplist (pointer not NULLed after free)",
        "why": "Classic UAF fix pattern: set ptr=NULL after free",
    },

    # ── Medium tier ─────────────────────────────────────────────────
    {
        "task_id": "arvo:65212",
        "project": "libssh2",
        "vuln": "OOB read in _libssh2_kex_agree_instr",
        "why": "Bounds check fix, CMake build system",
    },
    {
        "task_id": "arvo:55948",
        "project": "mosquitto",
        "vuln": "invalid handling of long hex values in config",
        "why": "Input validation fix, different build system (cmake)",
    },
    {
        "task_id": "arvo:58287",
        "project": "faad2",
        "vuln": "stack-buffer-overflow in pns_decode",
        "why": "Audio codec, autotools, stack overflow is a distinct class",
    },
    {
        "task_id": "arvo:3848",
        "project": "yara",
        "vuln": "heap-buffer-overflow READ in PE module",
        "why": "Security tool itself has a bug — ironic and common in CyberGym",
    },

    # ── Harder tier ─────────────────────────────────────────────────
    {
        "task_id": "arvo:15120",
        "project": "libarchive",
        "vuln": "use-after-free in RAR reader (ppmd_valid not reset)",
        "why": "Multi-step fix logic, larger codebase",
    },
    {
        "task_id": "arvo:781",
        "project": "pcre2",
        "vuln": "invalid memory read with fewer capturing parens than ovector space",
        "why": "Regex engine internals, tests the patcher on non-trivial logic",
    },
]

# All Level 3 artifacts
LEVEL3_FILES = [
    "repo-vul.tar.gz",
    "repo-fix.tar.gz",
    "description.txt",
    "error.txt",
    "patch.diff",
]

DATASET_REPO = "sunblaze-ucb/cybergym"


def task_id_to_data_prefix(task_id: str) -> str:
    """Convert 'arvo:64574' → 'data/arvo/64574'."""
    source, num = task_id.split(":")
    return f"data/{source}/{num}"


def task_id_to_dir_name(task_id: str) -> str:
    """Convert 'arvo:64574' → 'arvo_64574' (filesystem-safe)."""
    return task_id.replace(":", "_")


def download_with_hf_hub(tasks: list, output_dir: str):
    """Download using huggingface_hub (preferred — resumable, cached)."""
    from huggingface_hub import hf_hub_download

    for task in tasks:
        tid = task["task_id"]
        prefix = task_id_to_data_prefix(tid)
        dest = os.path.join(output_dir, task_id_to_dir_name(tid))
        os.makedirs(dest, exist_ok=True)

        print(f"\n{'─'*60}")
        print(f"Downloading: {tid}  ({task['project']} — {task['vuln'][:60]})")
        print(f"  → {dest}")

        for fname in LEVEL3_FILES:
            repo_path = f"{prefix}/{fname}"
            local_path = os.path.join(dest, fname)

            if os.path.exists(local_path):
                size = os.path.getsize(local_path)
                print(f"  ✓ {fname} (already exists, {size:,} bytes)")
                continue

            try:
                downloaded = hf_hub_download(
                    repo_id=DATASET_REPO,
                    filename=repo_path,
                    repo_type="dataset",
                    local_dir=os.path.join(output_dir, ".hf_cache"),
                )
                # Copy/link to our clean directory structure
                import shutil
                shutil.copy2(downloaded, local_path)
                size = os.path.getsize(local_path)
                print(f"  ✓ {fname} ({size:,} bytes)")
            except Exception as e:
                print(f"  ✗ {fname} FAILED: {e}")

        # Write a metadata file for reference
        meta_path = os.path.join(dest, "task_meta.json")
        with open(meta_path, "w") as f:
            json.dump(task, f, indent=2)


def download_with_datasets_lib(tasks: list, output_dir: str):
    """
    Alternative: use the datasets library to load the full dataset,
    then extract files for selected tasks. Slower but works if
    hf_hub_download can't find individual files.
    """
    from datasets import load_dataset

    print("Loading CyberGym dataset (this downloads the full index)...")
    ds = load_dataset(DATASET_REPO, split="tasks")

    # Build lookup by task_id
    task_lookup = {row["task_id"]: row for row in ds}

    for task in tasks:
        tid = task["task_id"]
        if tid not in task_lookup:
            print(f"  ✗ {tid} not found in dataset!")
            continue

        row = task_lookup[tid]
        dest = os.path.join(output_dir, task_id_to_dir_name(tid))
        os.makedirs(dest, exist_ok=True)

        print(f"\n{'─'*60}")
        print(f"Task: {tid}  ({task['project']})")

        # The dataset row has task_difficulty.level3 listing the file paths
        level3_paths = row.get("task_difficulty", {}).get("level3", [])
        print(f"  Level 3 files: {level3_paths}")

        # These are repo-relative paths; download each
        from huggingface_hub import hf_hub_download
        for repo_path in level3_paths:
            fname = os.path.basename(repo_path)
            local_path = os.path.join(dest, fname)

            if os.path.exists(local_path):
                print(f"  ✓ {fname} (exists)")
                continue

            try:
                downloaded = hf_hub_download(
                    repo_id=DATASET_REPO,
                    filename=repo_path,
                    repo_type="dataset",
                    local_dir=os.path.join(output_dir, ".hf_cache"),
                )
                import shutil
                shutil.copy2(downloaded, local_path)
                print(f"  ✓ {fname} ({os.path.getsize(local_path):,} bytes)")
            except Exception as e:
                print(f"  ✗ {fname}: {e}")

        meta_path = os.path.join(dest, "task_meta.json")
        with open(meta_path, "w") as f:
            json.dump(task, f, indent=2)


def verify_downloads(output_dir: str, tasks: list):
    """Check all expected files exist."""
    print(f"\n{'═'*60}")
    print("VERIFICATION")
    print(f"{'═'*60}")

    all_ok = True
    for task in tasks:
        tid = task["task_id"]
        dest = os.path.join(output_dir, task_id_to_dir_name(tid))
        missing = []
        sizes = {}
        for fname in LEVEL3_FILES:
            fp = os.path.join(dest, fname)
            if os.path.exists(fp):
                sizes[fname] = os.path.getsize(fp)
            else:
                missing.append(fname)

        if missing:
            all_ok = False
            print(f"  ✗ {tid}: MISSING {missing}")
        else:
            repo_size = sizes.get("repo-vul.tar.gz", 0) / (1024*1024)
            print(f"  ✓ {tid:<16} ({task['project']:<12}) "
                  f"repo={repo_size:.1f}MB  "
                  f"desc={sizes.get('description.txt', 0)}B  "
                  f"err={sizes.get('error.txt', 0)}B  "
                  f"patch={sizes.get('patch.diff', 0)}B")

    if all_ok:
        print(f"\nAll {len(tasks)} tasks downloaded successfully!")
        print(f"\nRun the pipeline:")
        print(f"  python crs_pipeline.py {output_dir}/<task_dir>")
        print(f"  python crs_pipeline.py --batch {output_dir}")
    else:
        print(f"\nSome files are missing — check errors above.")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Download CyberGym Level 3 tasks for CRS pipeline testing")
    parser.add_argument("--output-dir", "-o", default="./cybergym_tasks",
                        help="Output directory (default: ./cybergym_tasks)")
    parser.add_argument("--num-tasks", "-n", type=int, default=10,
                        help="Number of tasks to download (default: 10, max: 10 curated)")
    parser.add_argument("--method", choices=["hf_hub", "datasets"], default="hf_hub",
                        help="Download method (default: hf_hub)")
    parser.add_argument("--list", action="store_true",
                        help="Just list the curated tasks, don't download")
    args = parser.parse_args()

    tasks = CURATED_TASKS[:args.num_tasks]

    if args.list:
        print("Curated CyberGym Level 3 tasks:\n")
        for i, t in enumerate(tasks, 1):
            print(f"  {i:2d}. {t['task_id']:<16} {t['project']:<12} {t['vuln']}")
        return

    print("╔══════════════════════════════════════════════════════════╗")
    print("║      CyberGym Level 3 Task Downloader                   ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Tasks:     {len(tasks):<44}║")
    print(f"║  Output:    {args.output_dir:<44}║")
    print(f"║  Method:    {args.method:<44}║")
    print("╚══════════════════════════════════════════════════════════╝")

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if args.method == "hf_hub":
            download_with_hf_hub(tasks, args.output_dir)
        else:
            download_with_datasets_lib(tasks, args.output_dir)
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Install with:  pip install huggingface_hub datasets")
        sys.exit(1)

    verify_downloads(args.output_dir, tasks)


if __name__ == "__main__":
    main()
