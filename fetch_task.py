#!/usr/bin/env python3
"""
fetch_task.py — Download a SEC-bench task and clone the vulnerable repo.

Usage:
    # List all available ASAN tasks:
    python fetch_task.py --list

    # List only tasks for a specific project:
    python fetch_task.py --list --project faad2

    # Fetch a specific task:
    python fetch_task.py --id "faad2__faad2-CVE-2023-38857"

    # Fetch into a custom directory:
    python fetch_task.py --id "faad2__faad2-CVE-2023-38857" --out ./my_tasks

Requires: pip install datasets
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path


def load_dataset_cached():
    """Load SEC-bench dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)
    print("[fetch] Loading SEC-bench dataset from HuggingFace...")
    return load_dataset("SEC-bench/SEC-bench", split="test")


def list_tasks(ds, project_filter=None):
    """Print all available tasks."""
    count = 0
    for row in ds:
        iid = row["instance_id"]
        desc = row.get("bug_description", row.get("problem_statement", ""))
        repo = row.get("repo", "")

        if project_filter and project_filter.lower() not in iid.lower():
            continue

        # Check if it's an ASAN task
        is_asan = "addresssanitizer" in desc.lower() or "asan" in desc.lower()
        tag = "[ASAN]" if is_asan else "[OTHER]"

        print(f"  {tag} {iid}")
        print(f"       repo={repo}  desc={desc[:100]}...")
        print()
        count += 1

    print(f"  Total: {count} task(s)")
    if not project_filter:
        print(f"\n  Tip: use --project faad2 to filter by project")
        print(f"  Recommended ASAN projects: faad2, mruby, njs, gpac, imagemagick")


def fetch_task(ds, instance_id: str, out_dir: Path):
    """Download task metadata and clone the repo."""
    row = None
    for r in ds:
        if r["instance_id"] == instance_id:
            row = r
            break

    if row is None:
        print(f"ERROR: instance_id '{instance_id}' not found in dataset")
        print(f"  Use --list to see available tasks")
        sys.exit(1)

    task_dir = out_dir / instance_id.replace("/", "_")
    task_dir.mkdir(parents=True, exist_ok=True)

    # Save full metadata
    meta = dict(row)
    meta_path = task_dir / "task_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    # Save description (include sanitizer report if available)
    desc = meta.get("bug_description", meta.get("problem_statement", ""))
    sanitizer = meta.get("sanitizer_report", "")
    full_desc = desc
    if sanitizer:
        full_desc += "\n\n" + sanitizer
    (task_dir / "description.txt").write_text(full_desc)

    # Save harness (build/run commands from SEC-bench)
    harness = meta.get("harness", "")
    if harness:
        (task_dir / "harness_commands.txt").write_text(harness)

    # Save patch (for reference, not used by CRS)
    patch = meta.get("patch", "")
    if patch:
        (task_dir / "gold_patch.diff").write_text(patch)

    # Extract repo URL and commit
    repo_name = meta.get("repo", "")
    base_commit = meta.get("base_commit", "")

    if not repo_name:
        print(f"WARNING: No repo field found — you'll need to clone manually")
    else:
        repo_url = f"https://github.com/{repo_name}.git"
        repo_dir = task_dir / "repo"

        if repo_dir.exists():
            print(f"[fetch] Repo already cloned at {repo_dir}")
        else:
            print(f"[fetch] Cloning {repo_url} ...")
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "50", repo_url, str(repo_dir)],
                    check=True, timeout=120,
                )
            except subprocess.CalledProcessError:
                print(f"  Shallow clone failed, trying full clone...")
                subprocess.run(
                    ["git", "clone", repo_url, str(repo_dir)],
                    check=True, timeout=300,
                )

            if base_commit:
                print(f"[fetch] Checking out vulnerable commit: {base_commit[:12]}...")
                subprocess.run(
                    ["git", "checkout", base_commit],
                    cwd=str(repo_dir), check=True,
                )

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Task fetched: {instance_id}")
    print(f"  Directory:    {task_dir}")
    print(f"  Repo:         {repo_name}")
    print(f"  Commit:       {base_commit[:12] if base_commit else 'N/A'}")
    print(f"  Description:  {desc[:120]}...")
    print(f"{'='*60}")
    print(f"\n  Next step:")
    print(f"    python run_task.py --task-dir {task_dir}")
    if harness:
        print(f"\n  SEC-bench harness commands (saved to harness_commands.txt):")
        for line in harness.split("\n")[:15]:
            print(f"    {line}")


def main():
    p = argparse.ArgumentParser(description="Fetch SEC-bench tasks")
    p.add_argument("--list", action="store_true", help="List available tasks")
    p.add_argument("--project", default=None, help="Filter by project name")
    p.add_argument("--id", default=None, help="Instance ID to fetch")
    p.add_argument("--out", default="./secbench_tasks", help="Output directory")
    args = p.parse_args()

    ds = load_dataset_cached()

    if args.list:
        list_tasks(ds, args.project)
    elif args.id:
        fetch_task(ds, args.id, Path(args.out))
    else:
        print("Usage:")
        print("  python fetch_task.py --list                    # see all tasks")
        print("  python fetch_task.py --list --project faad2    # filter")
        print("  python fetch_task.py --id <instance_id>        # fetch one")


if __name__ == "__main__":
    main()
