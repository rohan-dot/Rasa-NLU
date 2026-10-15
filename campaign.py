#!/usr/bin/env python3
"""
campaign.py — runs the task files in order with microagent.py, verifies each with tests, keeps the tree green.

Per task:  snapshot → agent → acceptance command(s) from the task file → py_compile → full suite vs baseline
           → on failure ONE repair pass (the agent sees the failing output) → still failing: restore the snapshot,
           mark FAIL, continue with the next task.  Task 00 (map) produces MAP.md that every later task receives.
At the end: changed_files.txt, patch.diff, CAMPAIGN.md, and discver-fix.zip (changed src/ + tests/ files).

Layout under --work (created by setup.sh):
  discver/   results/   refs/<name>/   tasks/NN_*.md   MAP.md (written by task 00)   runs/   CAMPAIGN.md

Usage:
  python3 campaign.py --work /path/to/work                      # all tasks in order
  python3 campaign.py --work ... --only 10,13                   # specific tasks by number prefix
  python3 campaign.py --work ... --from 20                      # resume from a task number
  python3 campaign.py --work ... --test-cmd "python -m pytest -q tests"   # override the full-suite command
Env: LITELLM_BASE_URL LITELLM_API_KEY AGENT_MODEL (+ AGENT_TLS_VERIFY / AGENT_CA_BUNDLE / AGENT_REASONING_EFFORT)
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
IGNORE_DIRS = {".git", ".microagent", ".microagent-team", ".crew", "node_modules", "__pycache__", ".venv", "venv",
               ".mypy_cache", ".pytest_cache", "dist", "build", ".idea", "output", "oss-crs"}
BINARY_SUFFIXES = (".pyc", ".zip", ".jar", ".class", ".png", ".jpg", ".jpeg", ".gz", ".tgz", ".tar", ".so", ".bin", ".pdf", ".whl")


def log(msg: str) -> None:
    print(msg, flush=True)


def now() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# ----------------------------------------------------------------------------- snapshot / diff (no git needed)
def snapshot(repo: Path) -> dict[str, bytes]:
    out = {}
    for directory, dirs, names in os.walk(repo):
        d = Path(directory)
        dirs[:] = [x for x in dirs if x not in IGNORE_DIRS]
        for name in names:
            p = d / name
            if name.endswith(BINARY_SUFFIXES) or p.is_symlink():
                continue
            try:
                out[p.relative_to(repo).as_posix()] = p.read_bytes()
            except OSError:
                continue
    return out


def changed(before: dict[str, bytes], after: dict[str, bytes]) -> list[str]:
    return sorted(set(k for k in after if before.get(k) != after[k]) | set(k for k in before if k not in after))


def restore(repo: Path, snap: dict[str, bytes], current: dict[str, bytes]) -> None:
    for rel, data in snap.items():
        p = repo / rel
        if not p.exists() or p.read_bytes() != data:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
    for rel in set(current) - set(snap):
        (repo / rel).unlink(missing_ok=True)


def unified(before: dict[str, bytes], after: dict[str, bytes]) -> str:
    chunks = []
    for rel in changed(before, after):
        a = before.get(rel, b"").decode("utf-8", errors="replace").splitlines(True)
        b = after.get(rel, b"").decode("utf-8", errors="replace").splitlines(True)
        chunks.extend(difflib.unified_diff(a, b, fromfile=f"a/{rel}", tofile=f"b/{rel}"))
        if chunks and not chunks[-1].endswith("\n"):
            chunks.append("\n\\ No newline at end of file\n")
    return "".join(chunks) or "(no changes)\n"


# ----------------------------------------------------------------------------- tests
def sh(cmd: str, cwd: Path, timeout: int = 1800) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + "\n" + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s: {cmd}"


def pytest_counts(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "error": 0}
    tail = "\n".join(output.splitlines()[-15:])
    for key, pat in (("passed", r"(\d+) passed"), ("failed", r"(\d+) failed"), ("error", r"(\d+) errors?")):
        m = re.search(pat, tail)
        if m:
            counts[key] = int(m.group(1))
    return counts


def full_suite(repo: Path, test_cmd: str) -> tuple[int, dict[str, int], str]:
    rc, out = sh(test_cmd, repo)
    return rc, pytest_counts(out), out


def acceptance_commands(task_text: str) -> list[str]:
    m = re.search(r"^##\s*Acceptance.*?\n```[a-z]*\n(.*?)```", task_text, re.S | re.M)
    if not m:
        return []
    return [ln.strip() for ln in m.group(1).splitlines() if ln.strip() and not ln.strip().startswith("#")]


def task_kind(task_text: str) -> str:
    return "map" if re.search(r"^##\s*Map\b", task_text, re.M) or re.search(r"^kind:\s*map", task_text, re.M) else "change"


# ----------------------------------------------------------------------------- campaign
def run_agent(args, work: Path, task_file: Path, out_dir: Path, extra_task: str | None = None) -> dict:
    cmd = [sys.executable, str(HERE / "microagent.py"), "--repo", str(work / "discver"), "--out", str(out_dir),
           "--model", args.model, "--max-turns", str(args.max_turns), "--bash-timeout", str(args.bash_timeout)]
    if (work / "results").is_dir():
        cmd += ["--results", str(work / "results")]
    refs = work / "refs"
    if refs.is_dir():
        for d in sorted(refs.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                cmd += ["--ref", f"{d.name}={d}"]
    if (work / "MAP.md").is_file():
        cmd += ["--context", str(work / "MAP.md")]
    if (work / "EVIDENCE.md").is_file():
        cmd += ["--results-hint", str(work / "EVIDENCE.md")]
    if extra_task:
        tmp = out_dir / "task.md"
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(task_file.read_text(encoding="utf-8") + "\n\n" + extra_task, encoding="utf-8")
        cmd += ["--task-file", str(tmp)]
    else:
        cmd += ["--task-file", str(task_file)]
    log(f"[campaign] agent: {task_file.name} -> {out_dir}")
    subprocess.run(cmd, check=False)
    rj = out_dir / "result.json"
    return json.loads(rj.read_text(encoding="utf-8")) if rj.is_file() else {"status": "error", "edits": [], "summary": "agent produced no result.json"}


def verify_task(repo: Path, task_text: str, test_cmd: str, baseline: dict[str, int]) -> tuple[bool, str]:
    report = []
    rc, out = sh("python -m py_compile src/*.py", repo)
    report.append(f"py_compile: exit {rc}" + ("" if rc == 0 else "\n" + out[-2000:]))
    if rc != 0:
        return False, "\n".join(report)
    for cmd in acceptance_commands(task_text):
        rc, out = sh(cmd, repo)
        report.append(f"acceptance `{cmd}`: exit {rc}\n" + "\n".join(out.splitlines()[-25:]))
        if rc != 0:
            return False, "\n".join(report)
    rc, counts, out = full_suite(repo, test_cmd)
    report.append(f"full suite `{test_cmd}`: exit {rc} {counts} (baseline {baseline})")
    regressions = counts["failed"] + counts["error"] > baseline["failed"] + baseline["error"]
    if regressions or (rc != 0 and baseline.get("rc", 1) == 0):
        report.append("\n".join(out.splitlines()[-40:]))
        return False, "\n".join(report)
    return True, "\n".join(report)


def main() -> None:
    ap = argparse.ArgumentParser(description="run the discver task campaign")
    ap.add_argument("--work", required=True)
    ap.add_argument("--tasks", help="task dir (default <work>/tasks)")
    ap.add_argument("--only", help="comma-separated task number prefixes, e.g. 10,13")
    ap.add_argument("--from", dest="from_", help="start at this task number prefix")
    ap.add_argument("--test-cmd", default="python -m pytest -q tests")
    ap.add_argument("--model", default=os.environ.get("AGENT_MODEL", ""))
    ap.add_argument("--max-turns", type=int, default=90)
    ap.add_argument("--bash-timeout", type=int, default=900)
    ap.add_argument("--no-repair", action="store_true")
    a = ap.parse_args()
    if "," in a.model:
        a.model = a.model.split(",")[0].strip()
    if not a.model:
        sys.exit("set AGENT_MODEL or pass --model")

    work = Path(a.work).expanduser().resolve()
    repo = work / "discver"
    tasks_dir = Path(a.tasks).resolve() if a.tasks else work / "tasks"
    for p, what in ((repo / "src", "discver/src"), (tasks_dir, "tasks/")):
        if not p.is_dir():
            sys.exit(f"missing {what}: {p} — run setup.sh first")
    tasks = sorted(t for t in tasks_dir.glob("*.md") if re.match(r"^\d+_", t.name))
    if a.only:
        wanted = {x.strip() for x in a.only.split(",")}
        tasks = [t for t in tasks if t.name.split("_")[0] in wanted]
    if a.from_:
        tasks = [t for t in tasks if int(t.name.split("_")[0]) >= int(a.from_)]
    if not tasks:
        sys.exit("no tasks selected")
    runs = work / "runs" / now()
    runs.mkdir(parents=True, exist_ok=True)
    campaign_md = work / "CAMPAIGN.md"
    if not campaign_md.exists():
        campaign_md.write_text(f"# CAMPAIGN — {work}\n\n", encoding="utf-8")

    log(f"[campaign] work={work}\n[campaign] model={a.model} tasks={[t.name for t in tasks]}\n[campaign] runs -> {runs}")
    origin = snapshot(repo)
    rc, baseline, out = full_suite(repo, a.test_cmd)
    baseline["rc"] = rc
    log(f"[campaign] baseline suite: exit {rc} {baseline}")
    with campaign_md.open("a", encoding="utf-8") as f:
        f.write(f"## run {runs.name} — baseline `{a.test_cmd}` exit {rc} {baseline}\n\n")
    if rc != 0:
        (work / "BASELINE_FAILURES.txt").write_text(out, encoding="utf-8")
        log("[campaign] baseline has failures — recorded in BASELINE_FAILURES.txt; tasks must not add to them")

    results = []
    for task_file in tasks:
        text = task_file.read_text(encoding="utf-8")
        kind = task_kind(text)
        before = snapshot(repo)
        out_dir = runs / task_file.stem
        res = run_agent(a, work, task_file, out_dir)
        verdict, detail = "", ""
        if kind == "map":
            if res.get("status") in ("finished", "no_edits") and res.get("summary"):
                (work / "MAP.md").write_text(res["summary"], encoding="utf-8")
                verdict, detail = "MAPPED", f"MAP.md written ({len(res['summary'])} chars)"
            else:
                verdict, detail = "FAIL", f"mapping task {res.get('status')}: {res.get('summary', '')[:500]}"
            after = snapshot(repo)
            if changed(before, after):
                restore(repo, before, after)
                detail += " — mapping task edited files; reverted"
        else:
            ok, detail = verify_task(repo, text, a.test_cmd, baseline)
            if ok and res.get("edits"):
                verdict = "PASS"
            elif not res.get("edits"):
                verdict, detail = "NO_EDITS", (res.get("summary") or "")[:1500]
            else:
                if not a.no_repair:
                    log(f"[campaign] {task_file.name}: verification failed — repair pass")
                    repair = ("\n\n## REPAIR PASS\nYour previous attempt made these edits: " + ", ".join(res.get("edits", [])) +
                              ".\nVerification FAILED with this output:\n```\n" + detail[-6000:] + "\n```\nFix the failure without "
                              "reverting the feature. If the failure is a pre-existing test unrelated to your change, say so in finish() with evidence.")
                    res2 = run_agent(a, work, task_file, out_dir / "repair", extra_task=repair)
                    ok, detail = verify_task(repo, text, a.test_cmd, baseline)
                    res["edits"] = sorted(set(res.get("edits", [])) | set(res2.get("edits", [])))
                    res["summary"] = (res.get("summary") or "") + "\n\nREPAIR PASS:\n" + (res2.get("summary") or "")
                verdict = "PASS" if ok else "FAIL"
            after = snapshot(repo)
            if verdict != "PASS":
                restore(repo, before, after)
                log(f"[campaign] {task_file.name}: {verdict} — repo restored to the state before this task")
        files = changed(before, snapshot(repo))
        results.append((task_file.name, verdict, files, res))
        log(f"[campaign] {task_file.name}: {verdict} ({len(files)} file(s) kept)")
        with campaign_md.open("a", encoding="utf-8") as f:
            f.write(f"### {task_file.name} — {verdict}\n\nFiles kept: {', '.join(files) or '(none)'}\n\n"
                    f"Verification:\n```\n{detail[-3000:]}\n```\n\nAgent summary:\n\n{(res.get('summary') or '')[:4000]}\n\n")

    final = snapshot(repo)
    files = changed(origin, final)
    (work / "changed_files.txt").write_text("\n".join(files) + ("\n" if files else ""), encoding="utf-8")
    (work / "patch.diff").write_text(unified(origin, final), encoding="utf-8")
    py = [f for f in files if re.match(r"^(src|tests)/.*\.py$", f) and (repo / f).exists()]
    if py:
        with zipfile.ZipFile(work / "discver-fix.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for f in py:
                z.write(repo / f, f)
            z.write(work / "patch.diff", "patch.diff")
            z.write(campaign_md, "CAMPAIGN.md")
            for extra in ("HANDOFF.md", "DIAGNOSIS.md"):
                if (work / extra).is_file():
                    z.write(work / extra, extra)
        log(f"[campaign] discver-fix.zip: " + ", ".join(py) + " + patch.diff + CAMPAIGN.md" + (" + HANDOFF.md" if (work / "HANDOFF.md").is_file() else ""))
    summary = "\n".join(f"- {n}: {v}" + (f" ({', '.join(fs)})" if fs else "") for n, v, fs, _ in results)
    with campaign_md.open("a", encoding="utf-8") as f:
        f.write(f"## summary {runs.name}\n\n{summary}\n\n")
    log("[campaign] summary:\n" + summary)
    log(f"[campaign] done. CAMPAIGN.md, patch.diff, changed_files.txt in {work}")


if __name__ == "__main__":
    main()
