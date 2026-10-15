#!/usr/bin/env python3
"""
autofix.py — one command: diagnose the latest real run, plan the fixes, run the campaign on that plan.

  python3 autofix.py --work ~/campaign_work --results /path/to/new_results

Steps (each logged, each resumable):
  1. rotate   the new results folder into <work>/results/latest (the old latest becomes prev/)
  2. map      task 00 re-reads the CURRENT code and the NEW run → MAP.md
  3. plan     a read-only planner agent reads MAP.md + results + what the last campaign shipped, and writes
              DIAGNOSIS.md (what ran, what died at runtime and where, per-harness and per-finding causes) and a PLAN:
              at most --max-tasks tasks, each USE_LIBRARY <existing task> or NEW <task text>, each backed by a quoted
              log line. NEW tasks are validated (Acceptance block, Change section) and written to <work>/tasks_auto/.
  4. fix      campaign.py runs <work>/tasks_auto in order with its usual verification (py_compile → task test → full
              suite vs baseline → repair pass → restore on FAIL) → CAMPAIGN.md, patch.diff, discver-fix.zip
  5. grade    if src/check_run.py exists, grade the diagnosed run with it (what the shipped code actually did)

Flags: --skip-rotate (results already in place) · --skip-map (MAP.md is fresh) · --dry-run (plan only, no campaign)
       --max-tasks N (default 8) · --max-turns N (default 90) · --test-cmd "<pytest command>"
Env:   LITELLM_BASE_URL LITELLM_API_KEY AGENT_MODEL (+ AGENT_TLS_VERIFY / AGENT_CA_BUNDLE / AGENT_REASONING_EFFORT)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(msg, flush=True)


def now() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def load_campaign_module(work: Path):
    for cand in (work / "campaign.py", HERE / "campaign.py"):
        if cand.is_file():
            spec = importlib.util.spec_from_file_location("campaign", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod, cand
    sys.exit("campaign.py not found next to autofix.py or in --work")


# ----------------------------------------------------------------------------- 1. rotate results
def locate_results_root(p: Path) -> Path | None:
    if (p / "orchestrator.log").is_file():
        return p
    hits = sorted(p.rglob("orchestrator.log"))
    return hits[0].parent if hits else None


def sha(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest() if p.is_file() else ""


def rotate_results(work: Path, new_results: Path) -> None:
    root = locate_results_root(new_results)
    if root is None:
        sys.exit(f"no orchestrator.log found under {new_results}")
    results = work / "results"
    latest, prev = results / "latest", results / "prev"
    results.mkdir(parents=True, exist_ok=True)
    if latest.is_dir() and sha(latest / "orchestrator.log") == sha(root / "orchestrator.log"):
        log(f"[autofix] results/latest already holds this run ({root}); no rotation")
        return
    if latest.is_dir():
        if prev.is_dir():
            keep = results / f"prev_{now()}"
            prev.rename(keep)
            log(f"[autofix] kept older prev/ as {keep.name}")
        latest.rename(prev)
        log("[autofix] results/latest -> results/prev")
    shutil.copytree(root, latest)
    log(f"[autofix] {root} -> results/latest ({', '.join(sorted(x.name for x in latest.iterdir()))})")


# ----------------------------------------------------------------------------- 2. map
def run_map(work: Path, campaign_py: Path, args) -> None:
    cmd = [sys.executable, str(campaign_py), "--work", str(work), "--tasks", str(work / "tasks"), "--only", "00",
           "--max-turns", str(args.max_turns), "--test-cmd", args.test_cmd]
    log("[autofix] map: " + " ".join(cmd))
    subprocess.run(cmd, check=False)
    if not (work / "MAP.md").is_file():
        sys.exit("MAP.md was not produced — see the campaign log above")


# ----------------------------------------------------------------------------- 3. plan
PLANNER_TEMPLATE = """# Task 01 — diagnose the latest real run and write the fix plan (read-only; the plan becomes the next campaign)

## Map
This task makes NO edits. Its finish() summary is parsed by autofix.py: the DIAGNOSIS and PLAN sections become
DIAGNOSIS.md, every `=== TASK <file> ===` block becomes a task file, and the campaign then runs those tasks in order.
Be exact and evidence-bound: a plan item without a quoted log line is discarded.

## Inputs
- MAP.md (injected above): file:line for every stage and the evidence block from results/latest and results/prev.
- results/latest = the run that executed the CURRENT src/ (the code in the repo root). results/prev = the run before.
- LATEST RUN EVIDENCE (injected): history, the previous campaign's per-task verdicts, and the files it shipped.
- The task library (bottom of this task): ready-made tasks. Prefer USE_LIBRARY when a library task addresses a cause
  and has not landed; do not re-plan one whose log signature is present and correct in results/latest.

## Do, in order
1. What actually ran. For each change the previous campaign shipped (its PASS list), find the change's log signature in
   latest/orchestrator.log: present → it ran; absent → it did not take effect (not wired / path not reached / exception).
   Quote every `[<module>] failed: <error>` line verbatim and resolve each to the src line that raised it (grep the
   message text in src/). These are regressions of shipped code and are fixed FIRST.
2. Fuzzing: for each harness in latest/health.json — verdict, execs, cov a→b, corpus a→b — give the causal chain with
   log line numbers (launch → seeds line → crash lines → stop/reap line → verdict) and name the single most likely cause.
3. Patching: for each entry in latest/patches/patch_index.json — status, attempts, test_calls, skip_cause, reason
   verbatim — the log lines around its last turn, and the cause (diff mismatch / out of scope / parse / sidecar build /
   replay / tests). Distinguish "never reached the builder" from "builder ran and failed" from "agent never edited".
4. Plan: at most {MAX_TASKS} tasks, ordered: (a) repairs of shipped modules that died at runtime (number them 05–09);
   (b) the change that decides the primary harness's verdict (10–19); (c) patch-path fixes (20–29); (d) extras (30+).
   Each item is `USE_LIBRARY <file>` (copied as is) or `NEW <file>` (full task text in the library format: Goal /
   Study first / Change (src/) / Tests to add / Acceptance (a fenced block with ONE pytest command) / Do not touch).
   A NEW task has exactly one behavioural change, tests with fakes only (no real fuzzer, no network), edits only under
   src/ and tests/, no operator flags, health verdicts never more lenient, the sidecar invocations
   (apply-patch-build / run-pov / run-test) and crash_dedup._get_crash_signature untouched, and every new or changed
   module wrapped at its call site so an exception logs `[<module>] failed: <error>` WITH the traceback and the run
   continues as today. A repair task for a runtime failure must quote the failing line and name the src line.

## Output (finish summary — exactly this shape; nothing else before DIAGNOSIS)
DIAGNOSIS
shipped changes: <one line per shipped change: ran / did not take effect (why) / died at runtime (error → src line)>
harnesses: <one line per harness: verdict, numbers, causal chain with line numbers, cause>
patching: <one line per finding: status/reason → cause>
root causes, ranked: <1..n, each with its evidence line>
PLAN
1. USE_LIBRARY <file> — <cause it addresses> — evidence: <quoted log line>
2. NEW <file> — <cause> — evidence: <quoted log line>
=== TASK <file> ===
<full task text for each NEW item>
=== END TASK ===

## Task library (file — goal)
{LIBRARY_INDEX}
"""


def library_index(tasks_dir: Path) -> str:
    rows = []
    for t in sorted(tasks_dir.glob("*.md")):
        if not re.match(r"^\d+_", t.name) or t.name.startswith("00_") or t.name.startswith("01_"):
            continue
        text = t.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^##\s*Goal\s*\n(.+)", text, re.M)
        goal = (m.group(1).strip() if m else text.splitlines()[0].lstrip("# ").strip())[:220]
        rows.append(f"- {t.name} — {goal}")
    return "\n".join(rows) or "(empty)"


def previous_campaign_summary(work: Path) -> str:
    cm = work / "CAMPAIGN.md"
    if not cm.is_file():
        return "(no previous campaign)"
    text = cm.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"^## summary ", text, flags=re.M)
    last = parts[-1] if len(parts) > 1 else text[-4000:]
    return "## summary " + last[-6000:]


def write_planner_inputs(work: Path, args) -> tuple[Path, Path]:
    hint = [(work / "EVIDENCE.md").read_text(encoding="utf-8", errors="replace") if (work / "EVIDENCE.md").is_file() else ""]
    hint.append("\n\nPREVIOUS CAMPAIGN (what was shipped to the container before results/latest ran)\n" + previous_campaign_summary(work))
    cf = work / "changed_files.txt"
    if cf.is_file():
        hint.append("\nfiles shipped:\n" + cf.read_text(encoding="utf-8", errors="replace")[-3000:])
    hint_path = work / "PLANNER_HINT.md"
    hint_path.write_text("".join(hint), encoding="utf-8")
    task_path = work / "planner_task.md"
    task_path.write_text(PLANNER_TEMPLATE.replace("{MAX_TASKS}", str(args.max_tasks))
                         .replace("{LIBRARY_INDEX}", library_index(work / "tasks")), encoding="utf-8")
    return task_path, hint_path


def run_planner(work: Path, task_path: Path, hint_path: Path, out_dir: Path, args) -> str:
    microagent = work / "microagent.py" if (work / "microagent.py").is_file() else HERE / "microagent.py"
    cmd = [sys.executable, str(microagent), "--repo", str(work / "discver"), "--out", str(out_dir),
           "--max-turns", str(args.max_turns), "--bash-timeout", "900",
           "--task-file", str(task_path), "--context", str(work / "MAP.md"), "--results-hint", str(hint_path)]
    if (work / "results").is_dir():
        cmd += ["--results", str(work / "results")]
    refs = work / "refs"
    if refs.is_dir():
        for d in sorted(refs.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                cmd += ["--ref", f"{d.name}={d}"]
    env = os.environ.copy()
    env.setdefault("AGENT_MAX_TOKENS", "16384")   # the plan carries whole task files
    log("[autofix] planner: " + " ".join(cmd))
    subprocess.run(cmd, check=False, env=env)
    rj = out_dir / "result.json"
    if not rj.is_file():
        sys.exit("planner produced no result.json")
    res = json.loads(rj.read_text(encoding="utf-8"))
    if res.get("status") not in ("finished", "no_edits") or not res.get("summary"):
        sys.exit(f"planner did not finish: {res.get('status')} — {str(res.get('summary'))[:500]}")
    return res["summary"]


def materialize_plan(work: Path, summary: str, campaign_mod, args) -> list[Path]:
    tasks_auto = work / "tasks_auto"
    if tasks_auto.exists():
        shutil.move(str(tasks_auto), str(work / f"tasks_auto_{now()}"))
    tasks_auto.mkdir()
    blocks = {m.group(1): m.group(2).strip() + "\n"
              for m in re.finditer(r"=== TASK ([\w\-.]+\.md) ===\n(.*?)\n=== END TASK ===", summary, re.S)}
    head = summary.split("=== TASK ", 1)[0].strip()
    (work / "DIAGNOSIS.md").write_text(f"# DIAGNOSIS — {now()}\n\n{head}\n", encoding="utf-8")
    items = re.findall(r"^\s*\d+\.\s+(USE_LIBRARY|NEW)\s+([\w\-.]+\.md)", summary, re.M)
    written: list[Path] = []
    for kind, name in items[: args.max_tasks]:
        if not re.match(r"^\d+_", name):
            name = f"{50 + len(written):02d}_{name}"
        dest = tasks_auto / name
        if kind == "USE_LIBRARY":
            src = work / "tasks" / name
            if not src.is_file():
                src = HERE / "tasks" / name
            if not src.is_file():
                log(f"[autofix] plan item {name}: not in the library — skipped")
                continue
            shutil.copy(src, dest)
        else:
            text = blocks.get(name)
            if not text:
                log(f"[autofix] plan item NEW {name}: no task block in the summary — skipped")
                continue
            if campaign_mod.task_kind(text) != "change" or not campaign_mod.acceptance_commands(text) \
                    or not re.search(r"^##\s*Change", text, re.M):
                log(f"[autofix] plan item NEW {name}: missing Change section or Acceptance command — skipped")
                continue
            dest.write_text(text, encoding="utf-8")
        written.append(dest)
        log(f"[autofix] task {kind:11s} {name}")
    (work / "PLAN.md").write_text("\n".join(f"- {p.name}" for p in written) + "\n", encoding="utf-8")
    if not written:
        sys.exit("the plan produced no runnable tasks — read DIAGNOSIS.md and the planner transcript")
    return written


# ----------------------------------------------------------------------------- 4. fix, 5. grade
def run_campaign(work: Path, campaign_py: Path, args) -> None:
    cmd = [sys.executable, str(campaign_py), "--work", str(work), "--tasks", str(work / "tasks_auto"),
           "--max-turns", str(args.max_turns), "--test-cmd", args.test_cmd]
    log("[autofix] campaign: " + " ".join(cmd))
    subprocess.run(cmd, check=False)


def grade(work: Path) -> None:
    cr = work / "discver" / "src" / "check_run.py"
    if cr.is_file() and (work / "results" / "latest").is_dir():
        log("[autofix] grading results/latest with src/check_run.py (what the shipped code actually did):")
        subprocess.run([sys.executable, str(cr), str(work / "results" / "latest")], check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="diagnose the latest run, plan the fixes, run the campaign")
    ap.add_argument("--work", required=True)
    ap.add_argument("--results", help="folder (or zip-extracted tree) holding the NEW run's orchestrator.log")
    ap.add_argument("--skip-rotate", action="store_true")
    ap.add_argument("--skip-map", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="diagnose and plan only")
    ap.add_argument("--max-tasks", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=90)
    ap.add_argument("--test-cmd", default="python -m pytest -q tests")
    args = ap.parse_args()
    if not os.environ.get("AGENT_MODEL"):
        sys.exit("set AGENT_MODEL (and LITELLM_BASE_URL / LITELLM_API_KEY) first")
    work = Path(args.work).expanduser().resolve()
    for p in (work / "discver" / "src", work / "tasks"):
        if not p.is_dir():
            sys.exit(f"missing {p} — run setup.sh first")
    campaign_mod, campaign_py = load_campaign_module(work)
    runs = work / "runs" / f"{now()}_autofix"
    runs.mkdir(parents=True, exist_ok=True)

    if args.results and not args.skip_rotate:
        rotate_results(work, Path(args.results).expanduser().resolve())
    elif not (work / "results" / "latest" / "orchestrator.log").is_file():
        sys.exit("results/latest/orchestrator.log missing — pass --results <new run folder>")
    if not args.skip_map:
        run_map(work, campaign_py, args)
    task_path, hint_path = write_planner_inputs(work, args)
    summary = run_planner(work, task_path, hint_path, runs / "01_plan", args)
    written = materialize_plan(work, summary, campaign_mod, args)
    log(f"[autofix] DIAGNOSIS.md and PLAN.md written; {len(written)} task(s) in tasks_auto/")
    if args.dry_run:
        log("[autofix] dry run — stopping before the campaign. Review DIAGNOSIS.md and tasks_auto/, then rerun without --dry-run (add --skip-rotate --skip-map).")
        return
    run_campaign(work, campaign_py, args)
    grade(work)
    log(f"[autofix] done. Read {work / 'DIAGNOSIS.md'} and {work / 'CAMPAIGN.md'}; ship {work / 'discver-fix.zip'} with HANDOFF.md.")


if __name__ == "__main__":
    main()
