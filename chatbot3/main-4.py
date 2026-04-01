"""
crs/main.py — CRS v2 Entry Point (byte-based pipeline)

Usage:
    python -m crs.main --task-dir ./data/arvo/1065 --output-dir ./crs_results_1065
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from crs.code_intelligence import build_context
from crs.config import cfg
from crs.data_loader import CyberGymTask, load_task_from_local, load_tasks_from_dir
from crs.harness_finder import find_harness
from crs.harness_synthesizer import synthesize_harness, refine_harness
from crs.harness_runner import (
    FuzzTargetBuild, RunResult, build_fuzz_target, execute_pipeline, run_poc_bytes,
)
from crs.byte_strategies import ByteOrchestrator, PoCBytes
from crs.llm_router import LLMRouter


@dataclass
class EvalRecord:
    task_id: str
    project_name: str
    vuln_type: str
    strategy_used: str
    triggered: bool
    crash_type: str
    confidence: float
    harness_found: bool
    fuzz_target_built: bool
    poc_size_bytes: int
    time_elapsed: float
    notes: str


class Evaluator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[EvalRecord] = []

    def record(self, task, vuln_type, run_results, elapsed, harness_found, fuzz_target_built):
        triggered_results = [r for r in run_results if r.triggered]
        best = triggered_results[0] if triggered_results else (
            max(run_results, key=lambda r: r.poc.confidence) if run_results else None
        )

        if best and best.triggered:
            rec = EvalRecord(
                task_id=task.task_id, project_name=task.project_name,
                vuln_type=vuln_type, strategy_used=best.poc.strategy_name,
                triggered=True, crash_type=best.crash_type,
                confidence=best.poc.confidence, harness_found=harness_found,
                fuzz_target_built=fuzz_target_built,
                poc_size_bytes=len(best.poc.data),
                time_elapsed=elapsed, notes=best.poc.notes,
            )
        elif best:
            rec = EvalRecord(
                task_id=task.task_id, project_name=task.project_name,
                vuln_type=vuln_type, strategy_used=best.poc.strategy_name,
                triggered=False, crash_type="no_trigger",
                confidence=best.poc.confidence, harness_found=harness_found,
                fuzz_target_built=fuzz_target_built,
                poc_size_bytes=len(best.poc.data),
                time_elapsed=elapsed,
                notes=f"Tested {len(run_results)} PoC(s), none triggered",
            )
        else:
            rec = EvalRecord(
                task_id=task.task_id, project_name=task.project_name,
                vuln_type=vuln_type, strategy_used="none",
                triggered=False, crash_type="no_poc",
                confidence=0.0, harness_found=harness_found,
                fuzz_target_built=fuzz_target_built, poc_size_bytes=0,
                time_elapsed=elapsed, notes="Pipeline failed",
            )

        status = "\033[92mTRIGGERED\033[0m" if rec.triggered else "\033[91mMISS\033[0m"
        print(f"  [Eval] {task.task_id}: {status}  strategy={rec.strategy_used}  "
              f"crash={rec.crash_type}  time={elapsed:.1f}s")
        self.records.append(rec)
        return rec

    def save_report(self):
        json_path = self.output_dir / "results.json"
        json_path.write_text(json.dumps([asdict(r) for r in self.records], indent=2))
        print(f"  [Report] JSON → {json_path}")

        md_path = self.output_dir / "results.md"
        lines = ["# CRS v2 Results", "",
                 f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
                 "| task | project | vuln | status | strategy | crash | size | time |",
                 "|------|---------|------|--------|----------|-------|------|------|"]
        for r in self.records:
            icon = "✅" if r.triggered else "❌"
            lines.append(f"| {r.task_id} | {r.project_name} | {r.vuln_type} | "
                         f"{icon} | {r.strategy_used} | {r.crash_type} | "
                         f"{r.poc_size_bytes} | {r.time_elapsed:.1f}s |")
        triggered = sum(r.triggered for r in self.records)
        lines += ["", f"**{triggered}/{len(self.records)} triggered**"]
        md_path.write_text("\n".join(lines))
        print(f"  [Report] MD   → {md_path}")

    def print_summary(self):
        if not self.records:
            print("No results.")
            return
        triggered = sum(r.triggered for r in self.records)
        sep = "─" * 100
        print(f"\n  CRS v2 SUMMARY")
        print(f"  {sep}")
        for r in self.records:
            s = "\033[92mTRIGGERED\033[0m" if r.triggered else (
                "\033[93mBUILD FAIL\033[0m" if not r.fuzz_target_built else "\033[91mMISS\033[0m")
            print(f"  {r.task_id:<12} {r.project_name:<15} {r.vuln_type:<15} "
                  f"{s:<12} {r.strategy_used:<25} {r.crash_type:<22} {r.time_elapsed:.1f}s")
        print(f"  {sep}")
        print(f"  Triggered: {triggered}/{len(self.records)}  "
              f"Total: {sum(r.time_elapsed for r in self.records):.1f}s\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CRS v2 — Byte-based pipeline")
    p.add_argument("--task-dir", required=True)
    p.add_argument("--output-dir", default="./crs_results")
    p.add_argument("--model", default=cfg.LLM_MODEL)
    p.add_argument("--base-url", default=cfg.LLM_BASE_URL)
    p.add_argument("--api-key", default=cfg.LLM_API_KEY)
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--max-pocs", type=int, default=30)
    return p.parse_args(argv)


def run_task(task, router, evaluator, max_pocs=30):
    print(f"\n{'='*60}")
    print(f"  TASK : {task.task_id} | {task.project_name} | {task.project_language}")
    print(f"  DESC : {task.vulnerability_description[:100]}...")
    print(f"{'='*60}")

    t0 = time.time()

    # Step 1: Analyze
    print(f"\n  Step 1: Building code context...")
    context = build_context(task)

    # Step 2: Find or synthesize harness
    print(f"\n  Step 2: Finding fuzz harness...")
    harness = find_harness(task)
    if harness is None:
        print(f"  ⚠ No harness in repo — synthesizing via LLM...")
        harness = synthesize_harness(task, context, router)
    if harness is None:
        print(f"  ✗ No harness found or synthesized")
        evaluator.record(task, context.vuln_type, [], time.time()-t0, False, False)
        return

    source = "repo" if "synthesized" not in str(harness.harness_path) else "LLM"
    print(f"  ✓ Harness ({source}): {harness.harness_path.name}")
    print(f"    Calls: {harness.called_functions[:8]}")

    # Step 3: Build fuzz target (with refinement loop for synthesized harnesses)
    print(f"\n  Step 3: Building fuzz target...")
    MAX_ATTEMPTS = 3
    fuzz_build = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        fuzz_build = build_fuzz_target(task, harness, context)
        if fuzz_build.success:
            break
        if attempt < MAX_ATTEMPTS and "synthesized" in str(harness.harness_path):
            print(f"  ⚠ Build failed (attempt {attempt}) — refining harness...")
            refined = refine_harness(harness, fuzz_build.build_log, context, router, task, attempt)
            if refined:
                harness = refined
            else:
                break
        else:
            break

    if not fuzz_build or not fuzz_build.success:
        print(f"  ✗ Fuzz target build FAILED")
        evaluator.record(task, context.vuln_type, [], time.time()-t0, True, False)
        return

    print(f"  ✓ Fuzz target: {fuzz_build.binary_path}")

    # Step 4: Generate bytes
    print(f"\n  Step 4: Generating PoC bytes...")
    def _run_cb(poc):
        result = run_poc_bytes(poc, fuzz_build.binary_path)
        return result.triggered, result.run_log

    orchestrator = ByteOrchestrator(router, run_callback=_run_cb)
    poc_candidates = orchestrator.run(context, harness)[:max_pocs]

    if not poc_candidates:
        print(f"  ⚠ No PoC candidates generated")
        evaluator.record(task, context.vuln_type, [], time.time()-t0, True, True)
        return

    # Step 5: Test
    print(f"\n  Step 5: Testing {len(poc_candidates)} candidate(s)...")
    run_results = execute_pipeline(fuzz_build.binary_path, poc_candidates)

    triggered = sum(1 for r in run_results if r.triggered)
    if triggered:
        print(f"\n  ✓ SUCCESS — {triggered} PoC(s) triggered!")
    else:
        print(f"\n  ✗ No trigger")

    evaluator.record(task, context.vuln_type, run_results, time.time()-t0, True, True)


def main(argv=None):
    args = parse_args(argv)
    print("\033[96m")
    print("  " + "═" * 50)
    print("  CRS v2 — Cyber Reasoning System (byte pipeline)")
    print("  " + "═" * 50)
    print("\033[0m")
    print(f"  Model  : {args.model}")
    print(f"  URL    : {args.base_url}")
    print(f"  Output : {args.output_dir}")

    router = LLMRouter(args.model, args.base_url, args.api_key)
    task_path = Path(args.task_dir)
    evaluator = Evaluator(Path(args.output_dir))

    if (task_path / "repo-vul.tar.gz").exists():
        tasks = [load_task_from_local(task_path)]
    else:
        tasks = load_tasks_from_dir(task_path)

    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    print(f"  Tasks  : {len(tasks)}\n")

    for i, task in enumerate(tasks, 1):
        print(f"\n  [{i}/{len(tasks)}]")
        run_task(task, router, evaluator, args.max_pocs)

    evaluator.save_report()
    evaluator.print_summary()
    router.log_stats()


if __name__ == "__main__":
    main()
