"""
crs/main.py — CLI entry-point for the Cyber Reasoning System.

Usage examples
--------------
Single local task::

    python -m crs.main \\
        --task-dir ./data/arvo/1065 \\
        --output-dir ./crs_results \\
        --model google/gemma-3-27b-it \\
        --base-url http://localhost:8000/v1

Batch from HuggingFace::

    python -m crs.main \\
        --task-ids arvo:1065 arvo:1461 oss-fuzz:12345 \\
        --output-dir ./crs_results \\
        --model gpt-4o \\
        --api-key sk-… \\
        --strategies direct analyze pattern \\
        --max-tasks 10
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

# ── Sibling imports (Steps 1-6) ───────────────────────────────────────────
from crs import config
from crs.data_loader import (
    CyberGymTask,
    load_task_from_local,
    load_task_from_hf,
)
from crs.code_intelligence import build_context, CodeContext
from crs.llm_router import LLMRouter
from crs.poc_strategies import PoCOrchestrator
from crs.build_executor import (
    BuildResult,
    RunResult,
    build_project,
    compile_poc,
    run_poc,
)
from crs.fuzzer import try_fuzzing
from crs.evaluator import Evaluator, EvalRecord


# ── Colour helpers (duplicate-safe, evaluator also has its own) ───────────

_CYAN  = "\033[96m"
_BOLD  = "\033[1m"
_RESET = "\033[0m"


def _banner(text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{_CYAN}{_BOLD}{text}{_RESET}"


# ── Pipeline helpers ──────────────────────────────────────────────────────

def execute_pipeline(
    task: CyberGymTask,
    context: CodeContext,
    poc_results: list,
) -> List[RunResult]:
    """Build the project once, then compile + run every PoC candidate.

    Returns a list of ``RunResult`` objects (one per candidate).
    """

    # Build the vulnerable project
    build_result: BuildResult = build_project(task, context)
    if not build_result.success:
        print(f"  [Build] Project build FAILED — will still attempt PoC compilation")

    run_results: List[RunResult] = []
    for idx, poc in enumerate(poc_results):
        label = getattr(poc, "strategy", f"poc_{idx}")
        print(f"  [PoC {idx}] Compiling ({label}) …")

        compiled = compile_poc(poc, build_result, task)
        if not compiled.compiled:
            print(f"  [PoC {idx}] Compile failed — skipping execution")
            # Create a stub RunResult so the evaluator sees the attempt.
            stub = RunResult(
                triggered=False,
                returncode=-1,
                stdout="",
                stderr=getattr(compiled, "error", "compile failed"),
                crash_type="",
                confidence=0.0,
                poc_code=getattr(poc, "code", ""),
                strategy=label,
            )
            run_results.append(stub)
            continue

        print(f"  [PoC {idx}] Running …")
        result = run_poc(compiled, task)
        # Propagate strategy tag if the runner didn't set it.
        if not getattr(result, "strategy", ""):
            result.strategy = label
        run_results.append(result)

        tag = "TRIGGERED ✓" if result.triggered else "no trigger"
        print(f"  [PoC {idx}] {tag}  (exit={result.returncode})")

        # Early exit: no need to keep testing once we have a trigger.
        if result.triggered:
            print(f"  [PoC {idx}] Early-stopping — vulnerability confirmed.")
            break

    return run_results


# ── Core task runner ──────────────────────────────────────────────────────

def run_task(
    task: CyberGymTask,
    router: LLMRouter,
    evaluator: Evaluator,
    args: argparse.Namespace,
) -> EvalRecord:
    """End-to-end pipeline for a single CyberGym task."""

    start = time.time()

    desc_preview = getattr(task, "vulnerability_description", "")[:120]
    print(f"\n{'=' * 68}")
    print(f"  TASK : {task.task_id} | {task.project_name} | "
          f"{getattr(task, 'project_language', '?')}")
    print(f"  DESC : {desc_preview}…")
    print(f"{'=' * 68}")

    # ── Step 1: Code intelligence ─────────────────────────────────────
    context: CodeContext = build_context(task)
    top_names = [f.name for f, _ in context.ranked_files[:3]]
    print(f"  [Context] vuln_type={context.vuln_type}, "
          f"build={context.build_info.get('type', '?')}, "
          f"top_files={top_names}")

    # ── Step 2: Generate PoC candidates ───────────────────────────────
    orchestrator = PoCOrchestrator(router)

    # Honour --strategies filter if provided.
    requested = getattr(args, "strategies", None)
    poc_results = orchestrator.run(context, strategies=requested)
    print(f"  [Strategies] Generated {len(poc_results)} PoC candidate(s)")

    # ── Step 3: Build + Execute ───────────────────────────────────────
    run_results = execute_pipeline(task, context, poc_results)

    build_ok = any(
        getattr(r, "returncode", -1) != -1 for r in run_results
    )

    # ── Step 4: Optional fuzzing fallback ─────────────────────────────
    triggered_any = any(r.triggered for r in run_results)
    fuzzing_disabled = getattr(args, "no_fuzzing", False)

    if not triggered_any and config.FUZZING_ENABLED and not fuzzing_disabled:
        print("  [Fuzz] No trigger yet — attempting fuzzer fallback …")
        try:
            fuzz_poc = try_fuzzing(context, router=router, task=task)
            if fuzz_poc is not None:
                # Build the vulnerable project again if we need a fresh BuildResult
                fuzz_build: BuildResult = build_project(task, context)
                fuzz_compiled = compile_poc(fuzz_poc, fuzz_build, task)
                if fuzz_compiled.compiled:
                    fuzz_run = run_poc(fuzz_compiled, task)
                    fuzz_run.strategy = "fuzzing"
                    run_results.append(fuzz_run)
                    if fuzz_run.triggered:
                        print("  [Fuzz] TRIGGERED via fuzzer! ✓")
                    build_ok = build_ok or fuzz_build.success
        except Exception as exc:
            print(f"  [Fuzz] Fuzzer failed: {exc}")

    # ── Step 5: Record results ────────────────────────────────────────
    elapsed = time.time() - start
    record = evaluator.record(task, run_results, elapsed, build_ok)
    return record


# ── Argument parsing ──────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="crs",
        description="Cyber Reasoning System — CyberGym Level-1 PoC generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Single local task
  python -m crs.main --task-dir ./data/arvo/1065 --output-dir ./out

  # Batch from HuggingFace
  python -m crs.main --task-ids arvo:1065 arvo:1461 --output-dir ./out

  # Custom model endpoint
  python -m crs.main --task-dir ./data/arvo/1065 \\
      --model google/gemma-3-27b-it \\
      --base-url http://localhost:8000/v1 --api-key EMPTY
""",
    )

    # ── Task selection ────────────────────────────────────────────────
    task_grp = p.add_mutually_exclusive_group()
    task_grp.add_argument(
        "--task-dir",
        type=str,
        default=None,
        help="Path to a single local task directory (contains repo-vul.tar.gz + description).",
    )
    task_grp.add_argument(
        "--task-ids",
        nargs="+",
        type=str,
        default=None,
        help="One or more HuggingFace task IDs (e.g. arvo:1065 arvo:1461).",
    )

    # ── Output ────────────────────────────────────────────────────────
    p.add_argument(
        "--output-dir",
        type=str,
        default="./crs_results",
        help="Directory where results.json and results.md are written.",
    )

    # ── LLM backend ──────────────────────────────────────────────────
    p.add_argument(
        "--model",
        type=str,
        default=config.DEFAULT_MODEL,
        help="Model identifier (e.g. google/gemma-3-27b-it, gpt-4o).",
    )
    p.add_argument(
        "--base-url",
        type=str,
        default=config.DEFAULT_BASE_URL,
        help="OpenAI-compatible API base URL.",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=config.DEFAULT_API_KEY,
        help="API key (use 'EMPTY' for local vLLM).",
    )

    # ── Behaviour knobs ──────────────────────────────────────────────
    p.add_argument(
        "--no-fuzzing",
        action="store_true",
        default=False,
        help="Disable the fuzzing fallback even if AFL++/libFuzzer are available.",
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        type=str,
        default=None,
        help="Run only these PoC-generation strategies (e.g. direct analyze pattern).",
    )
    p.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to process (useful for batch caps).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=config.POC_RUN_TIMEOUT,
        help="Per-PoC execution timeout in seconds.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Verbose / debug logging.",
    )

    return p.parse_args(argv)


# ── Entry point ───────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Apply runtime overrides to global config.
    if args.timeout:
        config.POC_RUN_TIMEOUT = args.timeout
    if args.verbose:
        config.VERBOSE = True

    print(_banner("╔══════════════════════════════════════════════════════════╗"))
    print(_banner("║        CRS — Cyber Reasoning System  (Level 1)         ║"))
    print(_banner("╚══════════════════════════════════════════════════════════╝"))
    print(f"  Model   : {args.model}")
    print(f"  Base URL: {args.base_url}")
    print(f"  Output  : {args.output_dir}")
    print(f"  Fuzzing : {'disabled' if args.no_fuzzing else 'enabled (if available)'}")
    if args.strategies:
        print(f"  Strategies: {', '.join(args.strategies)}")

    # ── Initialise components ─────────────────────────────────────────
    router = LLMRouter(
        primary_model=args.model,
        primary_base_url=args.base_url,
        primary_api_key=args.api_key,
    )
    evaluator = Evaluator(Path(args.output_dir))

    # ── Load tasks ────────────────────────────────────────────────────
    tasks: List[CyberGymTask] = []
    if args.task_dir:
        task_path = Path(args.task_dir)
        if not task_path.exists():
            print(f"ERROR: task directory does not exist: {task_path}")
            sys.exit(1)
        tasks = [load_task_from_local(task_path)]
        print(f"  Loaded 1 task from {task_path}")
    elif args.task_ids:
        tasks = load_task_from_hf(args.task_ids)
        print(f"  Loaded {len(tasks)} task(s) from HuggingFace")
    else:
        print("ERROR: provide --task-dir or --task-ids")
        sys.exit(1)

    if args.max_tasks and len(tasks) > args.max_tasks:
        print(f"  Capping at --max-tasks={args.max_tasks}")
        tasks = tasks[: args.max_tasks]

    print(f"\n  Total tasks to run: {len(tasks)}")

    # ── Main loop ─────────────────────────────────────────────────────
    wall_start = time.time()

    for i, task in enumerate(tasks, 1):
        print(f"\n{'─' * 68}")
        print(f"  [{i}/{len(tasks)}]")
        try:
            run_task(task, router, evaluator, args)
        except KeyboardInterrupt:
            print("\n\n  ⚠ Interrupted by user — saving partial results …")
            break
        except Exception as exc:
            print(f"  FATAL ERROR on {getattr(task, 'task_id', '?')}: {exc}")
            traceback.print_exc()

    wall_elapsed = time.time() - wall_start

    # ── Reporting ─────────────────────────────────────────────────────
    evaluator.save_report()
    evaluator.print_summary()

    print(f"\n  Total wall-clock time: {wall_elapsed:.1f}s")

    # Print LLM usage stats if the router exposes them.
    if hasattr(router, "log_stats"):
        router.log_stats()

    # Exit code: 0 if at least one trigger, 1 otherwise.
    any_triggered = any(r.triggered for r in evaluator.records)
    sys.exit(0 if any_triggered else 1)


if __name__ == "__main__":
    main()
