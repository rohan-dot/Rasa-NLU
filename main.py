"""
main.py — gemma-fuzzer orchestrator.

Workflow:
1. Run LLM strategies BEFORE fuzzing (source audit, PoC generation)
2. Start LibFuzzer on the target harness
3. Monitor crashes and analyze with LLM
4. Periodically run strategy rounds (coverage seeds, variant analysis, 
   harness generation)
5. Run generated harnesses in parallel
6. Collect all artifacts

This is NOT just "LibFuzzer + LLM commentary". The LLM actively drives
bug discovery through 5 strategies modeled after AIxCC finalist CRSs.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from crash_analyzer import CrashAnalyzer
from fuzzer import LibFuzzerRunner
from llm_client import VLLMClient
from strategies import StrategyOrchestrator
from callgraph import build_callgraph
from agents import AgentOrchestrator

LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="gemma-fuzzer orchestrator")
    p.add_argument("--build-dir",   required=True)
    p.add_argument("--src-dir",     required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--seed-dir",    default=None)
    p.add_argument("--log-dir",     default="/var/log/gemma-fuzzer")
    p.add_argument("--harness",     required=True)
    p.add_argument("--vllm-host",   default="host.docker.internal")
    p.add_argument("--vllm-port",   default="8000")
    p.add_argument("--vllm-model",  default="gpt-oss-120b")
    p.add_argument("--fuzz-timeout", type=int, default=3600)
    p.add_argument("--fuzz-jobs",    type=int, default=1)
    p.add_argument("--llm-seed-interval", type=int, default=120,
                   help="Seconds between LLM strategy rounds")
    return p.parse_args()


def setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(Path(log_dir) / "orchestrator.log"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=handlers,
    )


def run_generated_harness(
    harness_bin: str,
    output_dir: str,
    timeout: int,
    logger: logging.Logger,
) -> None:
    """Run a generated harness in a background thread."""
    harness_name = Path(harness_bin).stem
    crash_dir = Path(output_dir) / "crashes" / harness_name
    corpus_dir = Path(output_dir) / "corpus" / harness_name
    crash_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        harness_bin, str(corpus_dir),
        f"-artifact_prefix={crash_dir}/",
        f"-max_total_time={timeout}",
        "-detect_leaks=0",
        "-max_len=4096",
        "-timeout=30",
    ]

    logger.info("[harness-runner] Starting: %s (%ds)", harness_name, timeout)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout + 30,
            env={"ASAN_OPTIONS": "abort_on_error=1:detect_leaks=0"},
        )
        # Copy any crashes to main PoV dir
        pov_dir = Path(output_dir) / "povs"
        for crash_file in crash_dir.glob("crash-*"):
            shutil.copy2(crash_file, pov_dir / f"{harness_name}-{crash_file.name}")
            logger.info(
                "[harness-runner] CRASH from %s: %s",
                harness_name, crash_file.name,
            )
    except subprocess.TimeoutExpired:
        logger.info("[harness-runner] %s timed out (expected).", harness_name)
    except Exception as exc:
        logger.error("[harness-runner] %s failed: %s", harness_name, exc)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_dir)
    logger = logging.getLogger("gemma-fuzzer")
    logger.info("=" * 60)
    logger.info("gemma-fuzzer v0.2.0 (strategy engine)")
    logger.info("  harness:      %s", args.harness)
    logger.info("  build_dir:    %s", args.build_dir)
    logger.info("  fuzz_timeout: %ds", args.fuzz_timeout)
    logger.info("  vllm:         %s:%s (%s)",
                args.vllm_host, args.vllm_port, args.vllm_model)
    logger.info("=" * 60)

    output_dir = Path(args.output_dir)
    pov_dir = output_dir / "povs"
    corpus_dir = output_dir / "corpus"
    crash_dir = output_dir / "crashes"
    for d in [pov_dir, corpus_dir, crash_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Initialize components ─────────────────────────────────────

    llm = VLLMClient(
        host=args.vllm_host,
        port=args.vllm_port,
        model=args.vllm_model,
    )

    analyzer = CrashAnalyzer(
        llm=llm,
        src_dir=args.src_dir,
        output_dir=args.output_dir,
    )

    strategies = StrategyOrchestrator(
        llm=llm,
        src_dir=args.src_dir,
        build_dir=args.build_dir,
        output_dir=args.output_dir,
        harness_name=args.harness,
    )

    runner = LibFuzzerRunner(
        build_dir=args.build_dir,
        harness=args.harness,
        corpus_dir=str(corpus_dir),
        crash_dir=str(crash_dir),
        seed_dir=args.seed_dir,
        jobs=args.fuzz_jobs,
    )

    # ══════════════════════════════════════════════════════════════
    # PHASE 0: BUILD CODEBASE INTELLIGENCE (no LLM needed)
    # ══════════════════════════════════════════════════════════════

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  PHASE 0: Building Call Graph (ctags)            ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    call_graph = build_callgraph(args.src_dir)
    logger.info(call_graph.to_summary())

    # Find the binary for PoC verification
    binary_path = None
    for candidate in [
        Path(args.build_dir) / args.harness,
        Path(args.build_dir) / f"{args.harness}_fuzzer",
    ]:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            binary_path = str(candidate)
            break
    if not binary_path:
        for g in Path(args.build_dir).glob(f"*{args.harness}*"):
            if g.is_file() and os.access(g, os.X_OK):
                binary_path = str(g)
                break

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: PRE-FUZZING — Strategy Round + Multi-Agent Pipeline
    # ══════════════════════════════════════════════════════════════

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  PHASE 1: Pre-Fuzzing LLM Analysis              ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    # Strategy round 1: prescan + codebase map + cross-file audit + harness gen + seeds
    pre_results = strategies.run_round(str(corpus_dir))

    for r in pre_results:
        logger.info(
            "  [pre-fuzz] %s: %d findings (%.1fs)",
            r.strategy_name, r.findings, r.elapsed,
        )

    # Run the multi-agent pipeline if we have a binary and call graph
    if binary_path and call_graph.functions:
        logger.info("╔══════════════════════════════════════════════════╗")
        logger.info("║  Multi-Agent Pipeline (Scanner → Exploiter)     ║")
        logger.info("╚══════════════════════════════════════════════════╝")

        agent_orch = AgentOrchestrator(
            llm=llm,
            call_graph=call_graph,
            binary_path=binary_path,
            src_dir=args.src_dir,
            output_dir=args.output_dir,
        )
        pipeline_result = agent_orch.run_pipeline(
            risky_files=strategies.risky_files,
            max_scan_targets=8,
            max_exploit_targets=3,
        )

        # Add any generated harnesses to the strategies tracker
        for h in agent_orch.generated_harnesses:
            strategies.generated_harnesses.append(h)

        # Feed findings into crash context
        for exploit in agent_orch.all_exploits:
            if exploit.crashed:
                strategies.add_crash(
                    f"{exploit.finding.bug_type}: {exploit.finding.description}",
                    {
                        "crash_type": exploit.finding.bug_type,
                        "affected_function": exploit.finding.function,
                        "root_cause": exploit.finding.description,
                    },
                )
    else:
        logger.warning("Skipping multi-agent pipeline (no binary or empty call graph).")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: FUZZING + CONTINUOUS LLM STRATEGIES
    # ══════════════════════════════════════════════════════════════

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  PHASE 2: Fuzzing + Continuous LLM Strategies   ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    runner.start(duration=args.fuzz_timeout)
    logger.info("LibFuzzer launched for %d seconds.", args.fuzz_timeout)

    # Start any generated harnesses in background threads
    harness_threads: list[threading.Thread] = []
    for harness_bin in strategies.get_generated_harnesses():
        t = threading.Thread(
            target=run_generated_harness,
            args=(harness_bin, args.output_dir, args.fuzz_timeout, logger),
            daemon=True,
        )
        t.start()
        harness_threads.append(t)

    # ── Main loop ─────────────────────────────────────────────────

    crashes_processed = 0
    last_strategy_round = time.monotonic()

    while runner.is_running():
        time.sleep(5)

        # ── Process new crashes ──
        new_crashes = runner.get_new_crashes(since_idx=crashes_processed)
        for crash in new_crashes:
            logger.info(
                "CRASH: %s (%s)", crash.crash_type, crash.crash_file,
            )

            # Analyze with LLM
            report = analyzer.analyze_crash(
                crash_file=crash.crash_file,
                stack_trace=crash.stack_trace,
            )

            # Copy crash file to PoV dir
            if crash.crash_file and Path(crash.crash_file).exists():
                pov_dest = pov_dir / Path(crash.crash_file).name
                shutil.copy2(crash.crash_file, pov_dest)

            # Feed crash info to strategy engine for variant analysis
            summary = crash.crash_type
            if report and report.get("root_cause"):
                summary = f"{crash.crash_type}: {report['root_cause']}"
            strategies.add_crash(summary, report)

        crashes_processed += len(new_crashes)

        # ── Periodic strategy rounds ──
        now = time.monotonic()
        if (now - last_strategy_round) >= args.llm_seed_interval:
            round_results = strategies.run_round(str(corpus_dir))

            # Start any newly generated harnesses
            for harness_bin in strategies.get_generated_harnesses():
                if not any(
                    t.name == harness_bin for t in harness_threads
                ):
                    remaining = max(60, args.fuzz_timeout - int(now - last_strategy_round))
                    t = threading.Thread(
                        target=run_generated_harness,
                        args=(harness_bin, args.output_dir, remaining, logger),
                        daemon=True,
                        name=harness_bin,
                    )
                    t.start()
                    harness_threads.append(t)

            last_strategy_round = now

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: FINAL ANALYSIS
    # ══════════════════════════════════════════════════════════════

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  PHASE 3: Final Analysis                        ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    runner.wait()

    # Process remaining crashes
    final_crashes = runner.get_new_crashes(since_idx=crashes_processed)
    for crash in final_crashes:
        report = analyzer.analyze_crash(crash.crash_file, crash.stack_trace)
        if crash.crash_file and Path(crash.crash_file).exists():
            pov_dest = pov_dir / Path(crash.crash_file).name
            shutil.copy2(crash.crash_file, pov_dest)
        strategies.add_crash(crash.crash_type, report)

    # Wait for generated harness threads
    for t in harness_threads:
        t.join(timeout=10)

    # ── Summary ───────────────────────────────────────────────────

    total_crashes = len(runner.state.crashes)
    total_povs = len(list(pov_dir.glob("*")))
    total_bugs = len(list((output_dir / "bugs").glob("*")))
    total_seeds = len(list((output_dir / "seeds").glob("*")))
    total_harnesses = len(strategies.get_generated_harnesses())

    logger.info("=" * 60)
    logger.info("gemma-fuzzer finished")
    logger.info("  Total executions:      %d", runner.state.total_execs)
    logger.info("  Unique crashes:        %d", total_crashes)
    logger.info("  PoVs:                  %d", total_povs)
    logger.info("  Bug reports:           %d", total_bugs)
    logger.info("  LLM-generated seeds:   %d", total_seeds)
    logger.info("  Generated harnesses:   %d", total_harnesses)
    logger.info("  Strategy rounds:       %d", strategies.round_number)
    logger.info("")
    logger.info("Strategy results breakdown:")
    for r in strategies.results_log:
        logger.info(
            "  %-20s  %3d findings  (%5.1fs)",
            r.strategy_name, r.findings, r.elapsed,
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
