"""
integrate.py — one-call glue to run the sink-driven hunt INSIDE your existing
pipeline, reusing the call graph and harness your main.py already built.

Use this (instead of hunt.py) on large repos, where rebuilding the call graph
per run is wasteful. You pass in the objects main.py already has; this wires up
compile_fn / fuzz_fn / plugin and calls the hunt.

INTEGRATION — two lines in main.py:
    1) near the top imports:        from integrate import run_sink_hunt
    2) AFTER your call graph is built AND you have a harness binary:
           run_sink_hunt(llm, call_graph, src_dir, output_dir, harness_binary,
                         language=target_language)
       (rename the args to whatever those variables are called in your main.)

It degrades safely: no harness -> ranks sinks but can't verify; no LLM -> static
ranking; missing agents/languages modules -> clear warning, no crash.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("discver.integrate")


def run_sink_hunt(llm, call_graph, src_dir: str, output_dir: str,
                  harness_binary: str | None,
                  language: str = "c", plugin=None,
                  n_targets: int = 10, n_agents: int = 3,
                  fuzz_seconds: int = 90, run_fuzzer: bool = True,
                  sink_src_dir: str | None = None):
    """Run enumerate -> filter -> ensemble -> clean-room verify, reusing the
    call graph you already built.

    Args you already have in main: llm, call_graph, src_dir, output_dir, and a
    harness binary (your prebuilt fuzz target, or one your exploiter compiled).

    sink_src_dir: scope sink *scanning* to a subtree on a huge repo (defaults to
    src_dir). The call graph and verification still use the full src_dir.
    """
    from sinks import hunt_sinks   # local imports so importing this file is cheap

    compile_fn = None
    fuzz_fn = None
    try:
        import agents
        if run_fuzzer:
            fuzz_fn = agents.run_libfuzzer          # works for libFuzzer AND cargo-fuzz
        if language in ("c", "cpp", "c++"):
            compile_fn = (lambda code, name:
                          agents.compile_harness(code, name, src_dir, output_dir))
    except Exception as exc:
        logger.warning("[hunt] agents module unavailable (%s) — no concurrent fuzz / rebuild.", exc)

    if plugin is None:
        try:
            from languages import get_plugin
            plugin = get_plugin(language)
        except Exception:
            plugin = None

    if not harness_binary:
        logger.warning("[hunt] no harness binary — sinks will be ranked but no PoV "
                       "can be verified. Pass the fuzz target your pipeline located.")

    toolbox_kwargs = dict(
        src_dir=src_dir, output_dir=output_dir,
        harness_binary=harness_binary, call_graph=call_graph,
        compile_fn=compile_fn, input_mode="arg", language=language,
    )

    results = hunt_sinks(
        llm=llm, toolbox_kwargs=toolbox_kwargs,
        src_dir=(sink_src_dir or src_dir), plugin=plugin,
        n_targets=n_targets, n_agents=n_agents,
        fuzz_fn=fuzz_fn, fuzz_seconds=fuzz_seconds,
    )

    verified = 0
    for r in results:
        sink = r["sink"]
        for v in r["result"].verified:
            verified += 1
            logger.info("[hunt] VERIFIED %s in %s -> %s  (%s)",
                        sink.api, sink.function, v["pov"], v.get("judge_reason", ""))
    logger.info("[hunt] %d sinks driven, %d verified PoVs.", len(results), verified)
    return results
