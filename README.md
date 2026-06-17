# Phase 1.5: sink-driven ensemble hunt (additive)
    from integrate import run_sink_hunt
    run_sink_hunt(
        llm, call_graph, args.src_dir, args.output_dir, binary_path,
        language=plugin.name, plugin=plugin,
        n_targets=8, n_agents=3,
        fuzz_seconds=min(90, args.fuzz_timeout // 4),
    )
