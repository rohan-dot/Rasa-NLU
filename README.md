# discver: bring the run to HEALTHY and get one validated patch

You are in the discver repo on the machine that runs it under OSS-CRS. You can execute the real system — do that instead of reasoning about what it would do. Runtime LLM is GLM-5 via vLLM; do not change models. No feature flags or env switches: everything default-on and auto-detected. If this directory is not a git repo, `git init` and commit the current state before touching anything.

## Read before editing anything
1. CHANGELOG.md, README.md "Agent Status", AGENT.md, docs/PATCH_BACKLOG.md. They encode every failure so far and the rules below.
2. The last run's artifacts: orchestrator.log, health.json, patches/patch_index.json, unique_bugs/summary.json, crashes/<harness>/, REBUILD_OUT_DIR/.
3. From orchestrator.log, extract and quote in RUN_LOG.md: every `[llm-call]` and `[patch-turn]` line for the X509Fuzzer crash, every seed-generation line, and the harness-generation section. Do not edit code until you can state, from those lines, exactly why test_patch was called 0 times in 15 iterations.

## Known facts from the last run (verify each, don't assume)
- Overall DEAD. X509Fuzzer: 475 execs, saturated by a shallow crash. Its PoV hash da39a3ee5e6b4b0d3255bfef95601890afd80709 is sha1 of the empty input: getCnFromDn throws on empty/garbage DN on nearly every input, so the fuzzer never explores.
- GenFuzzer0 and SinkFuzzer1: ~50M execs, peak coverage 0. The harness never reaches instrumented target code: wrong classpath, no-op harness, or instrumentation includes missing mil.disa.*.
- GenFuzzer1 and SinkFuzzer0 healthy (~1350 cov) but corpus 0→2, new_units 0. LLM-generated seeds = 0; an earlier run reached ~3400 coverage with seeds wired in. Find why seed generation emitted nothing (look for invalid-JSON / guided-schema failures from GLM).
- Patch loop: X509 crash failed iters=15 tests=0. The static-* finding failed iters=14 tests=0 and gave up with "the build system cannot find a Makefile" — the model tried to build by hand instead of calling test_patch. That is a prompt/tool-contract failure, not a build problem.

## Phase 1: fuzzing health
Gate: health.json Overall HEALTHY, primary target HEALTHY, coverage > 0 on every harness kept.
- Shallow-crash saturation: record the crash once (it is a real finding, never hide it), then let fuzzing continue past it — Jazzer keep_going with stack-hash dedup, or reject trivially-invalid inputs in the harness before the call.
- Zero-coverage harnesses: determine whether the harness calls the target, whether target classes are on the classpath, and whether instrumentation includes cover mil.disa.**. Fix, or drop the harness with a logged reason. Never keep a harness that provably exercises nothing.
- Seeds: restore LLM seed output > 0, logged per harness.

## Phase 2: patch loop
Gate: patch_index.json shows the X509 crash with test_calls ≥ 1 and status validated, or UNVALIDATED with a specific reason.
- Fix what the [patch-turn]/[llm-call] lines show first: unparseable actions, unknown tool names, exploration without edits, truncation (finish_reason=length), or reasoning_present=true (thinking not disabled server-side).
- Prove every loop change with `python -m pytest tests/test_patch_loop_e2e.py -q` (fake LLM + fake libCRS, ~18s) before a real run.
- Then apply docs/PATCH_BACKLOG.md items only where log evidence supports them, in order: #4 navigation cap, #2 feed back the prior diff, #1 verify apply_edit before the oracle.
- The static-* skip is already committed; confirm it via [patch-route] lines.
- Validation is fixed: apply → rebuild → replay the specific PoV → tests. Nothing else is "validated". Never modify crash_dedup._get_crash_signature or the sidecar build path.

## Rules
- One change → one real run → read artifacts → next. Never stack unverified changes.
- Fail loud with a specific cause. No silent skip, no silent degrade.
- Match ids by exact prefix, never substring (a substring match on "fuzz" nearly eliminated Team Atlanta from AIxCC).
- py_compile + `pytest tests/` before every commit. One commit per change, one CHANGELOG entry per commit.
- RUN_LOG.md gets one entry per real run: commit hash, what changed, health.json summary, patch_index.json summary, decision for the next run.
- The artifacts are the only evidence. Never report success from your own reasoning.

## Stop
Stop and write FINAL_REPORT.md when both gates pass, or after 8 real runs without a gate passing. The report: what changed, what the artifacts show, and the single question a human should answer next.
