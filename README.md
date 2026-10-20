Five, in pipeline order, each written as the advantage it buys:

1. **Autonomous harness discovery** — finds every harness in the build and the source tree, compiles the ones the project's build skipped, and generates its own, each gated by a reachability probe (fail-closed: a harness that can't reach project code is never counted).
2. **Truthful fuzzing health** — probe → deep-phase with keep-going, mined + format-aware + LLM seeds and a dictionary; verdicts (HEALTHY / WEAK / SATURATED / DEAD) come from measured coverage and execs, never from wall-clock.
3. **Crash triage with evidence** — signature-based dedup, PoV replay, leads kept separate from reproducers; the `X509Utils.getCnFromDn` crash was found, reproduced with a stack trace, and located to file:line with no human in the loop.
4. **One fix queue for static and dynamic findings** — CWE-tagged static findings and fuzz crashes go through the same LLM fix loop, with root-cause context from the call graph and the trace.
5. **Independent patch validation and full accounting** — every candidate patch is judged by the libCRS oracle (apply-patch-build → run-pov → run-test), never by discver itself; every attempt records attempts, test calls, parse failures and a reason, so a failed loop is explainable instead of silent.

One honesty note for the talk: 1 and 2 describe the mechanisms, not yet a HEALTHY verdict on this target, and 5 has not yet produced a validated patch — those are what this week's run is for. If you want a sixth, the self-grading `check_run.py` (PASS / FAIL / NOT SEEN per change, from the results folder alone) is a nice closer.
