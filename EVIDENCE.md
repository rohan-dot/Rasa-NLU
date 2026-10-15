Two real runs are in the results root — `latest/` (the most recent run, which executed the CURRENT src/) and `prev/`
(the run before it). The results outrank this text: if `latest/` is newer than the runs described below, MAP.md's
reading of it is the truth about what the current code does, and the notes below are history.

prev/ (report_run_20260915-203913 — the WEAK run)
- Primary harness X509Fuzzer WEAK: ~300 executions, coverage flat 2043→2043, corpus 0→4, verdict
  "SATURATED by a shallow crash (throws on ~every input; each fork restart dominates runtime)". SinkFuzzer0 the same.
- GenFuzzer0: ~27.6M executions, coverage 10→10, zero crashes — a generated harness that never reached the target, burning a full budget.
- GenFuzzer1 / SinkFuzzer1 HEALTHY (2063→3417): the fuzzer goes deep once a harness actually reaches the code.
- Every harness printed an execution rate of 0.0/s next to millions of executions — the rate calculation is wrong.
- 181 LLM-generated seeds, yet corpus 0→4 on the primary harness.
- Fix loop: failed=7; every patch_index.json entry: status=failed, attempts=15, test_calls=0, root_cause="" — on two
  reproducer-backed crashes and five static-* findings. The loop never turned a model reply into apply_edit / test_patch.

latest/ (the DEAD run, after the previous code change)
- X509Fuzzer DEAD. Sequence in orchestrator.log: first stats line → `NEW CRASH` → coverage flat → the no-op harness
  detector stopped it. The harness was flat because it was saturated by the crash, not because it was a no-op.
  THE RULE MUST REQUIRE: coverage flat AND no observed crashes. (Task 10.)
- No patch was attempted: no reproduced crash reached the fix loop. The prev/ run shows the loop fails on its own when
  it does run — two independent problems.

Known facts
- Confirmed crash: index-out-of-bounds in mil.disa.common.util.agentmanager.X509Utils.getCnFromDn (X509Utils.java:81).
  Its reproducer file name is the SHA-1 of the empty string: the target crashes on EMPTY input. Fix = a one-line guard.
- Builder sidecar works ("build oracle path: SIDECAR"; apply-patch-build / run-pov / run-test). Do not touch its wiring.
- Ten real runs, zero validated patches. Every failure so far was plumbing or protocol, never the model.

Provided harnesses (from the project agentmanager-java8-master, in the fuzz build output under .../builds/<id>/address/)
- TWO provided harnesses exist: X509Fuzzer and GetPrettySignatureFuzzer. Both are primary — discover and drive both
  (task 18); do not assume a single harness. The last run appears to have driven only one; driving all provided harnesses
  is part of reliability.
- X509Fuzzer: `public static void fuzzerTestOneInput(FuzzedDataProvider data)`; `String dn = data.consumeRemainingAsString();`
  then `X509Utils.getCnFromDn(dn)` inside `try { } catch (IllegalArgumentException e) { }`. It deliberately lets
  ArrayIndexOutOfBounds / StringIndexOutOfBounds / NullPointerException propagate. INPUT LAYOUT: the whole fuzz input IS
  the DN string — seeds for this harness are plain UTF-8 text (no FuzzedDataProvider field encoding).
- This catch discipline is the template for generated harnesses (task 17): catch only the expected/declared exception,
  let the bug classes propagate; never a broad catch (Throwable|Exception).
- Target X509Utils.getCnFromDn(String) is a DN-string parser; its coverage ceiling is that method and what it calls.
  HEALTHY for X509Fuzzer means coverage actually GROWS past the empty-input crash, not a large number.
- Line numbers: the harness comment cites CodeQL flagging X509Utils.java:146 (CWE-193); our run's reproducer is EMPTY
  input (the :81 site). Treat the crash artifact in results/ as authoritative; :146 is a static candidate that may be the
  same site or a second one. The patch/RCA tasks should not assume a single line.

Invariants for every task
- Health verdicts become more accurate, never more lenient. A saturated harness fuzzing past its crash is WEAK until
  coverage actually grows. A harness with a crash is never "no-op".
- The normal (probe) launch command stays what it is today. Only the deep-phase relaunch and generated-harness launches add options.
- New modules are wrapped at their single call site: any exception logs `[<module>] failed: <error>` and the run continues as today.
- No feature flags or environment switches for the operator. Edits only under src/ and tests/.
- Nothing target-specific in src/: no `X509Fuzzer`, `getCnFromDn`, `CN=`, package names or DN examples in code. The
  provided harness names come from discovery (task 18); seed and dictionary examples are derived from the harness and
  target sources at runtime. The DN / CN= examples in the task files are TEST FIXTURES only. discver must run unchanged
  on the next codebase, which will ship different harnesses.
