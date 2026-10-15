# Task 16 — the generated-harness gate must fail CLOSED: a probe with no output is a rejection, not a pass

## Goal
Latest run: `:650 "probe produced no stats line (0 execs) — reach not judged, launching (fail-open)"`, after which
GenFuzzer1 and GenFuzzer2 each burned ~8M execs at coverage 2062→2062 before being reaped as degenerate
(`:879`, `:883`). A probe that produces no stats line means the generated harness did not run at all — that is the
clearest possible rejection, and the gate launched it anyway. Also `SinkFuzzer0: "target-call reach marker was never
written"` although it clearly runs (cov 2079→3434): the reach decision depends on a marker that is not produced, so
it cannot judge. Make the gate decide from evidence it can always obtain, and never fail open for a generated harness.

## Study first
- results: `grep -nE "fail-open|reach not judged|reach marker|probe produced|degenerate|reaped|covers [0-9]+ target|coverage_report" latest/orchestrator.log`
  — quote the lines; note what the probe command was and whether its output was captured anywhere.
- repo: the gate code from the previous campaign (`grep -rnE "fail-open|reach not judged|reach marker|harness_reach" src/`),
  MAP.md "harness generation" / "harness selection", and how the probe is launched (does it reuse fuzzer.py:612's
  command builder? does it capture stdout+stderr? what max_total_time/-runs does it pass?).

## Change (src/)
- Probe execution: capture the probe's full stdout and stderr to a file under the run's output dir; on 0 execs, log
  `harness-gen <name>: probe produced no execs → rejected (cmd: <argv>; first output line: <line>)` and do NOT launch.
  A rejected harness is regenerated once (with the probe output in the feedback, if task 17's feedback exists), then dropped.
- Reach decision: prefer Jazzer `--coverage_report=<file>` on the probe (written at exit) and require the target package
  in it; fall back to "coverage moved from the first stats line during the probe"; the file-marker mechanism is removed
  or made a third signal, never the only one. Log `harness-gen <name>: reach=<ok|fail> (<method>) → <admitted|rejected: reason>`.
- The provided harness(es) never go through the gate (name from discovery, never hardcoded).
- Safety wrapper stays: an exception inside the gate logs `[harness_gate] failed: <error>` with traceback — and for a
  GENERATED harness that still means "not admitted" (fail closed), for a provided harness "run as today".

## Tests to add (tests/test_harness_gate_fail_closed.py) — fakes only
- fake probe output with no stats line → rejected, not launched, the log line names the command and first output line.
- fake probe with stats and a fake coverage report containing the target method → admitted via `coverage_report`.
- no coverage report, but coverage moved during the probe → admitted via the fallback; flat → rejected.
- an exception inside the gate on a generated harness → not launched; on the provided harness → launched as today.
- the provided harness (from a fake discovery list) is never probed by the gate.

## Acceptance
```
python -m pytest -q tests/test_harness_gate_fail_closed.py
```

## Do not touch
The provided harness's launch; sidecar; crash_dedup._get_crash_signature; anything outside src/ and tests/.
