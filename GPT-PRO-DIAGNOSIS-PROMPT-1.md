# Regression diagnosis: the main fuzz harness is reported DEAD after a diff, and the fix loop never ran

I maintain a Python orchestrator (`src/`, tests in `tests/`) that runs Jazzer/libFuzzer harnesses against a Java codebase,
parses the fuzzer's stats lines, assigns each harness a health verdict (HEALTHY / WEAK / DEAD), and then runs an automated
fix loop for any crash it reproduced. A batch of changes went in between two runs; the run after them regressed. Find out
why, from the uploaded evidence, and give me the smallest fix.

## Uploaded files (project sources)
- `discver_code.zip` — `src/` and `tests/` exactly as the latest run used them (the rest of the repo is unchanged and not included). Only these may be changed.
- `patch.diff`, `changed_files.txt`, `CAMPAIGN.md`, `MAP.md` — exactly what changed between the two runs, with the test verdict per change. `MAP.md` has file:line locations for the main stages (fuzzer launch, stats parsing, health checker, seed generation, fix loop).
- `discver_results_latest.zip` — the LATEST run (main harness DEAD; fix loop produced nothing): `orchestrator.log`, `health.json`, `patches/patch_index.json`, `report_run_*.md`.
- `discver_results_prev.zip` — the PREVIOUS run, before the changes. In it the main harness (`X509Fuzzer`) was WEAK: a few hundred executions, coverage flat at ~2043, verdict "SATURATED by a shallow crash" (nearly every input hits the same exception at the entry point, so the process restarts constantly); the fix loop ran on 7 findings and failed all of them with `attempts=15, test_calls=0`. If only its report is present, use the report's health block and summary as the "before".
Read the zips directly and refer to files by their path inside the zip. If a file named here is missing, say so once and continue.

## Questions, in order — every claim with a quoted line (file:line or log line number); write MISSING plus the command that would produce it rather than inferring
1. **Did the new code run at all?** The changes add startup/decision log lines. Search the latest `orchestrator.log` for `fuzz thresholds|probe |deep phase|keep_going|no-op|seeds wired|parse_failures|patch-turn|llm-call`. If none appear, the run used stale code and the rest is moot — say that first.
2. **What made the main harness DEAD?** Quote its health block from both runs. Was it stopped early (by which code path, at what second / execution count), never launched, launched with a different command line, or did coverage parsing change so real coverage now reads as 0? Compare the harness's first stats lines in both logs.
3. **Which hunk caused it?** Walk `patch.diff` against the evidence from (2) and name the hunk. Test this hypothesis first: one change added a "no-op harness" detector that stops any harness whose coverage stays flat for a budget. The main harness's coverage is flat *because* it is saturated by one shallow crash — it has crashes — so that detector would stop the real target. Confirm or refute with log lines.
4. **Why did the fix loop produce nothing?** Read the latest `patches/patch_index.json` and grep the log for the fix loop. Distinguish "no reproduced crash existed because the harness was dead" (downstream of 2) from "the loop ran and failed" (quote its reasons and any counters).
5. **Minimal fix.** Prefer a guard over a revert — e.g. the no-op rule must not apply to a harness that has produced a crash, or whose first coverage value is non-trivial; the saturated case belongs to the keep-going path. Unified diff limited to `src/` and `tests/`, with a test that fails on the current code and passes with the fix. No refactors, no reformatting, no second change.
6. **Then, only then:** is the probe → keep-going mechanism present in the shipped code (grep `keep_going|ignore_crashes|probe`)? If absent or never reached, describe the smallest change that lets the main harness keep fuzzing past the known crash (Jazzer `--keep_going=N` relaunch after a saturated probe; replay the deep-phase corpus through the normal harness afterwards so crashing inputs become ordinary reproducers), as a separate second diff.

## Output
```
1. CODE RAN: yes/no — evidence
2. DEAD CAUSE: … — quoted lines
3. RESPONSIBLE HUNK: patch.diff lines … — why
4. FIX LOOP EMPTY BECAUSE: … — quoted lines
5. FIX (diff + test)
6. KEEP-GOING STATUS + second diff (optional)
7. NEXT RUN WILL SHOW: the exact startup line(s) and one grep per fix; what it looks like if the fix did not take effect
```
Ground rules: don't claim a result you didn't see in the uploaded files; health verdicts must stay accurate (never paper over a dead or saturated harness); don't touch `crash_dedup._get_crash_signature` or the builder-sidecar calls (`apply-patch-build` / `run-pov` / `run-test`); no feature flags or environment switches — the orchestrator auto-detects and logs the path it chose.
