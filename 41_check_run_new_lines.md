# Task 41 — check_run.py is blind to the newest changes: grade the patch-path, harness-source, entry-point and re-fuzz lines

## Goal
`src/check_run.py` (task 40) grades a results folder for the lines emitted by tasks 10–17 only. The latest campaign
landed changes it cannot see — relative patch paths (task 28), provided harness sources compiled by discver (task 29),
ranked entry points for generated harnesses (task 19) and the post-patch shadow re-fuzz (task 32) — so the next real
run would again be graded by reading the orchestrator log by hand. Add one check per change, with the same
`PASS` / `FAIL` / `NOT SEEN` + deciding-line output, plus a generic check that no safety wrapper fell back. Exit code
stays 0 (a report, not a gate).

## Study first
- repo: `src/check_run.py` — the check table, how a check names its success/failure patterns and where it looks, and
  how `NOT SEEN` is printed; `tests/test_check_run.py` for the fake-results-dir helper (reuse it, do not copy it).
- repo: the REAL format strings, never guesses:
  `grep -rn "candidate patch:\|not under project root\|source harnesses found\|compile failed:\|candidates ranked\|no admissible entry point\|\[refuzz\]\|\] failed:" src/`
  If a producer is absent from src/ (its task did not land), add the check anyway with the format written below;
  `NOT SEEN` is then the correct output for that line.
- results: `grep -nE "apply-patch-build|patch_apply_failed|harness-discovery|harness-gen|SIDECAR" latest/orchestrator.log | head -30`
  — the lines the next run must improve on.

## Change (src/check_run.py only)
Add these checks (each: what decides PASS, what decides FAIL, what prints NOT SEEN):
- **patch paths (28)** — PASS: a `[patch] candidate patch: <n> file(s), paths relative to <root>, first=<rel>` line AND
  the following `[sidecar] cmd=[... 'apply-patch-build' ...] exit=0`. FAIL: any sidecar stderr tail containing
  `No such file or directory` or a `[patch-turn] … class=patch_apply_failed` line, quoted. NOT SEEN: no candidate
  patch line (print the number of `diff_mismatch` lines if any).
- **harness sources (29)** — PASS: `harness-discovery: source harnesses found: [<a>]; built by project: [<b>];
  compiled by discver: [<c>]` where every name in `<a>` appears in the `harnesses: provided=[…]` line. FAIL: any
  `harness-discovery: <name>: compile failed:` line, quoted, or a found source missing from `provided=`. NOT SEEN:
  no `source harnesses found` line.
- **entry points (19)** — PASS: `harness-gen: <n> candidates ranked; top-k=[…]` AND at least one later
  `harness-gen <name>: reach=ok run=ok → admitted` line. FAIL: the ranked line present but every generated harness
  `→ rejected` (quote the first rejection reason). NOT SEEN: no ranked line; if
  `harness-gen: no admissible entry point — <reason>` is present, print it under NOT SEEN as the reason.
- **shadow re-fuzz (32)** — PASS: `[refuzz] <harness> on patch <id>: <execs> execs, cov <a>→<b>, crash gone=yes, …`
  with `<b>` > `<a>`. FAIL: `crash gone=no`, `cov` flat, or a `[refuzz] <harness>: shadow compile failed:` line.
  NOT SEEN: no `[refuzz]` line — print whether the trigger could have fired (any `build_ok=True` in the log).
- **safety wrappers** — FAIL if any `[<module>] failed:` line exists (`[harness_entry_points] failed`,
  `[harness_sources] failed`, `[refuzz] failed`, or any other `[…] failed:` of that shape): quote it and the first
  line of its traceback. PASS otherwise. This is the one check that says "the new code itself crashed".
- Summary additions: number of provided harnesses (and which were compiled by discver), generated harnesses
  admitted/rejected, and the patched-block verdict per harness when a re-fuzz ran.

## Tests to add (tests/test_check_run_new_lines.py) — fake results dirs built from the formats above
- a dir with all new lines in their PASS form → the five new checks PASS; the summary counts are right.
- a dir reproducing the latest run (apply error `No such file or directory`, one provided harness, every generated
  harness rejected, no `[refuzz]`) → patch paths FAIL with the error quoted, harness sources NOT SEEN, entry points FAIL
  with the rejection reason, re-fuzz NOT SEEN with "trigger could not fire".
- a dir with a `[harness_sources] failed: …` line and a traceback → the safety-wrapper check FAILs quoting it.
- the existing `tests/test_check_run.py` still passes unchanged (the old checks' output is byte-identical).

## Acceptance
```
python -m pytest -q tests/test_check_run.py tests/test_check_run_new_lines.py
```

## Do not touch
Everything in src/ except `check_run.py`; `tests/test_check_run.py`; anything outside src/ and tests/.
