# Task 25 — the applied edit is lost because `diff` exits 2 on a dangling symlink: fix `_diff_from_copy` only

## Goal
Latest run, the reproduced crash's finding: `:1314 "[ok] applied edit #1 to .../X509Utils.java"` →
`:1321 "[patch] source diff failed rc=2: diff: /opt/target/src/fuzztest/bazel-fuzztest: No such file or directory"`
(repeated for `bazel-out` / `bazel-testlogs` at :1322–1326) → `src/main.py:390–394 _diff_from_copy` returns `""` on
rc 2 → `src/main.py:439–444 build()` maps the empty diff to BUILD_BUG_IN_DEPENDENCY → `:1328 "tool=test_patch ->
bug in third-party dependency (skip)"`. The edit never reached the builder. GNU `diff -r` exits 2 ("trouble") when it
cannot follow a symlink, but it still prints the hunks for every file it could compare; the code discards that output.
Fix the diff computation so dangling symlinks cannot erase a real edit, and make an empty-diff-after-edit impossible to
misread again. `_diff_from_copy` makes no sidecar invocation: `build()`, `_build_sidecar` and the `apply-patch-build` /
`run-pov` / `run-test` calls stay byte-for-byte untouched.

## Study first
- repo: read `_diff_from_copy` in full (main.py near :390), its caller in `build()` (:439–444), how the work copy is
  created (does the copy carry the `bazel-*` symlinks? `shutil.copytree(symlinks=...)`?), and the exact `diff`
  argv (flags, which two roots, cwd). Note file:line for each.
- results: `grep -nE "source diff failed|rc=2|No such file or directory|bazel-|BUG_IN_DEPENDENCY|third-party|apply_edit" latest/orchestrator.log | head -30`
  and the same over prev/ — quote the ok→skip sequence in finish().
- `run`: `diff --help 2>&1 | grep -n "no-dereference"` to learn whether the box's diff supports `--no-dereference`
  (compare symlinks as symlinks instead of following them; a dangling link then produces no error). Note the result;
  the container's diff may differ, so the fix must work either way.

## Change (src/ — `_diff_from_copy` and its immediate caller only)
- Run diff with `--no-dereference` when the binary accepts it (probe once, cache the answer); otherwise unchanged flags.
- Treat exit code 2 as "diff produced with errors", not as failure: if stdout is non-empty, return it as the diff and
  log `[patch] diff reported errors (rc=2) but produced <n> hunk(s); stderr tail: <tail>`; if stdout is empty, return
  "" and log the stderr tail as the reason.
- An empty diff after a successful apply_edit is classified `diff_error` (rc 2, with the stderr tail) or
  `diff_mismatch` (rc 0, edited path not under the diffed root — log edited path and diffed root). Neither may ever
  become BUILD_BUG_IN_DEPENDENCY; that label is reserved for a crash site outside the project tree (task 26 if present).
  `patch_index` records `skip_cause=diff_error|diff_mismatch` with the detail.
- Keep `_diff_from_copy`'s signature and return type. No change to the work-copy creation unless the evidence shows the
  copy itself drops files (then say so in finish() and keep the change minimal).

## Tests to add (tests/test_patch_diff_rc2.py) — fakes only; use a real temp tree and the real `diff` binary
- two temp trees that differ by one edited line and contain a dangling symlink in both → `_diff_from_copy` returns a
  diff containing the edited line; no exception; the rc=2 warning line is logged when the binary lacks
  `--no-dereference`, and no warning when it has it.
- a fake diff runner returning rc=2 with empty stdout → returns "", logs the stderr tail, and the caller classifies
  `diff_error`, never BUILD_BUG_IN_DEPENDENCY.
- a successful apply_edit outside the diffed root (fake paths) → `diff_mismatch` with both paths in the log and reason.
- the sidecar invocation strings (apply-patch-build / run-pov / run-test argv builders) are byte-identical before and
  after — snapshot assertion taken from the current code.

## Acceptance
```
python -m pytest -q tests/test_patch_diff_rc2.py
```

## Do not touch
`build()`'s sidecar calls, `_build_sidecar`, the three sidecar invocations, crash_dedup._get_crash_signature; anything outside src/ and tests/.
