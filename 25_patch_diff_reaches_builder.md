# Task 25 — applied edits must reach the builder: fix the empty-diff "third-party dependency" skip

## Goal
In both real runs the fix loop applied an edit to the X509Utils crash site (apply_edit ok), then the builder returned
BUILD_BUG_IN_DEPENDENCY because `ContainerPatchBuilder._diff_from_copy(work_src)` came back empty, and the finding was
recorded as "skipped — vulnerable code is in a third-party dependency". The edit existed; the diff did not see it. Find
where the two trees diverge and fix the side that is wrong. This is discver's own Python (main.py / patch_generator.py),
not the sidecar commands: the `apply-patch-build`, `run-pov` and `run-test` invocations stay byte-for-byte unchanged.

## Study first — read-only until the divergence is proven, and say so in finish() before editing
- repo: MAP.md "sidecar/oracle" and "patch loop". Read in full: `_diff_from_copy` (main.py near :433), its call site
  (:433–438), `ContainerPatchBuilder.build` (:425), how the work copy (`work_src`, scratch copy) is created and where,
  and `apply_edit` in patch_generator.py — the exact path it writes to.
- results: `grep -nE "apply_edit|_diff_from_copy|BUG_IN_DEPENDENCY|third-party|work_src|scratch" latest/orchestrator.log`
  and the same over prev/ — quote the ok→skip sequence (latest:719→740; prev:1486→1528).
- Establish with file:line: (a) the directory apply_edit writes into; (b) the directory `_diff_from_copy` diffs, and
  against which baseline; (c) the relative-path / prefix form each side uses.

## Change (src/) — exactly ONE of these, whichever the evidence shows
- A. apply_edit writes outside the tree the builder diffs → make apply_edit write into (or be pointed at) the work copy
  the builder diffs (patch_generator.py). The builder is untouched.
- B. `_diff_from_copy` diffs the wrong tree, the wrong baseline, or strips the wrong prefix → fix that computation only
  (main.py), keeping its signature and return type. The subprocess calls are untouched.
- Either way: an empty diff after a successful apply_edit must never be labelled "third-party dependency" again. Log
  `[builder] empty diff after apply_edit: edited=<path> diffed_root=<root> — classification: diff_mismatch` and record
  `skip_cause=diff_mismatch` in patch_index.

## Tests to add (tests/test_patch_diff_reaches_builder.py) — fakes only
- a fake work copy + a fake apply_edit of one line → the diff the builder receives is non-empty, contains that line, and
  uses the path form the sidecar expects.
- the empty-diff-after-edit case logs the `diff_mismatch` line and never yields BUILD_BUG_IN_DEPENDENCY.
- the sidecar invocation strings (apply-patch-build / run-pov / run-test argv) are byte-identical before and after —
  assert against a snapshot taken from the current code.

## Acceptance
```
python -m pytest -q tests/test_patch_diff_reaches_builder.py
```

## Do not touch
The sidecar subprocess invocations themselves and crash_dedup._get_crash_signature; anything outside src/ and tests/.
