# Task 32 — re-fuzz the patched target without the sidecar's binaries: compile the patched classes and shadow them on the classpath

## Goal
Task 31 established that libCRS `apply-patch-build` returns only a `rebuild_id` — discver never receives the patched
binaries, so it cannot re-fuzz the official patched build. For the health question ("does the harness get past the
crash once it is patched?") discver does not need the official build. For a Java target it can build an UNOFFICIAL
patched classpath itself: compile only the patched source file(s) with javac against the original classpath (the
jars in the build dir) into `<workdir>/patched_classes/<finding>/`, then launch the same harness with that directory
FIRST on the Jazzer `--cp` (the JVM loads the first definition of a class it finds), for the remaining budget, with
the same seeds, dictionary and corpus. Report a separate health block titled
`<harness> (patched — unofficial build; validation is the sidecar's)`: HEALTHY only if coverage grows; a line
`crash gone: yes|no` from replaying the reproducer on the shadowed classpath; any new crash is a new finding tagged
with the patch id. The unpatched verdict is never replaced, and this result is never used as validation.
Trigger: the sidecar reported `build_ok=True` and `crash_gone=True` for the patch (validated, or validated except
tests). Not for findings whose site is outside the project (task 26).

## Study first
- repo: the generated-harness javac path and classpath derivation (task 29 / MAP.md "harness generation"); the
  provided harness wrapper — read `/opt/target/build/<harness>` style wrappers to see how `--cp=` and
  `--target_class=` are passed to `jazzer_driver`; how discver builds launch commands (fuzzer.py:612) and whether it
  can launch `jazzer_driver` directly with its own `--cp` instead of through the wrapper; the sidecar result fields
  (`build_ok`, `crash_gone`, `tests_pass`) from task 24's classification.
- results: `grep -nE "SIDECAR|build_ok|crash_gone|tests_pass|validated" latest/orchestrator.log | head -20`.

## Change (src/) — new module `src/post_patch_shadow.py`, one call site after a patch is validated
- `compile_patched_sources(patch, project_root, build_dir) -> shadow_dir`: the files touched by the patch, read from
  the patched work copy, compiled with javac `-cp <all jars in build_dir>` into the shadow dir; javac stderr tail
  logged on failure and the re-fuzz skipped (`[refuzz] <harness>: shadow compile failed: <tail>`).
- `shadowed_launch(harness_wrapper, shadow_dir, …) -> argv`: parse `--cp=` and `--target_class=` (and the agent path)
  from the harness wrapper, and build a direct `jazzer_driver` command with `--cp=<shadow_dir>:<original cp>` and the
  same remaining options; log the full command once: `[refuzz] <harness> cmd: <argv>`.
- Run through the existing fuzzing loop (stats parsing, health, crash handling — unchanged) for the remaining budget;
  render the separate health block; replay the reproducer once on the shadowed classpath for `crash gone`.
- Log `[refuzz] <harness> on patch <id>: <execs> execs, cov <a>→<b>, crash gone=<yes|no>, <n> new crash(es)`.
- Safety: wrapped; `[refuzz] failed: <error>` with traceback → the run ends as it does today.

## Tests to add (tests/test_post_patch_shadow.py) — fakes only (fake javac, fake wrapper file, fake runner)
- a fake wrapper with `--cp=a.jar:b.jar --target_class=X` and a fake patch touching `X509Utils.java` → javac is
  called with `-cp a.jar:b.jar` on that file, and the launch argv has `--cp=<shadow>:a.jar:b.jar` and `--target_class=X`.
- the report contains both health blocks; the unpatched verdict text is unchanged by the re-fuzz result.
- coverage grew on the shadowed run → HEALTHY in the patched block; flat → WEAK; `crash gone` reflects the replay.
- shadow compile failure → the skip line; no launch.
- the trigger: a patch with `build_ok=False` never starts a re-fuzz; `build_ok=True, crash_gone=True` does.

## Acceptance
```
python -m pytest -q tests/test_post_patch_shadow.py
```

## Do not touch
The sidecar invocations (apply-patch-build / run-pov / run-test) and the validation verdict; crash_dedup._get_crash_signature; the normal launch of the unpatched harness; anything outside src/ and tests/.
