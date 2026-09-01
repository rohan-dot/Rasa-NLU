Read PATCH_SURVEY.md first — your own prior investigation. Fix the build oracle. Add NO new environment variables or feature flags; everything must work as-is on a normal run.

ROOT CAUSE: ContainerPatchBuilder.build() (src/main.py:205) tries to rebuild the target itself by setting SRC=work_src and running "compile" with cwd=work_src. That cannot work: build.sh does `cd "$SRC/java-project"` and needs $OUT, $JAZZER_API_PATH and Dockerfile-staged deps, while work_src is a copy of /opt/target/src which is already the inner project. build_ok is never True, so no patch ever validates.

KEY INSIGHT — do not reimplement the OSS-Fuzz build. discver runs under OSS-CRS (`uv run oss-crs run --compose-file ...`), and OSS-CRS's libCRS already exposes exactly the operations needed, via a builder sidecar that restores a snapshot of the compiled target with its build environment, applies a patch diff, and does an incremental rebuild:
  - apply-patch-build : rebuild the target with a patch applied
  - run-pov           : run a PoV against the rebuilt target
  - run-test          : run the project's tests
Investigate how this repo can invoke those (look in oss-crs/, compose.yaml, run-fuzzer.sh, and any libCRS client available in the container). Rewire ContainerPatchBuilder to USE them as the primary path rather than staging a source copy and shelling out to `compile`.

REQUIREMENTS:
1. Primary path: produce a patch diff from the edited scratch copy and hand it to apply-patch-build; use the resulting rebuilt target for PoV replay. Prefer the sidecar's own run-test for the regression leg when available.
2. Auto-detect, don't flag: determine at startup whether the sidecar interface is reachable, and LOG WHICH BUILD PATH IS IN USE as one of the first lines of the run (sidecar / local-fallback / none). If not reachable, fall back to the existing local-build attempt with the layout corrected (SRC must be a parent dir containing <project>/, with $OUT and $JAZZER_API_PATH reconstructed from the environment the prebuilt harness was produced with — main.py already discovers that harness in Phase 0; capture what it needs there).
3. Build caching: the loop rebuilds up to MAX_TESTS times per crash. Ensure Maven's local repository / dependency cache is reused across rebuilds rather than re-resolving every time, or patch validation will not finish in budget.
4. Classify build failures with distinct logged causes (sidecar_unavailable, wrong_src_layout, harness_not_produced, build_error) surfaced at WARNING and in patch_index.json. A broken oracle must never silently degrade into a pile of UNVALIDATED emits.
5. Regression honesty: if no test mechanism is available, report the regression gate as NOT EXERCISED in the record — never as tests_pass=True.
6. Dependency-only bugs: when the vulnerable code is not in the editable source tree (e.g. java-json-sanitizer-cve, where the bug is in a third-party jar), record it as a distinct non-failure outcome, not a build failure.
7. AUDIT the _is_test_or_harness guard. A winning AIxCC team nearly lost the competition because their "skip fuzzing harness files" heuristic matched on "fuzz" in the path, and the organizers prefixed every project directory with "ossfuzz" — so the heuristic silently excluded the entire project from patching. Check whether our guard can over-match on this target's real paths, and make it match the harness precisely (by known harness path/name) rather than by substring. Log every file it excludes so this failure mode is visible.

TESTS: the existing 36 tests only exercise the stub _Builder (tests/test_patch_generator.py:70) — that is how this bug survived. Add tests asserting the real path: which build interface is selected and what is passed to it, each failure classification, the not-exercised regression reporting, and that _is_test_or_harness does not over-match paths containing "ossfuzz"/"fuzz" as a substring of the project dir.

Do NOT change the replay logic in crash_dedup._get_crash_signature or the all-PoV loop in validate_attempt — they are correct.

Verify with py_compile and the full suite. Update CHANGELOG. Report which build interface you wired, how you detected it, and what the guard audit found. Commit.
