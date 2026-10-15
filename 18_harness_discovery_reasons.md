# Task 18 — discovery found one provided harness of two: log why each candidate was excluded, and fix the filter

## Goal
Latest run `:15 "harnesses: provided=[X509Fuzzer] generated=[]"`. The project's build output also contains
`GetPrettySignatureFuzzer` (both `.java` files sit together in the project's fuzz directory and both are built by its
build script), and it has never appeared in any run. Discovery (`main.py:1143–1183 _discover_harnesses`) scans the
build dir, excludes data suffixes / jars / LLVM tools / jazzer*, then filters by a libFuzzer `-help` probe. Something
in that chain drops the second harness silently. Make every exclusion visible, then fix the one that is wrong.

## Study first
- repo: read `_discover_harnesses` in full and its `-help` probe (exit-code handling, timeout, environment, cwd, what
  counts as "usage text"); MAP.md "harness selection" (:1185 all_harnesses, :1190 auto_mode, :1192–1200 named harness,
  :1389 / :1399 primary runner).
- results: `grep -nE "harness-discovery|discover|provided=|-help|probe" latest/orchestrator.log | head -30`.
- Think through why a second wrapper could fail the probe while the first passes: non-zero exit with usage on stderr,
  a different classpath or missing jar for that harness, a probe timeout, a name caught by the `jazzer*` / suffix
  exclusions, or the build dir scanned being a different one than the operator's.

## Change (src/)
- Log `harness-discovery: scanned <dir>: <n> entries, <k> candidates, <p> provided` and, for every entry that is not
  admitted, `harness-discovery: <name> excluded: <reason>` where reason is one of: data suffix / jar / llvm tool /
  jazzer helper / not executable / help probe exit <rc> (stderr tail: …) / help probe timeout <s>s / no usage text.
- Probe fix: accept a non-zero exit when stdout+stderr contains libFuzzer/Jazzer usage text (e.g. `-max_total_time`,
  `libFuzzer`, `jazzer`); run the probe with the same environment and working directory as the real launch; a probe
  that times out is logged, not dropped — the wrapper is admitted as "unprobed" and the probe phase (task 11's probe
  → deep phase) decides.
- Every provided harness becomes a primary runner (probe → deep phase), each with its own seed set from the seed
  module (keyed by harness name, derived from each harness's own source — nothing target-specific in src/).

## Tests to add (tests/test_harness_discovery_reasons.py) — fakes only
- a fake build dir with two wrappers, one of whose `-help` exits 1 but prints usage → both provided; the scanned/
  candidates/provided line is correct.
- `llvm-symbolizer`, `jazzer_driver`, `x.jar`, `X509Fuzzer_seed_corpus.zip` → excluded with the right reasons logged.
- a wrapper whose probe times out → logged with the timeout reason and admitted as unprobed.
- two provided harnesses → two primary runners, each launched with its own seed directory.

## Acceptance
```
python -m pytest -q tests/test_harness_discovery_reasons.py
```

## Do not touch
Sidecar; crash_dedup._get_crash_signature; the normal launch command; anything outside src/ and tests/.
