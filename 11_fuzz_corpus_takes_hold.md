# Task 11 — seeds are written, but the fuzzer's corpus never takes hold: prove whether the files are loaded, then fix

## Goal
Latest run, primary harness: `:130 "seeds mined=47 format=79 llm=5 dropped=42"`, `:107 "47 / 79 seed(s) kept UNSCREENED"`,
yet 109,889,040 execs with coverage flat at 2043→2043 and corpus 0→4 "(effective 0, external+4)". Two explanations,
and the fuzzer's own startup output decides between them: (A) the fuzzer never loads the seed files (written to a
directory it does not read, written after launch, or in a layout it ignores) — libFuzzer prints
`INFO: N files found in <dir>` / `INFO: seed corpus: files: N` at startup; (B) it loads them and they add no coverage,
which would mean the target's code is not being counted. Determine which from results/latest, make the launch and the
logging prove it every run, and fix the wiring if it is (A).

## Study first
- results: `grep -nE "files found in|seed corpus|corpus_dir=|seed_dir=|\[seeds\]|UNSCREENED|dropped=|-dict|instrumentation|cov: 2043|corp: " latest/orchestrator.log | head -60`
  — quote the `[seeds] <harness> launch: corpus_dir=<n> file(s), seed_dir=<s>` line (fuzzer.py:658) and the launch line (:135) and any libFuzzer "files found" line, or state that none exists.
- repo: MAP.md "seeds" — the seed writer (pb_seeds.py:229 / :296 and the task-12 module), the launch (fuzzer.py:516 /
  :612 / :658), the live re-seeding of `seed-*` files (fuzzer.py:40–48, :91–101), and where health computes
  "effective" vs "external" corpus (run_health.py:152–153). Establish the ORDER of events: are the seed files on disk
  in the directory passed to the fuzzer BEFORE the process starts?
- Why "UNSCREENED": find the branch that skips the known-crash screening and what it logs.

## Change (src/)
- Launch invariant: the directory that holds the seed files is passed to the fuzzer as a corpus/seed directory, and the
  files exist before the process starts; log once per harness
  `[seeds] <harness>: <n> files in <dir> at launch; fuzzer reported <m> files found` where m is parsed from the
  fuzzer's `INFO: ... files found in <dir>` / `seed corpus: files:` lines. n>0 with m==0 is an ERROR line naming both
  directories — never silent. If the fuzzer reads a different directory than the writer uses, make them the same.
- "UNSCREENED" logs the specific reason screening was skipped (helper unavailable / timeout / exception), one line.
- If m>0 and coverage never rises above the first stats line within the probe window, log
  `[coverage] <harness>: <m> seed files loaded, cov unchanged at <c> — target code may not be instrumented; flags: <the Jazzer/libFuzzer flags from the launch line>`
  so the next run answers (B) without guessing.
- Health block: "effective corpus" = the `corp:` count from the fuzzer's own stats lines; "external" stays as today.

## Tests to add (tests/test_corpus_takes_hold.py) — fakes only
- seed files written → the launch argv contains their directory → fake fuzzer output `INFO: 5 files found in <dir>` →
  the log line reports n=5, m=5.
- files written but the fake fuzzer reports `0 files found` → ERROR line names both directories.
- files written after a fake launch (wrong order) → the launch invariant test fails on the current code if that is what
  the current code does; passes after the fix.
- m>0 and flat coverage through the probe → the `[coverage]` line is present with the flags.
- UNSCREENED path → its reason line is present.

## Acceptance
```
python -m pytest -q tests/test_corpus_takes_hold.py
```

## Do not touch
The sidecar; crash_dedup._get_crash_signature (call it, do not change it); the fuzzing flags themselves; anything outside src/ and tests/.
