# Task 61 — workshop retrieval: stop matching runtime frames and generic tokens; one plan line per target; language from OSS-Fuzz; GitHub-only fetches

## Goal
The first real dry run (`build_kb.py --dry-run-target <java-ant-byo>`, 4,100 recipes) shows the ranking decided by the
wrong evidence: `token overlap {string, buffer, bytes} … crash state …java.base/java.lang.StringCoding.encode (+11.60)`,
`{request, connect} … com.code_intelligence.jazzer.sanitizers.ServerSideRequestForgery.checkSsrf (+10.87)`. JDK and
Jazzer frames appear in nearly every Java crash state, tokens like `string`/`buffer`/`bytes`/`request` match everything,
and that overlap is worth +11 while idiom (+1.5), language (+1.0) and crash-type prior (+0.9) are worth +3 together. Six
of the ten plan lines target the same function; the string parser `getCnFromDn` sits 5th. Also: the build reports
`go=0, python=12, rust=3` because language is inferred from file extensions in crash states (Go/Rust/Python frames have
none), and `--fetch-fixes` hangs 120 s per `git://` repository the firewall blocks. Fix all five:
1. **Runtime frames out.** Before tokenizing a crash state, drop frames from the language runtime and fuzzer: Java
   `java.base/`, `java.`, `javax.`, `jdk.`, `sun.`, `com.code_intelligence.jazzer.`; C/C++ libFuzzer/AFL/honggfuzz/
   sanitizer frames (`__asan`, `__interceptor`, `fuzzer::`, `LLVMFuzzerTestOneInput`); Go `runtime.`, `testing.`;
   Rust `core::`, `std::`, `alloc::`, `libfuzzer_sys`; Python `/usr/lib/python`, `atheris`. Keep the Jazzer sanitizer
   name as the recipe's `subtype` (`SSRF`, `SqlInjection`, `Deserialization`, …) — it is a label, not code. A state
   with no project frame left keeps its crash type and gets `tokens=[]`.
2. **IDF weighting + stop tokens.** Token weight = log(N / df); tokens present in > 20 % of a language's recipes get
   weight 0 (that covers `string`, `buffer`, `bytes`, `byte`, `data`, `get`, `set`, `new`, `init`, `read`, `write`);
   minimum token length 3; the per-token weights are written to `kb/index.json`. Score = Σ weights of overlapping
   tokens, capped so that token overlap can be at most 2× the sum of the other signals — idiom and delta evidence
   must be able to outrank it.
3. **One line per target.** Aggregate recipes by target function: score = top-3 recipe matches + idiom + (delta prior
   when present), listing the best 3 OSV ids as evidence; the plan shows at least 5 distinct targets in the top-k
   when the profile has that many public functions. The dry run prints one line per target with the breakdown.
4. **Language from OSS-Fuzz.** New `--oss-fuzz-projects <dir>` (a sparse clone of `google/oss-fuzz`, `projects/*/project.yaml`):
   project → `language:`; fall back to the extension inference only when the project is unknown. Report the per-language
   counts before/after.
5. **Fetches.** `--fetch-fixes` tries only `https://github.com/` repositories (and `--fetch-hosts` to add others), a
   60 s timeout each, and skips every other host instantly with a count (`skipped non-github: 37`).

## Study first
- `src/workshop/{tokens,osv_ingest,retrieve,build_kb}.py` and `tests/test_workshop.py` (the fixtures to extend);
  the dry-run output above (quote it in the report); ROOTS `oss-fuzz-vulns` for real crash states per language.

## Change (src/workshop/ only)
- As specified; `profile.py` applies the same runtime-frame and stop-token rules to the target's own functions so both
  sides are tokenized identically. `build_kb.py --dry-run-target` output format: one line per target:
  `1. getCnFromDn(String)  score 9.4  [idiom string-parsing +1.5 | tokens {substring, index} +4.1 | prior Uncaught exception +0.6 | 3 recipes: OSV-… OSV-… OSV-…]`.

## Tests to add (tests/test_workshop_v2.py) — the task-55 fixture extended
- a crash state of only JDK + Jazzer frames → `tokens=[]`, `subtype=SSRF`; a mixed state keeps only project frames.
- a token present in 40 % of recipes has weight 0; a rare token outranks it.
- the java-ant-byo-like fixture (getCnFromDn with substring/indexOf; an LDAP connector; a byte-reading audit method)
  → `getCnFromDn` first, three distinct targets in the top 3, one line each.
- `--oss-fuzz-projects` fixture with a Go project → the Go record counted as go, not c.
- a `git://` fix URL is skipped without a fetch attempt; a GitHub one is attempted (fake git runner).
- `tests/test_workshop.py` still passes.

## Acceptance
```
python -m pytest -q tests/test_workshop.py tests/test_workshop_v2.py
```

## Do not touch
The strategy phase hooks (task 56) beyond the `retrieve.plan()` signature, which stays the same; anything outside src/ and tests/.
