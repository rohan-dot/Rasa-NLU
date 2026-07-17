
Let me ground the "wrong repo" question in the actual code before I answer, since that's the one that matters.

Taking your three questions in the order that matters, because the repo one is the whole ballgame.

## Why it keeps fuzzing the skeleton, not your code

Your code *is* being analyzed — it's just not being *fuzzed*. Those are two different halves of the CRS looking at two different things:

- The **analysis** half (call graph, prescan, cross-file audit, taint, the CodeQL/PMD/SpotBugs pass) reads all of `/opt/target/src`, and it clearly read *your* repo — that's where the LDAP-injection candidates in `performLDAPModify` / `doesAgentExistInPeople` / `exportAgentsOnCurrentInstance` and the 654 PMD / 2297 SpotBugs findings came from. Those are your `agentmanager-java8-master` LDAP code, not the skeleton.
- The **fuzzing** half only ever runs the two *prebuilt* harness binaries it finds in `/opt/target/build/` — `DecoderFuzzer` and `NativeBufFuzzer`. Those two harnesses call into `com.discver.skeleton`. So every *confirmed crash* is in the skeleton, because the skeleton is the only thing with a runnable harness.

I checked: the skeleton is nowhere in the CRS source (`grep skeleton/NativeBuf/DecoderFuzzer *.py` → nothing). It's not the CRS shipping a demo. It lives in the *target image* — whoever packaged `/opt/target` built the skeleton's `DecoderFuzzer`/`NativeBufFuzzer` into `build/` and dropped your source into `src/`, but **nobody wrote a Jazzer harness that calls your LDAP code.** `_find_binary()` in `fuzzer.py` just globs `build/` for the configured harness name (`DecoderFuzzer`) and runs it. It's doing exactly what it's told; it's told to run the skeleton harness.

So: your code is present for *reading*, absent for *fuzzing*. Present in `src/`, missing from `build/`.

## The blocker

**There is no fuzz harness that drives your code, and the CRS currently cannot build one.** Two things nail it shut, both visible in the logs:

1. `custom-harness compile: False` → Java is running in **prebuilt-only** mode. The exploiter says it out loud every run: *"Skipping harness compilation (java: custom-harness compilation unsupported (prebuilt-only))."*
2. `[java-harness] Jazzer API jar not found ($JAZZER_API_PATH unset and not on disk)` → the java-harness-agent that *would* synthesize a Jazzer harness for your entry points can't even start. Hence `Generated harnesses: 0` and `No fuzzable entry points found`.

That missing Jazzer jar is the single highest-leverage fix — it's the exact thing standing between you and arbitrary-repo fuzzing. Until a harness exists that calls into your code, coverage-guided fuzzing will *only* ever touch the skeleton, and everything on your actual repo stays at "unconfirmed static finding."

Fastest unblock, in order: set `JAZZER_API_PATH` to the Jazzer jar in the image (or install it) so harness synthesis runs; **or** hand-write one Jazzer harness that calls your LDAP filter/DN construction (`doesAgentExistInPeople` / `performLDAPModify` take attacker-influenced `dn`/filter strings — perfect fuzz entry point), drop it in the OSS-Fuzz project so it builds into `build/`, and point `harness` at it. `FORCE_HARNESS_COMPILE=1` won't save you here — Java compile is disabled, so it's the jar or a hand-written harness, not the override.

## What the reports read like

Mixed — trust the mechanical parts, discount the LLM-authored parts:

- **Solid:** the dedup summary (`unique_bugs/summary.json`, 2 unique, 0 non-reproducing), the PoV inputs, the ASAN traces, `health.json`, `crashreport.md`. These are grounded in real reproduction and are trustworthy.
- **Patchy:** the LLM bug narratives in `/output/bugs/bug-*.json`. You're getting repeated `WARNING: LLM returned invalid JSON — using raw report` — that's the glm52 JSON-adherence problem, so several "reports" are raw text fallbacks, not structured findings. Triage verdicts read low-confidence and self-contradicting across runs (33% `[V??+V]`, 0% `[??I→I]`) because 2/3 triage rounds got no usable model response.
- **Misleading if skimmed:** `Crash bug reports: 5` but only `2 unique bugs` — the count is inflated by multiple writers (analyzer + pipeline + dedup all emit). And because the only confirmed crashes are skeleton bugs, a PI reading `REPORT.md` top-to-bottom would think discver validated *your* repo. It didn't — it validated itself on the skeleton and produced *unconfirmed* leads on your code.

## The Java tooling in this build (what's new vs the C/C++ AIxCC path)

Honest caveat first: some of this lives in modules that aren't in the snapshot I can see (no `structured.py`, `harness_error.py`, or `semgrep_scan.py` in my copy), so they came in with your ChatGPT integration or sit elsewhere — I'm reading them off the logs, not the source. The Java-specific machinery that's new relative to the old flawfinder/cppcheck C pipeline:

- **tree-sitter-java call graph** (the `languages` module) — proper Java AST parsing; when the grammar's missing it silently drops to a regex fallback (that was your earlier bad run).
- **java-harness-agent** — Jazzer-based entry-point discovery + harness synthesis. This is the one that's broken (missing jar) and the reason you're stuck on the skeleton.
- **Java SAST fan-out** via the external runner — CodeQL (`java-security-and-quality.qls`, buildless `build-mode=none`), Semgrep, PMD, SpotBugs, plus the built-in `java-pattern` scanner. None existed on the C path.
- **Java-aware taint** (source→sink; caught the JNI `n → malloc → memcpy` on the native side) and **LLM cross-file audit** (the LDAP-injection detector on your agentmanager code).
- **Jazzer as the Java fuzz engine + Java crash triage/dedup** — parsing `ArrayIndexOutOfBounds` and JNI ASAN crashes, deduping by Java frame.

The irony worth saying out loud: the newest and most valuable of these (the SAST fan-out + cross-audit) are the *only* tools that actually reached your code this run — and they're the ones producing unconfirmed findings that need the harness to validate. Fix the Jazzer jar and the whole thing points at your repo instead of the scaffold.

Want me to draft that one Jazzer harness for `doesAgentExistInPeople` / `performLDAPModify` so you can drop it into the project and actually fuzz your LDAP code?
