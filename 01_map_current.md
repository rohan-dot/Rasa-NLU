# Task 01 — map the CURRENT code and the latest run it produced, whatever the target (read-only)

## Map
This task makes NO edits. Its finish() summary becomes MAP.md: it is handed to every later task as context AND it is the
evidence base for the final list of tasks, so it has to be complete and exact. Rules:
- file:line for every code claim; log line number + verbatim text for every evidence claim; nothing from memory.
- Where something cannot be determined, write `not determined — <what would answer it>`; never guess.
- You will be nudged to "make the edit". Ignore it: this task is read-only by design. Keep reading until the deliverable
  below is filled in, then call finish() with the whole deliverable as the summary.
- Size: the summary must stay under ~6,000 words. Summarize mechanisms; do not paste code; quote log lines, not log pages.

## Study first — the run (`results/latest/`, read-only)
The run is the most recent real execution of the CURRENT src/ on some target — which one is not known to this task:
read it from the first 40 lines of `orchestrator.log` (target name, language, mode, budget, model) and refer to it by
that name from then on. Nothing in this map may assume a language, a build system or a harness shape. If the layout is staged (`STAGED.md` present) read
`STAGED.md` and every `*_listing.txt` first; otherwise `list_files` the folder and note what is missing (no `/output`
copy? no SUBMIT_DIR? say so under OPEN QUESTIONS).
Read in full EVERY top-level file in `results/latest/` other than `orchestrator.log` (a flat log-dir copy has `patch_index.json`,
`report.json`, `crashreport.md`, `crash-summary.txt`, `health.txt` at the top level — those are the primary evidence; a 0-byte
`libcrs-pov.log` means the sidecar never ran a PoV). Then, when present: `health.json`, `REPORT.md`, `report_run_*.md` (first 200 lines), `patches/patch_index.json`,
`output/input_boundaries.json`, `output/components.json`, `output/llm_capabilities.json`, `static_findings.json`,
`libcrs-pov.log`, `libcrs-seed.log`, `crs-bug-candidate.log`, everything under `patches/`, `generated_harnesses/` (every
source + any compile/probe log beside it), `unique_bugs/`, the `.txt` traces in `crashes/`, every file in `hex/`,
the small text files in `run/SUBMIT_DIR` and `run/SHARED_DIR`.
Then grep `orchestrator.log` phase by phase and read only the matching lines plus 3 lines of context:
- startup/config: `Target language|adapter=|\[lang\]|DISCVER_|FORCE_|KB |\[strategy\]|guided_json|vLLM client|model=|budget|max_total_time|timeout`
- build/components: `\[build\]|\[components\]|cannot build|local_build|\$OUT|compile_commands|harness-discovery:|source harnesses found|built by project`
- boundary/ranking/generation: `\[boundary\]|input_boundaries|candidates ranked|harness-gen|GenFuzzer|RegenFuzzer|boundary harness|custom-harness compile|Skipping harness compilation|-help=1|failed libFuzzer probe|\[harness_reach\]|harness_repair|\[gate\]|static check`
- seeds/dict: `\[seeds\]|\[dict\]|seed_corpus|seeds mined|round-trip`
- fuzzing: `Starting libFuzzer|-fork|-jobs|-timeout|exec/s|cov:|Total execs|probe |deep phase|keep_going|no-op|noop|\[(DEAD|WEAK|HEALTHY)\]|SATURATED|run health|\[hang\]|NON-CRASH|timeout=|every child times out|slow \(`
- crashes/triage: `NEW CRASH|crash_dedup|signature|\[sidecar\]|run-pov|exit=|\[triage\]|Round [0-9]/|FINAL:|Triage confidence|\[arbiter\]|KB neighbours|\[triage_calibration\]|ANALYSIS LEAD|LIKELY VULNERABILITY|BUG_IN_DEPENDENCY|held|overruled`
- static/LLM leads: `\[scan\]|scanner|static_findings|\[confirm\]|confirm_static|LLM analysis findings|\[lead\]`
- patching: `patch|\[patch-turn\]|llm-call|root_cause|RCA|read_source|apply_edit|apply-patch-build|rebuild|run-test|candidate patch|Validated patches|Unvalidated patches|scope|diff_mismatch|skip_cause|persist\]|SIDECAR|ContainerPatchBuilder`
- submission/results: `\[submit|submission_counts|tier|FINAL RESULTS|Confirmed crashes:|Overall:|stopped|report written`
- failures: `\] failed:|Traceback|Error code: 4|Error code: 5|HTTP/1.1 4|HTTP/1.1 5|Client Error|timed out`
Build the TIMELINE from timestamps: first and last line of each phase above, minutes spent, outcome.
Build the PATCH CHAIN explicitly: for discver to produce a validated patch on any target, which conditions had to hold
in order (a crash found → reproduced by the oracle → triaged VALID → patch loop entered → RCA context built → patch
candidate produced → applied + rebuilt → PoV no longer crashes → tests pass → persisted/submitted). For each link say
`held` / `broke` with the verbatim log line that proves it and the file:line of the code that decided it. Stop at the
first broken link but still report the state of every later link (e.g. "patch loop never entered — no log line").

## Study first — the repo (root=repo; grep, then read the matching functions)
- The orchestrator (`src/main.py` and whatever it dispatches to): list every phase in CALL ORDER with file:line. For
  each phase: the module/function, whether it is wrapped (`[<module>] failed:` pattern), the env var or flag that gates
  it (and its default), and whether it RAN in this run (cite the log line number) — `ran` / `skipped (<why>)` / `no trace`.
- Language adapters (`src/lang/*.py`, registry, detection): for each of C, C++, Java, Go, Rust, Python, fill a capability
  table — detect, build-from-path, harness discovery, entry ranking, boundary discovery, harness generation, compile/link
  of generated harnesses, reach probe, crash/sanitizer parsing, patch context, static scanners, seeds — each
  `supported (file:line)` / `partial (what is missing)` / `missing`.
- Patch pipeline: the trigger (what a finding must have to enter the loop), inputs (RCA context: which sources, trace,
  diff), prompt builder, reply parser, attempts/iterations/time budget, every skip cause string, the oracle calls
  (apply-patch-build / run-pov / run-test — which container, which command, where the response is read), what makes a
  patch `validated`, scope check, persistence of candidates, submission tier for patches.
- Triage/submission: score computation, calibration (rounds vote, support), KB grounding, class gate, arbiter, tiers
  and thresholds, the no-PoV path.
- Fuzzing: the function that builds the fuzzer argv per language and the exact flags; probe/deep/health thresholds;
  hang guard; seeds/dict wiring; exec-rate sampling.
- Flags: `grep -rn "os.environ" src/` → table (name | default | file:line | what changes). Mark every flag whose
  DEFAULT disables a capability or requires the operator to set it to get the intended behaviour.
- Dead/duplicate: modules under `src/` that nothing on the orchestrator path imports or calls; second implementations of
  a stage. `grep -rn "^import\|^from" src/main.py` against `ls src/`.
- Campaign state: `cat ../CAMPAIGN.md | grep -nE "^## run|: (PASS|FAIL|NO_EDITS|MAPPED)"` — the verdict of every task
  from 50 upward, and for each FAIL/NO_EDITS whether its module exists in the tree anyway.
- Tests: `python -m py_compile src/*.py && python -m pytest -q tests 2>&1 | tail -15` (baseline; fix nothing).

## Deliverable (finish summary — this exact structure)
```
CODE MAP (current tree, commit/mtime of src/main.py)
pipeline:        n. <phase> | <function file:line> | wrapped? | gated by <flag=default> | in this run: ran/skipped/no trace (log L<n>)
language matrix: table language × capability → supported (file:line) / partial (<missing>) / missing
patch pipeline:  trigger → RCA → prompt → parse → apply-patch-build → run-pov → run-test → validated → persist → submit, each file:line;
                 skip causes (verbatim strings); attempt/iteration/time limits; what "validated" requires
triage/submit:   score file:line; calibration; KB grounding; class gate; arbiter; tiers + thresholds; no-PoV path
fuzzing:         argv builder file:line + flags per language; probe/deep/health thresholds; hang guard; seeds/dict; exec-rate
flags:           name | default | file:line | effect   (flag whose default disables something marked **)
dead/duplicate:  <modules never reached from the orchestrator path>
TESTS
baseline:        <last pytest lines>; CAMPAIGN.md verdicts 50–69 with FAIL/NO_EDITS explained
THE RUN (results/latest — name the target)
config:          target, mode (full/delta), language detected, budget, model(s), KB line, env vars seen, start/end, wall time
timeline:        phase | start | end | min | outcome
harnesses:       provided: name, how built, probe result | boundary: `[boundary] …` verbatim or "no line" | generated: name, compiled?, gate, reach, verdict
                 per harness: execs, exec/s, cov a→b, corpus a→b, health verdict, hang summary
findings:        per crash: signature, dedup group, sanitizer class, sidecar run-pov cmd+exit+stderr tail, triage rounds → FINAL + confidence,
                 KB neighbours, arbiter, submitted? tier; per static/LLM lead the same; hangs; FINAL RESULTS block verbatim
patching:        PATCH CHAIN link by link (held/broke, log L<n> verbatim, code file:line); patch_index.json entries
                 (id, status, attempts, test_calls, skip_cause, reason); patches/ contents; oracle calls + exits; [patch-turn]/llm-call counts;
                 what the patch agent received (RCA present? sources? trace?) and produced (candidate diff summary or "none")
submissions:     each with tier + confidence; which you judge false positives and why
failures:        every `[<module>] failed:` verbatim + module file:line; tracebacks; HTTP error classes with counts
budget:          top 5 wall-time consumers with minutes
GAPS (ranked: impact on this run's outcome, then generality across repos)
n. <gap> — evidence: L<n> "<verbatim>" — code: <file:line> — general / specific to this target's shape (say which shape: language, build system, input boundary) — fix shape in one line
OPEN QUESTIONS
<what could not be determined and which file or log would answer it>
```
