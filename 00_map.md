# Task 00 — map the code and the evidence (read-only)

## Map
This task makes NO edits. Its finish() summary becomes MAP.md and is given to every later task, so make it precise:
file:line for everything, nothing from memory.

## Study first
The results root holds two runs: `latest/` (the most recent real run — the one that executed the CURRENT src/, so its
log is the ground truth for what the current code does, including any `[seeds] failed:` / `[dict] failed:` /
`[harness_reach] failed:` lines from wrapped modules) and `prev/` (the run before it). Grep, then read only the matching lines:
- latest: `grep -nE "fuzz thresholds|probe |deep phase|keep_going|no-op|noop|saturated, not no-op|seeds|\[seeds\]|\[dict\]|\[harness_reach\]|failed:|NEW CRASH|stopped|FINAL RESULTS|run health|\[(DEAD|WEAK|HEALTHY)\]|SATURATED|patch-turn|llm-call|BUG_IN_DEPENDENCY|diff_mismatch|build oracle path|SIDECAR"` over `latest/orchestrator.log`;
  read `latest/health.json` and `latest/patches/patch_index.json` in full; `latest/report_run_*.md` lines 1–200.
  Quote every `[<module>] failed: <error>` line verbatim — each one is a shipped change that died at runtime.
- prev: the same grep over `prev/orchestrator.log`; `prev/patches/patch_index.json` in full.
Repo (root=repo), grep then read the matching functions:
- fuzzer launch: `jazzer|-fork|max_total_time|artifact_prefix|keep_going|ignore_crashes|run_fuzzer|runs/|subprocess|Popen`
- probe / deep phase, if present: `probe|deep|keep_going|DEDUP_TOKEN|saturat`
- no-op detector: `noop|no-op|flat|uninstrumented`
- seeds: `seed|corpus|pb_seeds|DISCVER_WIRE_SEEDS|dict`
- stats / health: `exec/s|execs|cov:|HEALTHY|WEAK|DEAD|SATURATED|peak coverage|liveness`
- harness generation: `GenFuzzer|SinkFuzzer|generate_harness|bug_swarm|fuzzerTestOneInput|javac`
- patch loop: `json\.loads|parse|apply_edit|test_patch|read_source|max_iter|attempts|root_cause|static-|patch-turn|llm-call`
- oracle / sidecar (do-not-touch zone): `_get_crash_signature|apply-patch-build|run-pov|run-test|SIDECAR|ContainerPatchBuilder`
- the LLM client used by the patch loop and by seed generation (shared or separate?)
Then `run`: `python -m py_compile src/*.py && python -m pytest -q tests 2>&1 | tail -15` (baseline; do not fix anything).

## Deliverable (finish summary — this exact structure)
```
CODE MAP
fuzzer launch:      <file:line of the function that builds the fuzzer command; in src/ or an external script? exact flags it passes>
probe / deep phase: <present? file:line; is the deep phase reachable from the normal run path? if not, what stops it?>
no-op detector:     <file:line; the exact condition; does it check crashes? the startup line it prints>
stats parsing:      <file:line that parses exec/cov lines; where exec rate is computed>
health checker:     <file:line that decides DEAD/WEAK/HEALTHY and emits the SATURATED verdict>
seeds:              <file:line generating LLM seeds; file:line writing them to a corpus dir; which dir the runner reads; how the harness consumes input (FuzzedDataProvider calls?)>
harness generation: <file:line of the generator; how a generated harness is compiled and launched; where its launch command is built (src/ or external)>
harness selection:  <file:line choosing primary vs background harnesses; budget split>
patch loop:         <file:line of the agent loop; prompt builder; reply parser (the json.loads/regex line); what happens on parse failure; attempts counter; max iterations; static-* handling; what fills root_cause>
LLM client:         <file:line; shared between patch loop and seeds? timeouts/max_tokens>
sidecar/oracle:     <file:line of ContainerPatchBuilder/sidecar calls and crash_dedup._get_crash_signature>  (DO NOT TOUCH)
dead/duplicate paths: <any second implementation of a stage that is NOT what the orchestrator calls>
TESTS
baseline:           <exact last lines of pytest: N passed / N failed, names of pre-existing failures>
test style:         <pytest? fixtures? how existing tests fake the LLM/fuzzer, with file:line to copy>
EVIDENCE
latest:             <one line per harness: execs, cov a→b, corpus a→b, verdict; the 3–5 log lines around any stop; every `[<module>] failed:` line verbatim; which of the expected new log lines (fuzz thresholds … AND no observed crashes / seeds mined= / dict / [lead] / [replay] / harnesses: provided=) are present and which are absent>
prev:               <one line per harness; one line per patch_index entry: crash_id, status, attempts, test_calls, skip_cause, reason verbatim>
<[patch-turn] / [llm-call] lines present? yes/no with counts, both runs>
```
