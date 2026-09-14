# discver — Claude Code operating manual (kumar-dev)

You are working on **discver**, an in-house Cyber Reasoning System (AIxCC-style, Java targets: Jazzer/libFuzzer fuzzing + LLM analysis + automated patching), running under OSS-CRS in containers on this machine. Unlike the previous coding setup (an edit-only box that could never run discver), **you can run discver, read its logs, and rebuild targets here.** That execution access is the whole reason this effort can now converge. Use it on every change.

## The one rule
No change is "done" until you have run an oracle yourself and pasted its output into RUNLOG.md: unit tests → `patch_replay` (minutes) → full run (long). A change you have not executed is a hypothesis, not a fix. Never report a hypothesis as a fix.

## Ground truth (as of 2026-09-14)

Works, verified on real runs:
- Crash discovery + reproduction: PoV replay with ASAN traces. `crash_dedup._get_crash_signature` is the oracle primitive — do not modify it.
- Dedup to `unique_bugs/summary.json`; stable finding IDs (`src/finding_id.py`).
- Run health checker: correctly diagnoses "harness SATURATED by a shallow crash" and "peak coverage 0".
- Builder sidecar detection: `build oracle path: SIDECAR (libCRS at /usr/local/bin/libCRS)`; libCRS exposes `apply-patch-build`, `run-pov`, `run-test`.
- Seed wiring (`DISCVER_WIRE_SEEDS` fix): when LLM seeds exist they reach the runners and coverage jumps (~3400 on GenFuzzer0/SinkFuzzer0 in earlier runs).
- Confirmed target crash: index-out-of-bounds in `mil.disa.common.util.agentmanager.X509Utils.getCnFromDn` (X509Utils.java:81). Intermittent second crash: null-pointer in `AgentUtils.createParentDirectories` (AgentUtils.java:74).

Does NOT work:
- **Patch generation has never reached `test_patch` in 8 runs** (history in RUNLOG.md). Run 8: `failed=2`, reasons not yet read. Run 7: `agent returned no reply (LLM timed out after retries)`, attempts=1, test_calls=0.
- **Run 8 regression: `LLM-generated seeds: 0`** → corpus 0→0 → X509Fuzzer, GenFuzzer0, SinkFuzzer1 reported DEAD with peak coverage 0. Run 7 (same day, earlier) had seeds and cov 2043 on X509Fuzzer. Something between run 7 and run 8 broke the seed path or the runtime LLM calls behind it. If run 8 carried the "Task B" patch-loop client change ([llm-call] logging, thinking off, adaptive retry), check first whether it leaked into the shared LLM client used by `src/pb_seeds.py`.
- Primary harness X509Fuzzer is saturated by the shallow getCnFromDn crash (~475 execs per run): deeper coverage is unreachable until that bug is patched or the corpus is seeded past it.
- `static-*` findings (no PoV) burn 15 patch iterations and can never be validated by PoV replay.
- Runtime LLM (GLM-5 on vLLM, offline) often returns non-JSON ("LLM returned invalid JSON — using raw report") and is slow on large prompts.

## Where things are (verify with `ls` before relying on any of this)
- Code: `src/` and `tests/` in the discver project dir (was `example/discver/`). Orchestrator `src/main.py` (phases 0–3, `ContainerPatchBuilder`); patcher `src/patch_generator.py`; seeds `src/pb_seeds.py`; dedup `src/crash_dedup.py`; scanner swarm `src/bug_swarm.py`; telemetry `src/reliability_metrics.py`.
- Full run: `uv run oss-crs run --compose-file example/discver/compose.yaml --fuzz-proj-path example/discver/test-targets/java-ant-byo --target-harness auto`
- Run output: `.oss-crs-workdir/crs_compose/<hash>/address/runs/<run-id>/crs/discver/java-ant-byo_<hash>/LOG_DIR/auto/discver/` → `orchestrator.log`, `patches/patch_index.json`, `health.json`, `crashes/<harness>/`, `unique_bugs/`, `report_run_*.md`; `REBUILD_OUT_DIR/` alongside LOG_DIR.
- Reference repos (read-only mechanism donors): `reference/atlantis` (Team Atlanta AIxCC CRS) and `reference/buttercup` (Trail of Bits AIxCC CRS) — fix these paths to wherever they were cloned.
- Runtime LLM endpoint: find it in the compose file / env and record it in RUNLOG.md.

## Triage greps (run these before forming any hypothesis)
```
cat patches/patch_index.json
grep -nE "llm-call|timed out|timeout|patch-turn" orchestrator.log
grep -niE "seed" orchestrator.log | head -60
grep -nE "build oracle path|build_ok|SIDECAR" orchestrator.log
grep -cE "invalid JSON|raw report" orchestrator.log
```

## Invariants (the reviewer blocks any diff that violates these)
1. No feature flags, no env switches. Ashok runs discver with zero configuration; everything auto-detects and logs which path it chose at startup.
2. Fail loud with a specific cause. No silent skip or degrade, no bare `except: continue`, no `if action is None: continue` without a log line carrying the raw reply.
3. `validated` means exactly: patch applied → rebuilt → the specific PoV re-run and no longer crashes → existing tests pass. Anything else is emitted as UNVALIDATED with a prominent label. Never report `tests_pass=True` when no test command ran; report NOT EXERCISED.
4. Never modify real target source; patches go to `output/patches/<finding-id>/` (patched files + `.orig` + `patch.diff` + `PATCH_NOTES.md`).
5. Do not modify `crash_dedup._get_crash_signature` or the sidecar builder wiring without a replay proving the replacement.
6. One behavioural change per cycle, plus its test, plus its log line. Observability-only additions may be bundled.
7. Design for a weak tool-caller. The runtime model is GLM-5, not a frontier model: small prompts (source slices around the sink), a strict JSON action schema with an example, validate→repair on parse failure, thinking off, hard caps on max_tokens, explicit timeouts logged at start, and never an identical retry after a timeout.

## The loop (every cycle)
1. `/triage <run-dir>` → log-analyst report with exact lines, appended to RUNLOG.md.
2. State one hypothesis and the single change that tests it.
3. `patch-engineer` or `fuzz-engineer` implements it with a test.
4. `reviewer` checks the diff against the invariants.
5. Prove it: `python -m py_compile src/*.py` → full test suite → `patch_replay` on a real run dir (minutes). Only a passing replay earns a full run.
6. Commit on branch `claude/discver-fix` with a message naming the run and hypothesis. Append the result to RUNLOG.md, including the oracle output.

## Build the fast oracle first
If `src/patch_replay.py` does not exist, building it is cycle 1. `python -m src.patch_replay --run-dir <LOG_DIR/auto/discver>` re-runs only patch generation on an existing run's crash + PoV, inside the discver container, with the real sidecar and the real runtime LLM, writing to `<run-dir>/patch_replay_<timestamp>/` (patches/, patch_index.json, orchestrator-style log). Reuse the Phase 3 functions; do not copy-paste them. Add `--scenario <file>` to drive the loop from a scripted list of LLM replies (in-process fake client) so the pipeline can be proven independent of GLM-5: if a scripted `read_source → apply_edit(bounds check) → test_patch` sequence validates the X509Utils:81 patch against the real sidecar, the pipeline works and only the model remains. Work out the invocation (`docker compose run`, or exec into the running container) and record the exact command in RUNLOG.md.

## Definition of done
1. A `validated` patch for X509Utils.java:81 emitted by a full run, with `patch.diff`, PoV replay no longer crashing, tests exercised.
2. `LLM-generated seeds > 0` and X509Fuzzer no longer DEAD (patched-and-refuzzed, or seeded past the shallow crash).
3. `static-*` findings take a build+tests-only path and are emitted UNVALIDATED with a "no reproducer" label within 2 iterations, not 15.
4. RUNLOG.md has one entry per run/replay with the oracle output pasted in.

## Handing changes back
The code owner (Rohan, edit-only box) merges from here. Every cycle is a commit on `claude/discver-fix`; at the end of a session run `git format-patch <base>..HEAD -o handback/` (or push to the agreed remote) and list the patch files in RUNLOG.md.

## Subagents in this repo
- `log-analyst` — read-only triage of a run dir; fixed report format.
- `patch-engineer` — owns the patch loop, builder wiring, and patch_replay; proves changes with tests + replay.
- `fuzz-engineer` — owns seeds, harness selection, health checker; first job is the seeds=0 regression.
- `reference-reader` — read-only over Atlantis/Buttercup; answers mechanism questions with file:line.
- `reviewer` — gates every diff against the invariants before a run is spent.

Use them through `/triage`, `/cycle`, `/reference`, `/full-run`. Keep the main conversation for orchestration and decisions.
