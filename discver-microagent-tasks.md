# discver patch loop — three microagent tasks (paste at `task>`)

Order: **B first** (ship to Ashok tonight), **A** while his run is going, **C** next. One commit per task.
Setup as usual: `LITELLM_*` exports → `python microagent.py --repo $(pwd) --verify` → `AGENT_MAX_TURNS=500 python microagent.py --repo $(pwd)`.

## What to ask Ashok for after the Task B run

- `grep -nE "llm-call|timed out|timeout|patch-turn" orchestrator.log`
- `cat patches/patch_index.json`
- Only if it still times out: vLLM server log lines containing `Aborted` or `timeout` from the same window, and whether swarm/coverage_seeds requests were in flight at the same time.

---

## TASK B — patch-loop LLM call path: observability, no-thinking, adaptive retry

Context: last run's patch_index.json says `agent returned no reply (LLM timed out after retries)`, attempts=1, test_calls=0. The runtime LLM is GLM-5 on vLLM. We need every patch-loop LLM call to be visible in orchestrator.log, cheap enough to finish, and never retried identically.

Survey first (read-only). Report as a numbered list BEFORE editing anything:
1. Every LLM call made from src/patch_generator.py (RCA, per-turn action, repair, judge). For each: client used, timeout value and where it is set, max_tokens, whether response_format / guided_json / extra_body is sent, any thinking/reasoning setting, and the retry wrapper (count, backoff, whether the identical request is resent).
2. The same settings for the analyzer/swarm calls, so the delta is visible.

Then implement, no env flags, no feature flags:
A. One INFO line before and one after every patch-loop LLM call:
   `[llm-call] phase=patch stage=<rca|turn|repair|judge> turn=<n> attempt=<k> model=<m> timeout_s=<t> max_tokens=<mt> prompt_chars=<c> guided=<bool> thinking=<bool>`
   `[llm-call] ... elapsed_s=<e> outcome=<ok|timeout|http_<code>|parse_error|empty> reply_chars=<r>`
   Emit the same fields to the discver.reliability logger.
B. Thinking off for all patch-loop calls: send `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` (vLLM convention for GLM/Qwen; harmless if the template ignores it). If a reply carries `reasoning_content`, ignore it and use `content`.
C. No guided JSON in the patch loop: remove response_format / guided_json from patch-loop calls if present and keep the existing validate→repair path. If not present, say so in the report and skip.
D. Timeouts as explicit module-level constants, logged once at patcher start: first RCA/turn call >= 600s, action turns >= 300s, repair >= 120s.
E. Adaptive retry on timeout — never resend an identical request:
   - attempt 1: full request.
   - attempt 2: brief trimmed (inlined source capped to ~120 lines around the sink; tool docs reduced to name + one line), max_tokens halved.
   - attempt 3: minimal (crash summary + PoV path + scratch path + tool names), max_tokens <= 800.
   - after 3: fail loud. Reason string, also written to patch_index.json: `LLM timed out 3x (timeout_s=600/600/600 elapsed_s=<e1>/<e2>/<e3> prompt_chars=<c1>/<c2>/<c3>)`.
F. max_tokens caps: RCA <= 1500, action turns <= 800, repair <= 400.
G. At patcher start, log how many analyzer/swarm LLM requests are still in flight, or log that this cannot be determined. If phase-3 analysis and patch generation can overlap on the same endpoint, make patch generation wait for analysis to finish and log the wait.

Do not touch: crash_dedup._get_crash_signature, the sidecar builder path from 8ec0471.

Tests: trimming (attempt-2 and attempt-3 prompts strictly smaller than attempt 1); retry policy (fake client that times out twice then answers is called with three different prompts); reason-string format.
Finish: `python -m py_compile src/*.py`, full suite, CHANGELOG entry, commit `patch loop: llm-call observability, no-thinking, adaptive timeout retry`.

---

## TASK A — local end-to-end test of the patch loop (fake LLM + fake libCRS)

Goal: run the REAL src/patch_generator.py end to end on this box — no container, no network — in under 90 seconds, for four scenarios. Every later patch-loop fix gets tested against this before it goes to Ashok.

Survey first (read-only). Report BEFORE writing fakes:
1. The exact JSON action format the loop parses (tool names, required fields, how test_patch is invoked).
2. How the LLM client is configured (base URL / model / key: env names, defaults).
3. How ContainerPatchBuilder discovers and invokes the libCRS sidecar: binary path, subcommands (apply-patch-build / run-pov / run-test), argument order, and what exit codes / stdout it expects from each.
The fakes must match this call surface exactly.

Build:
1. `tests/fakes/fake_openai_server.py` — stdlib-only HTTP server: POST /v1/chat/completions (+ GET /v1/models; minimal SSE only if the patcher streams). Driven by a scenario JSON: ordered steps `{delay_s, content, finish_reason}`, consumed in order, default step after the end. `delay_s` larger than the client timeout simulates a hang. Records every request (messages, max_tokens, response_format, extra_body, timestamp) to a JSONL so tests can assert on what the patcher sent. Started/stopped by a pytest fixture on a free port.
2. `tests/fakes/fake_libcrs` — executable with the same CLI surface as the real sidecar. Behaviour via marker files in a scratch dir: `FAIL_BUILD` → apply-patch-build exits non-zero with a plausible javac error; run-pov reports the crash unless the patched target file contains the sentinel line `// DISCVER_FIX_APPLIED`; `FAIL_TESTS` → run-test fails. Records each invocation to a JSONL.
3. A tiny Java-shaped fixture target: source tree with X509Utils.java containing getCnFromDn and the real index-out-of-bounds pattern, a PoV file, and a crash record in the same shape as unique_bugs/summary.json + crashes/<harness>/ from a real run (reuse real shapes if any exist in the repo; otherwise synthesize).
4. `tests/test_patch_loop_e2e.py` — runs the real patcher with the LLM base URL pointed at the fake server and sidecar discovery monkeypatched to the fake binary (monkeypatch the discovery function; no env flags). Monkeypatch the Task B timeout constants down to a few seconds. Scenarios:
   - `happy`: read_source → apply_edit inserting a bounds check plus the sentinel → test_patch. Assert: patch_index.json status validated, emitted_dir contains patch.diff and PATCH_NOTES.md, `[patch-turn]` lines logged, run-pov and run-test each invoked exactly once.
   - `llm_timeout`: first step `delay_s` > client timeout. Assert: total wall time < 3× timeout, second recorded request strictly smaller than the first, final reason contains "timed out" plus the elapsed/timeout numbers, test_calls reported honestly.
   - `garbage`: first step content is not JSON. Assert: a repair request is sent, the loop proceeds, nothing is silently dropped (attempts and test_calls consistent with the log).
   - `build_fail`: `FAIL_BUILD` present. Assert: fail-fast with `build_ok=False` in the reason within 2 turns, not 15.

Acceptance: `pytest tests/test_patch_loop_e2e.py` green in < 90s and added to the full suite. py_compile, CHANGELOG, commit `tests: end-to-end patch loop against fake LLM + fake libCRS`.

---

## TASK C — patch_replay entrypoint for the container

Goal: let the container operator re-run ONLY patch generation on an existing run's artifacts, in minutes, with the real sidecar and the real runtime LLM — instead of a full fuzz run per debug cycle.

Implement `src/patch_replay.py`:
- `python -m src.patch_replay --run-dir <LOG_DIR/auto/discver>` loads unique_bugs/summary.json + crashes/<harness>/ + PoV inputs from the run dir, rebuilds the patcher inputs exactly as Phase 3 does (call the same functions — no copy-paste), and runs patch generation, writing to `<run-dir>/patch_replay_<timestamp>/` (patches/, patch_index.json, an orchestrator-style log). Auto-detects the sidecar the same way main.py does and logs which path it chose.
- `--scenario <file>`: drive the loop from a Task A scenario file via an in-process fake client instead of the real LLM. With the happy-path scenario this proves the oracle end to end in the container, independent of GLM-5. (CLI args on a separate tool are fine; the main discver run still needs zero configuration.)
- Document both commands in AGENT.md in one paragraph, including the Ashok grep list above.

Tests: from a fixture run dir, the replay reconstructs the same patcher inputs as Phase 3 (assert equality). py_compile, full suite, CHANGELOG, commit `patch_replay: re-run patch generation on an existing run dir`.
