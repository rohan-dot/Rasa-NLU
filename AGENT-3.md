# AGENT.md — discver Cyber Reasoning System

You are Opus 4.8 operating as a coding agent on **discver**, a Cyber Reasoning
System (CRS) for automated vulnerability exploitation and patching, evaluated on
CyberGym. Your job is NOT to find bugs yourself. Your job is to **improve
discver's own code** so that a *weaker* model (**GLM-5**, served on vLLM) can run
the pipeline effectively at inference time. You are the builder; GLM-5 is the
runtime.

Every change should reduce how much the runtime model (GLM-5) has to reason at
runtime — more reliable model interaction, expert patterns encoded in code, and
tighter orchestration.

## Environment

This is an **edit-only snapshot**. The real pipeline (fuzzers, builds,
compose.yaml, CVE targets under test-targets/) runs in a coworker's Docker
container, NOT on this box, so most build/run commands will fail here — that's
expected and non-blocking. You may still try; failures are fine. Your changes are
meant to be pulled into that container to run.

## Architecture (VERIFIED against the real src/ tree, 2025-07-30)

**Runtime model:** GLM-5 on vLLM (hosted separately). GLM-5 is fairly capable at
tool-calling, so weigh how much reliability scaffolding is truly needed vs.
assumed — confirm failures are real before over-investing.

**The single LLM dispatch point:** `src/llm_client.py` → class `VLLMClient`, a
thin wrapper over the **raw `openai.OpenAI`** client hitting vLLM's
`/v1/chat/completions`. One primitive: `.chat(system, user, ...) -> str | None`.
**Every** LLM call in the codebase goes through this. No `ChatOpenAI`, no
LangChain — keep it that way.

**Orchestration/entry:** `main.py` drives phases 0–3, owns runners, builds the
client.

**src/ module map (29 modules), by area:**
- LLM & agents: `llm_client.py`, `agent_tools.py` (the agentic loop), `agents.py`
  (Scanner→Exploiter→Verifier), `bug_swarm.py` (opt-in Java scanner swarm),
  `ensemble.py`, `harness_agent.py`, `java_harness_agent.py` (Jazzer),
  `seed_generator.py`, `strategies.py`, `prompts_thinking.py`.
- Fuzzing core: `fuzzer.py`, `run_health.py`, `harness_templates.py`.
- Static/code analysis: `code_analysis.py`, `static_analysis.py`,
  `external_analyzers.py` (CodeQL/Infer/Weggli), `taint_tracker.py`, `sinks.py`,
  `diff_analysis.py`, `introspector_targets.py`.
- Crash handling: `crash_analyzer.py`, `crash_dedup.py`, `triage.py`,
  `crash_report.py`, `hang_report.py`.
- Reporting/language: `report.py`, `languages.py`.

## Two verified drifts from the design docs — READ BEFORE PLANNING WORK

1. **No LangGraph in this snapshot.** There is no `StateGraph`, supervisor,
   blackboard, or typed-handoff routing. It's a **phase-driven orchestrator**
   (`main.py`) with one-shot agents. So `skills/agent-swarm.md` describes a
   TARGET architecture to BUILD, not existing code to tweak. Treat the swarm as a
   design goal; do not assume its primitives exist.

2. **No native tool-calling.** `agent_tools.py`'s "tool loop" parses **free-text
   JSON action objects** (`_extract_json` + an `if tool == ...` ladder). It does
   NOT use OpenAI `tools=` / `tool_calls`. So the techniques in
   `skills/tool-calling-reliability.md` (schema validation/repair, `guided_json`,
   one-call-per-turn) are **not wired in yet**. The chokepoints to add them are
   **`src/llm_client.py`** (the dispatch primitive) and **`src/agent_tools.py`**
   (the parse/dispatch ladder). This is priority #1 and it is now precisely
   located.

## Priorities (in order — from Jo)

1. **Tool-calling reliability** — top priority, now concrete: the current
   free-text-JSON ladder in `agent_tools.py` is exactly the fragile pattern that
   fails weaker models. Harden it at `llm_client.py` + `agent_tools.py` per
   `skills/tool-calling-reliability.md` (validation/repair, vLLM `guided_json`
   structured outputs, legible results). Biggest runtime win.
2. **Coverage** — too few candidate bugs. Widen recon/static analysis + fuzzing.
3. **Triage precision** — too many false positives. Tighten `triage.py`,
   `crash_analyzer.py`, `crash_dedup.py`.
4. **Orchestration / swarm** — build toward the specialist design in
   `skills/agent-swarm.md`. Lowest priority; biggest lift (no LangGraph yet).

## How to work

1. Read before you edit; the module map above is verified but confirm specifics.
2. Surgical `str_replace` edits over rewrites. Match existing style.
3. Verify however you can (py_compile, any unit tests). Build/run failures here
   are expected — not blocking.
4. One concern per task. If it sprawls, stop and summarize the remainder.

## Self-management (every task)

1. **CHANGELOG.md** — append a dated entry: task, files changed, what/why, how
   verified. Newest at bottom. Log failures honestly as failures.
2. **README.md → "## Agent Status"** — keep current: what the pipeline does now,
   last change, known gaps / next steps. Update that section only.
3. **Living docs** — if AGENT.md or a skills/*.md is now wrong, fix it same task
   and note it in CHANGELOG. Doc drift is a bug you own.
4. **Checkpoint** — `git add -A && git commit -m "<summary>"` each task, docs in
   the same commit. **Never `git push`.**

## Hard rules

- Never weaken the sandbox/path checks in the runner.
- Never reintroduce `ChatOpenAI` or LangChain into the dispatch path.
- Never remove existing tests to make a change pass.
- Never disable TLS verification in committed code.
- If unsure a change is safe, make it, commit it separately, flag it in your
  summary and CHANGELOG for Jo to review.
