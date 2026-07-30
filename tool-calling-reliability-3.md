# Skill: Tool-Calling Reliability for Weak Models

**Goal:** make the runtime model's tool calls succeed at runtime without the model getting
smarter. Every technique here moves reliability into the *scaffold*, so a weak
model that emits imperfect calls still ends up executing the right action.

When you (Opus) implement any of these in discver, add a focused unit test and
commit it separately.

## Where this lives in discver (verified)

The current mechanism is NOT native tool-calling. `src/agent_tools.py` runs a
"tool loop" that parses **free-text JSON action objects** from the model
(`_extract_json` + an `if tool == ...` ladder). The single dispatch primitive is
`src/llm_client.py` → `VLLMClient.chat(system, user, ...)` over raw `openai.OpenAI`
against vLLM `/v1/chat/completions`.

So the two chokepoints for everything below are:
- **`src/llm_client.py`** — add optional structured/guided decoding here
  (`extra_body={"guided_json": <schema>}`) so the model's action JSON is
  schema-valid by construction. Keep it behind a flag; degrade gracefully.
- **`src/agent_tools.py`** — wrap `_extract_json` and the `if tool == ...` ladder
  with validate→repair (section 1) before dispatch, and make tool results
  legible (section 5). This is the highest-leverage single change.

Start here. Everything below is the how.

## 1. Validate every tool call against its schema, then repair

Do not pass raw model output straight to the tool. Between the model and the
dispatcher, insert a validate→repair step:

- Parse arguments as JSON. On `JSONDecodeError`, attempt one structured repair:
  strip markdown fences, fix trailing commas, extract the first balanced `{...}`.
- Validate against the tool's JSON Schema (use `jsonschema` or `pydantic`).
- On validation failure, **don't crash and don't silently drop the call**.
  Feed the error back as a `tool` message: the exact validation error + the
  schema, and let the model retry *once*. Weak models fix their own args well
  when handed the precise error.
- Cap repair attempts (e.g. 2) so you never loop forever. After the cap, return
  a clear failure the orchestrator can route around.

## 2. Flatten and simplify tool schemas

Weaker models fail more on nested/optional-heavy schemas. For each discver tool:

- Prefer flat objects of primitives over nested objects.
- Avoid deep `oneOf`/`anyOf`; split an overloaded tool into two narrow tools.
- Give every parameter a one-line `description` with a concrete example value.
- Make as few parameters `required` as possible; give the rest sane defaults in
  code, not in the model's head.
- Name tools by verb+object (`read_source_file`, not `file`). Ambiguous names
  cause wrong-tool selection.

## 3. Constrained / structured decoding at the vLLM layer

vLLM supports guided decoding (outlines / lm-format-enforcer / xgrammar). Where
a tool's arguments have a fixed shape, pass the JSON Schema as a `guided_json`
constraint on the completion. This makes malformed JSON *impossible* rather than
merely caught — the strongest lever for a weak model. Wire this through the raw
`openai.OpenAI` call's `extra_body={"guided_json": <schema>}`. Keep it optional
behind a flag so it degrades gracefully if the backend lacks the feature.

## 4. One tool call per turn for the weak runtime

Parallel multi-tool calls multiply the failure surface for a weak model. In the runtime
loop, prefer serial single-call turns: the model picks one tool, you execute,
return the result, it picks the next. Slower, far more reliable. (Opus building
the pipeline can parallelize; the runtime model running it should not.)

## 5. Make tool results legible

A weak model reasons better over clean tool output. Standardize every tool's
return: a status line (`[ok]` / `[error] <reason>`), then the payload, truncated
with an explicit `... (truncated, N more lines)` marker. Never return raw
tracebacks as the whole payload — extract the relevant line.

## 6. Fail loud, route smart

When a tool ultimately can't be satisfied, return a typed failure the
orchestrator understands (e.g. `TOOL_FAILED:read:file_not_found`) so a critic or
supervisor agent can decide the next move, rather than the failure vanishing
into free text the model may ignore.

## Test checklist (add these as you build)

- Malformed JSON args → repaired or cleanly rejected, never crashes.
- Missing required arg → model gets the schema back and retries.
- Wrong-type arg (string for int) → caught by validation.
- Truncation marker present on oversized output.
- `guided_json` path produces schema-valid args under load.
