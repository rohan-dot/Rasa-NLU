# lora_review_agents

Two agents, one runner, no framework.

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `report_agent.py` | the LoRA project directory (code, configs, results, logs, notebooks) + the `.pptx` deck | `report.md` — an evidence-grounded audit |
| 2 | `architecture_agent.py` | `report.md` (+ read-only spot checks of the repo) | `ARCHITECTURE.md` (diagnosis → design → Mermaid diagram → data/LoRA plan → eval → roadmap) and a `scaffold/` code skeleton |
| both | `run_pipeline.py` | | everything above in one `--out-dir` |

Only hard dependency is the raw `openai.OpenAI` client, so it works against **vLLM** (Gemma etc.),
**OpenAI**, or **Anthropic's OpenAI-compatible endpoint** (`--base-url https://api.anthropic.com/v1/`,
key in `LLM_API_KEY`). No LangChain.

## Install

```bash
pip install openai python-pptx          # pypdf and pandas are optional (PDF decks, parquet tables)
```

## Run

```bash
# vLLM on the cluster
export LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=google/gemma-3-27b-it
python run_pipeline.py --root /path/to/lora_project --ppt /path/to/deck.pptx --out-dir agent_outputs

# OpenAI
export LLM_API_KEY=sk-... LLM_MODEL=gpt-4.1
python run_pipeline.py --root /path/to/lora_project --out-dir agent_outputs

# step by step
python report_agent.py --root ... --ppt ... --out-dir agent_outputs
python architecture_agent.py --report agent_outputs/report.md --root ... --out-dir agent_outputs
python run_pipeline.py --root ... --out-dir agent_outputs --skip-report      # re-run only Agent 2

# dry run: inventory only, no LLM calls
python report_agent.py --root ... --inventory-only
```

### LiteLLM proxy (e.g. gpt-5.6 routed through LiteLLM)

```bash
export LLM_BASE_URL=https://litellm.yourlab.edu:4000      # LiteLLM proxy; with or without /v1 both work
export LLM_API_KEY=sk-...                                  # LiteLLM virtual key (or master key)
export LLM_MODEL=gpt-5.6                                   # the model_name from LiteLLM's config
export LLM_CA_BUNDLE=/path/to/lab-ca.pem                   # TLS: lab/corporate CA (see below)
python run_pipeline.py --root /path/to/lora_project --ppt deck.pptx --out-dir agent_outputs
```

Recognised env aliases, first non-empty wins:

| setting | env vars checked (in order) |
|---|---|
| base URL | `LLM_BASE_URL`, `LITELLM_BASE_URL`, `LITELLM_PROXY_URL`, `OPENAI_BASE_URL`, `OPENAI_API_BASE` |
| API key | `LLM_API_KEY`, `LITELLM_API_KEY`, `LITELLM_KEY`, `OPENAI_API_KEY` |
| model | `LLM_MODEL`, `AGENT_MODEL`, `LITELLM_MODEL`, `OPENAI_MODEL` |
| CA bundle | `LLM_CA_BUNDLE`, `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS` |
| skip TLS verify | `LLM_INSECURE=1` (or `LLM_SSL_VERIFY=false`) — last resort, prints a warning |

TLS note: the OpenAI SDK talks HTTP through `httpx`, which on its own only honours `SSL_CERT_FILE`/`SSL_CERT_DIR` —
a CA exported as `REQUESTS_CA_BUNDLE` or `CURL_CA_BUNDLE` would be ignored. This package reads all of them and
builds the client with that bundle explicitly (`--ca-bundle` / `--insecure` on the CLI do the same).
`HTTPS_PROXY` / `NO_PROXY` are honoured as usual; put the LiteLLM host in `NO_PROXY` if it is internal.

gpt-5.x / o-series models reject `temperature` and want `max_completion_tokens`; the client detects the 400 and
retries without `temperature` / with the renamed parameter. Pass `--temperature -1` to skip that first round trip.
LiteLLM forwards `tools`/`tool_choice`, so native tool calling is used; the json fallback kicks in only if a
route rejects tools.

`--ppt` is only needed when the deck lives outside `--root`; decks inside the project are found automatically.
`--hint "one line of context"` is passed to both agents (e.g. the task, hardware limits, "no external APIs").

## Outputs (`--out-dir`)

```
inventory.md / inventory.json   deterministic pre-pass: every file classified, adapter_config.json,
                                trainer_state.json loss/eval curves, dataset row counts, framework hints
slides_outline.md               the deck: titles, bullets, tables, chart series values, speaker notes
report_notes.md                 Agent 1's notes (every fact cites a file) + coverage summary
report.md                       the audit (10 sections; claim-vs-code table; bottleneck hypotheses)
arch_notes.md                   Agent 2's diagnosis + decision notes
ARCHITECTURE.md                 the design; architecture.mmd is the Mermaid diagram alone
scaffold/                       MANIFEST.json + one generated file per entry
trace_*.jsonl, *_run.json       every tool call, token usage, which tool transport was used
```

## How the agents work

* **Inventory first, then exploration.** `repo_tools.py` classifies files, parses `adapter_config.json`,
  summarizes `trainer_state.json`, counts dataset rows and extracts the deck before any model call, so the
  agent starts with a map instead of `ls`.
* **Tools are read-only and sandboxed** to `--root` (plus the explicit `--ppt` files). Weight files are never
  read. Every result is capped in size; long files are paged with `start_line`/`max_lines`.
* **Notes survive, tool outputs don't.** When the conversation exceeds `--context-chars`, the oldest tool
  outputs are replaced by stubs; `save_note` entries are kept and feed the final synthesis call.
* **Two tool transports.** `--tool-mode auto` (default) uses native function calling and falls back to a
  json text protocol if the server rejects `tools` — so a vLLM server started without
  `--enable-auto-tool-choice --tool-call-parser ...` still works. Force with `--tool-mode json`.
* **Long outputs.** The report / architecture calls keep asking the model to continue when it hits
  `--synthesis-max-tokens`.
* **Step limits.** `--report-max-turns` / `--arch-max-turns` (or `--max-turns` per script) force a summary.

## Knobs for open-weight models

| Situation | Flag |
|---|---|
| 32k-context model | `--context-chars 100000` |
| chat template has no system role (Gemma 2) | `--no-system-role` |
| server rejects `temperature` / `max_tokens` | handled automatically (dropped / renamed) ; `--temperature -1` omits it up front |
| thinking models on vLLM | `--extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'` |
| anything else the API needs | `--extra-body '<json>'` is passed through verbatim |

## Test without a model

```bash
python tests/smoke_test.py        # builds a mock LoRA project (+ a real .pptx) and replays a scripted LLM
python tests/make_fixture.py /tmp/mock_lora && python report_agent.py --root /tmp/mock_lora --inventory-only
```

## Files

```
agent_core.py          LLM client (OpenAI-compatible), tool registry, agent loop, context trimming, tracing
repo_tools.py          sandboxed file tools (list/read/grep/notebook/pptx/pdf/json/table) + inventory pre-pass
report_agent.py        Agent 1
architecture_agent.py  Agent 2
run_pipeline.py        both
tests/                 fixture builder + offline smoke tests
```
