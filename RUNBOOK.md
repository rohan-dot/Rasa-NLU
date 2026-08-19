# FCG Extraction — RUNBOOK

Operational guide. The pipeline runs against any OpenAI-compatible endpoint:
Gemma-4-31B on vLLM (the target runtime), or Opus/Sonnet via LiteLLM (unchanged
code, different env vars).

## 0. Repo layout

```
fcg-checklist-pipeline/
├── AGENT.md                     builder/runtime spec — read first
├── RUNBOOK.md                   this file
├── fcg_extract.py               the extractor (single dispatch point)
├── validate_extract.py          verbatim-provenance validator
├── checklist_enriched.json      91 typed items with routes (28 extract / 63 workflow)
├── skills/
│   └── fcg-extraction.md        extraction specialist skill
└── data/
    ├── fcg/<COUNTRY>.json        one FCG file per country (stem = alpha-3 ideal)
    ├── svc_rmk.txt              SVC_RMK, tab-separated, cp1252
    ├── meis1.json meis2.json meis3.json    MEIS airfield arrays
    ├── airports.csv             OurAirports (ICAO resolution)
    └── countries_codes_and_coordinates.csv  name -> alpha2/alpha3
    (created at runtime: data/work/checkpoint.json, data/work/dossiers/*.txt)
```

## 1. Serve Gemma-4-31B on vLLM

Sized for the model's memory. On 80GB cards, one GPU is plenty; on 24GB cards the
unquantized 31B does NOT fit one card — use a quantized (AWQ/GPTQ int4) checkpoint
on one card, or tensor-parallel across enough cards (TP count MUST equal the
number of GPUs in CUDA_VISIBLE_DEVICES).

```
# single big GPU, or quantized on a small one:
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model <GEMMA4_PATH> --port 8001 --max-model-len 32768

# multi-GPU for unquantized 31B on smaller cards (e.g. 4x):
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model <GEMMA4_PATH> --port 8001 --max-model-len 32768 --tensor-parallel-size 4
```

Confirm and grab the exact model id:
```
curl http://localhost:8001/v1/models
```

Guided decoding (the reason a weak model produces valid JSON here) is native to
vLLM — no extra setup.

## 2. Preflight (always, before a full run)

```
LITELLM_BASE_URL=http://localhost:8001/v1 LITELLM_API_KEY=dummy AGENT_MODEL=<id-from-curl> python fcg_extract.py --verify --checklist checklist_enriched.json --fcg-dir data/fcg
```
Prints `[verify] OK ...` or the exact failure (bad URL/key/model/TLS). One cheap
call — catches the mistakes that otherwise waste a whole run.

## 3. Full extraction (single line)

```
LITELLM_BASE_URL=http://localhost:8001/v1 LITELLM_API_KEY=dummy AGENT_MODEL=<id-from-curl> FCG_MAX_TOKENS=6000 FCG_CONCURRENCY=2 python fcg_extract.py --checklist checklist_enriched.json --fcg-dir data/fcg --svc data/svc_rmk.txt --meis data/meis1.json data/meis2.json data/meis3.json --airports data/airports.csv --codes data/countries_codes_and_coordinates.csv --out fcg_extract_gemma.csv
```

Against the LiteLLM gateway instead (Opus/Sonnet), swap the first three env vars:
```
LITELLM_BASE_URL=https://llai-proxy.llan.ll.mit.edu/v1 LITELLM_API_KEY=sk-... AGENT_MODEL=claude-opus-4-8 AGENT_TLS_VERIFY=false ... (rest identical)
```

## 4. Validate (the quality metric)

```
python validate_extract.py --csv fcg_extract_gemma.csv --dossiers data/work/dossiers
```
Reports per-country fill (split FCG vs ANCILLARY), verbatim-verified %, fields NA
in every country (route-vocabulary gaps), and dossier truncation warnings. This
output is the artifact to show stakeholders: fill rate + verbatim % is a
defensible quality number.

## 4b. Gold standard: grade Gemma against Claude

The strong model (Opus/Sonnet) reads the same routed dossiers and, being far more
capable, understands that FCG wording like "LEAD-TIME AND VALIDITY" answers the
"diplomatic clearance lead time" item without being told. Its output is a
reference to grade the weak runtime model against.

```
# 1. Gold set (strong model via gateway). Fresh checkpoint so it's purely this model:
rm -f data/work/checkpoint.json
LITELLM_BASE_URL=https://llai-proxy.llan.ll.mit.edu/v1 LITELLM_API_KEY=sk-... AGENT_MODEL=claude-opus-4-8 AGENT_TLS_VERIFY=false python fcg_extract.py --checklist checklist_enriched.json --fcg-dir data/fcg --svc data/svc_rmk.txt --meis data/meis1.json data/meis2.json data/meis3.json --airports data/airports.csv --codes data/countries_codes_and_coordinates.csv --out gold.csv

# 2. Candidate set (Gemma via vLLM). Fresh checkpoint again:
rm -f data/work/checkpoint.json
LITELLM_BASE_URL=http://localhost:8001/v1 LITELLM_API_KEY=dummy AGENT_MODEL=<gemma-id> python fcg_extract.py --checklist checklist_enriched.json --fcg-dir data/fcg --svc data/svc_rmk.txt --meis data/meis1.json data/meis2.json data/meis3.json --airports data/airports.csv --codes data/countries_codes_and_coordinates.csv --out gemma.csv

# 3. Grade:
python compare_extracts.py --gold gold.csv --candidate gemma.csv --out diffs.csv
```

Output buckets every extractable field: agree / DISAGREE (both answered
differently — review) / MISSED (gold found it, Gemma didn't) / EXTRA (Gemma only).
It prints overall agreement %, recall on gold's answers, the weakest fields
(route/prompt candidates), and the actual differing text so you can judge who's
right. `--out diffs.csv` saves every difference for offline review.

IMPORTANT: the gold set is a strong *reference*, not ground truth — Claude can
miss or over-quote too. Use MISSED to find weak-model gaps, but read DISAGREE and
EXTRA both ways: sometimes the weak model (or the ancillary source) is right and
gold missed it. Spot-check a sample by hand before trusting the score wholesale.

## 5. Tuning knobs (env vars)

| var | default | when to change |
|-----|---------|----------------|
| FCG_GUIDED | 1 | set 0 only if the gateway rejects guided_json |
| FCG_MAX_TOKENS | 6000 | raise if answers truncate; if a reasoning model burns tokens (finish_reason=length), raise to 12000 |
| FCG_CONCURRENCY | 2 | raise for throughput if the server has headroom |
| FCG_BUDGET_FCG | 60000 | lower if full-doc fallback dossiers truncate context |
| FCG_BUDGET_MEIS / _SVC | 20000 / 10000 | lower to fit a smaller context window |
| BATCH_SIZE (in code) | 20 | drop to 10 if a weak model still struggles with big field sets |

## 6. Common failures (all seen in practice)

- **429 "budget exceeded"** — LiteLLM cap. Note it's often USER-level, pooling
  spend across all your keys; a fresh key won't help. Raise the budget on the
  user in the dashboard. The pipeline aborts immediately (won't spin on retries).
- **"Unterminated string" / bad JSON** — should not happen with FCG_GUIDED=1. If
  it does, the gateway isn't honoring guided_json; check it passes extra_body
  through, or drop BATCH_SIZE.
- **Everything NA in Phase 1** — check a `data/work/dossiers/<C>_phase1.txt`
  audit file. Tiny/empty routed sections → routes miss the real keys (fix in
  checklist_enriched.json). Healthy dossier but NA → the data genuinely isn't in
  the FCG; it'll fill in Phase 2.
- **CUDA OOM at vLLM start** — the model doesn't fit the GPU(s). Quantize or add
  cards. 24GB cards cannot hold an unquantized 31B on one device.
- **"World size (N) > available GPUs (M)"** — TP count exceeds visible GPUs. Make
  CUDA_VISIBLE_DEVICES list exactly N cards.

## 7. Checkpoint

`data/work/checkpoint.json` resumes completed batches (keyed on country + phase +
batch + fields-hash + dossier-hash). It does NOT key on model. So: to compare
models on identical inputs, `rm data/work/checkpoint.json` between runs; to simply
finish an interrupted run, keep it.
