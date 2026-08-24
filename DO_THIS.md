# DO_THIS — Route Fix + Re-run, Start to Finish

Plain walkthrough of the whole loop. Run every command from the repo root
(the folder that has `fcg_extract.py` in it). Fill in the two placeholders once:

- `GATEWAY_URL` = your LiteLLM base URL, must end in `/v1`
  (you've been using `https://llai-proxy.llan.ll.mit.edu/v1`)
- `YOUR_KEY`    = your LiteLLM API key (starts `sk-`)

Everything Claude here is your local Claude on LiteLLM. Claude is the BUILDER:
it fixes the shared routes and sets the quality bar. Gemma is the RUNTIME that
will eventually run everything. Fixing routes helps BOTH models, because routes
are shared plumbing (which slice of the document a model gets), not model code.

---

## What this whole thing is doing (read once)

Your weakest fields scored badly because their **routes hit nothing** — they
fell back to the full 60k-char document, which every model extracts poorly from.
The `FALLBACK` line in `data/work/dossiers/FRA_phase1.txt` lists them.

The fix: point those routes at the REAL FCG key names. That's a data edit in
`checklist_enriched.json`, no code change. It permanently improves the pipeline
for whichever model runs it — most of all Gemma, since weak models benefit most
from being handed the exact right paragraph.

Two separate questions, don't confuse them:
- "Did the route fix improve extraction?" → compare old-Claude vs new-Claude (below).
- "Is Gemma good enough to take over?" → serve Gemma, run the same pipeline,
  compare against the Claude gold set. Can only be done once Gemma is served.

---

## STEP 1 — Dump the real FCG keys to a file

```
python dump_keys.py data/fcg/FRA.json > fcg_keys.txt
```

Check it:
```
cat fcg_keys.txt
```
You should see key PATHS (like `aircraft.airport.airportDetails[].runwayDetailsList[].Width`)
and a flat key list. If `FRA.json` isn't the name in `data/fcg/`, use any country
file there — they share the schema.

---

## STEP 2 — Fix the routes

Two options. Pick ONE.

### Option A — let Claude do it (delegation)
The task file `route_task.md` is already in this repo (see STEP 2 appendix below
if it's missing). Point the agent at Claude and run it:

```
export LITELLM_BASE_URL=GATEWAY_URL
export LITELLM_API_KEY=YOUR_KEY
export AGENT_MODEL=claude-opus-4-8
export AGENT_TLS_VERIFY=false
```
Then run your agent against `route_task.md` (check its flag first):
```
python microagent.py --help
python microagent.py --task route_task.md
```

### Option B — have it done for you (reliable)
Paste the contents of `fcg_keys.txt` into the chat and ask for the rewritten
`checklist_enriched.json`. This avoids any agent-tooling quirks. Recommended if
Option A's agent output looks off.

After either option, confirm the JSON is still valid:
```
python -c "import json; json.load(open('checklist_enriched.json')); print('valid')"
```

---

## STEP 3 — Preflight (one cheap call — always do this before a full run)

```
LITELLM_BASE_URL=GATEWAY_URL LITELLM_API_KEY=YOUR_KEY AGENT_MODEL=claude-opus-4-8 AGENT_TLS_VERIFY=false FCG_GUIDED=0 python fcg_extract.py --verify --checklist checklist_enriched.json --fcg-dir data/fcg
```
Want: `[verify] OK ...`. If it says budget exceeded, stop and get the RO31337
user budget raised — nothing else will work until then.

Note `FCG_GUIDED=0`: the gateway routes to Bedrock, which rejects the guided_json
field. Keep guided OFF for every Claude/gateway run. (Only turn it ON — by
dropping the flag — when running against Gemma on vLLM directly.)

---

## STEP 4 — Re-run extraction with the fixed routes (Claude)

Delete the checkpoint first (MANDATORY — or it reuses old answers), then run the
whole thing on ONE line:

```
rm -f data/work/checkpoint.json
```
```
LITELLM_BASE_URL=GATEWAY_URL LITELLM_API_KEY=YOUR_KEY AGENT_MODEL=claude-opus-4-8 AGENT_TLS_VERIFY=false FCG_GUIDED=0 FCG_MAX_TOKENS=6000 FCG_CONCURRENCY=2 python fcg_extract.py --checklist checklist_enriched.json --fcg-dir data/fcg --svc data/svc_rmk.txt --meis data/meis1.json data/meis2.json data/meis3.json --airports data/airports.csv --codes data/countries_codes_and_coordinates.csv --out gold_v2.csv
```

---

## STEP 5 — Check the fix worked

Validate the new run (verbatim/hallucination check + fill rates):
```
python validate_extract.py --csv gold_v2.csv --dossiers data/work/dossiers
```
Then confirm the previously-broken fields no longer fall back — open the audit
file and look at the `FALLBACK` line; it should now be much shorter:
```
grep FALLBACK data/work/dossiers/FRA_phase1.txt
```

Compare old vs new to see exactly which fields the route fix changed:
```
python compare_extracts.py --gold gold.csv --candidate gold_v2.csv --out route_fix_diffs.csv
```
(Here `gold.csv` = your original Claude run, `gold_v2.csv` = after the fix. The
"differences" are the improvements. This proves the routes got better — it is
NOT a Gemma test.)

---

## STEP 6 — (LATER, when Gemma is served) the real model comparison

Serve Gemma-4 on vLLM (see RUNBOOK section 1 — needs enough GPU; 24GB cards can't
hold the unquantized 31B on one device, use tensor-parallel across enough cards
or a quantized checkpoint). Then, with guided decoding ON (drop FCG_GUIDED):

```
curl http://127.0.0.1:8000/v1/models          # confirm up, grab exact model id
rm -f data/work/checkpoint.json
```
```
LITELLM_BASE_URL=http://127.0.0.1:8000/v1 LITELLM_API_KEY=dummy AGENT_MODEL=gemma-4-31B-it FCG_MAX_TOKENS=6000 FCG_CONCURRENCY=2 python fcg_extract.py --checklist checklist_enriched.json --fcg-dir data/fcg --svc data/svc_rmk.txt --meis data/meis1.json data/meis2.json data/meis3.json --airports data/airports.csv --codes data/countries_codes_and_coordinates.csv --out gemma.csv
```
Grade Gemma against the Claude gold standard (use the fixed one, gold_v2):
```
python compare_extracts.py --gold gold_v2.csv --candidate gemma.csv --out gemma_vs_gold.csv
```
The agreement % here is your real answer to "can Gemma take over from Claude."

---

## Cheat sheet — the mistakes that bit you before

- `rm -f data/work/checkpoint.json` before EVERY re-run that should be fresh, or
  old cached answers get reused and you'll think nothing changed.
- Keep the extraction command on ONE line — multi-line prefixes drop flags silently.
- `FCG_GUIDED=0` for Claude/gateway (Bedrock rejects it); ON for Gemma/vLLM.
- Base URL ends in `/v1`, never `/v1/models`.
- 429 "budget" = LiteLLM user cap (RO31337), not a code bug. Raise the budget.
- Env vars set on their own line don't reach python — use the one-line prefix
  form shown here, or `export` them first.

---

## STEP 2 appendix — route_task.md contents (if the file is missing)

Save this as `route_task.md`:

```
Task: Fix broken extraction routes in checklist_enriched.json.

Read AGENT.md and skills/fcg-extraction.md first for the routing design.
checklist_enriched.json has 91 items; ~28 are type "extract" with route/scope
key-candidate arrays that direct the extractor to the matching FCG subtree. Some
routes match nothing, so those items fall back to the whole document and extract
poorly.

Inputs:
- fcg_keys.txt : the real key paths from the FCG JSON (structure, depth <=4).
- data/work/dossiers/FRA_phase1.txt : its "===== FALLBACK" line lists every item
  whose route hit nothing.

Do this:
1. For each item in the FALLBACK list, find the real key path in fcg_keys.txt that
   holds its answer (e.g. Runway_WBC -> the path containing runwayDetailsList /
   RawWeightBearingCapacity; Ops_Hrs -> the hours key; TERPS -> the terps key).
2. Update ONLY that item's route array (and scope if needed) in
   checklist_enriched.json to the real key names. Use "keyname$" for exact-key
   match, plain substring otherwise.
3. If an item's answer genuinely isn't in the FCG (e.g. AIR Card payment method),
   set its type to "reference" instead of inventing a route, and say why.
4. Do NOT change any workflow item, do NOT edit any .py file, do NOT change item
   ids or the file structure. Routes only.
5. Output a changelog: for each item, old route -> new route -> the key path matched.

Constraint: every new route must correspond to a key path that actually appears
in fcg_keys.txt. If you can't find a match, leave the route unchanged and list it
as unresolved.
```
