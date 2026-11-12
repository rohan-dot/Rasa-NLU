# How to run — your paths, copy-paste

Everything runs from `/exp/FY26/aim/ro31337/AIA` (the folder that holds `lora_review_agents/` and the deck).
Two launchers have your paths built in, so there is nothing long to paste:

| What | Command |
|---|---|
| **New** — fine-tuned vs base model inside agentic RAG | `bash lora_review_agents/run_rag.sh` |
| October — audit report + architecture | `bash lora_review_agents/run_audit.sh` |

## 0. Once

```bash
cd /exp/FY26/aim/ro31337/AIA
unzip -o lora_review_agents.zip                     # use the name your browser gave the download
cp lora_review_agents/env.sh.example lora_review_agents/env.sh
nano lora_review_agents/env.sh                      # LiteLLM URL, key, your gpt-5.6 model name, CA bundle
```

Both launchers read `env.sh` by themselves. If you already export `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`
and `LLM_CA_BUNDLE` in your shell, skip `env.sh`.

## 1. Fine-tuned vs base model inside agentic RAG

```bash
cd /exp/FY26/aim/ro31337/AIA
bash lora_review_agents/run_rag.sh --check       # 1. runner endpoint, GPU stack, paths — seconds, changes nothing
bash lora_review_agents/run_rag.sh --limit 5     # 2. five questions through every stage — minutes
bash lora_review_agents/run_rag.sh               # 3. the real run: 200 sampled questions (--limit 0 = all)
```

After step 2 read two short files before starting step 3:

- `rag_outputs/setup_notes.md` — which adapter, base model and question file the runner picked, and why
- `rag_outputs/sample_prompts.md` — the first prompt of each model, exactly as sent. The fine-tuned model's
  prompt must look like the training text in `train_sft.py`.

If either is wrong, fix it and go again — only the affected answers are regenerated:

```bash
bash lora_review_agents/run_rag.sh --limit 5 --adapter /path/to/adapter_dir      # or --eval-file FILE, --base-model PATH
nano rag_outputs/rag_config.json                                                  # prompt templates, max_new_tokens, stop strings
bash lora_review_agents/run_rag.sh --limit 5 --redo setup                         # let the runner read the project again
```

The result is `rag_outputs/comparison.md`; the headline is also printed at the end of the run.

What step 3 does, in order (each stage keeps its output, so re-running the same command continues where it stopped):

| Stage | Who | Output in `rag_outputs/` |
|---|---|---|
| setup | runner (GPT-5.6) reads `train.sh`, `train_sft.py`, the adapter, the data | `rag_config.json`, `setup_notes.md` |
| index | this code (BM25 over `af_epubs`) | `index/` |
| retrieve | runner searches, rewrites the query, picks passages — once per question | `evidence.jsonl` |
| generate | the fine-tuned model, then the base model, on the **same** passages (GPU) | `predictions_*.jsonl`, `sample_prompts.md` |
| judge | runner scores each answer against the reference, not knowing which model wrote it | `judgments.jsonl` |
| report | this code: metrics, paired statistics, examples | `comparison.md`, `per_example.csv` |

Useful additions:

```bash
bash lora_review_agents/run_rag.sh --with-closed-book      # also answer WITHOUT retrieval: shows what retrieval itself adds
bash lora_review_agents/run_rag.sh --limit 0               # the whole evaluation set
bash lora_review_agents/run_rag.sh --retrieval single      # one plain BM25 query instead of the agentic loop
OUT_DIR=rag_outputs_v2 bash lora_review_agents/run_rag.sh  # a separate output folder
```

**If `--check` says torch / transformers / peft are missing:** only the generate stage needs them. Run it in the
environment the adapter was trained in (`setup_notes.md` quotes how `train.sh` sets that up), then come back:

```bash
bash lora_review_agents/run_rag.sh                      # here: stops at generate, everything before it is saved
# ...in the training environment, same folder:
bash lora_review_agents/run_rag.sh --stages generate
# ...back here:
bash lora_review_agents/run_rag.sh                      # judges and writes the report
```

## 2. Audit report + architecture (the October pipeline)

```bash
cd /exp/FY26/aim/ro31337/AIA
bash lora_review_agents/run_audit.sh                         # explores quanta/, lora/, corpus/ and the deck
bash lora_review_agents/run_audit.sh --report-from-notes     # if agent_outputs/report_notes.md exists: skip the 20-minute exploration
```

Output: `agent_outputs/report.md`, `agent_outputs/ARCHITECTURE.md`, `agent_outputs/scaffold/`.

## 3. When something goes wrong

| You see | Do this |
|---|---|
| a lone `>` and nothing happens | the shell is waiting for the rest of a command: Ctrl-C and run it again (the launchers avoid this) |
| `CERTIFICATE_VERIFY_FAILED` | set `LLM_CA_BUNDLE` in `env.sh` to the lab CA bundle (try `/etc/pki/tls/certs/ca-bundle.crt`) |
| `No model given` | `LLM_MODEL` is not set — fill in `env.sh` |
| `the generate stage needs torch + transformers + peft` | see "If `--check` says torch … are missing" above |
| `no CUDA GPU is visible` | run the generate stage on a GPU node; pick one GPU with `export CUDA_VISIBLE_DEVICES=0` |
| `GPU out of memory` | it halves the batch size by itself; if it still fails add `--load-in-4bit` or `--ctx-k 3` |
| `eval file … is outside the directories given` | the message names the folder: append `--root data=<that folder>` |
| `could not load … --base-model /local/path` | the base model is not in the local Hugging Face cache: append `--base-model /path/to/Mistral-7B-Instruct-v0.3` |
| the runner seems slow | `--workers 10` (parallel runner calls); reasoning effort is already `low` |
| stopped half-way / lost connection | run the same command again — every stage resumes |

`python lora_review_agents/rag_compare.py --help` lists every flag; `README.md` explains how it works.
