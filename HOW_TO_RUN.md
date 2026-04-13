# BioASQ Task 14 — How to Run
## For whoever is running this while I'm on vacation

### Step 1: Start vLLM (Terminal 1)
```bash
CUDA_VISIBLE_DEVICES=0,1 \
python -m vllm.entrypoints.openai.api_server \
    --model /panfs/g52-panfs/exp/FY25/models/gemma-3-27b-it \
    --served-model-name gemma-3-27b-it \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --host 127.0.0.1 --port 8000
```
Wait until you see "Uvicorn running on http://127.0.0.1:8000"

### Step 2: Run the solver (Terminal 2)
```bash
cd /exp/FY26/CAITT/ro31337/bioasq13

pip install requests sentence-transformers faiss-cpu numpy rank-bm25

python bioasq_final.py \
    --test-input BioASQ-task14bPhaseA-testset1.json \
    --training training13b.json \
    --model gemma-3-27b-it \
    --embed-device cpu \
    -o submission.json
```

### Step 3: Check output
It produces TWO files:
- `submission.json` — submit to **Phase B** (answers)
- `submission_phaseA.json` — submit to **Phase A+** (documents + snippets)

### Step 4: Evaluate before submitting
```bash
python quick_eval.py --pred submission.json --gold 13B4_golden.json
```

### Troubleshooting
- **"Cannot reach vLLM"** → vLLM isn't running or hasn't finished loading. Wait for "Uvicorn running" in Terminal 1.
- **Hangs at "Loading embedding model"** → GPU is full. Add `--embed-device cpu`
- **429 errors from PubMed** → script handles this automatically with backoff. Just let it run.
- **Takes too long** → each question takes 2-5 min. 80 questions = ~3-6 hours.

### When new test batches drop
BioASQ releases test batches during the competition. When a new one drops:
1. Download it from the BioASQ participants area
2. Run the same command with the new test file:
```bash
python bioasq_final.py \
    --test-input BioASQ-task14bPhaseA-testset2.json \
    --training training13b.json \
    --model gemma-3-27b-it \
    --embed-device cpu \
    -o submission_batch2.json
```
3. Submit `submission_batch2.json` (Phase B) and `submission_batch2_phaseA.json` (Phase A+)
4. You have 24 hours per batch

### File overview
| File | What it does |
|------|-------------|
| `bioasq_final.py` | The solver — run this |
| `quick_eval.py` | Checks answers against gold |
| `training13b.json` | Few-shot examples from last year |
