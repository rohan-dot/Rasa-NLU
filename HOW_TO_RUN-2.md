# BioASQ Task 14 — How to Run

## What's the difference?

Phase A+ = questions only → return documents + snippets + exact/ideal answers
Phase B  = questions + gold snippets → return exact/ideal answers only

| Phase | Gold snippets? | Returns | Script | Needs embedding? |
|-------|---------------|---------|--------|-----------------|
| A+    | NO            | docs + snippets + answers | bioasq_phaseA.py | YES (cpu) |
| B     | YES           | answers only | bioasq_phaseB.py | NO |

## Step 1: Start vLLM

    CUDA_VISIBLE_DEVICES=0,1 \
    python -m vllm.entrypoints.openai.api_server \
        --model /panfs/g52-panfs/exp/FY25/models/gemma-3-27b-it \
        --served-model-name gemma-3-27b-it \
        --dtype bfloat16 --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.85 --max-model-len 8192 \
        --host 127.0.0.1 --port 8000

Wait for "Uvicorn running"

## Step 2a: Phase A+ (drops first, no gold snippets)

    pip install requests sentence-transformers faiss-cpu numpy rank-bm25

    python bioasq_phaseA.py \
        --test-input BioASQ-task14bPhaseA-testset1.json \
        --training training13b.json \
        --model gemma-3-27b-it \
        --embed-device cpu \
        -o phaseA_submission.json

Submit phaseA_submission.json → BioASQ Phase A+
Takes 3-5 hours. You have 24 hours.

## Step 2b: Phase B (drops ~24hrs later, gold snippets provided)

    python bioasq_phaseB.py \
        --test-input BioASQ-task14bPhaseB-testset1.json \
        --training training13b.json \
        --model gemma-3-27b-it \
        -o phaseB_submission.json

Submit phaseB_submission.json → BioASQ Phase B
Takes 1-2 hours. You have 24 hours.

## For batch 2, 3, 4: change the input file number

    python bioasq_phaseA.py -t ...testset2.json -tr training13b.json --model gemma-3-27b-it --embed-device cpu -o phaseA_b2.json
    python bioasq_phaseB.py -t ...testset2.json -tr training13b.json --model gemma-3-27b-it -o phaseB_b2.json

## Format Rules (from BioASQ guidelines)
- factoid exact_answer: list of up to 5 terms, no synonyms
- list exact_answer: flat list, max 100 items, max 100 chars each
- yesno exact_answer: "yes" or "no"
- summary: no exact_answer
- ideal_answer: max 200 words

## Troubleshooting
- Cannot reach vLLM: wait for it to load
- Hangs at embedding: add --embed-device cpu (Phase A only)
- 429 from PubMed: auto-retry, just wait
