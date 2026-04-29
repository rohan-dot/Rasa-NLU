pip install sentence-transformers

python bioasq_agentic_local.py \
    --test-input 13B1_golden.json \
    --training training13b.json \
    --db /scratch/ro31337/pubmed_index.db \
    --model gemma-4-31b-it \
    --embed-device cpu \
    -o batch4_local.json 2>&1 | tee local_run.log
