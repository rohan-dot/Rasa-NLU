cp /scratch/ro31337/pubmed_index.db /tmp/pubmed_index.db

python bioasq_agentic_local.py \
    --test-input 13B1_golden.json \
    --training training13b.json \
    --db /tmp/pubmed_index.db \
    --model gemma-4-31b-it \
    --embed-device cpu \
    -o batch4_local.json 2>&1 | tee local_run.log
