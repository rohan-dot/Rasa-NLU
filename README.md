python bioasq_sqlite_agent.py \
    --test-input 13B1_golden.json \
    --training training13b.json \
    --db /scratch/ro31337/pubmed_index.db \
    --model gemma-4-31b-it \
    --embed-device cpu \
    -o sqlite_results.json 2>&1 | tee sqlite_run.log
