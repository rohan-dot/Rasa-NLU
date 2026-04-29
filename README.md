python bioasq_agentic_local.py \
    --test-input BioASQ-task14bPhaseA-testset4.json \
    --training training13b.json \
    --db /scratch/ro31337/pubmed_index.db \
    --model gemma-3-27b-it \
    --embed-device cpu \
    -o batch4_local.json
