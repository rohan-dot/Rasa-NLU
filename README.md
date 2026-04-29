python bioasq_agentic_local.py \
    --test-input 13B1_golden.json \
    --training training13b.json \
    --faiss-index /scratch/da32459/data/rag-cpu/pubmed_2026.faiss \
    --bm25-index NONE \
    --corpus-db /scratch/da32459/data/rag-cpu/corpus/pubmed_2026.db \
    --model gemma-4-31b-it \
    --embed-device cpu \
    -o batch4_local.json
