python bioasq_agentic_local.py \
    --test-input 13B1_golden.json \
    --training training13b.json \
    --faiss-index /scratch/da32459/data/rag-cpu/pubmed-2026.IVF8192-Flat.NeuML-pubmedbert-base-embeddings.faiss \
    --bm25-index NONE \
    --corpus-db /scratch/da32459/data/rag-cpu/corpus/pubmed_2026.db \
    --model gemma-4-31b-it \
    --embed-device cpu \
    -o batch4_local.json
