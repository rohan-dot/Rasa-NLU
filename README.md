python bioasq_agentic_local.py \
    --test-input 13B1_golden.json \
    --training training13b.json \
    --faiss-index /scratch/da32459/data/rag-cpu/pubmed-2026.IVF8192-Flat.NeuML-pubmedbert-base-embeddings.faiss \
    --bm25-index /scratch/da32459/data/rag-cpu/indexes/pubmed_2026.bm25 \
    --corpus-db /scratch/da32459/data/rag-cpu/corpus/pubmed_2026.db \
    --model gemma-4-31b-it \
    --embed-device cpu \
    -o batch4_local_v2.json




xx

python bioasq_sqlite_agent.py \
    --test-input 13B1_golden.json \
    --training training13b.json \
    --db /scratch/ro31337/pubmed_index.db \
    --model gemma-4-31b-it \
    --embed-device cpu \
    -o sqlite_results.json 2>&1 | tee sqlite_run.log
