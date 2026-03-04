# BioASQ 13b — Full Pipeline

Hybrid retrieval + LLM answering pipeline for BioASQ Task 13b.
No runtime paper fetching — everything is pre-indexed offline.

## Architecture

```
Training JSON
     ↓
01_fetch_corpus.py
  - Extracts all PMIDs
  - Fetches PubMed abstracts in bulk (batch=200, rate-limited)
  - Attempts PMC full-text for ~50-60% of papers (BioC JSON API)
  - Saves gold training snippets as separate corpus
     ↓
data/
  papers.jsonl              (one paper per line, abstract + optional full text)
  training_snippets.jsonl   (gold evidence spans from training QA pairs)
  training_questions.json   (for few-shot retrieval at test time)
     ↓
02_build_index.py
  - Section-aware chunking (Results/Discussion prioritised)
  - MedCPT Article Encoder embeddings (768-dim, normalised)
  - FAISS HNSW index (cosine similarity via inner product)
  - BM25 index (keyword matching for rare terms, gene names etc.)
  - Two separate indexes: papers + training snippets
     ↓
index/
  papers_hnsw.faiss
  papers_chunks.pkl
  papers_bm25.pkl
  snippets_hnsw.faiss
  snippets_chunks.pkl
  snippets_bm25.pkl
     ↓
03_retrieve_and_answer.py
  - MedCPT Query Encoder (asymmetric: different model from article encoder)
  - Parallel retrieval from papers + snippets indexes
  - Hybrid reranking: 0.7*dense + 0.3*BM25 + section boost
  - 3 few-shot examples from most similar training questions (by type)
  - LLM answering with type-specific JSON prompts
  - BioASQ submission JSON output
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Step 1: Fetch corpus (one-time, ~30-60 min depending on corpus size)
python 01_fetch_corpus.py \
  --training BioASQ-training13b.json \
  --out data/

# Step 2: Build indexes (one-time, ~15-30 min)
python 02_build_index.py \
  --data data/ \
  --out  index/

# Step 3: Run on test set
export OPENAI_API_KEY=sk-...
python 03_retrieve_and_answer.py \
  --test  BioASQ-13b-testset.json \
  --index index/ \
  --train_questions data/training_questions.json \
  --out   submissions/submission.json \
  --model gpt-4o
```

## Key Design Decisions

### Why two indexes?
- **Papers index**: full text (PMC) or abstract — the actual literature
- **Snippets index**: gold evidence spans from training set — already validated by BioASQ annotators, thematically overlap with test questions year over year

### Why MedCPT asymmetric encoding?
MedCPT uses different encoders for queries vs articles. Using the article encoder for queries
(a common mistake) significantly degrades retrieval quality. Always:
- Query → `ncats/MedCPT-Query-Encoder`
- Chunks → `ncats/MedCPT-Article-Encoder`

### Why hybrid BM25 + dense?
Dense models miss exact matches for rare gene names (e.g. BRCA2, TP53 variants),
drug names, and numeric values. BM25 catches these. 70/30 split is a good default.

### Why section-aware chunking?
BioASQ answers predominantly live in Results and Discussion sections, not abstracts.
Boosting these at retrieval time meaningfully improves answer quality.

### Few-shot by question type
BioASQ has 4 question types with very different output formats:
- `yesno`   → "yes"/"no" + explanation
- `factoid` → exact entity + synonyms list
- `list`    → list of entities
- `summary` → paragraph answer
Few-shots from same type significantly improve format compliance.

## Expected Index Sizes (15-30k papers)

| Asset                  | Size     |
|------------------------|----------|
| papers.jsonl           | ~200 MB  |
| papers_hnsw.faiss      | ~350 MB  |
| papers_chunks.pkl      | ~180 MB  |
| snippets_hnsw.faiss    | ~80 MB   |
| snippets_chunks.pkl    | ~50 MB   |
| Total RAM at query time| ~1-2 GB  |
