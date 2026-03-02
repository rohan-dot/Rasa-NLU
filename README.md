# BioASQ Agentic Retrieval System — Refactored

## What Changed (vs QA_2_Try.py)

### 1. Multi-Index Architecture (per-paper FAISS)
- **Before**: Single unified FAISS index for all documents
- **After**: Each BioASQ paper gets its own FAISS index, stored in `./bioasq_paper_indices/<paper_id>/`
- A lightweight **master routing index** over paper abstracts enables fast paper selection before retrieval
- Paper IDs are extracted from PubMed URLs (e.g. `pubmed/12345678` → `12345678`)

### 2. Intelligent Paper Router
- Before retrieval, the system uses the master abstract index to find the top-K most relevant papers
- Only those papers are queried, reducing noise and improving precision
- Configurable via `ROUTER_TOP_K_PAPERS` (default: 5)

### 3. Parallel Multi-Paper Retrieval
- Selected papers are queried in parallel using `ThreadPoolExecutor`
- Results are deduplicated and merged before reranking
- Falls back to querying all papers if parallel retrieval fails

### 4. Fixed Graph Traversal (no more stuck states)
- **Hard loop caps** on every cycle:
  - `MAX_RETRIEVAL_LOOPS = 4` — total retrieval attempts per turn
  - `MAX_REWRITE_LOOPS = 2` — query rewriting attempts
  - `MAX_DECOMPOSE_LOOPS = 2` — decomposition attempts
  - `MAX_REFLECT_LOOPS = 2` — reflection/self-critique cycles
- Loop counters are tracked in state and reset each new user turn
- Every conditional edge has a guaranteed fallback to `generate_draft` or `finalize_answer`
- The graph **cannot** cycle indefinitely

### 5. Improved Prompts for Gemma-3-27b-it
- `GENERATE_PROMPT` now has biomedical-specific grounding rules
- Added `GENERATE_PROMPT_BIOASQ` for type-aware generation (factoid/list/yesno/summary)
- `GRADE_PROMPT` explicitly handles code/ID questions and low-snippet scenarios
- `REFLECT_PROMPT` produces structured JSON with clear action directives

### 6. BioASQ Evaluation Harness
- `evaluate_bioasq()` loads a dataset, builds indices, runs all questions through the pipeline
- Results are saved as JSON with predicted vs ideal/exact answers
- Run with: `python QA_BioASQ_Refactored.py dataset.json [max_questions]`

### 7. Preserved Components
- vLLM + Gemma-3-27b-it backend (unchanged)
- LangChain StateGraph framework
- Cross-encoder reranking
- HyDE for short questions
- Neighbor expansion for context completeness
- Long-term memory (optional, flag-gated)

---

## Requirements

```bash
pip install langchain langchain-community langchain-openai faiss-cpu \
            sentence-transformers rank_bm25 pydantic langgraph
```

vLLM must be serving `gemma-3-27b-it` on `http://127.0.0.1:8000/v1`.

---

## Usage

### Evaluate on BioASQ dataset
```bash
python QA_BioASQ_Refactored.py /path/to/bioasq_dataset.json
# or limit to first N questions:
python QA_BioASQ_Refactored.py /path/to/bioasq_dataset.json 50
```

### Interactive mode
```bash
python QA_BioASQ_Refactored.py
```

---

## Configuration

All tuneable parameters are at the top of the file:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `ROUTER_TOP_K_PAPERS` | 5 | How many papers to select per question |
| `DENSE_TOP_K` | 12 | Dense retrieval top-K per paper |
| `FINAL_TOP_K` | 8 | Final snippets after reranking |
| `MAX_RETRIEVAL_LOOPS` | 4 | Hard cap on retrieval cycles |
| `MAX_REWRITE_LOOPS` | 2 | Hard cap on query rewrites |
| `MAX_DECOMPOSE_LOOPS` | 2 | Hard cap on decomposition |
| `MAX_REFLECT_LOOPS` | 2 | Hard cap on reflection cycles |
| `CROSS_ENCODER_MODEL` | `ms-marco-MiniLM-L-6-v2` | Set to `None` to skip reranking |
| `ENABLE_LONGTERM_MEMORY` | `False` | Toggle conversation memory |

---

## Graph Flow

```
START → prepare_question → [plan_question?] → call_retriever → retrieve (tool)
      → capture_context → grade_retrieval
                           ├── generate_draft → reflect
                           │                    ├── finalize_answer → write_memory → END
                           │                    ├── apply_reflection_queries → call_retriever (loop)
                           │                    └── ask_user → END
                           ├── rewrite_question → call_retriever (loop, max 2)
                           └── decompose_question → call_retriever (loop, max 2)
```

Every loop path has a hard counter that forces termination after the cap is reached.
