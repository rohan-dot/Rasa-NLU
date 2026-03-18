# BioASQ 14b — Agentic QA System

**No PubMed API key, no email, no internet needed for retrieval.**  
All context comes from FAISS retrieval over your 5729 training questions (~50k+ snippets).

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                   LangGraph StateGraph                      │
│                                                            │
│  ┌─────────────────┐    ┌─────────────────┐               │
│  │ Retrieve Context │───▶│  Build Prompt   │               │
│  │                  │    │                  │               │
│  │ Phase A+: FAISS  │    │ • Few-shot demos │               │
│  │   snippet search │    │   (similar Qs    │               │
│  │   over training  │    │   from training) │               │
│  │                  │    │ • Type-specific  │               │
│  │ Phase B: gold    │    │   instructions   │               │
│  │   snippets       │    │                  │               │
│  └─────────────────┘    └────────┬────────┘               │
│                                  ▼                         │
│  ┌─────────────────┐    ┌─────────────────┐               │
│  │  Parse Answer   │◀───│ Generate Answer  │               │
│  │                  │    │                  │               │
│  │ • Extract EXACT  │    │ Gemma 27B via    │               │
│  │ • Extract IDEAL  │    │ vLLM (raw OpenAI │               │
│  │ • Format for     │    │ client)          │               │
│  │   submission     │    │                  │               │
│  └─────────────────┘    └─────────────────┘               │
└────────────────────────────────────────────────────────────┘
```

## Two FAISS Indexes

| Index | Content | Purpose |
|-------|---------|---------|
| Question index | 5729 training question embeddings | Few-shot demo retrieval (same question type) |
| Snippet index | ~50k+ training snippet embeddings | Context retrieval (replaces PubMed) |

Both use `intfloat/e5-base-v2` with query/passage prefixes for asymmetric search.

## Quick Start

```bash
pip install -r requirements.txt --break-system-packages

# Set your vLLM endpoint
export VLLM_BASE_URL="http://g52lambda02:8000/v1"

# Phase A+ (today's batch — retrieve from training + generate answers)
python main.py --test BioASQ-task14bPhaseA-testset1.json \
               --train trainining14b.json

# Quick sanity check (3 questions, no FAISS build)
python main.py --test BioASQ-task14bPhaseA-testset1.json \
               --train trainining14b.json \
               --limit 3 --no-faiss

# Phase B (when gold snippets are released)
python main.py --phase B \
               --test BioASQ-task14b-testset1-phaseB.json \
               --train trainining14b.json

# Evaluate on training data itself
python main.py --phase B \
               --test trainining14b.json \
               --train trainining14b.json \
               --eval --limit 50
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | vLLM URL, model, FAISS parameters |
| `data_loader.py` | JSON loader, dual FAISS index builder, submission formatter |
| `llm_client.py` | Raw OpenAI client for vLLM/Gemma (no LangChain wrapper) |
| `agent.py` | 4-node LangGraph pipeline |
| `main.py` | CLI entry point |

## Output Format

Matches BioASQ submission spec exactly:

```json
{
  "questions": [
    {
      "id": "69ac745885870e396d000065",
      "type": "factoid",
      "body": "What is the prevalence of primary adrenal insufficiency?",
      "ideal_answer": "Primary adrenal insufficiency has a prevalence of...",
      "exact_answer": [["93-140 per million"]]
    }
  ]
}
```
