# BioASQ 14b — Agentic QA System

An agentic pipeline for BioASQ Task 14b biomedical question answering using **LangGraph** + **vLLM/Gemma 3 27B** + **PubMed retrieval** + **FAISS few-shot retrieval**.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph                      │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Reformulate  │───▶│   Retrieve   │───▶│   Extract    │  │
│  │    Query      │    │  Documents   │    │  Snippets    │  │
│  │  (Gemma LLM)  │    │  (PubMed)    │    │ (keyword TF) │  │
│  └──────┬───────┘    └──────────────┘    └──────┬───────┘  │
│         │ Phase B: skip retrieval ──────────────▶│          │
│         │                                        ▼          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Build Context                            │   │
│  │  • Gold snippets (Phase B) or retrieved snippets     │   │
│  │  • FAISS few-shot examples from training data        │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Generate Answer (Gemma via vLLM)            │   │
│  │  • Type-specific prompts (yesno/factoid/list/summary) │   │
│  │  • Raw OpenAI client (avoids ChatOpenAI tool_choice)  │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Validate & Parse Answer                     │   │
│  │  • Extract EXACT / IDEAL from LLM output              │   │
│  │  • Format per BioASQ submission spec                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | All configuration (vLLM URL, model, paths, parameters) |
| `data_loader.py` | Load BioASQ JSON, FAISS index builder, submission formatter |
| `retriever.py` | PubMed search + fetch via NCBI E-utilities, snippet extraction |
| `llm_client.py` | Raw OpenAI client for vLLM/Gemma (no LangChain wrapper) |
| `agent.py` | LangGraph StateGraph with 6 nodes |
| `main.py` | CLI entry point for all phases + evaluation |
| `requirements.txt` | Python dependencies |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

### 2. Configure vLLM

Edit `config.py` or set environment variables:

```bash
export VLLM_BASE_URL="http://g52lambda02.example.com:8000/v1"
export VLLM_MODEL="google/gemma-3-27b-it"
export PUBMED_EMAIL="your@email.com"
# Optional: NCBI API key for higher rate limits
export PUBMED_API_KEY="your_ncbi_api_key"
```

### 3. Run

```bash
# Phase A+ (retrieve documents + generate answers — for today's batch 1)
python main.py --phase A+ \
    --test BioASQ-task14bPhaseA-testset1.json \
    --train trainining14b.json

# Phase B (when gold docs/snippets are released tomorrow)
python main.py --phase B \
    --test BioASQ-task14b-testset1-phaseB.json \
    --train trainining14b.json

# Quick debug: process first 5 questions, skip FAISS
python main.py --phase A+ --test testset.json --limit 5 --no-faiss

# Evaluate on training data
python main.py --phase B --test trainining14b.json --eval --limit 50
```

## Key Design Decisions

1. **Raw OpenAI client** (not LangChain `ChatOpenAI`): LangChain silently injects `tool_choice` params that cause Gemma on vLLM to return empty content. All generation nodes use `openai.OpenAI` directly.

2. **FAISS few-shot retrieval**: Training questions are embedded with `intfloat/e5-base-v2` and indexed. For each test question, the most similar training examples (same question type) are retrieved as few-shot demonstrations.

3. **Type-specific prompts**: Each question type (yesno, factoid, list, summary) gets a tailored prompt with format instructions matching BioASQ's exact submission schema.

4. **Two-path routing**: Phase B skips document retrieval and uses gold snippets directly; Phase A/A+ runs the full PubMed retrieval pipeline.

5. **PubMed query reformulation**: The LLM rewrites natural language questions into optimized PubMed search queries before retrieval.

## Submission Format

Output JSON follows the official BioASQ format exactly:

```json
{
  "questions": [
    {
      "id": "69ac745885870e396d000065",
      "type": "factoid",
      "body": "What is the prevalence of primary adrenal insufficiency?",
      "ideal_answer": "The prevalence of primary adrenal insufficiency...",
      "exact_answer": [["93-140 per million"]]
    }
  ]
}
```

## Timing

- **Batch 1 released**: Wednesday March 18, 2026 10:00 GMT
- **Phase A/A+ due**: Thursday March 19, 2026 10:00 GMT (24h)
- **Phase B gold released**: Thursday March 19, 2026 11:00 GMT
- **Phase B due**: Friday March 20, 2026 11:00 GMT (24h)
