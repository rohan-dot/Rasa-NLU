"""
BioASQ 14b Agentic QA System - Configuration
=============================================
Adjust these settings for your HPC cluster environment.
"""

import os

# ── vLLM / Gemma Configuration ──────────────────────────────────────────
# Use the FULL hostname (not localhost) to avoid vLLM connectivity issues.
VLLM_BASE_URL = os.environ.get(
    "VLLM_BASE_URL", "http://g52lambda02.example.com:8000/v1"
)
VLLM_MODEL = os.environ.get("VLLM_MODEL", "google/gemma-3-27b-it")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

# Generation parameters
MAX_TOKENS = 2048          # Keep modest to fit context window
TEMPERATURE = 0.3          # Low for factual biomedical QA
TOP_P = 0.9

# ── PubMed / Retrieval Configuration ────────────────────────────────────
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_EMAIL = os.environ.get("PUBMED_EMAIL", "your@email.com")
PUBMED_API_KEY = os.environ.get("PUBMED_API_KEY", "")  # Optional, raises rate limit
PUBMED_MAX_RESULTS = 10    # BioASQ allows max 10 documents
PUBMED_MAX_SNIPPETS = 10   # BioASQ allows max 10 snippets

# ── Embedding / FAISS Configuration ─────────────────────────────────────
EMBEDDING_MODEL = "intfloat/e5-base-v2"
FAISS_TOP_K = 10           # Top-k snippets to retrieve from training data

# ── LangGraph Agent Configuration ───────────────────────────────────────
MAX_RETRIES = 2            # Max LLM retries per node on failure
AGENT_TIMEOUT = 300        # Seconds per question timeout

# ── File Paths ──────────────────────────────────────────────────────────
TRAINING_DATA_PATH = os.environ.get(
    "TRAINING_DATA_PATH", "trainining14b.json"
)
TEST_DATA_PATH = os.environ.get(
    "TEST_DATA_PATH", "BioASQ-task14bPhaseA-testset1.json"
)
# Phase B test set (with gold docs/snippets) — set when available
PHASE_B_DATA_PATH = os.environ.get("PHASE_B_DATA_PATH", "")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
PHASE_A_OUTPUT = os.path.join(OUTPUT_DIR, "bioasq_phaseA_submission.json")
PHASE_APLUS_OUTPUT = os.path.join(OUTPUT_DIR, "bioasq_phaseAplus_submission.json")
PHASE_B_OUTPUT = os.path.join(OUTPUT_DIR, "bioasq_phaseB_submission.json")

# ── Few-shot Configuration ──────────────────────────────────────────────
# Number of training examples to include as few-shot demonstrations
FEW_SHOT_EXAMPLES = 3

# ── Logging ─────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
