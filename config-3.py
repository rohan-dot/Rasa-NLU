"""
BioASQ 14b Agentic QA System - Configuration
=============================================
No PubMed API needed — all retrieval from training data via FAISS.
"""

import os

# ── vLLM / Gemma Configuration ──────────────────────────────────────────
# Use the FULL hostname (not localhost) to avoid vLLM connectivity issues.
VLLM_BASE_URL = os.environ.get(
    "VLLM_BASE_URL", "http://g52lambda02:8000/v1"
)
VLLM_MODEL = os.environ.get("VLLM_MODEL", "google/gemma-3-27b-it")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

# Generation parameters
MAX_TOKENS = 2048
TEMPERATURE = 0.3
TOP_P = 0.9

# ── Embedding / FAISS Configuration ─────────────────────────────────────
EMBEDDING_MODEL = "intfloat/e5-base-v2"
SNIPPET_TOP_K = 10        # Top-k training snippets as context
FEW_SHOT_EXAMPLES = 3     # Few-shot training questions as demos

# ── LLM Retry ──────────────────────────────────────────────────────────
MAX_RETRIES = 2

# ── File Paths ──────────────────────────────────────────────────────────
TRAINING_DATA_PATH = os.environ.get("TRAINING_DATA_PATH", "trainining14b.json")
TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH", "BioASQ-task14bPhaseA-testset1.json")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
