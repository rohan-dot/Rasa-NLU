"""
BioASQ 14b Agentic QA System - Configuration
"""
import os

# ── vLLM / Gemma ────────────────────────────────────────────────────────
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://g52lambda02:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "google/gemma-3-27b-it")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

MAX_TOKENS = 2048
TEMPERATURE = 0.3
TOP_P = 0.9
MAX_RETRIES = 2

# ── Embedding / FAISS ───────────────────────────────────────────────────
EMBEDDING_MODEL = "intfloat/e5-base-v2"
SNIPPET_TOP_K = 10
FEW_SHOT_EXAMPLES = 3

# ── BioASQ PubMed Service (no API key needed) ───────────────────────────
BIOASQ_PUBMED_URL = "http://bioasq.org:8000/pubmed"
PUBMED_MAX_ARTICLES = 10

# ── File Paths ──────────────────────────────────────────────────────────
TRAINING_DATA_PATH = os.environ.get("TRAINING_DATA_PATH", "trainining14b.json")
TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH", "BioASQ-task14bPhaseA-testset1.json")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
