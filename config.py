"""Central configuration for the FCG checklist pipeline.

Everything is overridable via environment variables so the same code runs
against LiteLLM (Claude) today and can be repointed at the vLLM/Gemma
endpoint on the air-gapped cluster without code changes.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# LLM endpoint (OpenAI-compatible). Reads FCG_* vars first, then the
# LITELLM_*/AGENT_* names from Jo's existing MIT LL gateway runbook, so the
# same four exports used elsewhere work here unchanged.
# ---------------------------------------------------------------------------
def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


LLM_BASE_URL = _env("FCG_LLM_BASE_URL", "LITELLM_BASE_URL",
                    default="http://localhost:4000/v1")
LLM_API_KEY = _env("FCG_LLM_API_KEY", "LITELLM_API_KEY", default="sk-local")
LLM_MODEL = _env("FCG_LLM_MODEL", "AGENT_MODEL", default="claude-opus-4-8")

# AGENT_TLS_VERIFY=false disables cert verification (self-signed gateway cert)
TLS_VERIFY = _env("FCG_TLS_VERIFY", "AGENT_TLS_VERIFY",
                  default="true").strip().lower() not in ("false", "0", "no")

TEMPERATURE = float(os.environ.get("FCG_TEMPERATURE", "0"))
MAX_TOKENS = int(os.environ.get("FCG_MAX_TOKENS", "2000"))       # hard cap: prevents runaway generation
CONCURRENCY = int(os.environ.get("FCG_CONCURRENCY", "4"))        # async semaphore width
RETRIES = int(os.environ.get("FCG_RETRIES", "3"))
RETRY_BACKOFF_S = float(os.environ.get("FCG_RETRY_BACKOFF_S", "3"))

# Some proxies reject response_format; client auto-falls-back and remembers.
TRY_JSON_MODE = os.environ.get("FCG_TRY_JSON_MODE", "1") == "1"

# ---------------------------------------------------------------------------
# Paths (relative to repo root unless overridden)
# ---------------------------------------------------------------------------
ROOT = Path(os.environ.get("FCG_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"

TRANSCRIPTS_DIR = Path(os.environ.get("FCG_TRANSCRIPTS_DIR", DATA / "transcripts"))
CHECKLIST_DOCX = Path(os.environ.get("FCG_CHECKLIST_DOCX", DATA / "checklist" / "checklist.docx"))
FCG_COUNTRY_DIR = Path(os.environ.get("FCG_COUNTRY_DIR", DATA / "fcg_countries"))

WORK = Path(os.environ.get("FCG_WORK_DIR", DATA / "work"))       # checkpoints / intermediates
OUT = Path(os.environ.get("FCG_OUT_DIR", DATA / "out"))          # final deliverables

# Stage artifacts (fixed names so stages find each other)
CHECKLIST_ITEMS_JSONL = WORK / "checklist_items.jsonl"           # preprocess output
TRANSCRIPTS_CLEAN_DIR = WORK / "transcripts_clean"               # preprocess output
STAGE1_FLAGGED_JSON = WORK / "stage1_flagged.json"
STAGE2_VERIFIED_JSON = WORK / "stage2_verified.json"
STAGE2_SCHEMA_JSON = WORK / "stage2_schema.json"
STAGE3_CHECKPOINT_JSONL = WORK / "stage3_checkpoint.jsonl"       # per-country resume log
FINAL_CSV = OUT / "fcg_checklist_extract.csv"

# ---------------------------------------------------------------------------
# Behavior knobs
# ---------------------------------------------------------------------------
COUNTRY_DOC_CHAR_CAP = int(os.environ.get("FCG_COUNTRY_DOC_CHAR_CAP", "12000"))
TRANSCRIPT_CHUNK_CHARS = int(os.environ.get("FCG_TRANSCRIPT_CHUNK_CHARS", "12000"))
MIN_CHECKLIST_ITEMS_EXPECTED = int(os.environ.get("FCG_MIN_ITEMS", "10"))


def ensure_dirs() -> None:
    for p in (WORK, OUT, TRANSCRIPTS_CLEAN_DIR):
        p.mkdir(parents=True, exist_ok=True)
