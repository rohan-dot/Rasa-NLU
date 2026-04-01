"""
crs/config.py — Central configuration for the Cyber Reasoning System.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


def _resolve_work_dir() -> Path:
    p = Path(os.environ.get("CRS_WORK_DIR", Path.home() / ".crs_workdir"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _tool_on_path(name: str) -> bool:
    """Check whether an executable is reachable via PATH."""
    return shutil.which(name) is not None


@dataclass
class CRSConfig:
    """Central configuration object for the Cyber Reasoning System."""

    # — LLM backend ————————————————————————————————————————
    LLM_BASE_URL: str = field(
        default_factory=lambda: os.environ.get(
            "OPENAI_BASE_URL", "http://g52lambda02.llan.ll.mit.edu:8000/v1"
        )
    )

    LLM_API_KEY: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "EMPTY")
    )

    LLM_MODEL: str = field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "gemma-3-27b-it")
    )

    MAX_TOKENS: int = 4096   # bumped from 2048 to avoid truncated PoC code
    MAX_RETRIES: int = 3

    # — Build / run limits ————————————————————————————————
    BUILD_TIMEOUT: int = 300   # seconds — big projects (GraphicsMagick) need more
    RUN_TIMEOUT: int = 60      # seconds

    # — Filesystem ————————————————————————————————————————
    WORK_DIR: Path = field(default_factory=_resolve_work_dir)

    # — Dependency management ————————————————————————————
    AUTO_INSTALL_DEPS: bool = False   # set True only in containers/VMs

    # — Sanitizer / fuzzer flags ——————————————————————————
    USE_SANITIZERS: bool = True
    FUZZING_ENABLED: bool = True  # runtime-checked below

    # — Derived (set in __post_init__) ————————————————————
    AFL_AVAILABLE: bool = field(init=False, default=False)
    LIBFUZZER_AVAILABLE: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        # Ensure WORK_DIR exists even if caller supplied a custom Path
        self.WORK_DIR = Path(self.WORK_DIR)
        self.WORK_DIR.mkdir(parents=True, exist_ok=True)

        # Check fuzzer availability at runtime
        self.AFL_AVAILABLE = _tool_on_path("afl-fuzz")
        self.LIBFUZZER_AVAILABLE = _tool_on_path("clang")

        if self.FUZZING_ENABLED and not (self.AFL_AVAILABLE or self.LIBFUZZER_AVAILABLE):
            print(
                "[config] FUZZING_ENABLED=True but neither AFL++ nor libFuzzer "
                "found in PATH. Fuzzing will be skipped at runtime."
            )

    def task_work_dir(self, task_id: str) -> Path:
        safe = task_id.replace(":", "_").replace("/", "_")
        d = self.WORK_DIR / safe
        d.mkdir(parents=True, exist_ok=True)
        return d


# ── Singleton ──────────────────────────────────────────────────────────────
cfg = CRSConfig()

# Module-level aliases so other modules can do `from crs.config import WORK_DIR`
WORK_DIR          = cfg.WORK_DIR
BUILD_TIMEOUT     = cfg.BUILD_TIMEOUT
RUN_TIMEOUT       = cfg.RUN_TIMEOUT
USE_SANITIZERS    = cfg.USE_SANITIZERS
FUZZING_ENABLED   = cfg.FUZZING_ENABLED
FUZZING_TIMEOUT   = 120

LLM_MODEL         = cfg.LLM_MODEL
LLM_BASE_URL      = cfg.LLM_BASE_URL
LLM_API_KEY       = cfg.LLM_API_KEY
MAX_TOKENS        = cfg.MAX_TOKENS
MAX_RETRIES       = cfg.MAX_RETRIES

DEFAULT_MODEL     = cfg.LLM_MODEL
DEFAULT_BASE_URL  = cfg.LLM_BASE_URL
DEFAULT_API_KEY   = cfg.LLM_API_KEY

POC_RUN_TIMEOUT   = cfg.RUN_TIMEOUT
VERBOSE           = False
