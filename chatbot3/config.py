"""
CRS Configuration — Step 1
All runtime settings live here as a frozen dataclass.
Values are read from environment variables when available,
falling back to sensible defaults for a local vLLM setup.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


def _resolve_work_dir() -> Path:
    """Return WORK_DIR from env or default, creating it if needed."""
    p = Path(os.environ.get("CRS_WORK_DIR", Path.home() / ".crs_workdir"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _tool_on_path(name: str) -> bool:
    """Check whether an executable is reachable via PATH."""
    return shutil.which(name) is not None


@dataclass
class CRSConfig:
    """Central configuration object for the Cyber Reasoning System."""

    # ── LLM backend ──────────────────────────────────────────────
    LLM_BASE_URL: str = field(
        default_factory=lambda: os.environ.get(
            "OPENAI_BASE_URL", "http://localhost:8000/v1"
        )
    )
    LLM_API_KEY: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "EMPTY")
    )
    LLM_MODEL: str = field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "google/gemma-3-27b-it")
    )
    MAX_TOKENS: int = 4096
    MAX_RETRIES: int = 3

    # ── Build / run limits ───────────────────────────────────────
    BUILD_TIMEOUT: int = 120   # seconds
    RUN_TIMEOUT: int = 30      # seconds

    # ── Filesystem ───────────────────────────────────────────────
    WORK_DIR: Path = field(default_factory=_resolve_work_dir)

    # ── Sanitizer / fuzzer flags ─────────────────────────────────
    USE_SANITIZERS: bool = True
    FUZZING_ENABLED: bool = True  # runtime-checked below

    # ── Derived (set in __post_init__) ───────────────────────────
    AFL_AVAILABLE: bool = field(init=False, default=False)
    LIBFUZZER_AVAILABLE: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        # Ensure WORK_DIR exists even if caller supplied a custom Path
        self.WORK_DIR = Path(self.WORK_DIR)
        self.WORK_DIR.mkdir(parents=True, exist_ok=True)

        # Probe for fuzzer tooling
        self.AFL_AVAILABLE = _tool_on_path("afl-fuzz")
        self.LIBFUZZER_AVAILABLE = _tool_on_path("clang") and _tool_on_path("llvm-config")

        if self.FUZZING_ENABLED and not (self.AFL_AVAILABLE or self.LIBFUZZER_AVAILABLE):
            print(
                "[config] FUZZING_ENABLED=True but neither AFL++ nor libFuzzer found "
                "in PATH. Fuzzing will be skipped at runtime."
            )

    # ── Convenience ──────────────────────────────────────────────
    def task_work_dir(self, task_id: str) -> Path:
        """Return (and create) a per-task working directory."""
        d = self.WORK_DIR / task_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def summary(self) -> str:
        lines = [
            "╔══ CRS Configuration ══╗",
            f"  LLM endpoint : {self.LLM_BASE_URL}",
            f"  LLM model    : {self.LLM_MODEL}",
            f"  API key set  : {'yes' if self.LLM_API_KEY != 'EMPTY' else 'no (using EMPTY)'}",
            f"  MAX_TOKENS   : {self.MAX_TOKENS}",
            f"  MAX_RETRIES  : {self.MAX_RETRIES}",
            f"  BUILD_TIMEOUT: {self.BUILD_TIMEOUT}s",
            f"  RUN_TIMEOUT  : {self.RUN_TIMEOUT}s",
            f"  WORK_DIR     : {self.WORK_DIR}",
            f"  Sanitizers   : {'ON' if self.USE_SANITIZERS else 'OFF'}",
            f"  Fuzzing flag : {'ON' if self.FUZZING_ENABLED else 'OFF'}",
            f"  AFL++ found  : {self.AFL_AVAILABLE}",
            f"  libFuzzer ok : {self.LIBFUZZER_AVAILABLE}",
            "╚════════════════════════╝",
        ]
        return "\n".join(lines)


# Module-level singleton — import and use directly.
cfg = CRSConfig()


if __name__ == "__main__":
    print(cfg.summary())
