"""Shared type definitions used across CRS modules.

These are the canonical dataclasses. In a full build, each step's module
re-exports or directly uses these.  The fuzzer only needs the shapes
defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Step 1 — data_loader types
# ---------------------------------------------------------------------------
@dataclass
class CyberGymTask:
    task_id: str
    project_name: str
    vuln_type: str               # e.g. "buffer-overflow", "use-after-free"
    description: str             # CVE-style prose
    repo_path: Path              # extracted repo-vul.tar.gz root
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step 2 — code_intelligence types
# ---------------------------------------------------------------------------
@dataclass
class CodeSnippet:
    filepath: str
    start_line: int
    end_line: int
    content: str
    relevance_score: float = 0.0


@dataclass
class CodeContext:
    task: CyberGymTask
    top_snippets: list[CodeSnippet] = field(default_factory=list)
    build_info: dict[str, Any] = field(default_factory=dict)
    # build_info may contain:
    #   "entry_points": [Path, ...],
    #   "include_dirs": [str, ...],
    #   "source_files": [str, ...],
    #   "makefile": str | None


# ---------------------------------------------------------------------------
# Step 3 — llm_router types
# ---------------------------------------------------------------------------
class LLMRouter:
    """Thin wrapper around the OpenAI-compatible chat endpoint."""

    def query(self, prompt: str, *, max_tokens: int = 2048,
              temperature: float = 0.4) -> str:
        """Send *prompt* and return the assistant's text reply."""
        raise NotImplementedError("Stub — replaced by real router at runtime")


# ---------------------------------------------------------------------------
# Step 4 — poc_strategies types
# ---------------------------------------------------------------------------
@dataclass
class PoCResult:
    poc_code: str                # C/C++ source of the PoC
    poc_path: Path | None        # path to saved file (may be None until saved)
    strategy_name: str           # e.g. "llm_direct", "libfuzzer", "afl++"
    confidence: float            # 0.0 – 1.0
    crash_input_path: Path | None = None   # raw crash file, if from fuzzer
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step 5 — build_executor types
# ---------------------------------------------------------------------------
@dataclass
class BuildResult:
    success: bool
    binary_path: Path | None = None
    include_dirs: list[str] = field(default_factory=list)
    lib_paths: list[str] = field(default_factory=list)
    object_files: list[str] = field(default_factory=list)
    compile_log: str = ""
    link_log: str = ""
