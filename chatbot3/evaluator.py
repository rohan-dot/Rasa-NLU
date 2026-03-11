"""
crs/evaluator.py — Score PoC results and produce evaluation reports.

Consumes RunResult objects from the build/execute pipeline and emits
structured EvalRecords that are persisted as JSON + Markdown.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Sibling imports (Steps 1-6 assumed present)
# ---------------------------------------------------------------------------
from crs.data_loader import CyberGymTask
from crs.build_executor import RunResult


# ── ANSI helpers ──────────────────────────────────────────────────────────

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _colour(text: str, code: str) -> str:
    """Wrap *text* in an ANSI colour escape (no-op when not a tty)."""
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{_RESET}"


# ── Data structures ───────────────────────────────────────────────────────

@dataclass
class EvalRecord:
    """One row in the final evaluation report."""

    task_id: str
    project_name: str
    vuln_type: str
    strategy_used: str          # strategy that produced the best/triggered PoC
    triggered: bool
    crash_type: str             # e.g. "ASAN heap-buffer-overflow", "SEGV", …
    confidence: float
    build_success: bool
    time_elapsed: float         # wall-clock seconds from start to result
    poc_code: str               # the PoC source that worked (or best attempt)
    notes: str = ""


# ── Evaluator ─────────────────────────────────────────────────────────────

class Evaluator:
    """Collects per-task results and writes evaluation artefacts."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[EvalRecord] = []

    # ------------------------------------------------------------------ #
    #  Recording                                                          #
    # ------------------------------------------------------------------ #

    def record(
        self,
        task: CyberGymTask,
        run_results: List[RunResult],
        elapsed: float,
        build_success: bool,
    ) -> EvalRecord:
        """Create an EvalRecord from a finished task run and append it.

        Selection logic:
          • If any RunResult has ``triggered=True`` → pick the first one.
          • Otherwise → pick the result with the highest ``confidence``.
        """

        triggered_runs = [r for r in run_results if r.triggered]
        best: Optional[RunResult] = None

        if triggered_runs:
            best = triggered_runs[0]
        elif run_results:
            best = max(run_results, key=lambda r: getattr(r, "confidence", 0.0))

        # Derive vuln_type from task metadata when available.
        vuln_type = getattr(task, "vulnerability_type", "unknown")
        if vuln_type == "unknown":
            # Fallback: inspect the description for common CWE patterns.
            desc_lower = getattr(task, "vulnerability_description", "").lower()
            for label, keywords in _VULN_HEURISTICS.items():
                if any(k in desc_lower for k in keywords):
                    vuln_type = label
                    break

        rec = EvalRecord(
            task_id=getattr(task, "task_id", "unknown"),
            project_name=getattr(task, "project_name", "unknown"),
            vuln_type=vuln_type,
            strategy_used=getattr(best, "strategy", "none") if best else "none",
            triggered=any(r.triggered for r in run_results),
            crash_type=getattr(best, "crash_type", "") if best else "",
            confidence=getattr(best, "confidence", 0.0) if best else 0.0,
            build_success=build_success,
            time_elapsed=round(elapsed, 2),
            poc_code=getattr(best, "poc_code", "") if best else "",
            notes=_build_notes(run_results),
        )

        self.records.append(rec)

        # Live feedback
        status = _colour("TRIGGERED", _GREEN) if rec.triggered else _colour("MISS", _RED)
        print(f"  [Eval] {rec.task_id}: {status}  "
              f"strategy={rec.strategy_used}  time={rec.time_elapsed:.1f}s")
        return rec

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    def save_report(self) -> None:
        """Write ``results.json`` and ``results.md`` into *output_dir*."""

        # ---- JSON ----
        json_path = self.output_dir / "results.json"
        serialisable = []
        for r in self.records:
            d = asdict(r)
            # Truncate very long PoC blobs so the JSON stays reasonable.
            if len(d.get("poc_code", "")) > 4000:
                d["poc_code"] = d["poc_code"][:4000] + "\n// … truncated …"
            serialisable.append(d)
        json_path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")

        # ---- Markdown ----
        md_path = self.output_dir / "results.md"
        lines = [
            "# CRS Evaluation Results",
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "| task_id | project | vuln_type | triggered | strategy | crash_type | time (s) |",
            "|---------|---------|-----------|-----------|----------|------------|----------|",
        ]
        for r in self.records:
            trig = "✅" if r.triggered else "❌"
            lines.append(
                f"| {r.task_id} | {r.project_name} | {r.vuln_type} "
                f"| {trig} | {r.strategy_used} | {r.crash_type} | {r.time_elapsed:.1f} |"
            )

        triggered_count = sum(1 for r in self.records if r.triggered)
        total = len(self.records)
        pct = (triggered_count / total * 100) if total else 0
        lines += [
            "",
            f"**Summary**: {triggered_count} / {total} tasks triggered ({pct:.1f}%)",
        ]
        md_path.write_text("\n".join(lines), encoding="utf-8")

        print(f"\n  [Report] JSON  → {json_path}")
        print(f"  [Report] MD    → {md_path}")

    # ------------------------------------------------------------------ #
    #  Console summary                                                    #
    # ------------------------------------------------------------------ #

    def print_summary(self) -> None:
        """Pretty-print a coloured console table."""

        if not self.records:
            print("\n  No tasks were evaluated.")
            return

        # Header
        header = (
            f"{'task_id':<20} {'project':<20} {'vuln_type':<22} "
            f"{'status':<14} {'strategy':<14} {'crash':<26} {'time':>7}"
        )
        sep = "─" * len(header)

        print(f"\n{_colour(sep, _CYAN)}")
        print(_colour(f"  CRS EVALUATION SUMMARY", _BOLD))
        print(_colour(sep, _CYAN))
        print(f"  {_colour(header, _BOLD)}")
        print(f"  {sep}")

        for r in self.records:
            if r.triggered:
                status = _colour("TRIGGERED", _GREEN)
            elif not r.build_success:
                status = _colour("BUILD FAIL", _YELLOW)
            else:
                status = _colour("MISS", _RED)

            row = (
                f"  {r.task_id:<20} {r.project_name:<20} {r.vuln_type:<22} "
                f"{status:<24} {r.strategy_used:<14} "        # 24 accounts for ANSI
                f"{r.crash_type:<26} {r.time_elapsed:>6.1f}s"
            )
            print(row)

        # Footer totals
        triggered_count = sum(1 for r in self.records if r.triggered)
        build_fail = sum(1 for r in self.records if not r.build_success)
        total = len(self.records)
        pct = (triggered_count / total * 100) if total else 0

        print(f"  {sep}")
        print(
            f"  {_colour('Triggered', _GREEN)}: {triggered_count}/{total} ({pct:.1f}%)   "
            f"{_colour('Build failures', _YELLOW)}: {build_fail}   "
            f"Total time: {sum(r.time_elapsed for r in self.records):.1f}s"
        )
        print(_colour(sep, _CYAN))


# ── Private helpers ───────────────────────────────────────────────────────

_VULN_HEURISTICS: dict[str, list[str]] = {
    "heap-buffer-overflow":  ["heap-buffer-overflow", "heap buffer overflow"],
    "stack-buffer-overflow": ["stack-buffer-overflow", "stack buffer overflow"],
    "use-after-free":        ["use-after-free", "use after free"],
    "null-deref":            ["null pointer", "null dereference", "nullptr"],
    "integer-overflow":      ["integer overflow", "int overflow"],
    "out-of-bounds-read":    ["out-of-bounds read", "oob read"],
    "out-of-bounds-write":   ["out-of-bounds write", "oob write"],
    "double-free":           ["double free", "double-free"],
    "uninitialized-memory":  ["uninitialized", "uninitialised"],
    "divide-by-zero":        ["divide by zero", "division by zero"],
}


def _build_notes(run_results: List[RunResult]) -> str:
    """Summarise what happened across all run results."""
    parts: list[str] = []
    for i, r in enumerate(run_results):
        tag = "TRIG" if r.triggered else "miss"
        strat = getattr(r, "strategy", "?")
        stderr_snip = getattr(r, "stderr", "")[:120].replace("\n", " ")
        parts.append(f"[{i}] {tag} strategy={strat} stderr={stderr_snip!r}")
    return " | ".join(parts)
