"""
introspector_targets.py — Reachability-aware target selection.

Your call graph already scores "dangerous" functions by keyword + caller
count (code_analysis.find_dangerous_functions). That's a static heuristic:
it doesn't know which functions the *harness can actually reach*, nor which
are already covered. Fuzz Introspector does — it computes per-function
reachability from the fuzzer entry point, cyclomatic complexity, and
coverage. Feeding that in lets the scanner spend its budget on functions
that are (a) reachable and (b) not already exercised.

Contract:
  - Consumes an existing Fuzz Introspector run's summary JSON. We do NOT
    invoke introspector here (it's tied to your build/coverage setup).
  - Returns None on ANY parse failure so the caller falls back to the
    existing heuristic. Never raises, never blocks the pipeline.
  - When it succeeds, prefer MERGING with the heuristic (see
    rank_targets) rather than replacing it — reachability and
    bug-density are complementary signals.

⚠ Schema caveat: Fuzz Introspector's summary schema has changed across
releases (summary.json / *.data.yaml / all_functions arrays). The parser
below targets the common "all_functions" list with reachability/coverage
fields and is intentionally defensive. Confirm field names against YOUR
installed version on first run — if fields are missing it returns None,
which is safe but means you silently get no uplift. Log-check the first run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("gemma-fuzzer.introspector")


@dataclass
class ReachInfo:
    name: str
    reachable: bool
    covered: bool
    complexity: int
    reached_by_fuzzers: int


def load_introspector_summary(summary_path: str) -> dict | None:
    """Load a Fuzz Introspector summary JSON into {func_name: ReachInfo}.
    Returns None if the file is missing or the schema isn't recognized."""
    p = Path(summary_path)
    if not p.exists():
        logger.info("[introspector] no summary at %s — skipping.", summary_path)
        return None
    try:
        data = json.loads(p.read_text(errors="replace"))
    except Exception as exc:
        logger.warning("[introspector] unreadable summary: %s", exc)
        return None

    funcs = _extract_function_list(data)
    if not funcs:
        logger.warning("[introspector] schema not recognized — "
                       "falling back to heuristic. Check field names.")
        return None

    out: dict[str, ReachInfo] = {}
    for f in funcs:
        name = f.get("function_name") or f.get("name")
        if not name:
            continue
        cov = f.get("function_coverage") or f.get("coverage") or 0
        try:
            covered = float(str(cov).rstrip("%")) > 0.0
        except (ValueError, TypeError):
            covered = bool(cov)
        reached_by = (f.get("reached_by_fuzzers")
                      or f.get("reached-by-fuzzers") or [])
        n_reached = len(reached_by) if isinstance(reached_by, list) \
            else int(reached_by or 0)
        out[name] = ReachInfo(
            name=name,
            reachable=n_reached > 0 or bool(f.get("is_reachable")),
            covered=covered,
            complexity=int(f.get("cyclomatic_complexity")
                           or f.get("i_count") or 0),
            reached_by_fuzzers=n_reached,
        )
    logger.info("[introspector] loaded reachability for %d functions.",
                len(out))
    return out or None


def _extract_function_list(data) -> list:
    """Find the per-function array across a few known schema shapes."""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in ("all_functions", "function_list", "functions"):
        if isinstance(data.get(key), list):
            return data[key]
    # Some versions nest under a per-fuzzer or "analyses" block.
    for v in data.values():
        if isinstance(v, dict):
            for key in ("all_functions", "function_list", "functions"):
                if isinstance(v.get(key), list):
                    return v[key]
    return []


def rank_targets(
    heuristic_scored: list[tuple[str, int]],
    reach: dict | None,
    max_targets: int = 8,
    drop_covered: bool = True,
) -> list[tuple[str, int]]:
    """Merge the existing heuristic scoring with introspector reachability.

    heuristic_scored: output of call_graph.find_dangerous_functions()
                      -> [(func_name, score), ...] already sorted desc.
    reach:            output of load_introspector_summary(), or None.

    If reach is None, returns heuristic_scored unchanged (pure fallback).
    Otherwise:
      - reachable, not-yet-covered functions get boosted,
      - unreachable functions get demoted (can't be hit anyway),
      - already-covered functions optionally dropped (fuzzer's on them).
    """
    if not reach:
        return heuristic_scored[:max_targets]

    adjusted: list[tuple[str, int]] = []
    for name, score in heuristic_scored:
        info = reach.get(name)
        if info is None:
            # Not in introspector data (e.g. inlined/static) — keep as-is
            # but slightly demote: we can't confirm reachability.
            adjusted.append((name, score - 2))
            continue
        if drop_covered and info.covered and info.reachable:
            # Live fuzzer already exercises it — don't spend scanner budget.
            continue
        boost = 0
        if info.reachable:
            boost += 8              # reachable-from-fuzzer is the key signal
        else:
            boost -= 12             # unreachable: bug here isn't triggerable
        if info.reachable and not info.covered:
            boost += 6              # reachable but uncovered = the sweet spot
        boost += min(info.complexity // 5, 6)  # complex code hides bugs
        adjusted.append((name, score + boost))

    adjusted.sort(key=lambda x: -x[1])
    logger.info("[introspector] merged ranking: %d candidates -> top %d.",
                len(adjusted), min(max_targets, len(adjusted)))
    return adjusted[:max_targets]
