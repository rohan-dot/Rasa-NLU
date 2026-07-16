"""
external_analyzers.py — Heavyweight interprocedural static analysis.

Additive layer on top of static_analysis.py. Adds three tools that reach
where flawfinder/cppcheck can't:

  - CodeQL   : interprocedural taint *paths* (source -> sink), SARIF output.
               The path is the real value; we surface it separately.
  - Infer    : interprocedural null-deref / UAF / leak via biabduction.
  - Weggli   : C/C++ security query patterns; no build required.

Design contract (same as static_analysis.py):
  - Never raises. Every tool is wrapped; a missing binary or a bad build
    logs and returns nothing rather than crashing the pipeline.
  - Reuses StaticFinding + findings_to_context from static_analysis so the
    downstream shape is identical — no dataclass drift.

Honest caveats (read before trusting output):
  - CodeQL and Infer need to observe a BUILD. If you don't pass a build
    command (or a prebuilt CodeQL DB / infer-out), they log and skip. This
    is deliberate: a silent guessed build would be worse than no analysis.
  - Weggli is the only one that runs on a bare box with no build.
  - None of this has been run end-to-end in the authoring environment
    (no network / no tool binaries there). Syntax + import are verified;
    the parsers need one real run per tool to confirm against your
    installed versions. Treat first-run output as unconfirmed.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Reuse the exact finding shape + formatter the pipeline already consumes.
from static_analysis import StaticFinding, findings_to_context  # noqa: F401

logger = logging.getLogger("gemma-fuzzer.external")

_TIMEOUT = 900  # heavyweight tools; generous but bounded


# ══════════════════════════════════════════════════════════════════
# CodeQL — interprocedural taint PATHS (the reason to run it at all)
# ══════════════════════════════════════════════════════════════════

@dataclass
class TaintPath:
    """A source->sink data-flow path from CodeQL. This is richer than a
    single-line finding and is what actually helps the scanner reason."""
    rule: str
    cwe: str
    severity: str
    message: str
    steps: list = field(default_factory=list)  # list[(file, line, note)]

    @property
    def source(self):
        return self.steps[0] if self.steps else ("", 0, "")

    @property
    def sink(self):
        return self.steps[-1] if self.steps else ("", 0, "")


def run_codeql(
    src_dir: str,
    codeql_db: str | None = None,
    build_command: str | None = None,
    query_suite: str = "cpp-security-and-quality.qls",
    language: str = "cpp",
) -> tuple[list[StaticFinding], list[TaintPath]]:
    """Run CodeQL. Returns (flat findings, taint paths).

    Resolution order:
      1. If codeql_db is given and exists -> analyze it directly (cheapest).
      2. Else if `codeql` on PATH and build_command given -> create a DB
         then analyze.
      3. Else -> log why, return ([], []). Never guesses a build.
    """
    if not shutil.which("codeql"):
        logger.info("[codeql] codeql not on PATH — skipping.")
        return [], []

    tmp_db = None
    try:
        if codeql_db and Path(codeql_db).exists():
            db = codeql_db
        elif build_command:
            tmp_db = tempfile.mkdtemp(prefix="codeql_db_")
            logger.info("[codeql] Creating DB (build=%r)...", build_command)
            proc = subprocess.run(
                ["codeql", "database", "create", tmp_db,
                 f"--language={language}", f"--command={build_command}",
                 f"--source-root={src_dir}", "--overwrite"],
                capture_output=True, text=True, timeout=_TIMEOUT,
            )
            if proc.returncode != 0:
                logger.warning("[codeql] DB create failed: %s",
                               (proc.stderr or "")[:300])
                return [], []
            db = tmp_db
        else:
            logger.info("[codeql] No prebuilt DB and no build_command — "
                        "skipping (won't guess a build).")
            return [], []

        sarif = tempfile.mktemp(suffix=".sarif")
        logger.info("[codeql] Analyzing DB with %s...", query_suite)
        proc = subprocess.run(
            ["codeql", "database", "analyze", db, query_suite,
             "--format=sarifv2.1.0", f"--output={sarif}",
             "--rerun", "--threads=0"],
            capture_output=True, text=True, timeout=_TIMEOUT,
        )
        if proc.returncode != 0:
            logger.warning("[codeql] analyze failed: %s",
                           (proc.stderr or "")[:300])
            return [], []

        return _parse_sarif(sarif, src_dir)

    except subprocess.TimeoutExpired:
        logger.warning("[codeql] timed out after %ds.", _TIMEOUT)
        return [], []
    except Exception as exc:
        logger.warning("[codeql] failed: %s", exc)
        return [], []
    finally:
        if tmp_db:
            shutil.rmtree(tmp_db, ignore_errors=True)


def _parse_sarif(sarif_path: str, src_dir: str) -> tuple[list, list]:
    """Parse SARIF v2.1.0 into flat findings + taint paths.

    NOTE: SARIF is stable across CodeQL versions, but rule metadata
    (tags, security-severity) placement drifts a little. Kept defensive.
    """
    try:
        data = json.loads(Path(sarif_path).read_text(errors="replace"))
    except Exception as exc:
        logger.warning("[codeql] could not read SARIF: %s", exc)
        return [], []

    findings: list[StaticFinding] = []
    paths: list[TaintPath] = []

    for run in data.get("runs", []):
        # Build rule_id -> metadata map from driver + extensions.
        rules = {}
        tool = run.get("tool", {})
        for comp in [tool.get("driver", {})] + tool.get("extensions", []):
            for rule in comp.get("rules", []):
                rid = rule.get("id")
                if rid:
                    rules[rid] = rule

        for res in run.get("results", []):
            rid = res.get("ruleId", "")
            rule = rules.get(rid, {})
            cwe = _cwe_from_rule(rule)
            sev = _sev_from_rule(rule, res)
            msg = (res.get("message", {}).get("text", "") or "")[:160]

            floc = _first_location(res)
            findings.append(StaticFinding(
                tool="codeql", file=floc[0], line=floc[1],
                severity=sev, cwe=cwe, function="",
                message=f"{rid}: {msg}" if rid else msg,
            ))

            # Extract taint path(s) from codeFlows, if present.
            for cflow in res.get("codeFlows", []):
                for tflow in cflow.get("threadFlows", []):
                    steps = []
                    for loc in tflow.get("locations", []):
                        pl = (loc.get("location", {})
                                 .get("physicalLocation", {}))
                        art = pl.get("artifactLocation", {}).get("uri", "")
                        ln = pl.get("region", {}).get("startLine", 0)
                        note = (loc.get("location", {})
                                   .get("message", {}).get("text", "") or "")
                        steps.append((_rel(art, src_dir), ln, note[:60]))
                    if steps:
                        paths.append(TaintPath(
                            rule=rid, cwe=cwe, severity=sev,
                            message=msg, steps=steps,
                        ))

    _sort_by_sev(findings)
    paths.sort(key=lambda p: {"high": 0, "medium": 1, "low": 2}
               .get(p.severity, 9))
    logger.info("[codeql] %d findings, %d taint paths.",
                len(findings), len(paths))
    return findings, paths


def taint_paths_to_context(paths: list, max_paths: int = 8) -> str:
    """Format CodeQL taint paths as scanner context. The step chain is the
    high-value part — it hands the LLM a ground-truth data-flow instead of
    asking it to guess one."""
    if not paths:
        return ""
    lines = [f"## CodeQL Interprocedural Taint Paths ({len(paths)} total)\n"]
    for p in paths[:max_paths]:
        src_f, src_l, _ = p.source
        snk_f, snk_l, _ = p.sink
        cwe = f" {p.cwe}" if p.cwe else ""
        lines.append(f"[{p.severity.upper()}]{cwe} {p.rule}: "
                     f"{src_f}:{src_l} → {snk_f}:{snk_l}")
        # Show the intermediate hops (capped) — this is what the scanner
        # can't reconstruct on its own.
        for f, l, note in p.steps[1:-1][:4]:
            suffix = f"  ({note})" if note else ""
            lines.append(f"        ↳ {f}:{l}{suffix}")
    if len(paths) > max_paths:
        lines.append(f"\n... and {len(paths) - max_paths} more paths.")
    return "\n".join(lines)


def _cwe_from_rule(rule: dict) -> str:
    for tag in rule.get("properties", {}).get("tags", []):
        t = tag.lower()
        if "cwe/cwe-" in t:
            num = t.split("cwe-")[-1].strip("/")
            return f"CWE-{num}"
    return ""


def _sev_from_rule(rule: dict, res: dict) -> str:
    props = rule.get("properties", {})
    score = props.get("security-severity")
    if score:
        try:
            s = float(score)
            return "high" if s >= 7.0 else "medium" if s >= 4.0 else "low"
        except ValueError:
            pass
    level = res.get("level", rule.get("defaultConfiguration", {})
                    .get("level", "warning"))
    return {"error": "high", "warning": "medium",
            "note": "low"}.get(level, "medium")


def _first_location(res: dict) -> tuple[str, int]:
    locs = res.get("locations", [])
    if not locs:
        return "", 0
    pl = locs[0].get("physicalLocation", {})
    uri = pl.get("artifactLocation", {}).get("uri", "")
    line = pl.get("region", {}).get("startLine", 0)
    return uri, line


# ══════════════════════════════════════════════════════════════════
# Infer — interprocedural null-deref / UAF / leak
# ══════════════════════════════════════════════════════════════════

# Infer bug_type -> (severity, cwe)
_INFER_MAP = {
    "NULL_DEREFERENCE": ("high", "CWE-476"),
    "NULLPTR_DEREFERENCE": ("high", "CWE-476"),
    "USE_AFTER_FREE": ("high", "CWE-416"),
    "USE_AFTER_DELETE": ("high", "CWE-416"),
    "MEMORY_LEAK": ("medium", "CWE-401"),
    "MEMORY_LEAK_C": ("medium", "CWE-401"),
    "BUFFER_OVERRUN_L1": ("high", "CWE-787"),
    "BUFFER_OVERRUN_L2": ("high", "CWE-787"),
    "BUFFER_OVERRUN_L3": ("medium", "CWE-787"),
    "INTEGER_OVERFLOW_L1": ("high", "CWE-190"),
    "INTEGER_OVERFLOW_L2": ("medium", "CWE-190"),
    "UNINITIALIZED_VALUE": ("medium", "CWE-457"),
    "DEAD_STORE": ("low", ""),
}


def run_infer(
    src_dir: str,
    build_command: str | None = None,
    infer_out: str | None = None,
) -> list[StaticFinding]:
    """Run Infer, or parse an existing infer-out/report.json.

    Like CodeQL: needs to observe a build unless a prior infer-out exists.
    Never guesses a build.
    """
    if not shutil.which("infer") and not infer_out:
        logger.info("[infer] infer not on PATH — skipping.")
        return []

    out_dir = None
    try:
        if infer_out and Path(infer_out).exists():
            report = Path(infer_out) / "report.json"
        elif build_command and shutil.which("infer"):
            out_dir = tempfile.mkdtemp(prefix="infer_out_")
            logger.info("[infer] Running (build=%r)...", build_command)
            proc = subprocess.run(
                ["infer", "run", "-o", out_dir, "--", *build_command.split()],
                capture_output=True, text=True,
                timeout=_TIMEOUT, cwd=src_dir,
            )
            if proc.returncode != 0:
                logger.warning("[infer] run failed: %s",
                               (proc.stderr or "")[:300])
            report = Path(out_dir) / "report.json"
        else:
            logger.info("[infer] No infer-out and no build_command — "
                        "skipping (won't guess a build).")
            return []

        if not report.exists():
            logger.warning("[infer] no report.json produced.")
            return []

        raw = json.loads(report.read_text(errors="replace"))
        findings = []
        for item in raw:
            btype = item.get("bug_type", "")
            sev, cwe = _INFER_MAP.get(btype, ("low", ""))
            findings.append(StaticFinding(
                tool="infer",
                file=item.get("file", ""),
                line=int(item.get("line", 0) or 0),
                severity=sev, cwe=cwe,
                function=item.get("procedure", "") or "",
                message=f"{btype}: {(item.get('qualifier','') or '')[:110]}",
            ))
        _sort_by_sev(findings)
        logger.info("[infer] %d findings.", len(findings))
        return findings

    except subprocess.TimeoutExpired:
        logger.warning("[infer] timed out after %ds.", _TIMEOUT)
        return []
    except Exception as exc:
        logger.warning("[infer] failed: %s", exc)
        return []
    finally:
        if out_dir:
            shutil.rmtree(out_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════
# Weggli — pattern queries, NO build required (runs on a bare box)
# ══════════════════════════════════════════════════════════════════

# (pattern, cwe, severity, human note). Memory-safety oriented.
_WEGGLI_QUERIES = [
    ('{ _ $n = _; memcpy(_,_,$n); }', "CWE-787", "medium",
     "memcpy size from local, no bound seen"),
    ('{ strcpy(_,_); }', "CWE-120", "high", "strcpy (unbounded)"),
    ('{ strcat(_,_); }', "CWE-120", "high", "strcat (unbounded)"),
    ('{ sprintf(_,_); }', "CWE-787", "high", "sprintf (unbounded)"),
    ('{ gets(_); }', "CWE-242", "high", "gets() — never safe"),
    ('{ _ *$p = malloc(_); $p->_; }', "CWE-476", "medium",
     "malloc result deref without NULL check"),
    ('{ $x = alloca(_); }', "CWE-770", "medium",
     "alloca with computed size"),
    ('{ _ $n = strlen(_); $t[$n] = _; }', "CWE-787", "medium",
     "index at strlen (off-by-one risk)"),
]


def run_weggli(src_dir: str, max_hits_per_query: int = 40) -> list[StaticFinding]:
    """Run each weggli pattern. No build needed — this is the tool that
    actually works on a bare standalone box."""
    if not shutil.which("weggli"):
        logger.info("[weggli] weggli not on PATH — skipping.")
        return []

    findings: list[StaticFinding] = []
    for pattern, cwe, sev, note in _WEGGLI_QUERIES:
        try:
            proc = subprocess.run(
                ["weggli", "-C", pattern, src_dir],
                capture_output=True, text=True, timeout=180,
            )
        except subprocess.TimeoutExpired:
            logger.info("[weggli] query timed out: %s", pattern[:40])
            continue
        except Exception as exc:
            logger.info("[weggli] query error (%s): %s", pattern[:40], exc)
            continue

        hits = 0
        for f, l in _parse_weggli(proc.stdout):
            findings.append(StaticFinding(
                tool="weggli", file=f, line=l, severity=sev,
                cwe=cwe, function="", message=note,
            ))
            hits += 1
            if hits >= max_hits_per_query:
                break

    _sort_by_sev(findings)
    logger.info("[weggli] %d findings across %d queries.",
                len(findings), len(_WEGGLI_QUERIES))
    return findings


def _parse_weggli(stdout: str) -> list[tuple[str, int]]:
    """weggli prints 'path:line' header lines before each match snippet.
    Parse those; ignore the snippet body."""
    out = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        # header looks like:  /abs/path/file.c:123:
        if ":" not in line:
            continue
        parts = line.rsplit(":", 2)
        if len(parts) >= 2 and parts[-2].isdigit() and \
                (parts[0].endswith((".c", ".h", ".cc", ".cpp", ".cxx"))
                 or "/" in parts[0]):
            try:
                out.append((parts[0], int(parts[-2])))
            except ValueError:
                continue
    return out


# ══════════════════════════════════════════════════════════════════
# Aggregate
# ══════════════════════════════════════════════════════════════════

def run_all_external(
    src_dir: str,
    codeql_db: str | None = None,
    infer_out: str | None = None,
    build_command: str | None = None,
) -> tuple[list[StaticFinding], list[TaintPath]]:
    """Run every available external tool. Returns (findings, codeql_paths).

    Every tool degrades independently — a missing binary or absent build
    just contributes nothing. Weggli will run on a bare box; CodeQL/Infer
    contribute only if given a DB/out-dir or a build command.
    """
    findings: list[StaticFinding] = []
    paths: list[TaintPath] = []

    cq_findings, cq_paths = run_codeql(
        src_dir, codeql_db=codeql_db, build_command=build_command)
    findings.extend(cq_findings)
    paths.extend(cq_paths)

    findings.extend(run_infer(
        src_dir, build_command=build_command, infer_out=infer_out))
    findings.extend(run_weggli(src_dir))

    _sort_by_sev(findings)
    logger.info("[external] %d total findings, %d taint paths.",
                len(findings), len(paths))
    return findings, paths


# ── shared helpers ────────────────────────────────────────────────

def _sort_by_sev(findings: list) -> None:
    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: order.get(f.severity, 9))


def _rel(uri: str, src_dir: str) -> str:
    uri = uri.replace("file://", "")
    try:
        return str(Path(uri).relative_to(Path(src_dir)))
    except Exception:
        return uri
