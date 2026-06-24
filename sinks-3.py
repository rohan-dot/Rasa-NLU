"""
sinks.py — Sink enumeration + exploitability filtering for the discver CRS.

Gondar's data says recall comes from the FRONT END: enumerate security-sensitive
call sites ("sinks"), then cut the thousands of candidates down to a few hundred
ACTIONABLE ones with an exploitability-assessment agent — keeping ~96% of the
truly exploitable while dropping the noise. This module is that front end, and it
feeds `target_funcs` straight into ensemble.solve_ensemble (the "reach the sink"
signal).

Pipeline:
    enumerate_sinks(src)            -> all candidate sink call sites (offline, fast)
    filter_exploitable(llm, sinks)  -> ranked, LLM-vetted actionable sinks
    hunt_sinks(...)                 -> drive solve_ensemble at each top sink

REPO-AGNOSTIC: language is detected by extension; sink patterns ship per-language
(C/C++, Rust, Go, Python, JS/TS) and you can extend them or pass your `languages`
plugin. No build assumptions, no hardcoded project. Degrades without an LLM
(falls back to severity-ranked static selection) and without external tools.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("discver.sinks")

# Per-language sink patterns: (compiled regex, label, base severity 0..1).
# These are the standard SAST sink families — memory-unsafe APIs, command/exec,
# deserialization, dynamic eval. Defensive, well-known sets.
_RAW = {
    "c": [
        (r"\bgets\s*\(", "gets", 0.95),
        (r"\b(strcpy|strcat|stpcpy)\s*\(", "strcpy", 0.85),
        (r"\b(sprintf|vsprintf)\s*\(", "sprintf", 0.85),
        (r"\b(memcpy|memmove|bcopy)\s*\(", "memcpy", 0.7),
        (r"\b(strncpy|strncat|snprintf)\s*\(", "bounded-copy", 0.55),
        (r"\b(alloca)\s*\(", "alloca", 0.7),
        (r"\b(malloc|calloc|realloc)\s*\(", "alloc", 0.5),
        (r"\b(system|popen|execl|execlp|execle|execv|execvp|execve)\s*\(", "exec", 0.9),
        (r"\b(scanf|sscanf|fscanf)\s*\(", "scanf", 0.6),
        (r"\b(free)\s*\(", "free", 0.45),
    ],
    "rust": [
        (r"\bunsafe\b", "unsafe-block", 0.6),
        (r"\bget_unchecked(_mut)?\s*\(", "get_unchecked", 0.85),
        (r"\b(from_raw_parts|from_raw_parts_mut)\s*\(", "from_raw_parts", 0.85),
        (r"\b(transmute)\s*\(", "transmute", 0.8),
        (r"\bcopy_nonoverlapping\s*\(", "copy_nonoverlapping", 0.8),
        (r"\.offset\s*\(", "ptr-offset", 0.7),
        (r"\.unwrap\s*\(\s*\)", "unwrap", 0.4),
        (r"\bCommand::new\s*\(", "exec", 0.85),
        (r"\[\s*[A-Za-z_]\w*\s*\]", "index", 0.45),
    ],
    "go": [
        (r"\bunsafe\.Pointer\b", "unsafe-pointer", 0.8),
        (r"\bexec\.Command\s*\(", "exec", 0.9),
        (r"\bos\.Open(File)?\s*\(", "path-open", 0.55),
        (r"\b(Query|Exec)\s*\(\s*\"?.*\+", "sql-concat", 0.8),
    ],
    "python": [
        (r"\beval\s*\(", "eval", 0.9),
        (r"\bexec\s*\(", "exec", 0.9),
        (r"\bos\.system\s*\(", "os.system", 0.9),
        (r"shell\s*=\s*True", "shell-true", 0.85),
        (r"\b(pickle|cPickle)\.loads?\s*\(", "pickle", 0.85),
        (r"\byaml\.load\s*\(", "yaml.load", 0.8),
        (r"\b__import__\s*\(", "dynamic-import", 0.6),
    ],
    "js": [
        (r"\beval\s*\(", "eval", 0.9),
        (r"\bchild_process\b|\bexecSync?\s*\(", "exec", 0.9),
        (r"\bnew Function\s*\(", "new-function", 0.8),
        (r"\bvm\.runIn", "vm-run", 0.8),
    ],
}
_PATTERNS = {lang: [(re.compile(p), lbl, sev) for p, lbl, sev in rows]
             for lang, rows in _RAW.items()}

_EXT_LANG = {
    ".c": "c", ".h": "c", ".cc": "c", ".cpp": "c", ".cxx": "c", ".hpp": "c", ".hh": "c",
    ".rs": "rust", ".go": "go", ".py": "python", ".js": "js", ".ts": "js", ".jsx": "js",
}
_SKIP = (".git", "/node_modules", "/target/debug/", "/target/release/",
         "/fuzz/target/", "/.cargo/", "/vendor/", "/registry/", "/deps/")
#  NOTE: never bare "/target/" — it matches OSS-CRS's /opt/target/src and skips everything.
_DEFRX = {
    "c": re.compile(r"^[A-Za-z_].*?\b([A-Za-z_]\w*)\s*\([^;]*\)\s*\{?\s*$"),
    "rust": re.compile(r"\bfn\s+([A-Za-z_]\w*)"),
    "go": re.compile(r"\bfunc\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)"),
    "python": re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)"),
    "js": re.compile(r"\bfunction\s+([A-Za-z_]\w*)|\b([A-Za-z_]\w*)\s*=\s*(?:async\s*)?\("),
}
_CTRL_KW = {"if", "for", "while", "switch", "return", "sizeof", "do", "else", "case"}


@dataclass
class Sink:
    function: str       # enclosing function — the target to drive coverage toward
    file: str
    line: int
    api: str
    snippet: str
    score: float = 0.0
    why: str = ""


def _iter_source_files(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in _EXT_LANG and not any(s in str(p) for s in _SKIP):
            yield p


def _enclosing_function(lines: list[str], idx: int, lang: str) -> str:
    rx = _DEFRX.get(lang)
    if rx is None:
        return "?"
    for j in range(idx, max(-1, idx - 400), -1):
        m = rx.search(lines[j])
        if not m:
            continue
        name = next((g for g in m.groups() if g), None)
        if not name or name in _CTRL_KW:
            continue
        if lang == "c" and lines[j].rstrip().endswith(";"):
            continue            # a call/declaration, not a definition
        return name
    return "?"


def enumerate_sinks(src_dir: str, plugin=None, extra_patterns=None,
                    max_sinks: int = 400, allowed_files=None) -> list[Sink]:
    """Fast, offline pass: find sink call sites and their enclosing functions.
    `extra_patterns`: optional list of (regex_str, label, severity). `plugin`:
    optional language plugin exposing body_risk_patterns()/prescan_patterns()."""
    root = Path(src_dir).resolve()
    patterns = {lang: list(rows) for lang, rows in _PATTERNS.items()}
    if extra_patterns:
        for ps, lbl, sev in extra_patterns:
            for lang in patterns:
                patterns[lang].append((re.compile(ps), lbl, sev))
    if plugin is not None:
        for meth in ("body_risk_patterns", "prescan_patterns"):
            try:
                for item in (getattr(plugin, meth)() or []):
                    ps, lbl = item[0], (item[1] if len(item) > 1 else "plugin-sink")
                    sev = float(item[2]) / 10 if len(item) > 2 else 0.6
                    lang = getattr(plugin, "name", "c")
                    patterns.setdefault(lang, []).append((re.compile(ps), str(lbl), min(sev, 0.95)))
            except Exception:
                pass

    allowed_norm = None
    if allowed_files:
        allowed_norm = tuple(str(a).replace("\\", "/").lstrip("./")
                             for a in allowed_files if a)

    sinks, seen = [], set()
    for f in _iter_source_files(root):
        lang = _EXT_LANG.get(f.suffix.lower())
        rows = patterns.get(lang)
        if not rows:
            continue
        if allowed_norm is not None:                       # scope to the target's own files
            fp = str(f).replace("\\", "/")
            if not any(fp.endswith(a) for a in allowed_norm):
                continue
        try:
            lines = f.read_text(errors="replace").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines):
            for rx, label, sev in rows:
                if rx.search(line):
                    fn = _enclosing_function(lines, i, lang)
                    key = (str(f), fn, label)
                    if key in seen:
                        continue
                    seen.add(key)
                    sinks.append(Sink(function=fn, file=str(f.relative_to(root)),
                                      line=i + 1, api=label, snippet=line.strip()[:200],
                                      score=sev))
                    if len(sinks) >= max_sinks:
                        logger.info("[sinks] hit max_sinks=%d", max_sinks)
                        return sinks
    logger.info("[sinks] enumerated %d candidate sinks", len(sinks))
    return sinks


_FILTER_SYSTEM = """\
You are a vulnerability triage analyst. You are given candidate "sinks" — security-
sensitive call sites. For each, decide whether an attacker who controls the fuzz
input could PLAUSIBLY drive it into a real bug (memory corruption, command/code
injection, unsafe deserialization). Keep the promising ones, drop the clearly-safe
ones (constant arguments, fully validated, unreachable from input). Be decisive and
brief. Respond ONLY with a JSON array, one object per input index:
[{"i":0,"keep":true,"score":0.0,"why":"one short reason"}]
"""


def filter_exploitable(llm, sinks: list[Sink], src_dir: str = "",
                       call_graph=None, keep: int = 80, batch: int = 15) -> list[Sink]:
    """Exploitability-assessment agent. Vets sinks in batches; keeps the top `keep`
    by model score. Falls back to severity ranking if no LLM is available."""
    if not sinks:
        return []
    if llm is None or not llm.is_available():
        ranked = sorted(sinks, key=lambda s: -s.score)
        for s in ranked:
            s.why = s.why or "(static severity ranking; no LLM)"
        return ranked[:keep]

    vetted: list[Sink] = []
    for start in range(0, len(sinks), batch):
        chunk = sinks[start:start + batch]
        listing = "\n".join(
            f'{k}. api={s.api} func={s.function} {s.file}:{s.line}  >> {s.snippet}'
            for k, s in enumerate(chunk)
        )
        reply = llm.chat(system=_FILTER_SYSTEM, user=listing, max_tokens=1200, temperature=0.1)
        verdicts = _parse_array(reply)
        if not verdicts:
            vetted.extend(chunk)                      # don't lose them if parse fails
            continue
        by_i = {int(v.get("i", -1)): v for v in verdicts if isinstance(v, dict)}
        for k, s in enumerate(chunk):
            v = by_i.get(k)
            if v is None:
                vetted.append(s)
                continue
            if not v.get("keep", True):
                continue
            try:
                s.score = max(s.score, float(v.get("score", s.score)))
            except (TypeError, ValueError):
                pass
            s.why = str(v.get("why", ""))[:160]
            vetted.append(s)
    ranked = sorted(vetted, key=lambda s: -s.score)
    logger.info("[sinks] %d/%d sinks survived exploitability filter", len(ranked), len(sinks))
    return ranked[:keep]


def _parse_array(text: str):
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, list) else None
    except json.JSONDecodeError:
        return None


def hunt_sinks(llm, toolbox_kwargs: dict, src_dir: str,
               plugin=None, n_targets: int = 10, n_agents: int = 3,
               fuzz_fn=None, fuzz_seconds: int = 90, keep: int = 80,
               stop_after: int | None = None, allowed_files=None) -> list[dict]:
    """End-to-end front end: enumerate -> filter -> drive the ensemble at each top
    sink. Returns [{"sink": Sink, "result": EnsembleResult}]. This is the single
    entry point to call from your orchestrator."""
    from ensemble import solve_ensemble        # lazy import to avoid cycles

    sinks = enumerate_sinks(src_dir, plugin=plugin, allowed_files=allowed_files)
    ranked = filter_exploitable(llm, sinks, src_dir=src_dir, keep=keep)
    out = []
    found = 0
    for sink in ranked[:n_targets]:
        tk = dict(toolbox_kwargs)
        tk["target_funcs"] = tuple(set(tk.get("target_funcs", ())) | {sink.function})
        task = (f"Reach and trigger a bug at the {sink.api} sink in function "
                f"`{sink.function}` ({sink.file}:{sink.line}). Code: {sink.snippet}. "
                f"Why suspected: {sink.why or 'security-sensitive sink reachable from input'}.")
        logger.info("[sinks] hunting %s in %s (score %.2f)", sink.api, sink.function, sink.score)
        res = solve_ensemble(llm, tk, task, n_agents=n_agents,
                             fuzz_fn=fuzz_fn, fuzz_seconds=fuzz_seconds)
        out.append({"sink": sink, "result": res})
        if res.success:
            found += 1
            if stop_after and found >= stop_after:
                break
    logger.info("[sinks] hunt complete: %d sinks driven, %d produced verified PoVs",
                len(out), found)
    return out
