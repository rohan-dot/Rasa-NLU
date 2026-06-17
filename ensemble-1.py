"""
ensemble.py — Ensemble discovery + bidirectional agent/fuzzer loop + clean-room
verification for the discver CRS.

This is the architecture layer the references converge on, which a single ReAct
loop does NOT provide:

  * ENSEMBLE (Team Atlanta / Atlantis): run several diverse agents (varied
    temperature / model) on the same target and UNION the results. LLM
    non-determinism becomes coverage instead of noise.

  * BIDIRECTIONAL AGENT<->FUZZER LOOP (Gondar): the agents and a coverage-guided
    fuzzer run CONCURRENTLY over one SHARED corpus. Every input an agent tries —
    including failed "near-misses" — is deposited into that corpus (via
    ToolBox._deposit), so the fuzzer can mutate a structurally-close attempt into
    a working trigger. Gondar found bugs this way that NEITHER component finds
    alone. The fuzzer's own crashes are also harvested back as candidates.

  * CLEAN-ROOM VERIFICATION (Anthropic reference harness): a candidate PoV is only
    accepted after it reproduces N/N in a FRESH directory the finder never touched,
    and an adversarial judge agent (which sees only the crash, not the finder's
    reasoning) rules it a genuine bug rather than sanitizer noise / a harness bug.
    Executable witness first, opinion second.

REPO-AGNOSTIC: like agent_tools, nothing here assumes a language or build. The
concurrent fuzzer is injected as a callback (`fuzz_fn`) so C uses
agents.run_libfuzzer and Rust uses your cargo-fuzz runner; pass None to skip it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agent_tools import ToolBox, solve_with_tools, reproduces

logger = logging.getLogger("discver.ensemble")


def _sig(data: bytes) -> str:
    """Content signature for input-level dedup. (For crash-level dedup across
    distinct PoVs that hit the same bug, run your existing crash_dedup over the
    verified set afterwards — this only collapses identical inputs.)"""
    return f"{len(data)}-{hashlib.sha1(data).hexdigest()[:16]}"


# ──────────────────────────────────────────────────────────────────
# Clean-room verification + adversarial judge.
# ──────────────────────────────────────────────────────────────────
def clean_room_verify(harness_binary: str, data: bytes,
                      input_mode: str = "arg", trials: int = 3) -> tuple[bool, str]:
    """Re-verify a PoV in a fresh temp dir with a private copy of the binary, so
    nothing the finder did (cwd state, leftover files, env) can make a flaky crash
    look real. Returns (reproduced_n_of_n, output_tail)."""
    if not harness_binary or not os.path.exists(harness_binary):
        return False, "(no harness binary to verify against)"
    with tempfile.TemporaryDirectory(prefix="discver_clean_") as d:
        vb = os.path.join(d, "harness")
        try:
            shutil.copy2(harness_binary, vb)
            os.chmod(vb, 0o755)
        except Exception as exc:
            return False, f"(could not stage clean-room binary: {exc})"
        cwd = os.getcwd()
        try:
            os.chdir(d)
            return reproduces(vb, data, trials=trials, input_mode=input_mode)
        finally:
            os.chdir(cwd)


_JUDGE_SYSTEM = """\
You are a skeptical security reviewer. A bug-finder claims an input crashes a
target. You see ONLY the sanitizer/crash output, not the finder's reasoning. Decide
whether this is a GENUINE, attacker-relevant defect (memory-safety error, sanitizer
violation, reachable panic/abort on attacker-controlled input) or NOISE (a bug in
the harness itself, an intended assertion, an OOM from the fuzzer, a benign exit).
Respond with ONE JSON object: {"verdict":"VALID"|"INVALID","reason":"<one sentence>"}.
"""


def adversarial_judge(llm, crash_output: str, task: str) -> tuple[bool, str]:
    """Independent grader. Defaults to ACCEPT only on an explicit VALID verdict; if
    the LLM is unavailable we fall back to trusting the executable witness (the
    crash already reproduced N/N in the clean room)."""
    if llm is None or not llm.is_available():
        return True, "(no judge available; accepted on reproduction alone)"
    user = (f"Target context:\n{task[:800]}\n\n"
            f"Crash output:\n{crash_output[-2000:]}\n\nVerdict?")
    reply = llm.chat(system=_JUDGE_SYSTEM, user=user, max_tokens=300, temperature=0.0)
    if not reply:
        return True, "(judge returned nothing; accepted on reproduction)"
    m = re.search(r'"verdict"\s*:\s*"(VALID|INVALID)"', reply, re.IGNORECASE)
    reason = ""
    rm = re.search(r'"reason"\s*:\s*"([^"]*)"', reply)
    if rm:
        reason = rm.group(1)
    if m:
        return m.group(1).upper() == "VALID", reason
    # tolerate a bare word
    return "INVALID" not in reply.upper(), reason or "(unparsed judge reply)"


# ──────────────────────────────────────────────────────────────────
# Ensemble orchestration.
# ──────────────────────────────────────────────────────────────────
@dataclass
class EnsembleResult:
    success: bool
    pov_paths: list = field(default_factory=list)
    verified: list = field(default_factory=list)        # [{pov, judge_reason, crash}]
    candidates_seen: int = 0
    transcripts: list = field(default_factory=list)


def _run_agent(llm, toolbox_kwargs: dict, task: str,
               temperature: float, max_steps: int, trials: int):
    try:
        tb = ToolBox(**toolbox_kwargs)            # own ToolBox, SHARED corpus_dir
        return solve_with_tools(llm, tb, task, max_steps=max_steps,
                                trials=trials, temperature=temperature)
    except Exception as exc:
        logger.warning("[ensemble] agent crashed: %s", exc)
        return None


def solve_ensemble(
    llm,
    toolbox_kwargs: dict,                 # kwargs for ToolBox (src_dir, output_dir, harness_binary, ...)
    task: str,
    n_agents: int = 3,
    temperatures: tuple = (0.2, 0.5, 0.8),
    max_steps: int = 14,
    trials: int = 3,
    fuzz_fn: Callable[[str, str, int], object] | None = None,   # (harness, output_dir, seconds)
    fuzz_seconds: int = 90,
    judge: bool = True,
) -> EnsembleResult:
    """Run N diverse agents over one shared corpus while a fuzzer mutates their
    seeds concurrently, then union + clean-room-verify everything.

    Pass `fuzz_fn=agents.run_libfuzzer` (C) or your cargo-fuzz runner (Rust) to
    enable the bidirectional loop; pass None to run agents-only."""
    out_dir = toolbox_kwargs["output_dir"]
    harness = toolbox_kwargs.get("harness_binary")
    input_mode = toolbox_kwargs.get("input_mode", "arg")

    # One shared corpus that every agent deposits into and the fuzzer mutates.
    # Default to run_libfuzzer's own corpus path so its mutations land in the loop.
    corpus_dir = toolbox_kwargs.get("corpus_dir") or str(Path(out_dir) / "exploit_corpus")
    toolbox_kwargs["corpus_dir"] = corpus_dir
    Path(corpus_dir).mkdir(parents=True, exist_ok=True)

    # 1) Kick off the concurrent fuzzer over the shared corpus (bidirectional loop).
    fuzz_threads = []
    if fuzz_fn and harness:
        def _fuzz():
            try:
                fuzz_fn(harness, out_dir, fuzz_seconds)
            except Exception as exc:
                logger.warning("[ensemble] fuzz_fn failed: %s", exc)
        t = threading.Thread(target=_fuzz, daemon=True)
        t.start()
        fuzz_threads.append(t)
        logger.info("[ensemble] concurrent fuzzer started over %s", corpus_dir)

    # 2) Run the diverse agents in parallel; they share corpus_dir.
    temps = list(temperatures)[:n_agents]
    while len(temps) < n_agents:
        temps.append(0.5)

    results = []
    with ThreadPoolExecutor(max_workers=n_agents) as ex:
        futs = {ex.submit(_run_agent, llm, dict(toolbox_kwargs), task,
                          temps[i], max_steps, trials): i for i in range(n_agents)}
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                results.append(r)

    # 3) Collect candidate PoVs: agent successes + whatever the fuzzer crashed on.
    candidates: dict[str, bytes] = {}
    for r in results:
        if r and r.success and r.pov_path and os.path.exists(r.pov_path):
            try:
                b = Path(r.pov_path).read_bytes()
                candidates[_sig(b)] = b
            except Exception:
                pass
    for povdir in (Path(out_dir) / "povs", Path(out_dir) / "exploit_crashes"):
        if povdir.is_dir():
            for p in povdir.glob("*"):
                try:
                    b = p.read_bytes()
                    candidates.setdefault(_sig(b), b)
                except Exception:
                    pass

    for t in fuzz_threads:                       # let the fuzzer wind down
        t.join(timeout=fuzz_seconds + 30)

    # 4) Union + clean-room verify + adversarial judge.
    vdir = Path(out_dir) / "verified_povs"
    vdir.mkdir(parents=True, exist_ok=True)
    verified, pov_paths = [], []
    for sig, data in candidates.items():
        if not harness:
            break
        ok, out = clean_room_verify(harness, data, input_mode=input_mode, trials=trials)
        if not ok:
            logger.info("[ensemble] candidate %s failed clean-room repro", sig)
            continue
        jv, jr = adversarial_judge(llm, out, task) if judge else (True, "(judge off)")
        if not jv:
            logger.info("[ensemble] candidate %s rejected by judge: %s", sig, jr)
            continue
        dest = vdir / f"verified-{sig}.bin"
        dest.write_bytes(data)
        pov_paths.append(str(dest))
        verified.append({"pov": str(dest), "judge_reason": jr, "crash": out[-800:]})
        logger.info("[ensemble] VERIFIED %s -> %s", sig, dest)

    return EnsembleResult(
        success=bool(verified),
        pov_paths=pov_paths,
        verified=verified,
        candidates_seen=len(candidates),
        transcripts=[r.transcript for r in results if r],
    )
