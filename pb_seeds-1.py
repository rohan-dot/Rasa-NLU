"""
pb_seeds.py — property-based seed generation for discver.

Adapted from the PBFuzz generator/parameter-space design.

Idea: instead of asking the LLM for individual seed bytes, ask it once for
  (a) a PARAMETER SPACE  — typed knobs describing the input shape, and
  (b) a GENERATOR        — generate(**params) -> (bytes, used_params)
then sample that space thousands of times locally. One LLM call becomes
many structurally-valid, constraint-guided inputs.

Wire-up: seed_generator.py calls generate_seeds(...) below and writes the
returned bytes into the runner's seed dir (the one DISCVER_WIRE_SEEDS fixed).

Flag: DISCVER_PB_SEEDS (default off).
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("discver.pb_seeds")

ENABLED = os.environ.get("DISCVER_PB_SEEDS", "").lower() in ("1", "true", "yes")
GEN_TIMEOUT_SEC = float(os.environ.get("DISCVER_PB_GEN_TIMEOUT", "5"))

# Integer boundaries worth hitting — these are where bugs live.
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1
UINT32_MAX, UINT64_MAX = 2**32 - 1, 2**64 - 1
HUGE_POS, HUGE_NEG = 1e308, -1e308
TINY_POS, TINY_NEG = 1e-308, -1e-308
MACHINE_EPS = 2.220446049250313e-16


# --------------------------------------------------------------------------
# Parameter space validation
# --------------------------------------------------------------------------

VALID_TYPES = {"int_range", "float_range", "categorical", "bool",
               "segments", "base_seed"}


def validate_param_space(space: dict[str, Any]) -> list[str]:
    """Return a list of problems; empty list means valid.

    Kept permissive on purpose: a weak model will produce slightly-off
    specs, and we would rather repair than reject the whole space.
    """
    problems: list[str] = []
    if not isinstance(space, dict):
        return ["parameter space must be a JSON object"]

    for name, spec in space.items():
        if not isinstance(spec, dict) or "type" not in spec:
            problems.append(f"{name}: must be an object with a 'type' field")
            continue
        t = spec["type"]
        if t not in VALID_TYPES:
            problems.append(f"{name}: unknown type {t!r} "
                            f"(expected one of {sorted(VALID_TYPES)})")
        elif t in ("int_range", "float_range"):
            if "min" not in spec or "max" not in spec:
                problems.append(f"{name}: {t} needs 'min' and 'max'")
            elif spec["max"] < spec["min"]:
                problems.append(f"{name}: max < min")
        elif t == "categorical":
            if not spec.get("values"):
                problems.append(f"{name}: categorical needs a non-empty 'values' list")
        elif t == "segments":
            cr = spec.get("count_range")
            if not isinstance(cr, dict) or "min" not in cr or "max" not in cr:
                problems.append(f"{name}: segments needs count_range {{min,max}}")
        elif t == "base_seed":
            p = spec.get("seed_file_path")
            if not p or not os.path.isfile(p):
                problems.append(f"{name}: base_seed path missing or unreadable: {p!r}")
    return problems


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def sample_from_space(space: dict[str, Any], seed: int) -> dict[str, Any]:
    """Draw one concrete parameter set.

    Boundary bias escalates with the iteration number: early iterations are
    mostly uniform (explore the space), later ones increasingly favour edge
    values, which is where memory-safety bugs actually sit.
    """
    random.seed(seed)
    params: dict[str, Any] = {"seed": seed}
    edge_prob = min(0.01 * seed, 0.6)

    def pick_int(spec):
        lo, hi = int(spec.get("min", 0)), int(spec.get("max", 100))
        if lo > hi:
            lo, hi = hi, lo
        cands = [lo, hi, lo + 1 if lo < hi else lo, hi - 1 if lo < hi else hi,
                 -1, 0, 1, INT32_MIN, INT32_MAX, INT64_MIN, INT64_MAX,
                 UINT32_MAX, UINT64_MAX]
        cands = [c for c in cands if lo <= c <= hi]
        if cands and random.random() < edge_prob:
            return random.choice(cands)
        return random.randint(lo, hi)

    def pick_float(spec):
        lo, hi = float(spec.get("min", 0.0)), float(spec.get("max", 1.0))
        if lo > hi:
            lo, hi = hi, lo
        span = hi - lo
        near_lo = lo + max(span * 1e-12, MACHINE_EPS) if span > 0 else lo
        near_hi = hi - max(span * 1e-12, MACHINE_EPS) if span > 0 else hi
        cands = [lo, hi, near_lo, near_hi, -0.0, 0.0,
                 HUGE_NEG, HUGE_POS, TINY_NEG, TINY_POS]
        cands = [c for c in cands if lo <= c <= hi]
        if cands and random.random() < edge_prob:
            return random.choice(cands)
        return random.uniform(lo, hi)

    def pick_categorical(spec):
        values = spec.get("values") or ["default"]
        return random.choice(values)

    def pick_segments(spec):
        cr = spec.get("count_range", {"min": 0, "max": 4})
        n = random.randint(int(cr.get("min", 0)), int(cr.get("max", 4)))
        inner = spec.get("segment_params", {}) or {}
        return [{k: _pick(v) for k, v in inner.items()} for _ in range(n)]

    def _pick(spec):
        if not isinstance(spec, dict):
            return spec                      # already concrete
        t = spec.get("type")
        if t == "int_range":
            return pick_int(spec)
        if t == "float_range":
            return pick_float(spec)
        if t == "categorical":
            return pick_categorical(spec)
        if t == "bool":
            return random.choice([True, False])
        if t == "segments":
            return pick_segments(spec)
        if t == "base_seed":
            return spec                      # generator reads the file itself
        return spec

    for name, spec in (space or {}).items():
        params[name] = _pick(spec)
    return params


# --------------------------------------------------------------------------
# Generator loading + bounded execution
# --------------------------------------------------------------------------

def load_generator(code: str) -> Callable:
    """exec() the model-written generator and hand back its generate().

    The generator is model-authored code executed in-process. It runs with
    this process's privileges — acceptable because discver already executes
    model-directed commands, but do not point this at untrusted generators.
    """
    ns: dict[str, Any] = {}
    preamble = "import random, struct, os, io, json, math, base64\n"
    exec(compile(preamble + code, "<pb_generator>", "exec"), ns)  # noqa: S102
    fn = ns.get("generate")
    if not callable(fn):
        raise ValueError("generator code defines no callable generate()")
    return fn


def _call_with_timeout(fn: Callable, timeout: float, **params):
    """Run the generator on a worker thread so an infinite loop can't wedge
    the whole seed phase. Returns (payload, used_params)."""
    box: dict[str, Any] = {}

    def target():
        try:
            box["out"] = fn(**params)
        except BaseException as e:  # noqa: BLE001 - report, don't crash the run
            box["err"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"generator exceeded {timeout}s")
    if "err" in box:
        raise box["err"]
    out = box.get("out")
    if isinstance(out, tuple) and len(out) == 2:
        payload, used = out
    else:                                     # tolerate bare-bytes generators
        payload, used = out, dict(params)
    if isinstance(payload, str):
        payload = payload.encode("utf-8", errors="ignore")
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(f"generate() returned {type(payload).__name__}, want bytes")
    return bytes(payload), used


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

@dataclass
class SeedBatch:
    seeds: list[bytes] = field(default_factory=list)
    used_params: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    generated: int = 0
    duplicates: int = 0
    failures: int = 0

    def summary(self) -> str:
        return (f"pb_seeds: generated={self.generated} "
                f"duplicates={self.duplicates} failures={self.failures}")


def generate_seeds(generator_code: str,
                   param_space: dict[str, Any],
                   concrete_params: list[dict] | None = None,
                   max_iters: int = 200) -> SeedBatch:
    """Two-phase seed generation.

    Phase 1 runs the model's concrete guesses (its hypothesised triggering
    configurations). Phase 2 samples the parameter space with escalating
    boundary bias. Phase 1 first matters: if the model reasoned correctly,
    the answer is found in a handful of runs instead of hundreds.
    """
    batch = SeedBatch()

    problems = validate_param_space(param_space)
    if problems:
        # Report but continue: a partly-valid space still generates seeds,
        # and the caller can feed these problems back for a repair round.
        batch.errors.extend(problems)
        log.warning("parameter space problems: %s", "; ".join(problems))

    try:
        gen = load_generator(generator_code)
    except Exception as e:  # noqa: BLE001
        batch.errors.append(f"generator load failed: {e!r}")
        log.error("generator load failed: %r", e)
        return batch

    seen: set[str] = set()

    def _emit(params: dict, phase: str) -> None:
        try:
            payload, used = _call_with_timeout(gen, GEN_TIMEOUT_SEC, **params)
        except Exception as e:  # noqa: BLE001
            batch.failures += 1
            batch.errors.append(f"{phase} {params.get('seed')}: {e!r}")
            return
        digest = payload[:4096]
        key = digest.hex()
        if key in seen:
            batch.duplicates += 1
            return
        seen.add(key)
        batch.seeds.append(payload)
        batch.used_params.append(used)
        batch.generated += 1

    # Phase 1 — the model's concrete hypotheses.
    for i, cp in enumerate(concrete_params or []):
        p = dict(cp)
        p.setdefault("seed", i)
        _emit(p, "batch")

    # Phase 2 — sample the space.
    start = len(concrete_params or [])
    for it in range(start, max_iters):
        params = sample_from_space(param_space, seed=it)
        key = json.dumps(params, sort_keys=True, default=str)
        if key in seen:
            batch.duplicates += 1
            continue
        seen.add(key)
        _emit(params, "sample")

    log.info(batch.summary())
    return batch


def write_seeds(batch: SeedBatch, seed_dir: str) -> int:
    """Write seeds into the runner's corpus dir. Returns count written."""
    os.makedirs(seed_dir, exist_ok=True)
    n = 0
    for i, payload in enumerate(batch.seeds):
        with open(os.path.join(seed_dir, f"pb-{i:05d}"), "wb") as fh:
            fh.write(payload)
        n += 1
    log.info("pb_seeds: wrote %d seeds to %s", n, seed_dir)
    return n
