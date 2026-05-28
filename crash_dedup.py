"""
crash_dedup.py — Crash deduplication by stack signature.

LibFuzzer often finds the same bug 50 times. Reporting 50 "crashes"
is noise. We group crashes by their signature (crash type + top stack
frames) so we report N unique bugs, not N hundred duplicates.

Signature = sanitizer error type + the top 2-3 non-library stack frames.
Two crashes with the same signature are the same bug.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("gemma-fuzzer.dedup")


@dataclass
class UniqueCrash:
    signature: str
    crash_type: str        # "heap-buffer-overflow", "SEGV", etc.
    top_frame: str         # the function where it crashed
    location: str          # file:line if available
    example_input: str     # path to one input that triggers it
    count: int = 1         # how many inputs hit this same bug
    all_inputs: list = field(default_factory=list)


# Sanitizer error patterns
ERROR_TYPES = [
    "heap-buffer-overflow", "stack-buffer-overflow", "global-buffer-overflow",
    "heap-use-after-free", "stack-use-after-return", "use-after-poison",
    "double-free", "alloc-dealloc-mismatch", "memory-leak",
    "SEGV", "stack-overflow", "FPE", "null-deref",
    "negative-size-param", "calloc-overflow", "out-of-memory",
    "undefined-behavior", "integer-overflow", "shift-exponent",
]


def deduplicate(crash_inputs: list[str], harness_binary: str,
                output_dir: str) -> list[UniqueCrash]:
    """Run each crash input, extract signature, group duplicates."""
    if not crash_inputs or not os.path.exists(harness_binary):
        return []

    by_signature: dict[str, UniqueCrash] = {}

    for inp in crash_inputs:
        if not os.path.exists(inp):
            continue

        crash_type, top_frame, location = _get_crash_signature(harness_binary, inp)
        if not crash_type:
            continue

        # Signature = type + top frame (the bug identity)
        sig_raw = f"{crash_type}|{top_frame}"
        sig = hashlib.sha256(sig_raw.encode()).hexdigest()[:12]

        if sig in by_signature:
            by_signature[sig].count += 1
            by_signature[sig].all_inputs.append(inp)
        else:
            by_signature[sig] = UniqueCrash(
                signature=sig,
                crash_type=crash_type,
                top_frame=top_frame,
                location=location,
                example_input=inp,
                count=1,
                all_inputs=[inp],
            )

    unique = list(by_signature.values())
    unique.sort(key=lambda c: -c.count)

    logger.info("[dedup] %d crash inputs → %d unique bugs.",
                len(crash_inputs), len(unique))
    for c in unique:
        logger.info("[dedup]   %s in %s (%d inputs) — %s",
                   c.crash_type, c.top_frame, c.count, c.location)

    return unique


def _get_crash_signature(harness_binary: str, crash_input: str) -> tuple:
    """Run the input, parse ASAN output for type + crash location."""
    env = os.environ.copy()
    env["ASAN_OPTIONS"] = "abort_on_error=1:detect_leaks=0:symbolize=1"

    try:
        result = subprocess.run(
            [harness_binary, crash_input],
            capture_output=True, timeout=15, env=env,
        )
        output = (result.stderr or b"").decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return ("timeout", "unknown", "")
    except Exception:
        return ("", "", "")

    # Crash type
    crash_type = "unknown-crash"
    for et in ERROR_TYPES:
        if et in output:
            crash_type = et
            break
    if "AddressSanitizer" not in output and "ERROR" not in output and "SUMMARY" not in output:
        if result.returncode == 0:
            return ("", "", "")  # no crash

    # Top non-library stack frame (the bug's location)
    top_frame = "unknown"
    location = ""
    # ASAN frames: "    #0 0x... in function_name /path/file.c:line"
    frame_re = re.compile(r'#\d+\s+0x[0-9a-f]+\s+in\s+(\S+)\s+([^\s:]+:\d+)?')
    for m in frame_re.finditer(output):
        func = m.group(1)
        loc = m.group(2) or ""
        # Skip sanitizer/libc internal frames
        if any(s in func.lower() for s in [
            "asan", "__interceptor", "sanitizer", "libfuzzer",
            "llvmfuzzer", "__libc", "fuzzer::"
        ]):
            continue
        top_frame = func
        location = loc
        break

    return (crash_type, top_frame, location)


def save_unique_crashes(unique: list[UniqueCrash], output_dir: str):
    """Write a deduplicated crash report."""
    import json

    report_dir = Path(output_dir) / "unique_bugs"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for c in unique:
        summary.append({
            "signature": c.signature,
            "crash_type": c.crash_type,
            "function": c.top_frame,
            "location": c.location,
            "duplicate_count": c.count,
            "example_input": c.example_input,
        })

    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("[dedup] Wrote %d unique bugs to %s/summary.json",
                len(unique), report_dir)
    return str(report_dir / "summary.json")
