"""
CRS Step 6 — Optional Fuzzing Module
=====================================

Last-resort PoC generation via lightweight fuzzing.  Checks for AFL++ or
libFuzzer availability, generates harnesses/seeds with the LLM, and runs
a time-boxed fuzzing session.

The module is **fully optional**: every public function returns ``None``
(or a graceful dict) when no fuzzer is found or when fuzzing is disabled
via ``config.FUZZING_ENABLED``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from crs import config
from crs.types import (
    BuildResult,
    CodeContext,
    CyberGymTask,
    LLMRouter,
    PoCResult,
)

logger = logging.getLogger(__name__)

# ── helpers ────────────────────────────────────────────────────────────────

_SUBPROCESS_DEFAULTS: dict[str, Any] = dict(
    capture_output=True,
    text=True,
)


def _run(cmd: list[str], *, timeout: int = 30, cwd: Path | None = None,
         env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Thin wrapper around ``subprocess.run`` that always enforces a timeout."""
    merged_env = {**os.environ, **(env or {})}
    return subprocess.run(
        cmd,
        **_SUBPROCESS_DEFAULTS,
        timeout=timeout,
        cwd=cwd,
        env=merged_env,
    )


def _task_workdir(task_id: str) -> Path:
    """Return (and create) the per-task working directory."""
    d = config.WORK_DIR / task_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ═══════════════════════════════════════════════════════════════════════════
# 1. Availability Check
# ═══════════════════════════════════════════════════════════════════════════

def check_fuzzer_availability() -> dict[str, bool]:
    """Probe the host for AFL++ and/or libFuzzer support.

    Returns
    -------
    dict with keys ``"afl++"``, ``"libfuzzer"``, ``"available"``.
    """
    has_afl = shutil.which("afl-fuzz") is not None

    # libFuzzer needs clang that supports -fsanitize=fuzzer.
    has_libfuzzer = False
    clang_path = shutil.which("clang")
    if clang_path:
        try:
            with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as tmp:
                tmp.write(
                    '#include <stdint.h>\n#include <stddef.h>\n'
                    'int LLVMFuzzerTestOneInput(const uint8_t *d, size_t s)'
                    '{ return 0; }\n'
                )
                tmp_path = tmp.name
            out_path = tmp_path + ".bin"
            result = _run(
                [clang_path, "-fsanitize=fuzzer,address", "-o", out_path, tmp_path],
                timeout=30,
            )
            if result.returncode == 0 and Path(out_path).exists():
                has_libfuzzer = True
                Path(out_path).unlink(missing_ok=True)
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.debug("libFuzzer probe failed: %s", exc)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return {
        "afl++": has_afl,
        "libfuzzer": has_libfuzzer,
        "available": has_afl or has_libfuzzer,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Fuzzing Harness Generator (libFuzzer)
# ═══════════════════════════════════════════════════════════════════════════

_HARNESS_PROMPT_TEMPLATE = """\
You are a security researcher writing a libFuzzer harness.

Vulnerability description
-------------------------
{description}

Relevant code snippets
----------------------
{top_snippets}

Task
----
Write a C function with signature:
  int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {{ ... }}
that feeds the fuzzer-controlled bytes into the parsing / processing function
of the codebase that is most likely related to this vulnerability.

Rules:
  • Include any necessary headers.
  • Keep it under 50 lines.
  • Return 0 at the end.
  • Do NOT call exit() or abort() inside the harness.
  • Output ONLY the C source code — no markdown fences, no commentary.
"""


def generate_fuzzing_harness(context: CodeContext, router: LLMRouter) -> str:
    """Ask the LLM for a libFuzzer harness and persist it to disk.

    Returns
    -------
    The harness C source code as a string.
    """
    snippets_text = "\n---\n".join(
        f"// {s.filepath}:{s.start_line}-{s.end_line}\n{s.content}"
        for s in context.top_snippets[:6]
    ) or "(no snippets available)"

    prompt = _HARNESS_PROMPT_TEMPLATE.format(
        description=context.task.description,
        top_snippets=snippets_text,
    )

    harness_code = router.query(prompt, max_tokens=2048, temperature=0.3)

    # Strip markdown fences the LLM might emit despite instructions.
    harness_code = _strip_code_fences(harness_code)

    # Persist
    workdir = _task_workdir(context.task.task_id)
    harness_path = workdir / "fuzzer_harness.c"
    harness_path.write_text(harness_code, encoding="utf-8")
    logger.info("Harness written to %s (%d lines)", harness_path,
                harness_code.count("\n") + 1)

    return harness_code


def _strip_code_fences(code: str) -> str:
    """Remove optional ```c … ``` wrappers."""
    lines = code.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 3. LibFuzzer Runner
# ═══════════════════════════════════════════════════════════════════════════

_CRASH_TO_POC_PROMPT = """\
I have a crashing input file (raw bytes) that triggers a vulnerability when
passed to the function below.  Convert this into a standalone C program that:
  1. reads the crash file path from argv[1],
  2. passes its contents to the same vulnerable function,
  3. can be compiled with: gcc -o poc poc.c

Harness that found the crash:
{harness_code}

Crash file size: {crash_size} bytes (first 64 bytes hex): {crash_hex}

Output ONLY the C source code — no markdown fences, no commentary.
"""


def run_libfuzzer(
    harness_path: Path,
    build_result: BuildResult,
    task: CyberGymTask,
    timeout_seconds: int = 120,
    *,
    router: LLMRouter | None = None,
) -> PoCResult | None:
    """Compile the harness with libFuzzer, run it, and return a PoC if a
    crash is found.

    Parameters
    ----------
    harness_path
        Path to the ``fuzzer_harness.c`` written by :func:`generate_fuzzing_harness`.
    build_result
        Compilation artefacts from Step 5.
    task
        The current CyberGym task descriptor.
    timeout_seconds
        Wall-clock budget for the fuzzing run.
    router
        LLM router (used to convert a crash file into a standalone PoC).

    Returns ``None`` on any failure.
    """
    workdir = _task_workdir(task.task_id)
    crashes_dir = workdir / "crashes"
    crashes_dir.mkdir(exist_ok=True)
    corpus_dir = workdir / "libfuzzer_corpus"
    corpus_dir.mkdir(exist_ok=True)

    fuzzer_bin = workdir / "fuzzer_bin"

    # ── compile ────────────────────────────────────────────────────────
    include_flags: list[str] = []
    for inc in build_result.include_dirs:
        include_flags += ["-I", inc]

    lib_flags: list[str] = []
    for lp in build_result.lib_paths:
        lib_flags += ["-L", lp]

    # Link object files from the project build so the harness can call
    # into the actual codebase.
    obj_flags: list[str] = list(build_result.object_files)

    compile_cmd = [
        "clang",
        "-fsanitize=fuzzer,address,undefined",
        "-g", "-O1",
        str(harness_path),
        *include_flags,
        *lib_flags,
        *obj_flags,
        "-o", str(fuzzer_bin),
    ]

    logger.info("Compiling libFuzzer harness: %s", " ".join(compile_cmd))
    try:
        comp = _run(compile_cmd, timeout=60, cwd=workdir)
    except subprocess.TimeoutExpired:
        logger.warning("libFuzzer harness compilation timed out")
        return None

    if comp.returncode != 0:
        logger.warning("libFuzzer harness compilation failed:\n%s\n%s",
                        comp.stdout, comp.stderr)
        return None

    # ── run ────────────────────────────────────────────────────────────
    print(f"Starting libfuzzer for {task.task_id} ({timeout_seconds}s budget)...")
    logger.info("Running libFuzzer: %s", fuzzer_bin)

    run_cmd = [
        str(fuzzer_bin),
        str(corpus_dir),
        f"-max_total_time={timeout_seconds}",
        f"-artifact_prefix={crashes_dir}/",
    ]

    try:
        fuzz_run = _run(run_cmd, timeout=timeout_seconds + 30, cwd=workdir)
    except subprocess.TimeoutExpired:
        logger.info("libFuzzer hit external timeout — checking for crashes anyway")

    # ── harvest crashes ────────────────────────────────────────────────
    crash_files = sorted(
        p for p in crashes_dir.iterdir()
        if p.is_file() and p.stat().st_size > 0
    )

    if not crash_files:
        logger.info("libFuzzer finished — no crashes found")
        return None

    crash_file = crash_files[0]
    logger.info("libFuzzer crash found: %s (%d bytes)", crash_file,
                crash_file.stat().st_size)

    return _crash_to_poc(
        crash_file=crash_file,
        harness_path=harness_path,
        task=task,
        strategy_name="libfuzzer",
        router=router,
    )


def _crash_to_poc(
    crash_file: Path,
    harness_path: Path,
    task: CyberGymTask,
    strategy_name: str,
    router: LLMRouter | None,
) -> PoCResult:
    """Convert a raw crash file into a ``PoCResult``.

    If an LLM router is available the crash is turned into a standalone C
    program; otherwise the raw crash file is returned directly.
    """
    crash_bytes = crash_file.read_bytes()
    crash_hex = crash_bytes[:64].hex()
    harness_code = harness_path.read_text(encoding="utf-8")
    workdir = _task_workdir(task.task_id)

    poc_code: str
    if router is not None:
        prompt = _CRASH_TO_POC_PROMPT.format(
            harness_code=harness_code,
            crash_size=len(crash_bytes),
            crash_hex=crash_hex,
        )
        poc_code = _strip_code_fences(
            router.query(prompt, max_tokens=2048, temperature=0.2)
        )
    else:
        # Fallback: a minimal driver that reads the crash file.
        poc_code = _minimal_file_reader_poc(harness_code)

    poc_path = workdir / "poc_from_fuzzer.c"
    poc_path.write_text(poc_code, encoding="utf-8")

    return PoCResult(
        poc_code=poc_code,
        poc_path=poc_path,
        strategy_name=strategy_name,
        confidence=0.9,
        crash_input_path=crash_file,
    )


def _minimal_file_reader_poc(harness_code: str) -> str:
    """Emit a tiny C driver that reads a file and calls the harness entry."""
    return (
        '#include <stdio.h>\n'
        '#include <stdlib.h>\n'
        '#include <stdint.h>\n'
        '\n'
        '/* ---------- original harness (inlined) ---------- */\n'
        f'{harness_code}\n'
        '\n'
        'int main(int argc, char **argv) {\n'
        '    if (argc < 2) { fprintf(stderr, "Usage: %s <crash_file>\\n", argv[0]); return 1; }\n'
        '    FILE *f = fopen(argv[1], "rb");\n'
        '    if (!f) { perror("fopen"); return 1; }\n'
        '    fseek(f, 0, SEEK_END);\n'
        '    long sz = ftell(f);\n'
        '    rewind(f);\n'
        '    uint8_t *buf = (uint8_t *)malloc(sz);\n'
        '    fread(buf, 1, sz, f);\n'
        '    fclose(f);\n'
        '    LLVMFuzzerTestOneInput(buf, (size_t)sz);\n'
        '    free(buf);\n'
        '    return 0;\n'
        '}\n'
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. AFL++ Runner
# ═══════════════════════════════════════════════════════════════════════════

_SEED_PROMPT_TEMPLATE = """\
You are a security researcher preparing seed inputs for AFL++ fuzzing.

Vulnerability type : {vuln_type}
Project            : {project_name}
Description        : {description}

Generate exactly 3 minimal seed inputs.  For each seed, output ONE line of
hex bytes (e.g. ``4d5a90000300...``).  No commentary, no labels — just three
lines of hex.
"""


def run_afl(
    build_result: BuildResult,
    task: CyberGymTask,
    context: CodeContext,
    router: LLMRouter,
    timeout_seconds: int = 120,
) -> PoCResult | None:
    """Run AFL++ on the project binary and return a PoC if a crash is found.

    Returns ``None`` if AFL++ is unavailable, no suitable binary is found,
    or no crashes are produced within the time budget.
    """
    if not shutil.which("afl-fuzz"):
        logger.info("afl-fuzz not in PATH — skipping AFL++")
        return None

    # ── 1. locate target binary ────────────────────────────────────────
    binary = _resolve_afl_binary(build_result, context)
    if binary is None:
        logger.warning("No suitable binary entry point for AFL++")
        return None

    workdir = _task_workdir(task.task_id)

    # ── 2. seed corpus ─────────────────────────────────────────────────
    seeds_dir = workdir / "afl_seeds"
    seeds_dir.mkdir(exist_ok=True)
    _generate_afl_seeds(seeds_dir, task, context, router)

    # ── 3. output directory ────────────────────────────────────────────
    afl_out = workdir / "afl_out"
    if afl_out.exists():
        shutil.rmtree(afl_out)
    afl_out.mkdir()

    # ── 4. run ─────────────────────────────────────────────────────────
    print(f"Starting AFL++ for {task.task_id} ({timeout_seconds}s budget)...")
    afl_cmd = [
        "afl-fuzz",
        "-i", str(seeds_dir),
        "-o", str(afl_out),
        "-t", "5000",            # per-exec timeout in ms
        "-V", str(timeout_seconds),  # wall-clock limit (AFL++ flag)
        "--", str(binary), "@@",
    ]

    logger.info("AFL++ command: %s", " ".join(afl_cmd))

    afl_env = {
        "AFL_NO_UI": "1",        # headless
        "AFL_SKIP_CPUFREQ": "1",
        "AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES": "1",
    }

    try:
        _run(afl_cmd, timeout=timeout_seconds + 60, cwd=workdir, env=afl_env)
    except subprocess.TimeoutExpired:
        logger.info("AFL++ hit external timeout — harvesting crashes")

    # ── 5. harvest ─────────────────────────────────────────────────────
    crashes_glob = list((afl_out / "default" / "crashes").glob("id:*"))
    if not crashes_glob:
        # Some AFL++ versions use a flat layout.
        crashes_glob = list(afl_out.rglob("crashes/id:*"))

    if not crashes_glob:
        logger.info("AFL++ finished — no crashes found")
        return None

    crash_file = sorted(crashes_glob)[0]
    logger.info("AFL++ crash found: %s (%d bytes)", crash_file,
                crash_file.stat().st_size)

    # Reuse the harness path if one was generated for libFuzzer earlier;
    # otherwise synthesise a minimal one.
    harness_path = workdir / "fuzzer_harness.c"
    if not harness_path.exists():
        harness_code = generate_fuzzing_harness(context, router)
        harness_path.write_text(harness_code, encoding="utf-8")

    return _crash_to_poc(
        crash_file=crash_file,
        harness_path=harness_path,
        task=task,
        strategy_name="afl++",
        router=router,
    )


def _resolve_afl_binary(build_result: BuildResult,
                         context: CodeContext) -> Path | None:
    """Pick the best binary to fuzz with AFL++."""
    # Prefer the binary produced by the build executor.
    if build_result.binary_path and build_result.binary_path.exists():
        return build_result.binary_path

    # Fall back to entry_points recorded during code analysis.
    entry_points: list[Any] = context.build_info.get("entry_points", [])
    for ep in entry_points:
        p = Path(ep)
        if p.exists() and os.access(p, os.X_OK):
            return p

    return None


def _generate_afl_seeds(seeds_dir: Path, task: CyberGymTask,
                         context: CodeContext, router: LLMRouter) -> None:
    """Use the LLM to create 3 minimal seed files for AFL++."""
    prompt = _SEED_PROMPT_TEMPLATE.format(
        vuln_type=task.vuln_type,
        project_name=task.project_name,
        description=task.description[:1500],
    )

    raw = router.query(prompt, max_tokens=512, temperature=0.5)
    hex_lines = [
        ln.strip() for ln in raw.splitlines()
        if ln.strip() and all(c in "0123456789abcdefABCDEF" for c in ln.strip())
    ]

    # Always write at least one seed so AFL doesn't complain.
    if not hex_lines:
        hex_lines = ["00" * 32]

    for idx, hexstr in enumerate(hex_lines[:3]):
        try:
            data = bytes.fromhex(hexstr)
        except ValueError:
            data = b"\x00" * 32
        (seeds_dir / f"seed_{idx:02d}").write_bytes(data)

    logger.info("Wrote %d AFL++ seed files to %s", min(len(hex_lines), 3),
                seeds_dir)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def try_fuzzing(
    context: CodeContext,
    build_result: BuildResult,
    router: LLMRouter,
    task: CyberGymTask,
) -> PoCResult | None:
    """Try libFuzzer first, then AFL++.  Return the first successful PoC.

    Respects ``config.FUZZING_ENABLED``.  Returns ``None`` when fuzzing is
    disabled, unavailable, or produces no crashes.
    """
    if not config.FUZZING_ENABLED:
        logger.info("Fuzzing disabled via config — skipping")
        print("Fuzzing disabled via config — skipping")
        return None

    avail = check_fuzzer_availability()
    if not avail["available"]:
        print("Fuzzing not available — skipping")
        logger.info("No fuzzer (AFL++ or libFuzzer) found in PATH — skipping")
        return None

    logger.info("Fuzzer availability: %s", avail)
    timeout = config.FUZZING_TIMEOUT

    # ── libFuzzer ──────────────────────────────────────────────────────
    if avail["libfuzzer"]:
        try:
            harness_code = generate_fuzzing_harness(context, router)
            harness_path = _task_workdir(task.task_id) / "fuzzer_harness.c"
            result = run_libfuzzer(
                harness_path=harness_path,
                build_result=build_result,
                task=task,
                timeout_seconds=timeout,
                router=router,
            )
            if result is not None:
                logger.info("libFuzzer produced a PoC (confidence=%.2f)",
                            result.confidence)
                return result
        except Exception:
            logger.exception("libFuzzer run failed — falling through to AFL++")

    # ── AFL++ ──────────────────────────────────────────────────────────
    if avail["afl++"]:
        try:
            result = run_afl(
                build_result=build_result,
                task=task,
                context=context,
                router=router,
                timeout_seconds=timeout,
            )
            if result is not None:
                logger.info("AFL++ produced a PoC (confidence=%.2f)",
                            result.confidence)
                return result
        except Exception:
            logger.exception("AFL++ run failed")

    logger.info("Fuzzing completed — no crashes found by either backend")
    return None
