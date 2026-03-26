"""
crs/build_executor.py — Build and execution engine for the Cyber Reasoning System.
"""
from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from crs.code_intelligence import CodeContext
from crs.config import BUILD_TIMEOUT, RUN_TIMEOUT, USE_SANITIZERS, cfg
from crs.data_loader import CyberGymTask
from crs.poc_strategies import PoCResult

logger = logging.getLogger(__name__)

WORK_DIR = cfg.WORK_DIR

# ── Dataclasses ────────────────────────────────────────────────────────────

@dataclass
class BuildResult:
    success: bool
    binary_path: Optional[Path]
    include_dirs: List[Path]
    lib_path: Optional[Path]
    build_log: str


@dataclass
class PoCBuildResult:
    poc_result: PoCResult
    compiled: bool
    poc_binary: Optional[Path]
    compile_log: str


@dataclass
class RunResult:
    poc_build: PoCBuildResult
    triggered: bool
    crash_type: str
    sanitizer_output: str
    return_code: int
    run_log: str


# ── Helpers ────────────────────────────────────────────────────────────────

def _run_cmd(
    cmd: List[str],
    timeout: int = BUILD_TIMEOUT,
    cwd: Optional[Path] = None,
    env=None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        timeout=timeout,
        cwd=str(cwd) if cwd else None,
        env=env,
    )


# ── Sanitizer detection ────────────────────────────────────────────────────

_ASAN_PATTERNS = [
    r"AddressSanitizer",
    r"ERROR: AddressSanitizer",
    r"heap-buffer-overflow",
    r"stack-buffer-overflow",
    r"heap-use-after-free",
    r"SUMMARY: AddressSanitizer",
]

_UBSAN_PATTERNS = [
    r"runtime error:",
    r"UndefinedBehaviorSanitizer",
    r"SUMMARY: UndefinedBehaviorSanitizer",
]

_MSAN_PATTERNS = [
    r"MemorySanitizer",
    r"SUMMARY: MemorySanitizer",
    r"use-of-uninitialized-value",
]

_CRASH_PATTERNS = [
    r"Segmentation fault",
    r"Aborted",
    r"double free",
    r"corrupted size vs\. prev_size",
]


def is_triggered(return_code: int, output: str) -> Tuple[bool, str]:
    """
    Determine if a PoC triggered a sanitizer or crash.
    Returns (triggered, crash_type).
    """
    combined = output

    for pat in _ASAN_PATTERNS:
        if re.search(pat, combined, re.IGNORECASE):
            m = re.search(r"ERROR: AddressSanitizer: (\S+)", combined)
            crash_type = m.group(1) if m else "asan"
            return True, crash_type

    for pat in _UBSAN_PATTERNS:
        if re.search(pat, combined, re.IGNORECASE):
            return True, "ubsan"

    for pat in _MSAN_PATTERNS:
        if re.search(pat, combined, re.IGNORECASE):
            return True, "msan"

    for pat in _CRASH_PATTERNS:
        if re.search(pat, combined, re.IGNORECASE):
            return True, "crash"

    if return_code not in (0, 1) and return_code < 0:
        return True, f"signal_{abs(return_code)}"

    return False, ""


# ── Project build ──────────────────────────────────────────────────────────

def build_project(task: CyberGymTask, build_info: dict) -> BuildResult:
    """
    Attempt to build the vulnerable project with sanitizers.
    Falls back to a bare build if sanitizer build fails.
    """
    repo = Path(task.repo_path).resolve()
    work = cfg.task_work_dir(task.task_id)
    log_parts: list[str] = []

    # Collect include dirs from .h files
    include_dirs: list[Path] = []
    seen: set[str] = set()
    for h in sorted(repo.rglob("*.h")):
        d = h.parent
        ds = str(d)
        if ds not in seen:
            include_dirs.append(d)
            seen.add(ds)

    build_type = build_info.get("type", "unknown")

    # ── Helper to attempt a build ──────────────────────────────────────────
    def _attempt(cflags: str, label: str):
        import os, shutil
        env = os.environ.copy()
        env["CFLAGS"]   = cflags
        env["CXXFLAGS"] = cflags

        build_dir = work / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)
        build_dir.mkdir(parents=True, exist_ok=True)

        log = f"\n[build] Configuring with CFLAGS='{cflags}'\n"
        print(log.strip())

        try:
            if build_type == "cmake":
                r = _run_cmd(
                    ["cmake", "-B", str(build_dir), "-S", str(repo),
                     f"-DCMAKE_C_FLAGS={cflags}", f"-DCMAKE_CXX_FLAGS={cflags}"],
                    cwd=repo, env=env,
                )
                log += r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
                if r.returncode != 0:
                    return None, log + f"\nConfigure failed ({label})"

                print("[build] Building: cmake --build build -j4")
                r2 = _run_cmd(
                    ["cmake", "--build", str(build_dir), "-j4"],
                    cwd=repo, env=env,
                )
                log += r2.stdout.decode(errors="replace") + r2.stderr.decode(errors="replace")
                if r2.returncode != 0:
                    return None, log + f"\nBuild failed ({label})"

            elif build_type in ("autotools", "make"):
                if build_type == "autotools":
                    print("[build] Running bootstrap: autoreconf -fi")
                    _run_cmd(["autoreconf", "-fi"], cwd=repo, env=env)

                print(f"[build] Running configure")
                r = _run_cmd(
                    ["./configure", f"CFLAGS={cflags}", f"CXXFLAGS={cflags}"],
                    cwd=repo, env=env,
                )
                log += r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
                if r.returncode != 0:
                    return None, log + f"\nConfigure failed ({label})"

                print("[build] Building: make -j4")
                r2 = _run_cmd(["make", "-j4"], cwd=repo, env=env)
                log += r2.stdout.decode(errors="replace") + r2.stderr.decode(errors="replace")
                if r2.returncode != 0:
                    return None, log + f"\nBuild failed ({label})"

            else:
                print("[build] Building: make -j4")
                r = _run_cmd(["make", "-j4"], cwd=repo, env=env)
                log += r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
                if r.returncode != 0:
                    return None, log + f"\nBuild failed ({label})"

        except subprocess.TimeoutExpired:
            return None, log + "\nBuild timed out"
        except Exception as e:
            return None, log + f"\nBuild exception: {e}"

        # Find built library
        lib = None
        for pattern in ["**/*.so", "**/*.a", "**/*.dylib"]:
            candidates = sorted(build_dir.glob(pattern))
            if candidates:
                lib = candidates[0]
                break

        # Find binary
        binary = None
        for pattern in ["**/mosquitto", "**/jq", "**/xmllint", "**/a.out"]:
            candidates = sorted(build_dir.glob(pattern))
            if candidates:
                binary = candidates[0]
                break

        return (binary or lib), log

    # ── Try sanitizer build first ──────────────────────────────────────────
    if USE_SANITIZERS:
        print("[build] Configuring with CFLAGS='-fsanitize=address,undefined -g -O1'")
        result, log = _attempt("-fsanitize=address,undefined -g -O1", "sanitizer")
        log_parts.append(log)
        if result:
            lib = result if result.suffix in (".so", ".a", ".dylib") else None
            binary = result if result.suffix == "" else None
            print(f"[build] Result: OK  binary={binary}  lib={lib}")
            return BuildResult(
                success=True,
                binary_path=binary,
                include_dirs=include_dirs,
                lib_path=lib,
                build_log="\n".join(log_parts),
            )
        print("[build] Sanitizer build failed — retrying bare build")

    # ── Bare build fallback ────────────────────────────────────────────────
    result, log = _attempt("-g -O1", "bare")
    log_parts.append(log)
    if result:
        lib = result if result.suffix in (".so", ".a", ".dylib") else None
        binary = result if result.suffix == "" else None
        print(f"[build] Result: OK  binary={binary}  lib={lib}")
        return BuildResult(
            success=True,
            binary_path=binary,
            include_dirs=include_dirs,
            lib_path=lib,
            build_log="\n".join(log_parts),
        )

    print(f"[build] Result: FAILED  binary=None  lib=None")
    return BuildResult(
        success=False,
        binary_path=None,
        include_dirs=include_dirs,
        lib_path=None,
        build_log="\n".join(log_parts),
    )


# ── PoC compilation ────────────────────────────────────────────────────────

def compile_poc(
    poc_result: PoCResult,
    build_result: BuildResult,
    task: CyberGymTask,
) -> PoCBuildResult:
    """Compile a single PoC source file against the vulnerable project."""
    task_id       = getattr(task, "task_id", "unknown")
    strategy_name = getattr(poc_result, "strategy_name", "poc")

    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(strategy_name))

    work       = cfg.task_work_dir(task_id)
    poc_source = Path(poc_result.poc_path)       # fixed: was .source_path
    poc_binary = work / f"poc_{safe_name}_bin"

    # Determine compiler
    lang     = getattr(task, "project_language", "c")
    compiler = "g++" if lang.lower() in ("c++", "cpp", "cxx") else "gcc"

    # Base compile command
    cmd: list[str] = [compiler]
    cmd += ["-fsanitize=address,undefined", "-g", "-O1"]
    cmd += [str(poc_source)]
    cmd += ["-o", str(poc_binary)]

    # Add all include dirs from build result
    for d in build_result.include_dirs:
        cmd.append(f"-I{d}")

    # Also add common sub-dirs from repo root
    repo = Path(task.repo_path).resolve()
    for extra in (repo, repo / "include", repo / "src"):
        if extra.is_dir() and extra not in build_result.include_dirs:
            cmd.append(f"-I{extra}")

    # Link against built library if available
    if build_result.lib_path:
        lib = Path(build_result.lib_path)
        cmd.append(str(lib))
        cmd += [f"-L{lib.parent}", f"-Wl,-rpath,{lib.parent}"]
    elif build_result.binary_path:
        bin_dir = Path(build_result.binary_path).parent
        obj_files = sorted(bin_dir.glob("*.o"))
        if obj_files:
            cmd += [str(o) for o in obj_files]

    # Common system libs
    cmd += ["-lm", "-lpthread"]

    # ── Attempt 1: with sanitizers ─────────────────────────────────────────
    log = ""
    compiled = False
    print(f"[compile] {' '.join(cmd[:6])} ...")
    try:
        r = _run_cmd(cmd, timeout=BUILD_TIMEOUT, cwd=work)
        log = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
        compiled = r.returncode == 0
    except subprocess.TimeoutExpired:
        log = "compilation timed out"
        compiled = False
    except Exception as exc:
        log = f"compilation error: {exc}"
        compiled = False

    # ── Attempt 2: without sanitizers ─────────────────────────────────────
    if not compiled and "-fsanitize" in " ".join(cmd):
        print("[compile] Retrying without sanitizer flags")
        fallback_cmd = [c for c in cmd if not c.startswith("-fsanitize")]
        try:
            r = _run_cmd(fallback_cmd, timeout=BUILD_TIMEOUT, cwd=work)
            fb_log = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
            log += "\n--- fallback (no sanitizer) ---\n" + fb_log
            compiled = r.returncode == 0
        except Exception as exc:
            log += f"\nfallback error: {exc}"

    # ── Attempt 3: compile PoC with vulnerable source files directly ───────
    if not compiled:
        print("[compile] Retrying with vulnerable source files directly")
        vuln_repo = Path(task.repo_path).resolve()

        # Collect all include dirs from headers
        all_include_dirs: set[str] = set()
        for h in vuln_repo.rglob("*.h"):
            all_include_dirs.add(str(h.parent))

        # Collect source files (exclude tests and files with main)
        vuln_sources: list[str] = []
        for p in sorted(vuln_repo.rglob("*.c")):
            s = str(p)
            if any(x in s.lower() for x in ["test", "example", "fuzzing", "bench", "fuzz"]):
                continue
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                if re.search(r"\bint\s+main\s*\(", content):
                    continue  # skip files with their own main()
            except Exception:
                pass
            vuln_sources.append(s)

        vuln_sources = vuln_sources[:25]  # cap to avoid arg overload

        if vuln_sources:
            src_cmd = [compiler, "-fsanitize=address,undefined", "-g", "-O1"]
            src_cmd += [str(poc_source)]
            src_cmd += vuln_sources
            src_cmd += ["-o", str(poc_binary)]
            src_cmd += [f"-I{d}" for d in sorted(all_include_dirs)]
            src_cmd += ["-lm", "-lpthread"]

            try:
                r = _run_cmd(src_cmd, timeout=BUILD_TIMEOUT, cwd=work)
                src_log = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
                log += "\n--- src fallback ---\n" + src_log
                compiled = r.returncode == 0
                if compiled:
                    print("[compile] PoC compiled with vulnerable sources: OK")
            except Exception as exc:
                log += f"\nsrc fallback error: {exc}"

    status = "OK" if compiled else "FAILED"
    print(f"[compile] PoC '{safe_name}': {status}")

    return PoCBuildResult(
        poc_result=poc_result,
        compiled=compiled,
        poc_binary=poc_binary if compiled else None,
        compile_log=log[-4000:],
    )


# ── PoC runner ─────────────────────────────────────────────────────────────

def run_poc(poc_build: PoCBuildResult) -> RunResult:
    """Execute a compiled PoC and check for sanitizer triggers."""
    if not poc_build.compiled or poc_build.poc_binary is None:
        return RunResult(
            poc_build=poc_build,
            triggered=False,
            crash_type="compile_failed",
            sanitizer_output="",
            return_code=-1,
            run_log="PoC was not compiled.",
        )

    binary = Path(poc_build.poc_binary)
    print(f"[run] Executing {binary}")

    try:
        r = _run_cmd([str(binary)], timeout=RUN_TIMEOUT)
        output = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
        triggered, crash_type = is_triggered(r.returncode, output)
        print(f"[run] Result: {'triggered' if triggered else 'not triggered'}  rc={r.returncode}  crash_type={crash_type}")
        return RunResult(
            poc_build=poc_build,
            triggered=triggered,
            crash_type=crash_type,
            sanitizer_output=output[:8000],
            return_code=r.returncode,
            run_log=output[:4000],
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            poc_build=poc_build,
            triggered=False,
            crash_type="timeout",
            sanitizer_output="",
            return_code=-1,
            run_log="PoC execution timed out.",
        )
    except Exception as e:
        return RunResult(
            poc_build=poc_build,
            triggered=False,
            crash_type="error",
            sanitizer_output="",
            return_code=-1,
            run_log=str(e),
        )


# ── Full pipeline ──────────────────────────────────────────────────────────

def execute_pipeline(
    task: CyberGymTask,
    poc_results: list[PoCResult],
    build_result: BuildResult,
) -> list[RunResult]:
    """
    Compile and run all PoC candidates.
    Returns list of RunResult sorted: triggered first, then by confidence.
    """
    run_results: list[RunResult] = []

    print(f"  [Strategies] Generated {len(poc_results)} PoC candidate(s)")

    for idx, poc_result in enumerate(poc_results):
        print(f"\n  [PoC {idx}] Compiling ({poc_result.strategy_name}) ...")
        poc_build = compile_poc(poc_result, build_result, task)

        print(f"  [PoC {idx}] Running ...")
        run_result = run_poc(poc_build)

        tag = "TRIGGERED" if run_result.triggered else "no trigger"
        print(f"  [PoC {idx}] {tag}  (exit={run_result.return_code})")

        run_results.append(run_result)

        if run_result.triggered:
            print(f"\n  *** PoC {idx} TRIGGERED the vulnerability! ***")
            break

    return run_results
