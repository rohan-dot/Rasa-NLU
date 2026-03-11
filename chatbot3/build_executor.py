"""
crs/build_executor.py  —  Step 5: Build & Execution Harness

Compile the vulnerable project, compile each candidate PoC against it,
execute the PoC binaries, and detect crashes / sanitizer errors / hangs.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional

from crs.config import (
    BUILD_TIMEOUT,
    RUN_TIMEOUT,
    WORK_DIR,
)
from crs.data_loader import CyberGymTask
from crs.code_intelligence import CodeContext
from crs.poc_strategies import PoCResult

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BuildResult:
    """Outcome of building the vulnerable project from source."""
    success: bool
    binary_path: Optional[Path] = None
    include_dirs: List[Path] = field(default_factory=list)
    lib_path: Optional[Path] = None
    build_log: str = ""


@dataclass
class PoCBuildResult:
    """Outcome of compiling a single PoC source file."""
    poc_result: PoCResult
    compiled: bool = False
    poc_binary: Optional[Path] = None
    compile_log: str = ""


@dataclass
class RunResult:
    """Outcome of executing a compiled PoC binary."""
    poc_build: PoCBuildResult
    triggered: bool = False
    crash_type: str = ""
    sanitizer_output: str = ""
    return_code: int = 0
    run_log: str = ""


# ---------------------------------------------------------------------------
# Sanitizer / crash detection
# ---------------------------------------------------------------------------

_SANITIZER_PATTERNS: List[str] = [
    "ERROR: AddressSanitizer",
    "runtime error:",
    "ERROR: LeakSanitizer",
    "heap-buffer-overflow",
    "stack-buffer-overflow",
    "use-after-free",
    "SEGFAULT",
    "Segmentation fault",
    "Aborted",
]

_CRASH_RETURN_CODES = {
    1, 2, 3,
    -6,          # SIGABRT  (128+6 when unsigned; Python gives negative)
    -11,         # SIGSEGV
    134,         # 128 + 6  — some shells report SIGABRT this way
    139,         # 128 + 11 — SIGSEGV
}

_CRASH_TYPE_RE = re.compile(
    r"ERROR: AddressSanitizer:\s+(\S+)",
    re.IGNORECASE,
)

_UBSAN_TYPE_RE = re.compile(
    r"runtime error:\s+(.+)",
    re.IGNORECASE,
)


def is_triggered(return_code: int, output: str) -> Tuple[bool, str]:
    """Decide whether the PoC triggered a vulnerability.

    Returns
    -------
    (triggered, crash_type)
        *triggered* is ``True`` when at least one of the following holds:
          1. ``return_code`` is a known crash code **and** *output* contains a
             recognised sanitizer / crash string.
          2. ``output`` contains ``SUMMARY: AddressSanitizer``.
          3. ``output`` contains ``UndefinedBehaviorSanitizer``.

        *crash_type* is extracted from the first matching ASAN/UBSAN line, or
        falls back to a generic label derived from the signal.
    """
    crash_type = ""

    # --- rule 2 -----------------------------------------------------------
    if "SUMMARY: AddressSanitizer" in output:
        crash_type = _extract_crash_type(output)
        return True, crash_type or "AddressSanitizer"

    # --- rule 3 -----------------------------------------------------------
    if "UndefinedBehaviorSanitizer" in output:
        crash_type = _extract_ubsan_type(output)
        return True, crash_type or "UndefinedBehaviorSanitizer"

    # --- rule 1 -----------------------------------------------------------
    if return_code in _CRASH_RETURN_CODES:
        for pattern in _SANITIZER_PATTERNS:
            if pattern in output:
                crash_type = _extract_crash_type(output) or pattern
                return True, crash_type
        # Crash code but no sanitizer output → still counts for some signals
        # (e.g. bare SIGSEGV without ASAN). We treat these conservatively:
        # only flag if there is *some* recognisable crash artefact.

    return False, ""


def _extract_crash_type(output: str) -> str:
    m = _CRASH_TYPE_RE.search(output)
    return m.group(1) if m else ""


def _extract_ubsan_type(output: str) -> str:
    m = _UBSAN_TYPE_RE.search(output)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cmd(
    cmd: List[str],
    *,
    timeout: int,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    """Thin wrapper around subprocess.run with defaults."""
    merged_env = {**os.environ, **(env or {})}
    return subprocess.run(
        cmd,
        capture_output=True,
        timeout=timeout,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
    )


def _collect_header_dirs(root: Path) -> List[Path]:
    """Walk *root* and return every directory that contains at least one .h."""
    dirs: set[Path] = set()
    for dirpath, _, filenames in os.walk(root):
        if any(f.endswith((".h", ".hh", ".hpp")) for f in filenames):
            dirs.add(Path(dirpath))
    return sorted(dirs)


def _find_artifacts(root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Search for libraries (.a / .so) and binaries produced by the build.

    Returns
    -------
    (binary_path, lib_path)
    """
    search_dirs = [
        root / "src",
        root / "build",
        root / ".libs",
        root / "lib",
        root / "src" / ".libs",
        root,
    ]

    lib_path: Optional[Path] = None
    binary_path: Optional[Path] = None

    # Prefer static libs for easier linking.
    for sd in search_dirs:
        if not sd.is_dir():
            continue
        for f in sd.iterdir():
            if f.suffix == ".a" and lib_path is None:
                lib_path = f
            elif f.suffix in (".so", ".dylib") and lib_path is None:
                lib_path = f
            elif (
                f.is_file()
                and os.access(f, os.X_OK)
                and f.suffix not in (
                    ".c", ".cpp", ".h", ".hpp", ".o", ".py",
                    ".sh", ".mk", ".txt", ".md", ".a", ".so",
                    ".dylib", ".la", ".lo", ".log",
                )
                and binary_path is None
            ):
                binary_path = f

    # Broader recursive search if nothing found yet.
    if lib_path is None:
        for p in root.rglob("*.a"):
            lib_path = p
            break
    if lib_path is None:
        for p in root.rglob("*.so"):
            lib_path = p
            break
    if binary_path is None:
        for p in root.rglob("*"):
            if (
                p.is_file()
                and os.access(p, os.X_OK)
                and p.suffix not in (
                    ".c", ".cpp", ".h", ".hpp", ".o", ".py",
                    ".sh", ".mk", ".txt", ".md", ".a", ".so",
                    ".dylib", ".la", ".lo", ".log", ".m4",
                    ".ac", ".am", ".in", ".sub", ".guess",
                    ".configure", ".status",
                )
                and "config" not in p.name.lower()
                and "libtool" not in p.name.lower()
            ):
                binary_path = p
                break

    return binary_path, lib_path


# ---------------------------------------------------------------------------
# Project builder
# ---------------------------------------------------------------------------

def build_project(task: CyberGymTask, build_info: dict) -> BuildResult:
    """Build the vulnerable project in-place.

    Parameters
    ----------
    task : CyberGymTask
        Must expose ``repo_path: Path``.
    build_info : dict
        Expected keys (all optional):
          * ``configure_cmd`` – e.g. ``["./configure"]``
          * ``build_cmd``     – e.g. ``["make", "-j4"]``

    The function never raises; it always returns a ``BuildResult``.
    """
    repo = Path(task.repo_path).resolve()
    configure_cmd: List[str] = build_info.get("configure_cmd", [])
    build_cmd: List[str] = build_info.get("build_cmd", ["make", "-j4"])

    sanitizer_flags = "-fsanitize=address,undefined -g -O1"
    base_env = {
        "CC": "gcc",
        "CXX": "g++",
    }

    logs: List[str] = []

    # ---- autoreconf / autogen if needed ----------------------------------
    if configure_cmd and not (repo / "configure").exists():
        for bootstrap in ("autoreconf -fi", "./autogen.sh"):
            script = bootstrap.split()[0]
            if (repo / script).exists() or shutil.which(script):
                print(f"[build] Running bootstrap: {bootstrap}")
                try:
                    r = _run_cmd(
                        bootstrap.split(),
                        timeout=BUILD_TIMEOUT,
                        cwd=repo,
                        env=base_env,
                    )
                    logs.append(r.stdout.decode(errors="replace"))
                    logs.append(r.stderr.decode(errors="replace"))
                except Exception as exc:
                    logs.append(f"bootstrap failed: {exc}")

    # ---- configure -------------------------------------------------------
    def _configure(extra_cflags: str) -> bool:
        if not configure_cmd:
            return True
        env = {
            **base_env,
            "CFLAGS": extra_cflags,
            "CXXFLAGS": extra_cflags,
        }
        print(f"[build] Configuring with CFLAGS={extra_cflags!r}")
        try:
            r = _run_cmd(configure_cmd, timeout=BUILD_TIMEOUT, cwd=repo, env=env)
            logs.append(r.stdout.decode(errors="replace"))
            logs.append(r.stderr.decode(errors="replace"))
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            logs.append("configure timed out")
            return False
        except Exception as exc:
            logs.append(f"configure error: {exc}")
            return False

    configured = _configure(sanitizer_flags)
    if not configured:
        print("[build] Sanitizer configure failed — retrying bare build")
        configured = _configure("-g -O1")

    # ---- build -----------------------------------------------------------
    def _make(extra_cflags: str) -> bool:
        env = {
            **base_env,
            "CFLAGS": extra_cflags,
            "CXXFLAGS": extra_cflags,
        }
        print(f"[build] Building: {' '.join(build_cmd)}")
        try:
            r = _run_cmd(build_cmd, timeout=BUILD_TIMEOUT, cwd=repo, env=env)
            logs.append(r.stdout.decode(errors="replace"))
            logs.append(r.stderr.decode(errors="replace"))
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            logs.append("build timed out")
            return False
        except Exception as exc:
            logs.append(f"build error: {exc}")
            return False

    built = _make(sanitizer_flags)
    if not built:
        print("[build] Sanitizer build failed — retrying bare build")
        # Clean first so stale objects don't collide.
        try:
            _run_cmd(["make", "clean"], timeout=60, cwd=repo)
        except Exception:
            pass
        if configure_cmd:
            _configure("-g -O1")
        built = _make("-g -O1")

    # ---- cmake special path ---------------------------------------------
    if not built and (repo / "CMakeLists.txt").exists():
        print("[build] Attempting cmake build")
        build_dir = repo / "build"
        build_dir.mkdir(exist_ok=True)
        try:
            r = _run_cmd(
                [
                    "cmake", "..",
                    f"-DCMAKE_C_FLAGS={sanitizer_flags}",
                    f"-DCMAKE_CXX_FLAGS={sanitizer_flags}",
                    "-DCMAKE_BUILD_TYPE=Debug",
                ],
                timeout=BUILD_TIMEOUT,
                cwd=build_dir,
                env=base_env,
            )
            logs.append(r.stdout.decode(errors="replace"))
            logs.append(r.stderr.decode(errors="replace"))
            if r.returncode == 0:
                r2 = _run_cmd(
                    ["cmake", "--build", ".", "-j4"],
                    timeout=BUILD_TIMEOUT,
                    cwd=build_dir,
                    env=base_env,
                )
                logs.append(r2.stdout.decode(errors="replace"))
                logs.append(r2.stderr.decode(errors="replace"))
                built = r2.returncode == 0
        except Exception as exc:
            logs.append(f"cmake error: {exc}")

    # ---- collect results -------------------------------------------------
    binary_path, lib_path = _find_artifacts(repo)
    include_dirs = _collect_header_dirs(repo)

    success = built and (binary_path is not None or lib_path is not None)
    result = BuildResult(
        success=success,
        binary_path=binary_path,
        include_dirs=include_dirs,
        lib_path=lib_path,
        build_log="\n".join(logs)[-8000:],  # cap log size
    )
    status = "OK" if success else "FAILED"
    print(f"[build] Result: {status}  binary={binary_path}  lib={lib_path}")
    return result


# ---------------------------------------------------------------------------
# PoC compiler
# ---------------------------------------------------------------------------

def compile_poc(
    poc_result: PoCResult,
    build_result: BuildResult,
    task: CyberGymTask,
) -> PoCBuildResult:
    """Compile a single PoC source file against the vulnerable project.

    The PoC binary is placed under ``WORK_DIR / task_id /``.
    """
    task_id = getattr(task, "task_id", "unknown")
    strategy_name = getattr(poc_result, "strategy", "poc")
    # Sanitise strategy name for filesystem.
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(strategy_name))

    work = Path(WORK_DIR) / str(task_id)
    work.mkdir(parents=True, exist_ok=True)

    poc_source = Path(poc_result.source_path)
    poc_binary = work / f"poc_{safe_name}_bin"

    # Determine compiler.
    lang = getattr(task, "project_language", "c")
    compiler = "g++" if lang.lower() in ("c++", "cpp", "cxx") else "gcc"

    # --- assemble flags ---------------------------------------------------
    cmd: List[str] = [compiler]
    cmd += ["-fsanitize=address,undefined", "-g", "-O1"]
    cmd += [str(poc_source)]
    cmd += ["-o", str(poc_binary)]
    cmd += [f"-I{d}" for d in build_result.include_dirs]

    # Also include the repo root and common sub-dirs.
    repo = Path(task.repo_path).resolve()
    for extra in (repo, repo / "include", repo / "src"):
        if extra.is_dir() and extra not in build_result.include_dirs:
            cmd.append(f"-I{extra}")

    # --- link against library / object files ------------------------------
    if build_result.lib_path:
        lib = Path(build_result.lib_path)
        cmd.append(str(lib))
        # Add the library's directory to the search path too.
        cmd += [f"-L{lib.parent}", f"-Wl,-rpath,{lib.parent}"]
    elif build_result.binary_path:
        # Try to find .o files near the binary to link against.
        bin_dir = Path(build_result.binary_path).parent
        obj_files = sorted(bin_dir.glob("*.o"))
        if obj_files:
            cmd += [str(o) for o in obj_files]

    # Common system libs.
    cmd += ["-lm", "-lpthread"]

    # --- compile ----------------------------------------------------------
    print(f"[compile] {' '.join(cmd)}")
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

    # If sanitizer linking failed, retry without sanitizers.
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

    status = "OK" if compiled else "FAILED"
    print(f"[compile] PoC '{safe_name}': {status}")

    return PoCBuildResult(
        poc_result=poc_result,
        compiled=compiled,
        poc_binary=poc_binary if compiled else None,
        compile_log=log[-4000:],
    )


# ---------------------------------------------------------------------------
# PoC runner
# ---------------------------------------------------------------------------

def run_poc(poc_build: PoCBuildResult, task: CyberGymTask) -> RunResult:
    """Execute a compiled PoC binary and check for vulnerability triggers.

    The binary is run up to 3 times.  The vulnerability counts as triggered
    only if at least 2 out of 3 runs trigger it (determinism check).
    """
    assert poc_build.compiled and poc_build.poc_binary is not None

    binary = poc_build.poc_binary
    print(f"[run] Executing {binary}")

    env_extras = {
        # Make ASAN produce full output even on crash.
        "ASAN_OPTIONS": "detect_leaks=1:abort_on_error=1:print_legend=0",
        "UBSAN_OPTIONS": "print_stacktrace=1",
        # Ensure the library can be found at runtime.
    }
    if poc_build.poc_result and hasattr(poc_build.poc_result, "source_path"):
        # Guess LD_LIBRARY_PATH from build artefacts.
        repo = Path(task.repo_path).resolve()
        ld_dirs = set()
        for p in repo.rglob("*.so"):
            ld_dirs.add(str(p.parent))
        for p in repo.rglob(".libs"):
            ld_dirs.add(str(p))
        if ld_dirs:
            env_extras["LD_LIBRARY_PATH"] = ":".join(sorted(ld_dirs))

    def _single_run() -> Tuple[int, str]:
        try:
            r = _run_cmd(
                [str(binary)],
                timeout=RUN_TIMEOUT,
                env=env_extras,
            )
            output = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
            return r.returncode, output
        except subprocess.TimeoutExpired:
            return -9, "TIMEOUT: process did not finish within limit"
        except Exception as exc:
            return -1, f"execution error: {exc}"

    # --- first run --------------------------------------------------------
    rc, output = _single_run()
    triggered, crash_type = is_triggered(rc, output)

    # Handle timeout as a potential hang/infinite-loop bug.
    if rc == -9:
        crash_type = "timeout"
        triggered = True  # Treat hang as a trigger; determinism check below.

    all_logs = [output]

    # --- determinism check (2/3 majority) ---------------------------------
    if triggered:
        trigger_count = 1
        for attempt in range(2):
            rc_n, out_n = _single_run()
            all_logs.append(out_n)
            t_n, _ = is_triggered(rc_n, out_n)
            if t_n or rc_n == -9:
                trigger_count += 1
        triggered = trigger_count >= 2
        if not triggered:
            print(f"[run] Non-deterministic trigger ({trigger_count}/3) — marking as not triggered")

    # --- extract sanitizer-specific output --------------------------------
    sanitizer_output = ""
    for line in output.splitlines():
        if any(
            kw in line
            for kw in ("AddressSanitizer", "LeakSanitizer", "runtime error", "SUMMARY")
        ):
            sanitizer_output += line + "\n"

    result_str = "TRIGGERED" if triggered else "not triggered"
    print(f"[run] Result: {result_str}  rc={rc}  crash_type={crash_type}")

    return RunResult(
        poc_build=poc_build,
        triggered=triggered,
        crash_type=crash_type,
        sanitizer_output=sanitizer_output.strip(),
        return_code=rc,
        run_log="\n---\n".join(all_logs)[-8000:],
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def execute_pipeline(
    task: CyberGymTask,
    context: CodeContext,
    poc_results: List[PoCResult],
) -> List[RunResult]:
    """Orchestrate build → compile → run for every candidate PoC.

    Returns as soon as one PoC triggers the vulnerability (early-exit
    optimisation).  The full list of ``RunResult`` objects tried so far is
    returned.
    """
    print(f"\n{'='*60}")
    print(f"[pipeline] Starting for task {getattr(task, 'task_id', '?')}")
    print(f"[pipeline] {len(poc_results)} candidate PoC(s)")
    print(f"{'='*60}\n")

    # 1. Build the project.
    build_info = getattr(context, "build_info", {}) or {}
    build_result = build_project(task, build_info)

    if not build_result.success:
        print("[pipeline] WARNING: project build failed — will still try PoCs")

    # 2. Compile & run each PoC.
    run_results: List[RunResult] = []

    for idx, poc in enumerate(poc_results, 1):
        print(f"\n--- PoC {idx}/{len(poc_results)}: {getattr(poc, 'strategy', '?')} ---")

        # 2a. Compile.
        poc_build = compile_poc(poc, build_result, task)

        if not poc_build.compiled:
            print(f"[pipeline] PoC {idx} compilation failed — skipping execution")
            run_results.append(
                RunResult(
                    poc_build=poc_build,
                    triggered=False,
                    crash_type="",
                    sanitizer_output="",
                    return_code=-1,
                    run_log=poc_build.compile_log,
                )
            )
            continue

        # 2b. Run.
        run_result = run_poc(poc_build, task)
        run_results.append(run_result)

        # 2c. Early exit on success.
        if run_result.triggered:
            print(f"\n[pipeline] SUCCESS — PoC {idx} triggered: {run_result.crash_type}")
            return run_results

    n_tried = len(run_results)
    n_compiled = sum(1 for r in run_results if r.poc_build.compiled)
    n_triggered = sum(1 for r in run_results if r.triggered)
    print(f"\n[pipeline] Done. tried={n_tried}  compiled={n_compiled}  triggered={n_triggered}")
    return run_results
