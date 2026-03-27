"""
crs/build_executor.py — Build and execution engine for the Cyber Reasoning System.

Changes from original:
  - Generalized binary/lib discovery: finds any ELF executable, not hardcoded names
  - _auto_install_deps(): tries to install common C/C++ dev libraries before building
  - _parse_cmake_missing_packages(): reads cmake errors and disables missing features
  - _attempt() retries cmake with progressively more features disabled
  - Searches both build_dir AND repo for build artifacts
  - Meson build support in _attempt()
  - Increased BUILD_TIMEOUT awareness for large projects
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import stat
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


# ── Dependency auto-install ────────────────────────────────────────────────

# Common dev packages that CyberGym projects need.
# Maps a "missing signal" (string found in cmake/configure error output)
# to the apt package(s) that provide it.
_DEP_SIGNAL_TO_PKG: dict[str, list[str]] = {
    "openssl":      ["libssl-dev"],
    "ssl":          ["libssl-dev"],
    "cjson":        ["libcjson-dev"],
    "cJSON":        ["libcjson-dev"],
    "zlib":         ["zlib1g-dev"],
    "libxml2":      ["libxml2-dev"],
    "libxml":       ["libxml2-dev"],
    "curl":         ["libcurl4-openssl-dev"],
    "pcre":         ["libpcre3-dev"],
    "pcre2":        ["libpcre2-dev"],
    "libevent":     ["libevent-dev"],
    "libev":        ["libev-dev"],
    "libuv":        ["libuv1-dev"],
    "jansson":      ["libjansson-dev"],
    "sqlite":       ["libsqlite3-dev"],
    "sqlite3":      ["libsqlite3-dev"],
    "libpng":       ["libpng-dev"],
    "libjpeg":      ["libjpeg-dev"],
    "tiff":         ["libtiff-dev"],
    "freetype":     ["libfreetype6-dev"],
    "expat":        ["libexpat1-dev"],
    "bzip2":        ["libbz2-dev"],
    "lzma":         ["liblzma-dev"],
    "xz":           ["liblzma-dev"],
    "libyaml":      ["libyaml-dev"],
    "yaml":         ["libyaml-dev"],
    "protobuf":     ["libprotobuf-dev"],
    "glib":         ["libglib2.0-dev"],
    "icu":          ["libicu-dev"],
    "readline":     ["libreadline-dev"],
    "ncurses":      ["libncurses5-dev"],
    "flex":         ["flex"],
    "bison":        ["bison"],
    "autoconf":     ["autoconf"],
    "automake":     ["automake"],
    "libtool":      ["libtool"],
    "pkg-config":   ["pkg-config"],
    "cmake":        ["cmake"],
}

# Baseline dev packages to always attempt to install
_BASELINE_PKGS = [
    "build-essential", "cmake", "pkg-config", "autoconf", "automake",
    "libtool", "zlib1g-dev", "libssl-dev",
]


def _auto_install_deps(repo: Path, build_log: str = "") -> str:
    """
    Best-effort attempt to install common build dependencies.
    Returns a log string of what was attempted.
    """
    log_parts: list[str] = []

    # Check if we can run apt-get (might not have sudo / might not be Debian)
    if not shutil.which("apt-get"):
        return "[deps] apt-get not available — skipping dependency install\n"

    # Always try baseline packages
    pkgs_to_install: set[str] = set(_BASELINE_PKGS)

    # Scan CMakeLists.txt / configure.ac for hints about needed deps
    for hint_file in ["CMakeLists.txt", "configure.ac", "configure.in", "meson.build"]:
        hint_path = repo / hint_file
        if hint_path.exists():
            try:
                content = hint_path.read_text(encoding="utf-8", errors="replace").lower()
                for signal, apt_pkgs in _DEP_SIGNAL_TO_PKG.items():
                    if signal.lower() in content:
                        pkgs_to_install.update(apt_pkgs)
            except Exception:
                pass

    # Also scan build_log for "Could NOT find" or "missing" signals
    if build_log:
        log_lower = build_log.lower()
        for signal, apt_pkgs in _DEP_SIGNAL_TO_PKG.items():
            if signal.lower() in log_lower:
                pkgs_to_install.update(apt_pkgs)

    if not pkgs_to_install:
        return "[deps] No additional packages to install\n"

    pkg_list = sorted(pkgs_to_install)
    log_parts.append(f"[deps] Attempting to install: {' '.join(pkg_list)}")
    print(log_parts[-1])

    try:
        # Update package list first (quick, silent)
        subprocess.run(
            ["apt-get", "update", "-qq"],
            capture_output=True, timeout=60,
        )
        # Install packages (non-interactive)
        r = subprocess.run(
            ["apt-get", "install", "-y", "-qq", "--no-install-recommends"] + pkg_list,
            capture_output=True, timeout=120,
        )
        output = r.stderr.decode(errors="replace")
        if r.returncode == 0:
            log_parts.append("[deps] Package install succeeded")
        else:
            log_parts.append(f"[deps] Package install partial/failed: {output[:500]}")
    except subprocess.TimeoutExpired:
        log_parts.append("[deps] Package install timed out")
    except Exception as e:
        log_parts.append(f"[deps] Package install error: {e}")

    print(log_parts[-1])
    return "\n".join(log_parts) + "\n"


# ── CMake feature-disable parsing ──────────────────────────────────────────

# Maps cmake error signals to cmake flags that disable the problematic feature.
_CMAKE_DISABLE_FLAGS: list[tuple[str, list[str]]] = [
    # TLS/SSL
    ("could not find openssl",          ["-DWITH_TLS=OFF", "-DWITH_SSL=OFF", "-DENABLE_SSL=OFF"]),
    ("openssl",                         ["-DWITH_TLS=OFF", "-DWITH_SSL=OFF"]),
    # cJSON
    ("could not find cjson",            ["-DWITH_CJSON=OFF", "-DUSE_CJSON=OFF"]),
    ("cjson",                           ["-DWITH_CJSON=OFF"]),
    # Docs / man pages
    ("xsltproc",                        ["-DDOCUMENTATION=OFF", "-DWITH_DOCS=OFF", "-DBUILD_DOCS=OFF"]),
    ("docbook",                         ["-DDOCUMENTATION=OFF", "-DWITH_DOCS=OFF"]),
    ("doxygen",                         ["-DDOCUMENTATION=OFF", "-DBUILD_DOCS=OFF"]),
    # Testing
    ("could not find gtest",            ["-DBUILD_TESTING=OFF", "-DWITH_TESTS=OFF"]),
    ("gtest",                           ["-DBUILD_TESTING=OFF"]),
    ("ctest",                           ["-DBUILD_TESTING=OFF"]),
    # Misc optional features
    ("could not find systemd",          ["-DWITH_SYSTEMD=OFF"]),
    ("systemd",                         ["-DWITH_SYSTEMD=OFF"]),
    ("could not find dbus",             ["-DWITH_DBUS=OFF", "-DENABLE_DBUS=OFF"]),
    ("dbus",                            ["-DWITH_DBUS=OFF"]),
    ("websocket",                       ["-DWITH_WEBSOCKETS=OFF"]),
    ("libwebsocket",                    ["-DWITH_WEBSOCKETS=OFF"]),
    ("could not find curses",           ["-DBUILD_CURSES=OFF"]),
    ("could not find.*shared",          ["-DBUILD_SHARED_LIBS=OFF"]),
    # SRV (Mosquitto-specific but also common pattern)
    ("srv",                             ["-DWITH_SRV=OFF"]),
]

# Blanket "disable everything optional" flags to try as last resort
_CMAKE_BLANKET_DISABLE = [
    "-DBUILD_TESTING=OFF",
    "-DBUILD_TESTS=OFF",
    "-DDOCUMENTATION=OFF",
    "-DBUILD_DOCS=OFF",
    "-DWITH_DOCS=OFF",
    "-DWITH_TLS=OFF",
    "-DWITH_SSL=OFF",
    "-DENABLE_SSL=OFF",
    "-DWITH_WEBSOCKETS=OFF",
    "-DWITH_SRV=OFF",
    "-DWITH_SYSTEMD=OFF",
    "-DWITH_CJSON=OFF",
    "-DWITH_TESTS=OFF",
    "-DBUILD_EXAMPLES=OFF",
    "-DWITH_EXAMPLES=OFF",
    "-DBUILD_SHARED_LIBS=ON",
]


def _parse_cmake_disable_flags(cmake_log: str) -> list[str]:
    """Parse cmake error/warning output and return flags to disable missing features."""
    log_lower = cmake_log.lower()
    flags: set[str] = set()

    for signal, disable_flags in _CMAKE_DISABLE_FLAGS:
        if re.search(signal, log_lower):
            flags.update(disable_flags)

    return sorted(flags)


# ── Generalized artifact discovery ─────────────────────────────────────────

def _is_elf_executable(path: Path) -> bool:
    """Check if a file is an ELF executable (not a shared lib)."""
    try:
        if not path.is_file() or path.stat().st_size < 16:
            return False
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic != b"\x7fELF":
            return False
        # Exclude shared libraries
        if path.suffix in (".so", ".a", ".dylib", ".o"):
            return False
        if ".so." in path.name:
            return False
        # Must be executable
        return os.access(path, os.X_OK)
    except Exception:
        return False


def _find_build_artifacts(
    search_dirs: list[Path],
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Search directories for built library (.so/.a) and executable binary.
    Returns (binary, library).
    """
    lib: Optional[Path] = None
    binary: Optional[Path] = None

    # Directories/files to skip
    skip_patterns = {"test", "tests", "example", "examples", "doc", "docs",
                     "bench", "benchmark", "fuzz", "fuzzing", ".git"}

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Find libraries first (prefer .a for static linking with sanitizers)
        if lib is None:
            # Prefer .a files (static — better for sanitizer linking)
            for pattern in ["**/*.a", "**/*.so"]:
                candidates = []
                for c in sorted(search_dir.glob(pattern)):
                    # Skip test/example libs
                    parts_lower = {p.lower() for p in c.parts}
                    if parts_lower & skip_patterns:
                        continue
                    candidates.append(c)
                if candidates:
                    # Prefer largest lib (usually the main project lib)
                    lib = max(candidates, key=lambda p: p.stat().st_size)
                    break

        # Find executables
        if binary is None:
            for f in sorted(search_dir.rglob("*")):
                parts_lower = {p.lower() for p in f.parts}
                if parts_lower & skip_patterns:
                    continue
                if _is_elf_executable(f):
                    binary = f
                    break

    return binary, lib


# ── Project build ──────────────────────────────────────────────────────────

def build_project(task: CyberGymTask, build_info: dict) -> BuildResult:
    """
    Attempt to build the vulnerable project with sanitizers.
    Falls back to a bare build if sanitizer build fails.

    Build strategy:
      1. Auto-install common dependencies
      2. Try sanitizer build
      3. If cmake fails, parse errors → disable missing features → retry
      4. Fall back to bare build
      5. As last resort for cmake, retry with all optional features disabled
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

    # ── Step 0: Auto-install dependencies (opt-in) ──────────────────────────
    if cfg.AUTO_INSTALL_DEPS:
        dep_log = _auto_install_deps(repo)
    else:
        dep_log = "[deps] AUTO_INSTALL_DEPS=False — skipping package install\n"
    log_parts.append(dep_log)

    # ── Helper to attempt a build ──────────────────────────────────────────
    def _attempt(cflags: str, label: str, extra_cmake_flags: list[str] | None = None):
        env = os.environ.copy()
        env["CFLAGS"]   = cflags
        env["CXXFLAGS"] = cflags

        build_dir = work / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)
        build_dir.mkdir(parents=True, exist_ok=True)

        log = f"\n[build] Configuring with CFLAGS='{cflags}'"
        if extra_cmake_flags:
            log += f"  cmake_extras={extra_cmake_flags}"
        log += "\n"
        print(log.strip())

        try:
            if build_type == "cmake":
                cmake_cmd = [
                    "cmake", "-B", str(build_dir), "-S", str(repo),
                    f"-DCMAKE_C_FLAGS={cflags}",
                    f"-DCMAKE_CXX_FLAGS={cflags}",
                ]
                if extra_cmake_flags:
                    cmake_cmd += extra_cmake_flags

                r = _run_cmd(cmake_cmd, cwd=repo, env=env)
                log += r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
                if r.returncode != 0:
                    return None, log + f"\nConfigure failed ({label})"

                print(f"[build] Building: cmake --build {build_dir} -j4")
                r2 = _run_cmd(
                    ["cmake", "--build", str(build_dir), "-j4"],
                    cwd=repo, env=env,
                )
                log += r2.stdout.decode(errors="replace") + r2.stderr.decode(errors="replace")
                if r2.returncode != 0:
                    return None, log + f"\nBuild failed ({label})"

            elif build_type == "meson":
                print("[build] Meson: setting up build")
                r = _run_cmd(
                    ["meson", "setup", str(build_dir), str(repo),
                     f"-Dc_args={cflags}", f"-Dcpp_args={cflags}"],
                    cwd=repo, env=env,
                )
                log += r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
                if r.returncode != 0:
                    return None, log + f"\nMeson setup failed ({label})"

                print("[build] Meson: compiling")
                r2 = _run_cmd(
                    ["meson", "compile", "-C", str(build_dir)],
                    cwd=repo, env=env,
                )
                log += r2.stdout.decode(errors="replace") + r2.stderr.decode(errors="replace")
                if r2.returncode != 0:
                    return None, log + f"\nMeson compile failed ({label})"

            elif build_type in ("autotools", "make"):
                if build_type == "autotools":
                    print("[build] Running bootstrap: autoreconf -fi")
                    _run_cmd(["autoreconf", "-fi"], cwd=repo, env=env)

                configure_script = repo / "configure"
                if configure_script.exists():
                    print("[build] Running configure")
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
                # Unknown build system — try plain make
                print("[build] Building: make -j4 (unknown build system)")
                r = _run_cmd(["make", "-j4"], cwd=repo, env=env)
                log += r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
                if r.returncode != 0:
                    return None, log + f"\nBuild failed ({label})"

        except subprocess.TimeoutExpired:
            return None, log + "\nBuild timed out"
        except Exception as e:
            return None, log + f"\nBuild exception: {e}"

        # ── Find build artifacts (generalized) ─────────────────────────────
        # Search both the build dir and the repo (make-based projects build in-tree)
        binary, lib = _find_build_artifacts([build_dir, repo])

        return (binary or lib), log

    # ── Helper to unpack result ────────────────────────────────────────────
    def _make_build_result(result, log):
        if result is None:
            return None
        lib_val = result if result.suffix in (".so", ".a", ".dylib") else None
        bin_val = result if result.suffix not in (".so", ".a", ".dylib", ".o") else None
        print(f"[build] Result: OK  binary={bin_val}  lib={lib_val}")
        return BuildResult(
            success=True,
            binary_path=bin_val,
            include_dirs=include_dirs,
            lib_path=lib_val,
            build_log="\n".join(log_parts) + "\n" + log,
        )

    # ── Try 1: Sanitizer build ─────────────────────────────────────────────
    if USE_SANITIZERS:
        san_cflags = "-fsanitize=address,undefined -g -O1"
        print(f"[build] Attempt 1: sanitizer build")
        result, log = _attempt(san_cflags, "sanitizer")
        log_parts.append(log)
        br = _make_build_result(result, log)
        if br:
            return br

        # ── Try 1b: cmake with parsed disable flags ───────────────────────
        if build_type == "cmake":
            disable_flags = _parse_cmake_disable_flags(log)
            if disable_flags:
                print(f"[build] Attempt 1b: sanitizer + disable flags: {disable_flags}")
                result, log = _attempt(san_cflags, "sanitizer+disable", disable_flags)
                log_parts.append(log)
                br = _make_build_result(result, log)
                if br:
                    return br

            # ── Try 1c: cmake blanket disable ─────────────────────────────
            print(f"[build] Attempt 1c: sanitizer + blanket disable all optional features")
            result, log = _attempt(san_cflags, "sanitizer+blanket", _CMAKE_BLANKET_DISABLE)
            log_parts.append(log)
            br = _make_build_result(result, log)
            if br:
                return br

        # Re-install deps based on errors we've seen so far
        if cfg.AUTO_INSTALL_DEPS:
            full_log = "\n".join(log_parts)
            dep_log2 = _auto_install_deps(repo, build_log=full_log)
            log_parts.append(dep_log2)

        print("[build] Sanitizer build failed — retrying bare build")

    # ── Try 2: Bare build ──────────────────────────────────────────────────
    bare_cflags = "-g -O1"
    result, log = _attempt(bare_cflags, "bare")
    log_parts.append(log)
    br = _make_build_result(result, log)
    if br:
        return br

    # ── Try 2b: cmake bare with parsed disable ────────────────────────────
    if build_type == "cmake":
        disable_flags = _parse_cmake_disable_flags(log)
        if disable_flags:
            print(f"[build] Attempt 2b: bare + disable flags: {disable_flags}")
            result, log = _attempt(bare_cflags, "bare+disable", disable_flags)
            log_parts.append(log)
            br = _make_build_result(result, log)
            if br:
                return br

        # ── Try 2c: cmake bare blanket disable ────────────────────────────
        print(f"[build] Attempt 2c: bare + blanket disable")
        result, log = _attempt(bare_cflags, "bare+blanket", _CMAKE_BLANKET_DISABLE)
        log_parts.append(log)
        br = _make_build_result(result, log)
        if br:
            return br

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
    poc_source = Path(poc_result.poc_path)
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
