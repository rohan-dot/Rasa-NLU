"""
crs/harness_runner.py — Build the fuzz target and run PoC bytes against it.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from crs.byte_strategies import PoCBytes
from crs.code_intelligence import CodeContext
from crs.config import BUILD_TIMEOUT, RUN_TIMEOUT, cfg
from crs.data_loader import CyberGymTask
from crs.harness_finder import HarnessInfo


@dataclass
class FuzzTargetBuild:
    success: bool
    binary_path: Optional[Path]
    build_log: str


@dataclass
class RunResult:
    poc: PoCBytes
    triggered: bool
    crash_type: str
    sanitizer_output: str
    return_code: int
    run_log: str


_STANDALONE_DRIVER = """\
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);
extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) __attribute__((weak));

int main(int argc, char **argv) {
    if (LLVMFuzzerInitialize) {
        LLVMFuzzerInitialize(&argc, &argv);
    }
    FILE *f = stdin;
    if (argc > 1) {
        f = fopen(argv[1], "rb");
        if (!f) { perror("fopen"); return 1; }
    }
    size_t capacity = 65536, len = 0;
    uint8_t *buf = (uint8_t *)malloc(capacity);
    if (!buf) { perror("malloc"); return 1; }
    while (1) {
        size_t n = fread(buf + len, 1, capacity - len, f);
        len += n;
        if (n == 0) break;
        if (len == capacity) {
            capacity *= 2;
            buf = (uint8_t *)realloc(buf, capacity);
            if (!buf) { perror("realloc"); return 1; }
        }
    }
    if (f != stdin) fclose(f);
    LLVMFuzzerTestOneInput(buf, len);
    free(buf);
    return 0;
}
"""


def _run_cmd(cmd, timeout=BUILD_TIMEOUT, cwd=None, env=None):
    return subprocess.run(cmd, capture_output=True, timeout=timeout,
                          cwd=str(cwd) if cwd else None, env=env)


# ── Sanitizer detection ───────────────────────────────────────────────────

def is_triggered(return_code: int, output: str) -> Tuple[bool, str]:
    for pat in [r"AddressSanitizer", r"ERROR: AddressSanitizer",
                r"heap-buffer-overflow", r"stack-buffer-overflow",
                r"heap-use-after-free", r"SUMMARY: AddressSanitizer"]:
        if re.search(pat, output, re.IGNORECASE):
            m = re.search(r"ERROR: AddressSanitizer: (\S+)", output)
            return True, m.group(1) if m else "asan"
    for pat in [r"runtime error:", r"UndefinedBehaviorSanitizer",
                r"SUMMARY: UndefinedBehaviorSanitizer"]:
        if re.search(pat, output, re.IGNORECASE):
            return True, "ubsan"
    for pat in [r"MemorySanitizer", r"use-of-uninitialized-value",
                r"SUMMARY: MemorySanitizer"]:
        if re.search(pat, output, re.IGNORECASE):
            return True, "msan"
    for pat in [r"Segmentation fault", r"Aborted", r"double free"]:
        if re.search(pat, output, re.IGNORECASE):
            return True, "crash"
    if return_code not in (0, 1) and return_code < 0:
        return True, f"signal_{abs(return_code)}"
    return False, ""


# ── Build project ─────────────────────────────────────────────────────────

def _find_project_dir(repo: Path, build_type: str) -> Path:
    """Find the actual project directory containing the build files.

    Prefer files CLOSEST to the repo root (shortest path) — the main
    project's configure.ac is almost always shallower than subdirectory
    ones like TclMagick/configure.ac.
    """
    if build_type == "autotools":
        # Sort by depth (number of path parts), then alphabetically
        candidates = sorted(repo.rglob("configure.ac"),
                            key=lambda p: (len(p.parts), str(p)))
        if candidates:
            print(f"[build] Found {len(candidates)} configure.ac file(s), "
                  f"using shallowest: {candidates[0].parent}")
            return candidates[0].parent
        candidates = sorted(repo.rglob("configure"),
                            key=lambda p: (len(p.parts), str(p)))
        for sub in candidates:
            if sub.is_file():
                return sub.parent
    elif build_type == "cmake":
        candidates = sorted(repo.rglob("CMakeLists.txt"),
                            key=lambda p: (len(p.parts), str(p)))
        if candidates:
            return candidates[0].parent
    elif build_type == "make":
        candidates = sorted(repo.rglob("Makefile"),
                            key=lambda p: (len(p.parts), str(p)))
        for sub in candidates:
            if not any(x in str(sub).lower() for x in ["test", "doc", "example"]):
                return sub.parent
    return repo


def _build_project(repo: Path, build_info: dict, work: Path) -> Tuple[bool, str]:
    """Build the project with ASAN using its native build system."""
    build_type = build_info.get("type", "unknown")
    log_parts: list[str] = []
    env = os.environ.copy()
    san = "-fsanitize=address,undefined -g -O1"
    env["CFLAGS"] = san
    env["CXXFLAGS"] = san
    env["LDFLAGS"] = "-fsanitize=address,undefined"

    project_dir = _find_project_dir(repo, build_type)
    print(f"[build] Project dir: {project_dir}")
    print(f"[build] Build type: {build_type}")

    try:
        if build_type == "autotools":
            has_configure = (project_dir / "configure").exists()

            if (project_dir / "configure.ac").exists():
                print("[build] autoreconf -fi")
                r = _run_cmd(["autoreconf", "-fi"], cwd=project_dir, env=env,
                             timeout=120)
                autoreconf_ok = (r.returncode == 0)
                log_parts.append(r.stderr.decode(errors="replace"))
                if not autoreconf_ok:
                    if has_configure:
                        print("[build] autoreconf failed but configure exists, continuing...")
                    else:
                        print("[build] autoreconf failed and no configure script found")
                        return False, "\n".join(log_parts)

            if (project_dir / "configure").exists():
                print("[build] ./configure")
                r = _run_cmd(
                    ["./configure", f"CFLAGS={san}", f"CXXFLAGS={san}",
                     f"LDFLAGS=-fsanitize=address,undefined",
                     "--enable-static", "--disable-shared"],
                    cwd=project_dir, env=env, timeout=300,
                )
                log_parts.append(r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace"))
                if r.returncode != 0:
                    # Retry simpler
                    print("[build] Retrying configure with minimal flags...")
                    r = _run_cmd(
                        ["./configure", f"CFLAGS={san}", f"CXXFLAGS={san}"],
                        cwd=project_dir, env=env, timeout=300,
                    )
                    log_parts.append(r.stderr.decode(errors="replace"))
                    if r.returncode != 0:
                        print("[build] configure FAILED")
                        print("\n".join(log_parts)[-1500:])
                        return False, "\n".join(log_parts)

            print("[build] make -j4")
            r = _run_cmd(["make", "-j4"], cwd=project_dir, env=env, timeout=600)
            out = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
            log_parts.append(out)
            if r.returncode != 0:
                # Check if static libs were produced despite the error.
                # GraphicsMagick's make fails on utilities/gm (needs xmlNanoFTP
                # from libxml2) but the core .a libraries are already built.
                partial_libs = list(project_dir.rglob("*.a"))
                partial_libs = [l for l in partial_libs
                                if l.stat().st_size > 10000
                                and "CMakeFiles" not in str(l)]
                if partial_libs:
                    print(f"[build] make failed but found {len(partial_libs)} "
                          f"static lib(s) — treating as partial success")
                    for pl in partial_libs[:5]:
                        print(f"[build]   {pl.name} ({pl.stat().st_size} bytes)")
                else:
                    print(f"[build] make FAILED (no .a libs produced):")
                    print(out[-1500:])
                    return False, "\n".join(log_parts)

        elif build_type == "cmake":
            build_dir = work / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            cmake_source = _find_project_dir(repo, "cmake")
            print(f"[build] cmake -B build -S {cmake_source}")
            r = _run_cmd(
                ["cmake", "-B", str(build_dir), "-S", str(cmake_source),
                 f"-DCMAKE_C_FLAGS={san}", f"-DCMAKE_CXX_FLAGS={san}",
                 "-DBUILD_TESTING=OFF", "-DBUILD_SHARED_LIBS=OFF"],
                cwd=cmake_source, env=env,
            )
            log_parts.append(r.stderr.decode(errors="replace"))
            if r.returncode != 0:
                print("[build] cmake configure FAILED")
                return False, "\n".join(log_parts)
            r = _run_cmd(["cmake", "--build", str(build_dir), "-j4"],
                         cwd=cmake_source, env=env)
            log_parts.append(r.stderr.decode(errors="replace"))
            if r.returncode != 0:
                print("[build] cmake build FAILED")
                return False, "\n".join(log_parts)
        else:
            makefile_dir = _find_project_dir(repo, "make")
            print(f"[build] make -j4 in {makefile_dir}")
            r = _run_cmd(["make", "-j4"], cwd=makefile_dir, env=env, timeout=300)
            log_parts.append(r.stderr.decode(errors="replace"))
            if r.returncode != 0:
                return False, "\n".join(log_parts)

        print("[build] Project build SUCCESS")
        return True, "\n".join(log_parts)

    except subprocess.TimeoutExpired:
        return False, "\n".join(log_parts) + "\nBuild timed out"
    except Exception as e:
        return False, "\n".join(log_parts) + f"\nBuild error: {e}"


def _find_static_lib(repo: Path, work: Path) -> Optional[Path]:
    """Find the main .a static library (legacy — returns biggest)."""
    libs = _find_all_static_libs(repo, work)
    return libs[0] if libs else None


def _find_all_static_libs(repo: Path, work: Path) -> List[Path]:
    """Find ALL .a static libraries needed for linking.

    Projects like GraphicsMagick produce multiple .a files
    (libGraphicsMagick++.a, libGraphicsMagick.a, libpng.a) and the
    fuzz harness needs all of them.
    """
    skip = {"test", "tests", "example", "doc", "fuzz", "CMakeFiles"}
    all_libs: list[Path] = []
    seen: set[str] = set()

    for search in [work / "build", repo]:
        if not search.exists():
            continue
        for lib in sorted(search.rglob("*.a")):
            # Verify the file actually exists (might be stale from previous build)
            try:
                if not lib.is_file():
                    continue
            except OSError:
                continue
            # Skip test/doc directories
            if {p.lower() for p in lib.parts} & skip:
                continue
            # Skip duplicates by filename
            if lib.name in seen:
                continue
            seen.add(lib.name)
            all_libs.append(lib)

    # Sort: largest first (main project lib usually biggest)
    # Wrap stat in try/except for robustness against disappearing files
    def _safe_size(p: Path) -> int:
        try:
            return p.stat().st_size
        except OSError:
            return 0

    all_libs = [l for l in all_libs if _safe_size(l) > 0]
    all_libs.sort(key=_safe_size, reverse=True)

    for lib in all_libs:
        print(f"[build] Found static lib: {lib} ({lib.stat().st_size} bytes)")

    return all_libs


def _copy_data_files(repo: Path, target_dir: Path):
    """Copy data files the fuzz target needs (magic.mgc, .dict, etc.)."""
    for pattern in ["magic.mgc", "*.dict"]:
        for f in repo.rglob(pattern):
            dest = target_dir / f.name
            if not dest.exists():
                try:
                    shutil.copy2(f, dest)
                    print(f"[build] Copied: {f.name}")
                except Exception:
                    pass


# ── Build fuzz target ──────────────────────────────────────────────────────

def build_fuzz_target(
    task: CyberGymTask,
    harness: HarnessInfo,
    context: CodeContext,
) -> FuzzTargetBuild:
    repo = Path(task.repo_path).resolve()
    work = cfg.task_work_dir(task.task_id)
    log_parts: list[str] = []

    # Clean stale build artifacts from previous runs (e.g. old cmake build/ dir)
    stale_build = work / "build"
    if stale_build.exists():
        print(f"[build] Removing stale build dir: {stale_build}")
        shutil.rmtree(stale_build, ignore_errors=True)

    # Step 1: Build project
    print(f"\n[build] Building project...")
    build_ok, build_log = _build_project(repo, context.build_info, work)
    log_parts.append(build_log)

    static_libs = _find_all_static_libs(repo, work)
    if not static_libs:
        print("[build] No static library found")
        return FuzzTargetBuild(False, None, "\n".join(log_parts))

    # Step 2: Write driver
    driver_path = work / "standalone_driver.cpp"
    driver_path.write_text(_STANDALONE_DRIVER, encoding="utf-8")

    # Step 3: Collect includes
    include_dirs: set[str] = set()
    for h in repo.rglob("*.h"):
        include_dirs.add(str(h.parent))
        # Also add PARENT of the header's directory — needed for includes
        # like <magick/magick_config.h> where the -I must point to the
        # grandparent (graphicsmagick/) not the direct parent (magick/)
        if h.parent.parent != repo:
            include_dirs.add(str(h.parent.parent))
    include_dirs.add(str(repo))
    # Add the build system's source directory (where configure ran)
    source_dir = context.build_info.get("source_dir")
    if source_dir:
        include_dirs.add(source_dir)
    for name in ["include", "src", "lib"]:
        for d in repo.rglob(name):
            if d.is_dir():
                include_dirs.add(str(d))
    # Add all immediate subdirectories of repo (covers graphicsmagick/ under src-vul/)
    for d in repo.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            include_dirs.add(str(d))

    # Step 3b: Collect library search paths (.libs dirs from autotools)
    lib_dirs: set[str] = set()
    for d in repo.rglob(".libs"):
        if d.is_dir():
            lib_dirs.add(str(d))
    for lib in static_libs:
        lib_dirs.add(str(lib.parent))

    # Step 4: Compile — link ALL static libs, not just one
    fuzz_binary = work / "fuzz_target"
    cmd = ["g++", "-fsanitize=address,undefined", "-g", "-O1", "-fpermissive",
           "-include", "string.h",
           str(driver_path), str(harness.harness_path),
           "-o", str(fuzz_binary)]

    # Add all static libraries (order matters: C++ wrappers before C core)
    # Sort so that C++ libs (e.g. libGraphicsMagick++.a) come before C libs
    cpp_libs = [l for l in static_libs if "++" in l.name]
    c_libs   = [l for l in static_libs if "++" not in l.name]
    for lib in cpp_libs + c_libs:
        cmd.append(str(lib))

    cmd += [f"-I{d}" for d in sorted(include_dirs)]
    cmd += [f"-L{d}" for d in sorted(lib_dirs)]

    # Common system libraries — add them all; unused ones are silently ignored
    cmd += ["-lz", "-lm", "-lpthread", "-ldl", "-lrt"]
    # Image/media libraries that projects like GraphicsMagick often need
    for syslib in ["jpeg", "png", "tiff", "freetype", "bz2", "xml2",
                   "lcms2", "lzma", "webp", "gomp", "ltdl", "stdc++"]:
        cmd.append(f"-l{syslib}")

    print(f"[build] Compiling fuzz target...")
    print(f"[build] Linking {len(static_libs)} static lib(s): "
          f"{[l.name for l in static_libs]}")
    try:
        r = _run_cmd(cmd, timeout=BUILD_TIMEOUT, cwd=work)
        clog = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
        log_parts.append(clog)
        if r.returncode != 0:
            # Retry: drop failing -l flags one at a time
            print(f"[build] First compile attempt failed, retrying with minimal libs...")
            cmd_retry = [x for x in cmd if not any(
                x == f"-l{lib}" for lib in
                ["jpeg","png","tiff","freetype","bz2","xml2","lcms2",
                 "lzma","webp","gomp","ltdl"]
            )]
            r = _run_cmd(cmd_retry, timeout=BUILD_TIMEOUT, cwd=work)
            clog2 = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
            log_parts.append(clog2)
            if r.returncode != 0:
                print(f"[build] Compile FAILED:")
                print(clog2[-2000:])
                return FuzzTargetBuild(False, None, "\n".join(log_parts))
    except subprocess.TimeoutExpired:
        return FuzzTargetBuild(False, None, "\n".join(log_parts) + "\nCompile timed out")

    # Step 5: Copy data files
    _copy_data_files(repo, work)
    os.chmod(fuzz_binary, 0o755)

    print(f"[build] ✓ Fuzz target ready: {fuzz_binary}")
    return FuzzTargetBuild(True, fuzz_binary, "\n".join(log_parts))


# ── Run bytes ──────────────────────────────────────────────────────────────

def run_poc_bytes(poc: PoCBytes, fuzz_binary: Path) -> RunResult:
    print(f"[run] Feeding {len(poc.data)} bytes to {fuzz_binary.name}")
    try:
        r = subprocess.run(
            [str(fuzz_binary)], input=poc.data, capture_output=True,
            timeout=RUN_TIMEOUT, cwd=str(fuzz_binary.parent),
        )
        output = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
        triggered, crash_type = is_triggered(r.returncode, output)
        status = "TRIGGERED" if triggered else "no trigger"
        print(f"[run] Result: {status}  rc={r.returncode}  crash_type={crash_type}")
        if triggered:
            for line in output.split("\n"):
                if "ERROR:" in line or "SUMMARY:" in line:
                    print(f"[run]   {line.strip()}")
        return RunResult(poc=poc, triggered=triggered, crash_type=crash_type,
                        sanitizer_output=output[:8000], return_code=r.returncode,
                        run_log=output[:4000])
    except subprocess.TimeoutExpired:
        return RunResult(poc=poc, triggered=False, crash_type="timeout",
                        sanitizer_output="", return_code=-1, run_log="Timed out")
    except Exception as e:
        return RunResult(poc=poc, triggered=False, crash_type="error",
                        sanitizer_output="", return_code=-1, run_log=str(e))


def execute_pipeline(fuzz_binary: Path, poc_candidates: list[PoCBytes]) -> list[RunResult]:
    results: list[RunResult] = []
    print(f"\n  [Pipeline] Testing {len(poc_candidates)} PoC candidate(s)")
    for idx, poc in enumerate(poc_candidates):
        print(f"\n  [PoC {idx}] {poc.strategy_name} ({len(poc.data)} bytes)")
        result = run_poc_bytes(poc, fuzz_binary)
        results.append(result)
        if result.triggered:
            print(f"\n  *** PoC {idx} TRIGGERED: {result.crash_type} ***")
            break
    return results
