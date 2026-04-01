"""
crs/harness_runner.py — Build fuzz target and run PoC bytes against it.
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


def is_triggered(return_code, output):
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


def _find_project_dir(repo, build_type):
    if build_type == "autotools":
        for sub in sorted(repo.rglob("configure.ac")):
            return sub.parent
        for sub in sorted(repo.rglob("configure")):
            if sub.is_file(): return sub.parent
    elif build_type == "cmake":
        for sub in sorted(repo.rglob("CMakeLists.txt")):
            return sub.parent
    elif build_type == "make":
        for sub in sorted(repo.rglob("Makefile")):
            if not any(x in str(sub).lower() for x in ["test", "doc", "example"]):
                return sub.parent
    return repo


def _build_project(repo, build_info, work):
    build_type = build_info.get("type", "unknown")
    log_parts = []
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
            if (project_dir / "configure.ac").exists():
                print("[build] autoreconf -fi")
                r = _run_cmd(["autoreconf", "-fi"], cwd=project_dir, env=env)
                log_parts.append(r.stderr.decode(errors="replace"))

            if (project_dir / "configure").exists():
                print("[build] ./configure")
                r = _run_cmd(
                    ["./configure", f"CFLAGS={san}", f"CXXFLAGS={san}",
                     f"LDFLAGS=-fsanitize=address,undefined",
                     "--enable-static", "--disable-shared"],
                    cwd=project_dir, env=env, timeout=180)
                log_parts.append(r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace"))
                if r.returncode != 0:
                    r = _run_cmd(["./configure", f"CFLAGS={san}", f"CXXFLAGS={san}"],
                                 cwd=project_dir, env=env, timeout=180)
                    log_parts.append(r.stderr.decode(errors="replace"))
                    if r.returncode != 0:
                        print("[build] configure FAILED")
                        return False, "\n".join(log_parts)

            print("[build] make -j4")
            r = _run_cmd(["make", "-j4"], cwd=project_dir, env=env, timeout=300)
            out = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
            log_parts.append(out)
            if r.returncode != 0:
                print(f"[build] make FAILED (last 1000):")
                print(out[-1000:])
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
                cwd=cmake_source, env=env)
            log_parts.append(r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace"))
            if r.returncode != 0:
                print("[build] cmake configure FAILED")
                print(log_parts[-1][-1000:])
                return False, "\n".join(log_parts)
            print("[build] cmake --build")
            r = _run_cmd(["cmake", "--build", str(build_dir), "-j4"],
                         cwd=cmake_source, env=env, timeout=300)
            log_parts.append(r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace"))
            if r.returncode != 0:
                print("[build] cmake build FAILED")
                print(log_parts[-1][-1000:])
                # Don't return failure — partial builds may have produced .a files
                # that are enough for the fuzz target
                print("[build] Continuing despite build errors (partial .a files may exist)")
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


def _find_static_libs(repo, work):
    """Find ALL .a static libraries built by the project."""
    skip = {"test", "tests", "example", "doc", "fuzz", "fuzzing"}
    libs = []
    seen_names = set()
    for search in [work / "build", repo]:
        if not search.exists():
            continue
        for lib in sorted(search.rglob("*.a")):
            try:
                if not lib.exists():
                    continue
                if lib.stat().st_size < 1000:
                    continue  # skip tiny/empty archives
                parts_lower = {p.lower() for p in lib.parts}
                if parts_lower & skip:
                    continue
                # Avoid duplicates by name
                if lib.name in seen_names:
                    continue
                seen_names.add(lib.name)
                libs.append(lib)
            except (OSError, FileNotFoundError):
                continue
    if libs:
        print(f"[build] Found {len(libs)} static lib(s):")
        for l in libs:
            print(f"[build]   {l.name} ({l.stat().st_size} bytes)")
    return libs


def _copy_data_files(repo, target_dir):
    for pattern in ["magic.mgc", "*.dict", "*.xml"]:
        for f in repo.rglob(pattern):
            dest = target_dir / f.name
            if not dest.exists():
                try:
                    shutil.copy2(f, dest)
                    print(f"[build] Copied: {f.name}")
                except Exception:
                    pass


def build_fuzz_target(task, harness, context):
    repo = Path(task.repo_path).resolve()
    work = cfg.task_work_dir(task.task_id)
    log_parts = []

    # Step 1: Build project
    print(f"\n[build] Building project...")
    build_ok, build_log = _build_project(repo, context.build_info, work)
    log_parts.append(build_log)

    # Step 2: Find ALL static libs
    static_libs = _find_static_libs(repo, work)
    if not static_libs:
        print("[build] No static libraries found")
        return FuzzTargetBuild(False, None, "\n".join(log_parts))

    # Step 3: Write driver
    driver_path = work / "standalone_driver.cpp"
    driver_path.write_text(_STANDALONE_DRIVER, encoding="utf-8")

    # Step 4: Collect include dirs — be thorough
    include_dirs = set()
    # Add every directory containing a .h file AND its parent
    for h in repo.rglob("*.h"):
        include_dirs.add(str(h.parent))
        include_dirs.add(str(h.parent.parent))
    include_dirs.add(str(repo))
    # Add build dir and all its subdirs with headers
    build_dir = work / "build"
    if build_dir.exists():
        include_dirs.add(str(build_dir))
        for h in build_dir.rglob("*.h"):
            include_dirs.add(str(h.parent))
            include_dirs.add(str(h.parent.parent))
    # Standard subdirectory names
    for name in ["include", "src", "lib"]:
        for d in repo.rglob(name):
            if d.is_dir():
                include_dirs.add(str(d))

    # Step 5: Compile fuzz target with ALL .a libraries
    fuzz_binary = work / "fuzz_target"
    cmd = ["g++", "-fsanitize=address,undefined", "-g", "-O1", "-fpermissive",
           "-include", "string.h",
           str(driver_path), str(harness.harness_path),
           "-o", str(fuzz_binary)]
    cmd += [str(lib) for lib in static_libs]
    cmd += [f"-I{d}" for d in sorted(include_dirs)]
    cmd += ["-lz", "-lm", "-lpthread"]

    print(f"[build] Compiling fuzz target with {len(static_libs)} lib(s)...")
    try:
        r = _run_cmd(cmd, timeout=BUILD_TIMEOUT, cwd=work)
        clog = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
        log_parts.append(clog)
        if r.returncode != 0:
            print(f"[build] Compile FAILED:")
            print(clog[-2000:])

            # Retry without sanitizers
            print("[build] Retrying without sanitizers...")
            cmd2 = [c for c in cmd if not c.startswith("-fsanitize")]
            r = _run_cmd(cmd2, timeout=BUILD_TIMEOUT, cwd=work)
            clog2 = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
            log_parts.append(clog2)
            if r.returncode != 0:
                print(f"[build] Compile FAILED (no sanitizer):")
                print(clog2[-2000:])
                return FuzzTargetBuild(False, None, "\n".join(log_parts))
    except subprocess.TimeoutExpired:
        return FuzzTargetBuild(False, None, "\n".join(log_parts) + "\nCompile timed out")

    # Step 6: Copy data files
    _copy_data_files(repo, work)
    os.chmod(fuzz_binary, 0o755)

    print(f"[build] ✓ Fuzz target ready: {fuzz_binary}")
    return FuzzTargetBuild(True, fuzz_binary, "\n".join(log_parts))


def run_poc_bytes(poc, fuzz_binary):
    print(f"[run] Feeding {len(poc.data)} bytes to {fuzz_binary.name}")
    try:
        r = subprocess.run(
            [str(fuzz_binary)], input=poc.data, capture_output=True,
            timeout=RUN_TIMEOUT, cwd=str(fuzz_binary.parent))
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


def execute_pipeline(fuzz_binary, poc_candidates):
    results = []
    print(f"\n  [Pipeline] Testing {len(poc_candidates)} PoC candidate(s)")
    for idx, poc in enumerate(poc_candidates):
        print(f"\n  [PoC {idx}] {poc.strategy_name} ({len(poc.data)} bytes)")
        result = run_poc_bytes(poc, fuzz_binary)
        results.append(result)
        if result.triggered:
            print(f"\n  *** PoC {idx} TRIGGERED: {result.crash_type} ***")
            break
    return results
