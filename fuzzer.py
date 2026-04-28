"""
fuzzer.py — LibFuzzer process management.

Launches LibFuzzer against the target harness, monitors for crashes,
and collects crash artifacts for analysis.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("gemma-fuzzer.fuzzer")


@dataclass
class CrashInfo:
    """Represents a single crash found by LibFuzzer."""
    crash_file: str
    stack_trace: str
    crash_type: str        # e.g. "heap-buffer-overflow"
    timestamp: float
    dedup_key: str = ""    # first 3 frames for dedup


@dataclass
class FuzzerState:
    """Shared state for the fuzzer."""
    crashes: list[CrashInfo] = field(default_factory=list)
    seen_dedup_keys: set[str] = field(default_factory=set)
    total_execs: int = 0
    is_running: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class LibFuzzerRunner:
    """Manages a LibFuzzer process and harvests crashes."""

    def __init__(
        self,
        build_dir: str,
        harness: str,
        corpus_dir: str,
        crash_dir: str,
        seed_dir: str | None = None,
        jobs: int = 1,
    ):
        self.build_dir = Path(build_dir)
        self.harness = harness
        self.corpus_dir = Path(corpus_dir)
        self.crash_dir = Path(crash_dir)
        self.seed_dir = Path(seed_dir) if seed_dir else None
        self.jobs = jobs
        self.state = FuzzerState()
        self._proc: subprocess.Popen | None = None

        # Ensure directories exist
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.crash_dir.mkdir(parents=True, exist_ok=True)

    def _find_binary(self) -> Path:
        """Locate the harness binary in the build output."""
        # OSS-Fuzz names the binary after the harness
        candidates = [
            self.build_dir / self.harness,
            self.build_dir / f"{self.harness}_fuzzer",
        ]
        # Also glob for partial matches
        for g in self.build_dir.glob(f"*{self.harness}*"):
            if g.is_file() and os.access(g, os.X_OK):
                candidates.append(g)

        for c in candidates:
            if c.is_file() and os.access(c, os.X_OK):
                logger.info("Found harness binary: %s", c)
                return c

        raise FileNotFoundError(
            f"No executable harness matching '{self.harness}' in {self.build_dir}. "
            f"Available files: {list(self.build_dir.iterdir())}"
        )

    def _find_dict(self) -> Path | None:
        """Look for a fuzzer dictionary."""
        for pattern in [f"{self.harness}.dict", "*.dict"]:
            matches = list(self.build_dir.glob(pattern))
            if matches:
                logger.info("Using dictionary: %s", matches[0])
                return matches[0]
        return None

    def _find_seed_corpus(self) -> Path | None:
        """Look for a seed corpus archive or directory."""
        # OSS-Fuzz convention: <harness>_seed_corpus.zip
        for pattern in [
            f"{self.harness}_seed_corpus.zip",
            f"{self.harness}_seed_corpus",
        ]:
            matches = list(self.build_dir.glob(pattern))
            if matches:
                p = matches[0]
                if p.suffix == ".zip":
                    # Unzip into corpus dir
                    import zipfile
                    logger.info("Unpacking seed corpus: %s", p)
                    with zipfile.ZipFile(p) as zf:
                        zf.extractall(self.corpus_dir)
                    return self.corpus_dir
                elif p.is_dir():
                    return p
        return None

    def _build_command(self, binary: Path) -> list[str]:
        """Construct the LibFuzzer command line."""
        cmd = [
            str(binary),
            str(self.corpus_dir),
        ]

        # Add seed directories
        if self.seed_dir and self.seed_dir.exists() and any(self.seed_dir.iterdir()):
            cmd.append(str(self.seed_dir))

        # LibFuzzer flags
        cmd.extend([
            f"-artifact_prefix={self.crash_dir}/",
            "-print_final_stats=1",
            "-detect_leaks=0",           # ASAN leak detection is noisy
            f"-jobs={self.jobs}",
            f"-workers={self.jobs}",
            "-max_len=4096",
            "-timeout=30",               # per-input timeout
        ])

        # Dictionary
        dict_path = self._find_dict()
        if dict_path:
            cmd.append(f"-dict={dict_path}")

        return cmd

    def start(self, duration: int) -> None:
        """Start LibFuzzer for the given duration (seconds)."""
        binary = self._find_binary()
        self._find_seed_corpus()  # unpack seeds if available

        cmd = self._build_command(binary)
        cmd.append(f"-max_total_time={duration}")

        env = os.environ.copy()
        # ASAN options for better crash reporting
        env["ASAN_OPTIONS"] = (
            "abort_on_error=1:"
            "symbolize=1:"
            "detect_leaks=0:"
            "print_scariness=1:"
            "handle_abort=1"
        )

        logger.info("Starting LibFuzzer: %s", " ".join(cmd))
        self.state.is_running = True

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )

        # Monitor output in a thread
        monitor = threading.Thread(
            target=self._monitor_output, daemon=True
        )
        monitor.start()

    def _monitor_output(self) -> None:
        """Read LibFuzzer output, detect crashes, log progress."""
        assert self._proc is not None
        crash_buffer: list[str] = []
        in_crash = False

        for raw_line in self._proc.stdout:  # type: ignore
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            logger.debug("[libfuzzer] %s", line)

            # Detect crash start
            if "ERROR: AddressSanitizer:" in line or "SUMMARY:" in line:
                in_crash = True

            if in_crash:
                crash_buffer.append(line)

            # Detect crash end (artifact saved)
            if line.startswith("artifact_prefix=") or (
                "Test unit written to" in line
            ):
                in_crash = False
                if crash_buffer:
                    self._process_crash(crash_buffer)
                    crash_buffer = []

            # Parse stats
            if line.startswith("#") and "exec/s:" in line:
                try:
                    parts = line.split()
                    idx = parts.index("exec/s:")
                    self.state.total_execs = int(parts[1])
                except (ValueError, IndexError):
                    pass

        self._proc.wait()
        self.state.is_running = False
        logger.info(
            "LibFuzzer finished. Total execs: %d, Crashes: %d",
            self.state.total_execs, len(self.state.crashes),
        )

    def _process_crash(self, lines: list[str]) -> None:
        """Parse a crash from LibFuzzer output."""
        stack_trace = "\n".join(lines)

        # Extract crash type
        crash_type = "unknown"
        for line in lines:
            if "ERROR: AddressSanitizer:" in line:
                # e.g. "ERROR: AddressSanitizer: heap-buffer-overflow on ..."
                parts = line.split("AddressSanitizer:")
                if len(parts) > 1:
                    crash_type = parts[1].strip().split()[0]

        # Build a dedup key from the top 3 stack frames
        frames = []
        for line in lines:
            if line.strip().startswith("#") and " in " in line:
                # e.g. "#0 0x55... in FunctionName file.c:42"
                parts = line.split(" in ")
                if len(parts) > 1:
                    frames.append(parts[1].strip().split()[0])
                if len(frames) >= 3:
                    break
        dedup_key = f"{crash_type}|{'|'.join(frames)}"

        with self.state.lock:
            if dedup_key in self.state.seen_dedup_keys:
                logger.debug("Duplicate crash (skipped): %s", dedup_key)
                return
            self.state.seen_dedup_keys.add(dedup_key)

        # Find the crash file
        crash_files = sorted(
            self.crash_dir.glob("crash-*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        crash_file = str(crash_files[0]) if crash_files else "unknown"

        crash = CrashInfo(
            crash_file=crash_file,
            stack_trace=stack_trace,
            crash_type=crash_type,
            timestamp=time.time(),
            dedup_key=dedup_key,
        )

        with self.state.lock:
            self.state.crashes.append(crash)
        logger.info(
            "NEW CRASH [%s] %s — %s",
            crash_type, dedup_key, crash_file,
        )

    def get_new_crashes(self, since_idx: int = 0) -> list[CrashInfo]:
        """Return crashes found since the given index."""
        with self.state.lock:
            return list(self.state.crashes[since_idx:])

    def collect_crash_files(self) -> list[str]:
        """Return paths to all crash/leak/timeout files."""
        patterns = ["crash-*", "leak-*", "timeout-*", "oom-*"]
        files = []
        for p in patterns:
            files.extend(str(f) for f in self.crash_dir.glob(p))
        return sorted(files)

    def wait(self) -> None:
        """Wait for the fuzzer process to finish."""
        if self._proc:
            self._proc.wait()

    def is_running(self) -> bool:
        return self.state.is_running
