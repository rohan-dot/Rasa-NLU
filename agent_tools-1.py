"""
agent_tools.py — Agentic tool-use loop for the discver CRS.

WHY THIS EXISTS
---------------
The one-shot agents build a prompt, call the LLM once, and hope. The strongest
AIxCC CRSs and Anthropic's reference harness instead let the model INVESTIGATE
WITH REAL TOOLS and EXECUTE:

    read code + run static analyzers  ->  craft an input  ->  RUN it against the
    sanitizer binary  ->  read the real crash / coverage  ->  refine  ->  repeat
    ->  reproduce 3/3.

This module gives the LLM that loop. It mixes two kinds of tools:
  * LLM-reasoning helpers (read_function, grep, list_callers, read_file)
  * REAL bug-finding tools (semgrep, cppcheck, gdb, valgrind), executable
    oracles (run_input, coverage), and the ability to rewrite its own harness
    (write_and_compile_harness) or build structured inputs (python_exec).

REPO-AGNOSTIC BY DESIGN  (works on any target, any language, not just C/libxml)
-------------------------------------------------------------------------------
  * No hardcoded filenames, languages, or build steps.
  * Compilation is a CALLBACK (`compile_fn`) supplied by the caller, so C goes
    through agents.compile_harness and Rust goes through your language plugin.
  * Input delivery is configurable (`input_mode`: "arg" for libFuzzer/cargo-fuzz,
    "stdin" for AFL-style harnesses).
  * Every external tool is probed with shutil.which at runtime and degrades to a
    clear "not installed" message. The model is shown the live tool list up front
    and is told to call only those.

Fully additive: nothing here replaces your existing agents. A caller opts in via
`solve_with_tools(...)`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger("discver.agent_tools")

# Sanitizer / crash signatures = our oracles. A "crash" only counts if one of
# these fires (or a killing signal), never a plain non-zero exit — keeps junk
# out of the PoV pile across any language.
_SANITIZER_MARKERS = (
    "ERROR: AddressSanitizer",
    "ERROR: LeakSanitizer",
    "ERROR: MemorySanitizer",
    "ERROR: ThreadSanitizer",
    "runtime error:",                       # UBSAN
    "SUMMARY: UndefinedBehaviorSanitizer",
    "SUMMARY: AddressSanitizer",
    "SUMMARY: MemorySanitizer",
    "SUMMARY: ThreadSanitizer",
    "panicked at",                          # Rust panic (any thread name)
    "thread 'main' panicked",
    "AddressSanitizer:DEADLYSIGNAL",
    "ERROR: libFuzzer: deadly signal",      # libFuzzer's generic crash banner (covers
    "ERROR: libFuzzer:",                     # Rust panics intercepted by libFuzzer's handler)
)

# Source extensions across the languages a CRS might face. Used only for
# grep/read fallbacks — never to assume a language.
SOURCE_GLOBS = ("*.c", "*.cc", "*.cpp", "*.cxx", "*.h", "*.hpp", "*.hh",
                "*.rs", "*.go", "*.java", "*.py", "*.js", "*.ts")

_SKIP_DIRS = (".git", "/node_modules", "/target/debug", "/target/release")


def _which(name: str) -> bool:
    return shutil.which(name) is not None


def _tail(s: str, n: int = 3000) -> str:
    return s[-n:] if s else ""


# ──────────────────────────────────────────────────────────────────
# Executable oracle: run one input against the harness binary.
# input_mode "arg"  -> libFuzzer / cargo-fuzz : ./harness <file>
# input_mode "stdin"-> AFL-style persistent harness reading stdin
# ──────────────────────────────────────────────────────────────────
def run_one(binary: str, data: bytes, timeout: int = 20,
            input_mode: str = "arg") -> tuple[bool, str]:
    """Run the harness once on a single input. Returns (sanitizer_fired, output_tail)."""
    env = os.environ.copy()
    env["ASAN_OPTIONS"] = "abort_on_error=1:symbolize=1:detect_leaks=1:halt_on_error=1"
    env["UBSAN_OPTIONS"] = "print_stacktrace=1:halt_on_error=1"
    env.setdefault("RUST_BACKTRACE", "1")

    path = None
    try:
        if input_mode == "stdin":
            proc = subprocess.run([binary], input=data, capture_output=True,
                                  timeout=timeout, env=env)
        else:
            with tempfile.NamedTemporaryFile(prefix="discver_in_", delete=False) as fh:
                fh.write(data)
                path = fh.name
            proc = subprocess.run([binary, path], capture_output=True,
                                  timeout=timeout, env=env)
        out = (proc.stdout or b"").decode("utf-8", "replace")
        out += (proc.stderr or b"").decode("utf-8", "replace")
        fired = any(m in out for m in _SANITIZER_MARKERS)
        if not fired and proc.returncode is not None and proc.returncode < 0:
            fired = True            # killed by signal (SEGV/ABRT) with no clean exit
        return fired, _tail(out)
    except subprocess.TimeoutExpired:
        return False, "[timeout] no sanitizer crash within %ds (possible hang)" % timeout
    except Exception as exc:
        return False, f"[run error] {exc}"
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def reproduces(binary: str, data: bytes, trials: int = 3, timeout: int = 20,
               input_mode: str = "arg") -> tuple[bool, str]:
    """Cheap programmatic gate: a PoV must fire the sanitizer trials/trials times.
    Kills flaky / non-deterministic 'crashes' before they reach the verifier."""
    last = ""
    for _ in range(trials):
        fired, last = run_one(binary, data, timeout=timeout, input_mode=input_mode)
        if not fired:
            return False, last
    return True, last


# ──────────────────────────────────────────────────────────────────
# The toolbox the model may call.
# ──────────────────────────────────────────────────────────────────
@dataclass
class ToolBox:
    src_dir: str
    output_dir: str
    harness_binary: str | None = None                 # compiled sanitizer harness, if any
    call_graph: object | None = None                  # code_analysis.CallGraph (optional)
    static_context: str = ""                          # preloaded static/taint findings (optional)
    compile_fn: Callable[[str, str], tuple] | None = None  # (code, name) -> (bin_path|None, err)
    input_mode: str = "arg"                           # "arg" | "stdin"
    language: str = ""                                # optional hint: c/cpp/rust/go/...
    corpus_dir: str | None = None                     # SHARED fuzzer corpus (Gondar seed-sharing)
    target_funcs: tuple = ()                           # sink function names to drive "reach" toward
    _src_root: Path = field(init=False)
    _cov_funcs: set = field(default_factory=set, init=False)
    _cache: dict = field(default_factory=dict, init=False)
    _avail: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._src_root = Path(self.src_dir).resolve()
        if self.corpus_dir:
            Path(self.corpus_dir).mkdir(parents=True, exist_ok=True)

    def _deposit(self, data: bytes) -> None:
        """Gondar-style seed sharing: every input the agent tries (even near-misses)
        is written into the shared fuzzer corpus, so a CONCURRENTLY-running fuzzer can
        mutate a structurally-close attempt into a working trigger. This is where the
        bugs that neither the agent nor the fuzzer find alone come from."""
        if not self.corpus_dir or not data:
            return
        try:
            p = Path(self.corpus_dir) / f"agent-{hashlib.sha1(data).hexdigest()[:16]}"
            if not p.exists():
                p.write_bytes(data)
        except Exception:
            pass

    # -- helpers ---------------------------------------------------
    def _resolve(self, rel: str) -> str | None:
        if not rel:
            return None
        target = (self._src_root / rel).resolve()
        if not str(target).startswith(str(self._src_root)):
            return None
        return str(target)

    def available_tools(self) -> str:
        if not self._avail:
            self._avail = {
                "semgrep": _which("semgrep"),
                "cppcheck": _which("cppcheck"),
                "flawfinder": _which("flawfinder"),
                "gdb": _which("gdb"),
                "valgrind": _which("valgrind"),
            }
        live = [k for k, v in self._avail.items() if v]
        lines = [
            "Always-on: read_function, list_callers, grep, read_file, run_input, "
            "coverage, submit, give_up.",
            f"Installed external tools you MAY call: {', '.join(live) if live else '(none)'}.",
            f"write_and_compile_harness: {'available' if self.compile_fn else 'UNAVAILABLE (no compiler wired)'}.",
            f"python_exec: available (sandboxed).",
            f"run_input/coverage: {'available' if self.harness_binary else 'UNAVAILABLE (no harness binary; reason only, cannot verify)'}.",
        ]
        if self.language:
            lines.append(f"Target language: {self.language}.")
        return "\n".join(lines)

    # -- read code (LLM-reasoning helpers) -------------------------
    def read_function(self, name: str) -> str:
        if self.call_graph is not None:
            try:
                ctx = self.call_graph.get_function_context(name, self.src_dir)
                if ctx:
                    return ctx[:4000]
            except Exception:
                pass
        return self.grep(rf"\b{re.escape(name)}\b", max_hits=8)

    def list_callers(self, name: str) -> str:
        if self.call_graph is None:
            return "(no call graph available)"
        try:
            chains = self.call_graph.get_callers(name, depth=3)
            if not chains:
                return f"(no known callers of {name} — may be an entry point)"
            return "\n".join(" -> ".join(c) for c in chains[:10])
        except Exception as exc:
            return f"(caller lookup failed: {exc})"

    def grep(self, pattern: str, max_hits: int = 20) -> str:
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            return f"(bad regex: {exc})"
        hits: list[str] = []
        for g in SOURCE_GLOBS:
            for p in self._src_root.rglob(g):
                if any(s in str(p) for s in _SKIP_DIRS):
                    continue
                try:
                    for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                        if rx.search(line):
                            hits.append(f"{p.relative_to(self._src_root)}:{i}: {line.strip()[:160]}")
                            if len(hits) >= max_hits:
                                return "\n".join(hits)
                except Exception:
                    continue
        return "\n".join(hits) if hits else "(no matches)"

    def read_file(self, rel_path: str, start: int = 1, end: int = 120) -> str:
        target = self._resolve(rel_path)
        if target is None:
            return "(refused: path escapes source tree)"
        tp = Path(target)
        if not tp.is_file():
            return f"(no such file: {rel_path})"
        lines = tp.read_text(errors="replace").splitlines()
        start = max(1, start)
        chunk = lines[start - 1:min(end, start - 1 + 200)]
        return "\n".join(f"{start + i}: {ln}" for i, ln in enumerate(chunk))[:4000]

    # -- REAL static-analysis tools (multi-language, graceful) -----
    def semgrep(self, path: str = "") -> str:
        if not _which("semgrep"):
            return "(semgrep not installed)"
        target = self._resolve(path) or self.src_dir
        key = ("semgrep", target)
        if key in self._cache:
            return self._cache[key]
        cmd = ["semgrep", "scan", "--config", "auto", "--json", "--metrics=off",
               "--quiet", "--timeout", "20", "--max-target-bytes", "2000000", target]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=180, text=True)
            data = json.loads(proc.stdout or "{}")
            results = data.get("results", [])
            if not results:
                msg = "(semgrep: no findings)"
            else:
                rows = []
                for r in results[:25]:
                    rel = r.get("path", "?")
                    ln = r.get("start", {}).get("line", "?")
                    rule = r.get("check_id", "?").split(".")[-1]
                    m = (r.get("extra", {}).get("message", "") or "").strip().replace("\n", " ")
                    rows.append(f"{rel}:{ln}: [{rule}] {m[:140]}")
                msg = f"semgrep: {len(results)} findings (top {len(rows)}):\n" + "\n".join(rows)
        except json.JSONDecodeError:
            msg = "(semgrep produced no parseable output — rules may need a one-time fetch / network)"
        except subprocess.TimeoutExpired:
            msg = "(semgrep timed out — try a narrower path=<file>)"
        except Exception as exc:
            msg = f"(semgrep error: {exc})"
        self._cache[key] = msg
        return msg

    def cppcheck(self, path: str = "") -> str:
        if not _which("cppcheck"):
            return "(cppcheck not installed)"
        target = self._resolve(path) or self.src_dir
        key = ("cppcheck", target)
        if key in self._cache:
            return self._cache[key]
        cmd = ["cppcheck", "--enable=warning,style,performance,portability",
               "--inconclusive", "--quiet",
               "--template={file}:{line}: {severity}: {message} [{id}]", target]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=120, text=True)
            out = (proc.stderr or "").strip()
            msg = ("cppcheck:\n" + _tail(out, 2500)) if out else "(cppcheck: no findings)"
        except subprocess.TimeoutExpired:
            msg = "(cppcheck timed out — try a narrower path=<file>)"
        except Exception as exc:
            msg = f"(cppcheck error: {exc})"
        self._cache[key] = msg
        return msg

    # -- REAL dynamic crash-triage tools ---------------------------
    def gdb_backtrace(self, data: bytes) -> str:
        if not self.harness_binary:
            return "(no harness binary)"
        if not _which("gdb"):
            return "(gdb not installed)"
        if self.input_mode == "stdin":
            return "(gdb tool supports arg-mode harnesses only)"
        with tempfile.NamedTemporaryFile(prefix="discver_gdb_", delete=False) as fh:
            fh.write(data)
            path = fh.name
        cmd = ["gdb", "-batch", "-nx",
               "-ex", "run", "-ex", "bt -full", "-ex", "info registers",
               "--args", self.harness_binary, path]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=40, text=True)
            return _tail((proc.stdout or "") + (proc.stderr or ""), 2800)
        except subprocess.TimeoutExpired:
            return "(gdb timed out)"
        except Exception as exc:
            return f"(gdb error: {exc})"
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def valgrind(self, data: bytes) -> str:
        if not self.harness_binary:
            return "(no harness binary)"
        if not _which("valgrind"):
            return "(valgrind not installed)"
        with tempfile.NamedTemporaryFile(prefix="discver_vg_", delete=False) as fh:
            fh.write(data)
            path = fh.name
        cmd = ["valgrind", "--error-exitcode=99", "--leak-check=no",
               "--track-origins=yes", self.harness_binary]
        try:
            if self.input_mode == "stdin":
                proc = subprocess.run(cmd, input=data, capture_output=True, timeout=60, text=True)
            else:
                proc = subprocess.run(cmd + [path], capture_output=True, timeout=60, text=True)
            return _tail((proc.stdout or "") + (proc.stderr or ""), 2800)
        except subprocess.TimeoutExpired:
            return "(valgrind timed out)"
        except Exception as exc:
            return f"(valgrind error: {exc})"
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    # -- executable oracle + coverage feedback ---------------------
    def run_input(self, data: bytes) -> str:
        if not self.harness_binary or not os.path.exists(self.harness_binary):
            return "(cannot execute: no harness binary — reason, don't submit)"
        fired, out = run_one(self.harness_binary, data, input_mode=self.input_mode)
        self._deposit(data)   # near-miss or not, feed it to the shared corpus
        return f"[{'SANITIZER FIRED' if fired else 'no crash'}]\n{out}"

    def coverage(self, data: bytes) -> str:
        """Run the input under the harness's own coverage instrumentation and report
        which functions it reached — the directed-fuzzing signal. Works for any
        libFuzzer/cargo-fuzz harness (built with -fsanitize=fuzzer)."""
        if not self.harness_binary or not os.path.exists(self.harness_binary):
            return "(no harness binary for coverage)"
        self._deposit(data)
        with tempfile.NamedTemporaryFile(prefix="discver_cov_", delete=False) as fh:
            fh.write(data)
            path = fh.name
        env = os.environ.copy()
        env["ASAN_OPTIONS"] = "detect_leaks=0:halt_on_error=0"
        try:
            proc = subprocess.run([self.harness_binary, "-print_coverage=1", path],
                                  capture_output=True, timeout=30, env=env)
            out = (proc.stdout or b"").decode("utf-8", "replace")
            out += (proc.stderr or b"").decode("utf-8", "replace")
            funcs = set(re.findall(r"COVERED_FUNC:\s*\S+\s+in\s+(\S+)", out)) \
                or set(re.findall(r"COVERED_FUNC:\s*(\S+)", out))
            if funcs:
                new = funcs - self._cov_funcs
                self._cov_funcs |= funcs
                sample = sorted(new)[:15]
                hit = ""
                if self.target_funcs:
                    reached = sorted({t for t in self.target_funcs
                                      if any(t in f for f in funcs)})
                    hit = (f" REACHED TARGET SINK(S): {reached}." if reached
                           else " (target sink NOT yet reached — fix reachability first).")
                base = (f"reached {len(funcs)} functions ({len(new)} NEW vs best so far)."
                        + (f" New: {sample}" if sample else " No new functions this input."))
                return base + hit
            m = re.search(r"\bcov:\s*(\d+)", out)
            if m:
                return f"coverage edges: {m.group(1)} (no per-function data from this build)"
            return "(no coverage data — binary may not be coverage-instrumented)"
        except subprocess.TimeoutExpired:
            return "(coverage run timed out)"
        except Exception as exc:
            return f"(coverage error: {exc})"
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    # -- self-modifying harness + structured input construction ----
    def write_and_compile_harness(self, code: str, name: str = "agent_harness") -> str:
        """Let the agent REWRITE the harness when the current one can't reach the
        bug. Language-agnostic: delegates to the caller-supplied compile_fn."""
        if self.compile_fn is None:
            return "(no compiler wired — cannot build harnesses in this run)"
        try:
            bin_path, err = self.compile_fn(code, name)
        except Exception as exc:
            return f"(compile_fn raised: {exc})"
        if bin_path and os.path.exists(bin_path):
            self.harness_binary = bin_path
            self._cov_funcs.clear()
            return ("compiled OK. harness_binary updated — run_input/coverage now use "
                    "the new harness.")
        return "compile failed:\n" + _tail(err or "", 1500)

    def python_exec(self, script: str) -> str:
        """Run a model-authored Python script to build a structured input. The
        script must write the candidate bytes to the file at os.environ['DISCVER_OUT'].
        We then run that input through the harness and report the result.

        SECURITY: this executes model-authored code. Run the whole loop inside the
        same container/sandbox you already use for agents (gVisor/Docker, no host
        mounts, egress restricted). Do not run on a bare host."""
        with tempfile.TemporaryDirectory(prefix="discver_py_") as d:
            spath = os.path.join(d, "gen.py")
            opath = os.path.join(d, "out.bin")
            Path(spath).write_text(script)
            env = os.environ.copy()
            env["DISCVER_OUT"] = opath
            try:
                proc = subprocess.run(["python3", spath], capture_output=True,
                                      timeout=20, env=env, cwd=d, text=True)
            except subprocess.TimeoutExpired:
                return "(python_exec timed out)"
            except Exception as exc:
                return f"(python_exec error: {exc})"
            so = _tail((proc.stdout or "") + (proc.stderr or ""), 1200)
            if not os.path.exists(opath):
                return (f"script ran but wrote nothing to DISCVER_OUT.\nstdout/stderr:\n{so}\n"
                        "Remember: open(os.environ['DISCVER_OUT'],'wb').write(your_bytes)")
            data = Path(opath).read_bytes()
            run = self.run_input(data) if self.harness_binary else "(no harness to test it)"
            return (f"script produced {len(data)} bytes (head={data[:24].hex()}).\n"
                    f"stdout/stderr:\n{so}\nrun_input result:\n{run}")


# ──────────────────────────────────────────────────────────────────
# The ReAct loop.
# ──────────────────────────────────────────────────────────────────
@dataclass
class ToolResult:
    success: bool
    pov_path: str | None
    crash_output: str
    steps: int
    transcript: str


_SYSTEM = """\
You are a vulnerability researcher with EXECUTABLE access to a target and to real
analysis tools. You do not guess — you investigate, run tools, craft an input, RUN
it, read the real output, and refine until it crashes reliably.

Respond with EXACTLY ONE JSON object per turn, nothing else. Available tools (only
call the ones listed as available in the first message):

  {"tool":"read_function","name":"<fn>"}
  {"tool":"list_callers","name":"<fn>"}
  {"tool":"grep","pattern":"<regex>"}
  {"tool":"read_file","path":"<rel>","start":<int>,"end":<int>}
  {"tool":"semgrep","path":"<rel-or-empty>"}      # real multi-language static analysis
  {"tool":"cppcheck","path":"<rel-or-empty>"}     # real C/C++ static analysis
  {"tool":"run_input","input_hex":"<hex>"}        # RUN the sanitizer harness on these bytes
  {"tool":"coverage","input_hex":"<hex>"}         # which functions this input reaches
  {"tool":"gdb_backtrace","input_hex":"<hex>"}    # symbolized backtrace of a crash
  {"tool":"valgrind","input_hex":"<hex>"}         # memory errors without rebuilding
  {"tool":"python_exec","script":"<python>"}      # build a structured input; writes bytes to os.environ['DISCVER_OUT']
  {"tool":"write_and_compile_harness","code":"<src>"}  # rewrite the harness if you can't reach the bug
  {"submit":{"input_hex":"<hex>"}}                # claim this input is a PoV (verified 3/3 before accepted)
  {"give_up":{"reason":"<why>"}}

Strategy:
- Use semgrep/cppcheck to find candidate sinks fast; confirm with the source.
- Use coverage to see if your input even reaches the target; if it doesn't, fix the
  input or rewrite the harness rather than mutating blindly.
- Prefer run_input over reasoning in your head — the binary is ground truth.
- For structured formats, build the input with python_exec instead of typing hex.
- input_hex is the raw bytes the harness consumes, hex-encoded.
- Only submit after you have SEEN the sanitizer fire on that exact input.
"""


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        depth, end = 0, None
        for i, ch in enumerate(blob):
            depth += (ch == "{") - (ch == "}")
            if depth == 0:
                end = i + 1
                break
        if end:
            try:
                return json.loads(blob[:end])
            except json.JSONDecodeError:
                return None
    return None


def _hex_to_bytes(s: str) -> bytes | None:
    try:
        return bytes.fromhex(re.sub(r"\s+", "", s or ""))
    except (ValueError, TypeError):
        return None


def solve_with_tools(llm, toolbox: ToolBox, task: str,
                     max_steps: int = 14, trials: int = 3,
                     temperature: float = 0.4) -> ToolResult:
    """Run the agentic loop. Returns a 3/3-verified PoV or a clean failure."""
    transcript: list[str] = []

    def log(role: str, msg: str):
        transcript.append(f"### {role}\n{msg}")

    if llm is None or not llm.is_available():
        return ToolResult(False, None, "(LLM unavailable)", 0, "")

    context = task
    if toolbox.static_context:
        context += "\n\nStatic/taint signals:\n" + toolbox.static_context[:2000]

    user = (f"TOOLS AVAILABLE THIS RUN:\n{toolbox.available_tools()}\n\n"
            f"TARGET:\n{context}\n\nBegin. First action?")
    log("task", context)

    for step in range(1, max_steps + 1):
        reply = llm.chat(system=_SYSTEM, user=user, max_tokens=1400, temperature=temperature)
        if not reply:
            break
        log("model", reply)
        action = _extract_json(reply)
        if action is None:
            user = "That was not valid JSON. Respond with exactly one JSON action object."
            continue

        if "give_up" in action:
            return ToolResult(False, None, action["give_up"].get("reason", ""),
                              step, "\n\n".join(transcript))

        if "submit" in action:
            data = _hex_to_bytes(action["submit"].get("input_hex", ""))
            if data is None:
                user = "submit.input_hex was not valid hex. Try again."
                continue
            toolbox._deposit(data)
            if not toolbox.harness_binary:
                user = "Cannot verify a submit without a harness binary. Keep investigating."
                continue
            ok, out = reproduces(toolbox.harness_binary, data, trials=trials,
                                 input_mode=toolbox.input_mode)
            if ok:
                pov_dir = Path(toolbox.output_dir) / "povs"
                pov_dir.mkdir(parents=True, exist_ok=True)
                dest = pov_dir / f"agent-pov-{step}.bin"
                dest.write_bytes(data)
                log("verdict", f"VERIFIED {trials}/{trials} -> {dest}")
                return ToolResult(True, str(dest), out, step, "\n\n".join(transcript))
            user = (f"Did NOT reproduce {trials}/{trials}. Output:\n{_tail(out,1500)}\n"
                    "Use coverage/gdb_backtrace to understand why, then refine.")
            log("verdict", f"submit failed gate\n{_tail(out,500)}")
            continue

        tool = action.get("tool")
        try:
            if tool == "read_function":
                obs = toolbox.read_function(action.get("name", ""))
            elif tool == "list_callers":
                obs = toolbox.list_callers(action.get("name", ""))
            elif tool == "grep":
                obs = toolbox.grep(action.get("pattern", ""))
            elif tool == "read_file":
                obs = toolbox.read_file(action.get("path", ""),
                                        int(action.get("start", 1)),
                                        int(action.get("end", 120)))
            elif tool == "semgrep":
                obs = toolbox.semgrep(action.get("path", ""))
            elif tool == "cppcheck":
                obs = toolbox.cppcheck(action.get("path", ""))
            elif tool == "run_input":
                d = _hex_to_bytes(action.get("input_hex", ""))
                obs = "(input_hex invalid)" if d is None else toolbox.run_input(d)
            elif tool == "coverage":
                d = _hex_to_bytes(action.get("input_hex", ""))
                obs = "(input_hex invalid)" if d is None else toolbox.coverage(d)
            elif tool == "gdb_backtrace":
                d = _hex_to_bytes(action.get("input_hex", ""))
                obs = "(input_hex invalid)" if d is None else toolbox.gdb_backtrace(d)
            elif tool == "valgrind":
                d = _hex_to_bytes(action.get("input_hex", ""))
                obs = "(input_hex invalid)" if d is None else toolbox.valgrind(d)
            elif tool == "python_exec":
                obs = toolbox.python_exec(action.get("script", ""))
            elif tool == "write_and_compile_harness":
                obs = toolbox.write_and_compile_harness(action.get("code", ""))
            else:
                obs = f"(unknown tool: {tool})"
        except Exception as exc:
            obs = f"(tool error: {exc})"

        log("observation", _tail(obs, 2000))
        user = f"Observation:\n{_tail(obs, 2000)}\n\nNext action?"

    return ToolResult(False, None, "(max steps reached)", max_steps, "\n\n".join(transcript))
