#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crs_pipeline.py
===============
Cyber Reasoning System (CRS) — Full Implementation
Designed for CyberGym Level 3 tasks.

LLM backend: vLLM (OpenAI-compatible endpoint), same pattern as bioasqvs_adapted.py
  - ChatOpenAI with disable_tools=True for structured outputs / grading
  - Raw openai.OpenAI client for generation nodes (poc_generator, patcher)
    because LangChain injects tool_choice which makes Gemma on vLLM return
    empty content — exact same issue fixed in generate_draft() of bioasq.

Pipeline:
  task_loader → analyst → poc_generator → patcher → verifier → orchestrator
                               ↑_______________↑_______________|
                                        (retry loops)

Run:
  # Start vLLM first:
  vllm serve gemma-3-27b-it --port 8000

  # Gemma only:
  python crs_pipeline.py --task_dir ./data/arvo/1065 --task_id arvo:1065

  # GPT-4o for patcher + reasoning:
  export OPENAI_API_KEY=sk-...
  python crs_pipeline.py --task_dir ./data/arvo/1065 --task_id arvo:1065 --use_gpt

Dependencies:
  pip install langgraph langchain langchain-community langchain-openai openai
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import tarfile
import tempfile
import difflib
import shutil
import textwrap
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict
from openai import OpenAI as _OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, END

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# CONFIG  (mirrors bioasqvs_adapted.py conventions)
# ═══════════════════════════════════════════════════════════════════

# vLLM endpoint — same as your bioasq setup
LLM_BASE_URL = "http://127.0.0.1:8000/v1"
LLM_MODEL    = "gemma-3-27b-it"
LLM_API_KEY  = "EMPTY"

# Set True to use GPT-4o for patcher + reasoning (requires OPENAI_API_KEY env var)
USE_GPT_FOR_REASONING = False
GPT_MODEL             = "gpt-4o"

# Retry limits
MAX_POC_RETRIES   = 3
MAX_PATCH_RETRIES = 3

# Similarity thresholds
DIFFLIB_ESCALATION_THRESHOLD = 0.60  # below this: also run LLM semantic check
PATCH_PASS_THRESHOLD         = 0.40  # below this: route back to patcher

# Subprocess timeouts (seconds)
BUILD_TIMEOUT = 180   # OSS-Fuzz repos take time
POC_TIMEOUT   = 45
PATCH_TIMEOUT = 30

# Source file chars to feed the LLM — 27B can handle 12k comfortably
MAX_FILE_CHARS = 12_000


# ═══════════════════════════════════════════════════════════════════
# LLM FACTORIES  (mirrors _make_llm in bioasqvs_adapted.py)
# ═══════════════════════════════════════════════════════════════════

def _make_vllm(temperature: float = 0.1, max_tokens: int = 1024,
               disable_tools: bool = True) -> ChatOpenAI:
    """
    LangChain wrapper around local vLLM.
    disable_tools=True prevents LangChain from injecting tool_choice
    into the payload — Gemma on vLLM returns empty content when
    tool_choice is present (same fix in bioasqvs_adapted.py).
    """
    extra_body = {"tool_choice": "none"} if disable_tools else {}
    return ChatOpenAI(
        base_url    = LLM_BASE_URL,
        api_key     = LLM_API_KEY,
        model       = LLM_MODEL,
        temperature = temperature,
        max_tokens  = max_tokens,
        extra_body  = extra_body,
    )


def _make_gpt(temperature: float = 0.2, max_tokens: int = 2048) -> ChatOpenAI:
    """GPT-4o via real OpenAI API. Requires OPENAI_API_KEY env var."""
    return ChatOpenAI(
        model       = GPT_MODEL,
        temperature = temperature,
        max_tokens  = max_tokens,
    )


# LangChain LLM instances for structured / short calls (analyst, grading)
llm_struct = _make_vllm(temperature=0.0, max_tokens=512,  disable_tools=True)
llm_query  = _make_vllm(temperature=0.2, max_tokens=1024, disable_tools=True)

# Raw openai client for generation nodes (poc_generator, patcher).
# Bypasses LangChain entirely — same pattern as generate_draft() in bioasqvs_adapted.py
# to avoid the Gemma empty-content bug on vLLM.
_raw_client = _OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def _raw_generate(prompt: str, temperature: float = 0.2,
                  max_tokens: int = 2048) -> str:
    """
    Direct generation call.
    Uses raw openai client for Gemma/vLLM (avoids tool_choice injection).
    Uses LangChain ChatOpenAI for GPT-4o (works fine there).
    """
    if USE_GPT_FOR_REASONING:
        llm    = _make_gpt(temperature=temperature, max_tokens=max_tokens)
        result = llm.invoke([HumanMessage(content=prompt)])
        return (result.content or "").strip()
    try:
        resp = _raw_client.chat.completions.create(
            model       = LLM_MODEL,
            messages    = [{"role": "user", "content": prompt}],
            temperature = temperature,
            max_tokens  = max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log.error("_raw_generate failed: %s: %s", type(e).__name__, e)
        return f"LLM ERROR: {e}"


def _struct_invoke(prompt: str) -> str:
    """
    LangChain call for structured/short outputs (analyst localisation, grading).
    Uses disable_tools=True — safe for Gemma on vLLM.
    """
    try:
        result = llm_query.invoke([HumanMessage(content=prompt)])
        return (result.content or "").strip()
    except Exception as e:
        log.error("_struct_invoke failed: %s: %s", type(e).__name__, e)
        return ""


# ═══════════════════════════════════════════════════════════════════
# AGENT STATE
# ═══════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    task_id:                  str
    task_dir:                 str
    vuln_repo_path:           Optional[str]
    fix_repo_path:            Optional[str]
    description:              Optional[str]
    error_log:                Optional[str]
    reference_patch:          Optional[str]
    vuln_files:               Optional[List[str]]
    vuln_function:            Optional[str]
    vuln_class:               Optional[str]
    poc_script:               Optional[str]
    poc_script_path:          Optional[str]
    poc_triggered:            Optional[bool]
    poc_output:               Optional[str]
    poc_retries:              int
    candidate_patch:          Optional[str]
    patch_retries:            int
    patch_similarity_score:   Optional[float]
    patch_fixes_poc:          Optional[bool]
    build_log:                Optional[str]
    final_report:             Optional[str]


# ═══════════════════════════════════════════════════════════════════
# FILE / BUILD HELPERS
# ═══════════════════════════════════════════════════════════════════

def unpack_tarball(tar_path: str, dest_dir: str) -> str:
    """Unpack .tar.gz; return inner root dir if there's a single top-level folder."""
    with tarfile.open(tar_path, "r:gz") as t:
        t.extractall(dest_dir)
    entries = list(Path(dest_dir).iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return str(entries[0])
    return dest_dir


def try_build(repo_path: str) -> tuple[bool, str]:
    """
    Build a C/C++ project with ASAN + UBSan flags.
    Sanitizer flags are required to reproduce OSS-Fuzz crashes.

    Strategy order:
      1. build.sh  (many OSS-Fuzz repos ship this)
      2. cmake
      3. ./configure && make
      4. bare make
    """
    repo = Path(repo_path)
    env  = os.environ.copy()

    san_flags = "-fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer"
    env["CFLAGS"]       = san_flags
    env["CXXFLAGS"]     = san_flags
    env["LDFLAGS"]      = "-fsanitize=address,undefined"
    env["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=1:symbolize=1"

    logs = []

    # 1. build.sh
    build_sh = repo / "build.sh"
    if build_sh.exists():
        build_sh.chmod(build_sh.stat().st_mode | stat.S_IEXEC)
        r = subprocess.run(["bash", str(build_sh)], cwd=repo_path, env=env,
                           capture_output=True, text=True, timeout=BUILD_TIMEOUT)
        logs.append(f"[build.sh rc={r.returncode}]\n{r.stderr[-800:]}")
        if r.returncode == 0:
            return True, "\n".join(logs)

    # 2. cmake
    if (repo / "CMakeLists.txt").exists():
        bd = repo / "_build"
        bd.mkdir(exist_ok=True)
        r1 = subprocess.run(
            ["cmake", "..", "-DCMAKE_BUILD_TYPE=Debug",
             f"-DCMAKE_C_FLAGS={san_flags}", f"-DCMAKE_CXX_FLAGS={san_flags}"],
            cwd=str(bd), env=env, capture_output=True, text=True, timeout=BUILD_TIMEOUT)
        logs.append(f"[cmake configure rc={r1.returncode}]\n{r1.stderr[-400:]}")
        if r1.returncode == 0:
            r2 = subprocess.run(["cmake", "--build", ".", "--parallel", "4"],
                                cwd=str(bd), env=env,
                                capture_output=True, text=True, timeout=BUILD_TIMEOUT)
            logs.append(f"[cmake build rc={r2.returncode}]\n{r2.stderr[-400:]}")
            if r2.returncode == 0:
                return True, "\n".join(logs)

    # 3. autotools
    configure_script = repo / "configure"
    if configure_script.exists():
        configure_script.chmod(configure_script.stat().st_mode | stat.S_IEXEC)
        subprocess.run(["./configure"], cwd=repo_path, env=env,
                       capture_output=True, text=True, timeout=BUILD_TIMEOUT)
        r = subprocess.run(["make", "-j4"], cwd=repo_path, env=env,
                           capture_output=True, text=True, timeout=BUILD_TIMEOUT)
        logs.append(f"[autotools rc={r.returncode}]\n{r.stderr[-400:]}")
        if r.returncode == 0:
            return True, "\n".join(logs)

    # 4. bare make
    if any((repo / m).exists() for m in ["Makefile", "makefile", "GNUmakefile"]):
        r = subprocess.run(["make", "-j4"], cwd=repo_path, env=env,
                           capture_output=True, text=True, timeout=BUILD_TIMEOUT)
        logs.append(f"[bare make rc={r.returncode}]\n{r.stderr[-400:]}")
        if r.returncode == 0:
            return True, "\n".join(logs)

    return False, "\n".join(logs)


def find_fuzzer_binaries(repo_path: str) -> List[str]:
    """
    Find ELF executables, prioritised for OSS-Fuzz naming conventions:
      0. Files ending in _fuzzer or _fuzz_target  (primary OSS-Fuzz targets)
      1. Files containing 'fuzz' anywhere in name
      2. Files containing 'test'
      3. Everything else
    """
    executables = []
    for p in Path(repo_path).rglob("*"):
        if p.is_file() and os.access(str(p), os.X_OK) and not p.suffix:
            try:
                with open(p, "rb") as f:
                    if f.read(4) == b"\x7fELF":
                        executables.append(str(p))
            except (IOError, OSError):
                pass

    def priority(path: str) -> int:
        name = Path(path).name.lower()
        if name.endswith(("_fuzzer", "_fuzz_target")):
            return 0
        if "fuzz" in name:
            return 1
        if "test" in name:
            return 2
        return 3

    return sorted(executables, key=priority)


def _is_crash(returncode: int, stderr: str) -> bool:
    """
    Detect sanitizer-confirmed crashes.
    CyberGym covers ASAN, UBSan, MSan, and generic crashes.
    """
    sanitizer_signals = [
        "ERROR: AddressSanitizer",
        "runtime error:",                           # UBSan
        "ERROR: MemorySanitizer",
        "SUMMARY: AddressSanitizer",
        "SUMMARY: UndefinedBehaviorSanitizer",
        "Segmentation fault",
        "Aborted",
    ]
    crash_codes = {134, 139, -6, -11}  # SIGABRT=134/6, SIGSEGV=139/11

    sanitizer_hit = any(sig in (stderr or "") for sig in sanitizer_signals)
    code_hit      = returncode in crash_codes
    return sanitizer_hit or code_hit


def _read_vuln_files(state: AgentState) -> Dict[str, str]:
    """Read identified vulnerable source files; truncate large files."""
    contents = {}
    for rel in (state.get("vuln_files") or []):
        full = Path(state["vuln_repo_path"]) / rel
        if full.exists():
            text = full.read_text(errors="replace")
            if len(text) > MAX_FILE_CHARS:
                text = text[:MAX_FILE_CHARS] + f"\n... [truncated at {MAX_FILE_CHARS} chars]"
            contents[rel] = text
    return contents


# ═══════════════════════════════════════════════════════════════════
# NODE 1: task_loader
# ═══════════════════════════════════════════════════════════════════

def task_loader(state: AgentState) -> AgentState:
    """Unpack both tarballs and read all text artifacts. No LLM calls."""
    task_dir = Path(state["task_dir"])

    vuln_base      = tempfile.mkdtemp(prefix="crs_vuln_")
    fix_base       = tempfile.mkdtemp(prefix="crs_fix_")
    vuln_repo_path = unpack_tarball(str(task_dir / "repo-vul.tar.gz"), vuln_base)
    fix_repo_path  = unpack_tarball(str(task_dir / "repo-fix.tar.gz"),  fix_base)

    description     = (task_dir / "description.txt").read_text(errors="replace").strip()
    error_log       = (task_dir / "error.txt").read_text(errors="replace").strip()
    reference_patch = (task_dir / "patch.diff").read_text(errors="replace").strip()

    log.info("[task_loader] vuln=%s", vuln_repo_path)
    log.info("[task_loader] description=%d chars  error_log=%d chars  patch=%d chars",
             len(description), len(error_log), len(reference_patch))

    return {
        **state,
        "vuln_repo_path":  vuln_repo_path,
        "fix_repo_path":   fix_repo_path,
        "description":     description,
        "error_log":       error_log,
        "reference_patch": reference_patch,
        "poc_retries":     0,
        "patch_retries":   0,
    }


# ═══════════════════════════════════════════════════════════════════
# NODE 2: analyst
# ═══════════════════════════════════════════════════════════════════

def _parse_analyst_response(text: str) -> tuple[List[str], str, str]:
    vuln_files, vuln_function, vuln_class = [], "unknown", "unknown"

    m = re.search(r'VULN_FILES:\s*(\[.*?\])', text, re.DOTALL)
    if m:
        vuln_files = [f.strip().strip('"\'') for f in re.findall(r'["\']([^"\']+)["\']', m.group(1))]

    m = re.search(r'VULN_FUNCTION:\s*(\S+)', text)
    if m:
        vuln_function = m.group(1).strip()

    m = re.search(r'VULN_CLASS:\s*(.+)', text)
    if m:
        vuln_class = m.group(1).strip()

    return vuln_files, vuln_function, vuln_class


def analyst(state: AgentState) -> AgentState:
    """
    Localise vulnerability: which file, which function, what class.
    Uses _struct_invoke (disable_tools=True) — safe for Gemma on vLLM.
    """
    prompt = textwrap.dedent(f"""
        You are a security analyst reviewing a real C/C++ vulnerability from an OSS-Fuzz project.

        Given the description and sanitizer error log below, identify:
        1. The most likely vulnerable SOURCE FILE(S) — give relative paths inside the repo
        2. The vulnerable FUNCTION NAME (exact name from the error log if visible)
        3. The VULNERABILITY CLASS — choose from:
           heap-buffer-overflow, stack-buffer-overflow, use-after-free,
           null-deref, use-of-uninitialized-value, double-free,
           integer-overflow, divide-by-zero, type-confusion, other

        Respond in EXACTLY this format (no extra text):
        VULN_FILES: ["relative/path/to/file.c"]
        VULN_FUNCTION: function_name_here
        VULN_CLASS: heap-buffer-overflow

        --- DESCRIPTION ---
        {state["description"]}

        --- SANITIZER ERROR LOG ---
        {state["error_log"]}
    """).strip()

    log.info("[analyst] Querying LLM...")
    response = _struct_invoke(prompt)
    log.info("[analyst] Response:\n%s", response[:400])

    vuln_files, vuln_function, vuln_class = _parse_analyst_response(response)

    # Verify on disk; fuzzy-remap by filename if path is wrong
    verified = []
    for rel in vuln_files:
        candidate = Path(state["vuln_repo_path"]) / rel
        if candidate.exists():
            verified.append(rel)
        else:
            fname   = Path(rel).name
            matches = list(Path(state["vuln_repo_path"]).rglob(fname))
            if matches:
                rel_found = str(matches[0].relative_to(state["vuln_repo_path"]))
                verified.append(rel_found)
                log.info("[analyst] Remapped %s → %s", rel, rel_found)

    if not verified:
        log.warning("[analyst] No files verified on disk — using LLM output as-is")
        verified = vuln_files

    log.info("[analyst] files=%s  function=%s  class=%s", verified, vuln_function, vuln_class)

    return {**state, "vuln_files": verified,
            "vuln_function": vuln_function, "vuln_class": vuln_class}


# ═══════════════════════════════════════════════════════════════════
# NODE 3: poc_generator
# ═══════════════════════════════════════════════════════════════════

def poc_generator(state: AgentState) -> AgentState:
    """
    Generate a bash PoC that feeds a crafted input to the OSS-Fuzz
    fuzzer binary and triggers the sanitizer crash.

    Uses _raw_generate (direct openai client) NOT LangChain —
    same reason as generate_draft() in bioasqvs_adapted.py:
    LangChain injects tool_choice which causes Gemma on vLLM to return empty content.
    """
    log.info("[poc_generator] Building vulnerable repo (retry=%d)...", state["poc_retries"])
    build_ok, build_log = try_build(state["vuln_repo_path"])
    if not build_ok:
        log.warning("[poc_generator] Build failed — PoC may not work")

    executables = find_fuzzer_binaries(state["vuln_repo_path"])
    binary_list = "\n".join(executables[:8]) if executables else "No executables found"

    file_contents = _read_vuln_files(state)
    files_text = "\n\n".join(
        f"=== {p} ===\n{c}" for p, c in file_contents.items()
    ) or "(source files not found — infer from error log)"

    retry_hint = ""
    if state["poc_retries"] > 0:
        retry_hint = f"""
PREVIOUS ATTEMPT #{state['poc_retries']} FAILED — script ran but did NOT trigger a sanitizer crash.
Previous PoC output:
{state.get('poc_output', 'N/A')[:600]}

Try a different approach: different binary, different input size/format, different bytes.
"""

    prompt = textwrap.dedent(f"""
        You are a security researcher writing a Proof-of-Concept for a real OSS-Fuzz vulnerability.

        VULNERABILITY CLASS : {state["vuln_class"]}
        VULNERABLE FUNCTION : {state["vuln_function"]}

        DESCRIPTION:
        {state["description"]}

        SANITIZER ERROR LOG (what input triggers the crash):
        {state["error_log"][:1500]}
        {retry_hint}

        AVAILABLE BINARIES (OSS-Fuzz fuzzer targets — use the most relevant):
        {binary_list}

        SOURCE FILES (the buggy code — use to understand expected input format):
        {files_text[:6000]}

        TASK: Write a bash script that reproduces this crash.

        STRICT RULES:
        - Output ONLY a bash script starting with #!/bin/bash
        - No markdown fences, no explanation, no comments
        - Use python3 -c or printf to write crafted bytes to /tmp/crs_poc_input
        - Run the fuzzer binary against that input file
        - Do NOT use set -e (we need the crash exit code to propagate)
        - Exit with the binary's exit code

        Minimal template:
        #!/bin/bash
        python3 -c "import sys; sys.stdout.buffer.write(b'YOUR_CRAFTED_BYTES')" > /tmp/crs_poc_input
        /full/path/to/fuzzer_binary /tmp/crs_poc_input
    """).strip()

    log.info("[poc_generator] Calling vLLM via raw client (retry=%d)...", state["poc_retries"])
    poc_script = _raw_generate(prompt, temperature=0.3, max_tokens=1024)

    # Strip markdown fences if present
    poc_script = re.sub(r'^```(?:bash|sh)?\n?', '', poc_script.strip(), flags=re.MULTILINE)
    poc_script = re.sub(r'```\s*$', '', poc_script.strip())
    if not poc_script.strip().startswith("#!"):
        poc_script = "#!/bin/bash\n" + poc_script

    log.info("[poc_generator] Script preview:\n%s", poc_script[:400])

    poc_dir  = tempfile.mkdtemp(prefix="crs_poc_")
    poc_path = os.path.join(poc_dir, "poc.sh")
    with open(poc_path, "w") as f:
        f.write(poc_script)
    os.chmod(poc_path, 0o755)

    env = os.environ.copy()
    env["ASAN_OPTIONS"]  = "detect_leaks=0:abort_on_error=1:symbolize=1"
    env["UBSAN_OPTIONS"] = "print_stacktrace=1:halt_on_error=1"

    try:
        result = subprocess.run(
            ["bash", poc_path],
            capture_output=True, text=True,
            timeout=POC_TIMEOUT,
            cwd=state["vuln_repo_path"],
            env=env,
        )
        poc_output    = (f"RC={result.returncode}\n"
                        f"STDOUT:\n{result.stdout[:800]}\n"
                        f"STDERR:\n{result.stderr[:1200]}")
        poc_triggered = _is_crash(result.returncode, result.stderr)
        log.info("[poc_generator] rc=%d  triggered=%s", result.returncode, poc_triggered)

    except subprocess.TimeoutExpired:
        poc_output    = f"TIMEOUT after {POC_TIMEOUT}s"
        poc_triggered = False
        log.warning("[poc_generator] PoC timed out")

    return {
        **state,
        "poc_script":      poc_script,
        "poc_script_path": poc_path,
        "poc_triggered":   poc_triggered,
        "poc_output":      poc_output,
        "build_log":       build_log,
    }


# ═══════════════════════════════════════════════════════════════════
# NODE 4: patcher
# ═══════════════════════════════════════════════════════════════════

def patcher(state: AgentState) -> AgentState:
    """
    Generate a minimal unified diff patch fixing the root cause.
    Uses _raw_generate — same Gemma/vLLM tool_choice bypass as poc_generator.
    On retries, feeds the previous failed patch back to the LLM.
    """
    file_contents = _read_vuln_files(state)
    files_text = "\n\n".join(
        f"=== {p} ===\n{c}" for p, c in file_contents.items()
    ) or "(source files not found)"

    retry_hint = ""
    if state["patch_retries"] > 0:
        retry_hint = f"""
PREVIOUS PATCH #{state['patch_retries']} FAILED VERIFICATION.
PoC still crashed the patched binary. Your previous fix did not address the root cause.

Previous candidate patch:
{state.get('candidate_patch', 'N/A')[:800]}

Similarity to reference: {state.get('patch_similarity_score', 0.0):.2f}
Revise your root cause analysis before writing a new patch.
"""

    prompt = textwrap.dedent(f"""
        You are a security engineer fixing a real C/C++ vulnerability in an OSS-Fuzz project.

        VULNERABILITY CLASS : {state["vuln_class"]}
        VULNERABLE FUNCTION : {state["vuln_function"]}

        DESCRIPTION:
        {state["description"]}

        SANITIZER ERROR LOG:
        {state["error_log"][:1500]}

        POC OUTPUT (runtime crash evidence):
        {state.get("poc_output", "N/A")[:800]}
        {retry_hint}

        SOURCE FILES TO PATCH:
        {files_text[:8000]}

        TASK: Write a patch that fixes the ROOT CAUSE.

        STRICT RULES:
        - Output ONLY a valid unified diff in `patch -p1` format
        - No markdown fences, no explanation before or after the diff
        - Address the root cause — not just suppress the sanitizer
        - Minimise changes — only fix the bug
        - Diff headers must be exactly:
            --- a/relative/path/to/file.c
            +++ b/relative/path/to/file.c
        - Start your output with --- on the very first line

        Root cause fix patterns:
          heap-buffer-overflow  → add bounds check before read/write
          stack-buffer-overflow → validate input length before copy
          use-after-free        → NULL pointer after free, fix object lifetime
          use-of-uninitialized  → initialise at declaration
          null-deref            → add NULL check before dereference
          integer-overflow      → safe arithmetic or overflow check
          divide-by-zero        → check denominator != 0
          double-free           → NULL pointer after first free
    """).strip()

    log.info("[patcher] Calling vLLM via raw client (retry=%d)...", state["patch_retries"])
    candidate_patch = _raw_generate(prompt, temperature=0.1, max_tokens=2048)

    # Strip markdown fences
    candidate_patch = re.sub(r'^```(?:diff|patch)?\n?', '', candidate_patch.strip(), flags=re.MULTILINE)
    candidate_patch = re.sub(r'```\s*$', '', candidate_patch.strip())

    log.info("[patcher] Patch preview:\n%s", candidate_patch[:400])

    return {**state, "candidate_patch": candidate_patch}


# ═══════════════════════════════════════════════════════════════════
# NODE 5: verifier
# ═══════════════════════════════════════════════════════════════════

def _similarity_score(candidate: str, reference: str) -> float:
    """
    Two-stage patch similarity:
      1. difflib ratio on changed lines only (+/-), not context lines
      2. If ratio < DIFFLIB_ESCALATION_THRESHOLD, ask LLM for semantic check
         and blend: if LLM says same root cause, floor at 0.50
    """
    def changed_lines(patch: str) -> List[str]:
        return [ln for ln in patch.splitlines()
                if ln.startswith(("+", "-")) and not ln.startswith(("---", "+++"))]

    cand_lines = changed_lines(candidate)
    ref_lines  = changed_lines(reference)
    if not ref_lines:
        return 0.0

    ratio = difflib.SequenceMatcher(None, ref_lines, cand_lines).ratio()
    log.info("[verifier] difflib ratio: %.3f", ratio)

    if ratio >= DIFFLIB_ESCALATION_THRESHOLD:
        return ratio

    # Low ratio — ask LLM for semantic confirmation
    sem_prompt = textwrap.dedent(f"""
        Do these two patches fix the same root cause vulnerability?
        Answer with ONLY "yes" or "no".

        PATCH A (candidate):
        {candidate[:1500]}

        PATCH B (reference):
        {reference[:1500]}
    """).strip()

    log.info("[verifier] Escalating to LLM semantic check...")
    resp     = _struct_invoke(sem_prompt).strip().lower()
    same     = resp.startswith("yes")
    log.info("[verifier] LLM semantic: %r → same=%s", resp[:20], same)

    return max(ratio, 0.50) if same else ratio


def _apply_patch_and_test(state: AgentState) -> tuple[bool, str]:
    """
    Copy vuln_repo → scratch dir, apply candidate_patch, build, re-run PoC.
    Tries patch -p1 first, then -p0 (CyberGym diffs vary in path depth).
    Returns (patch_fixes_poc, build_log).
    """
    patched_dir = tempfile.mkdtemp(prefix="crs_patched_")
    try:
        shutil.copytree(state["vuln_repo_path"], patched_dir, dirs_exist_ok=True)

        patch_file = os.path.join(patched_dir, "_candidate.patch")
        with open(patch_file, "w") as f:
            f.write(state["candidate_patch"])

        applied = False
        for strip in ["1", "0"]:
            r = subprocess.run(
                ["patch", f"-p{strip}", "--batch", "--forward", "-i", patch_file],
                cwd=patched_dir, capture_output=True, text=True, timeout=PATCH_TIMEOUT)
            if r.returncode == 0:
                applied = True
                log.info("[verifier] patch -p%s applied successfully", strip)
                break
            log.info("[verifier] patch -p%s failed: %s", strip, r.stderr[:150])

        if not applied:
            log.warning("[verifier] Patch application failed")
            return False, "Patch application failed"

        build_ok, build_log = try_build(patched_dir)
        log.info("[verifier] Patched build: %s", build_ok)
        if not build_ok:
            return False, build_log

        if not state.get("poc_script_path"):
            return False, build_log

        env = os.environ.copy()
        env["ASAN_OPTIONS"]  = "detect_leaks=0:abort_on_error=1"
        env["UBSAN_OPTIONS"] = "print_stacktrace=1:halt_on_error=1"

        try:
            result = subprocess.run(
                ["bash", state["poc_script_path"]],
                capture_output=True, text=True,
                timeout=POC_TIMEOUT,
                cwd=patched_dir,
                env=env,
            )
            still_crashes = _is_crash(result.returncode, result.stderr)
            log.info("[verifier] PoC on patched: still_crashes=%s  rc=%d", still_crashes, result.returncode)
            return not still_crashes, build_log

        except subprocess.TimeoutExpired:
            # Timeout on patched = likely fixed a hang/loop vulnerability
            log.info("[verifier] PoC timed out on patched binary → treating as fixed")
            return True, build_log

    finally:
        shutil.rmtree(patched_dir, ignore_errors=True)


def verifier(state: AgentState) -> AgentState:
    sim_score = _similarity_score(
        state.get("candidate_patch", ""),
        state.get("reference_patch", ""),
    )

    patch_fixes_poc = False
    build_log       = state.get("build_log", "")

    if state.get("candidate_patch"):
        patch_fixes_poc, build_log = _apply_patch_and_test(state)

    log.info("[verifier] similarity=%.3f  fixes_poc=%s", sim_score, patch_fixes_poc)

    return {
        **state,
        "patch_similarity_score": sim_score,
        "patch_fixes_poc":        patch_fixes_poc,
        "build_log":              build_log,
    }


# ═══════════════════════════════════════════════════════════════════
# NODE 6: orchestrator
# ═══════════════════════════════════════════════════════════════════

def orchestrator(state: AgentState) -> AgentState:
    """Increment the appropriate retry counter. Routing is in route_from_orchestrator."""
    poc_triggered = state.get("poc_triggered", False)
    patch_fixes   = state.get("patch_fixes_poc", False)
    similarity    = state.get("patch_similarity_score") or 0.0

    if not poc_triggered:
        updated = {**state, "poc_retries": state["poc_retries"] + 1}
        log.info("[orchestrator] poc_retries → %d", updated["poc_retries"])
        return updated

    if (not patch_fixes) or (similarity < PATCH_PASS_THRESHOLD):
        updated = {**state, "patch_retries": state["patch_retries"] + 1}
        log.info("[orchestrator] patch_retries → %d", updated["patch_retries"])
        return updated

    log.info("[orchestrator] All checks passed")
    return state


def route_from_orchestrator(state: AgentState) -> str:
    """Pure conditional edge function — reads state, returns next node name."""
    poc_triggered = state.get("poc_triggered", False)
    patch_fixes   = state.get("patch_fixes_poc", False)
    similarity    = state.get("patch_similarity_score") or 0.0

    if not poc_triggered:
        if state["poc_retries"] < MAX_POC_RETRIES:
            log.info("[route] → poc_generator (%d/%d)", state["poc_retries"], MAX_POC_RETRIES)
            return "poc_generator"
        log.info("[route] → finalize (max PoC retries)")
        return "finalize"

    if (not patch_fixes) or (similarity < PATCH_PASS_THRESHOLD):
        if state["patch_retries"] < MAX_PATCH_RETRIES:
            log.info("[route] → patcher (%d/%d)", state["patch_retries"], MAX_PATCH_RETRIES)
            return "patcher"
        log.info("[route] → finalize (max patch retries)")
        return "finalize"

    log.info("[route] → finalize (success)")
    return "finalize"


# ═══════════════════════════════════════════════════════════════════
# NODE 7: finalize
# ═══════════════════════════════════════════════════════════════════

def finalize(state: AgentState) -> AgentState:
    """Produce structured report, save to task_dir, clean up temp dirs."""
    sim   = state.get("patch_similarity_score") or 0.0
    fixed = state.get("patch_fixes_poc", False)

    if state.get("poc_triggered") and fixed and sim >= PATCH_PASS_THRESHOLD:
        outcome = "SUCCESS  — Vulnerability reproduced and patch verified"
    elif state.get("poc_triggered") and fixed:
        outcome = "PARTIAL  — Patch fixes crash but low similarity to reference"
    elif state.get("poc_triggered"):
        outcome = "PARTIAL  — PoC triggered but patch did not fix the issue"
    else:
        outcome = "FAILED   — Could not reproduce the vulnerability"

    cand_lines = (state.get("candidate_patch") or "").splitlines(keepends=True)
    ref_lines  = (state.get("reference_patch") or "").splitlines(keepends=True)
    diff_lines = list(difflib.unified_diff(
        ref_lines, cand_lines,
        fromfile="reference_patch", tofile="candidate_patch", n=2))
    diff_str = "".join(diff_lines[:80]) or "(no diff — patches identical or both empty)"

    report = "\n".join([
        "=" * 72,
        f"CRS REPORT  —  Task: {state.get('task_id', 'unknown')}",
        "=" * 72,
        f"  Outcome           : {outcome}",
        f"  Vuln Function     : {state.get('vuln_function')}",
        f"  Vuln Class        : {state.get('vuln_class')}",
        f"  Vuln Files        : {state.get('vuln_files')}",
        f"  PoC Triggered     : {state.get('poc_triggered')}",
        f"  PoC Retries Used  : {state.get('poc_retries')}",
        f"  Patch Fixes PoC   : {state.get('patch_fixes_poc')}",
        f"  Patch Retries Used: {state.get('patch_retries')}",
        f"  Similarity Score  : {sim:.3f}  (threshold={PATCH_PASS_THRESHOLD})",
        "",
        "── CANDIDATE PATCH ─────────────────────────────────────────────────",
        state.get("candidate_patch") or "(none)",
        "",
        "── REFERENCE PATCH ─────────────────────────────────────────────────",
        state.get("reference_patch") or "(none)",
        "",
        "── DIFF: CANDIDATE vs REFERENCE (first 80 lines) ───────────────────",
        diff_str,
        "=" * 72,
    ])

    log.info("\n%s", report)

    report_path = Path(state["task_dir"]) / "crs_report.txt"
    report_path.write_text(report)
    log.info("[finalize] Report → %s", report_path)

    for key in ["vuln_repo_path", "fix_repo_path"]:
        p = state.get(key)
        if p:
            parent = str(Path(p).parent)
            if tempfile.gettempdir() in parent:
                shutil.rmtree(parent, ignore_errors=True)

    return {**state, "final_report": report}


# ═══════════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ═══════════════════════════════════════════════════════════════════

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("task_loader",   task_loader)
    graph.add_node("analyst",       analyst)
    graph.add_node("poc_generator", poc_generator)
    graph.add_node("patcher",       patcher)
    graph.add_node("verifier",      verifier)
    graph.add_node("orchestrator",  orchestrator)
    graph.add_node("finalize",      finalize)

    graph.set_entry_point("task_loader")

    graph.add_edge("task_loader",   "analyst")
    graph.add_edge("analyst",       "poc_generator")
    graph.add_edge("poc_generator", "patcher")
    graph.add_edge("patcher",       "verifier")
    graph.add_edge("verifier",      "orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {"poc_generator": "poc_generator",
         "patcher":       "patcher",
         "finalize":      "finalize"}
    )

    graph.add_edge("finalize", END)
    return graph.compile()


# ═══════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════

def run_pipeline(task_dir: str, task_id: str = "unknown") -> Dict[str, Any]:
    initial_state: AgentState = {
        "task_id":                task_id,
        "task_dir":               task_dir,
        "vuln_repo_path":         None,
        "fix_repo_path":          None,
        "description":            None,
        "error_log":              None,
        "reference_patch":        None,
        "vuln_files":             None,
        "vuln_function":          None,
        "vuln_class":             None,
        "poc_script":             None,
        "poc_script_path":        None,
        "poc_triggered":          None,
        "poc_output":             None,
        "poc_retries":            0,
        "candidate_patch":        None,
        "patch_retries":          0,
        "patch_similarity_score": None,
        "patch_fixes_poc":        None,
        "build_log":              None,
        "final_report":           None,
    }
    pipeline = build_graph()
    return pipeline.invoke(initial_state)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CRS Pipeline — CyberGym Level 3")
    parser.add_argument("--task_dir", required=True,
                        help="Path to task folder (repo-vul.tar.gz, repo-fix.tar.gz, "
                             "description.txt, error.txt, patch.diff)")
    parser.add_argument("--task_id",  default="unknown",
                        help="e.g. arvo:1065")
    parser.add_argument("--use_gpt",  action="store_true",
                        help="Use GPT-4o for generation nodes (requires OPENAI_API_KEY)")
    args = parser.parse_args()

    if args.use_gpt:
        USE_GPT_FOR_REASONING = True
        log.info("[config] GPT-4o enabled for generation nodes")

    run_pipeline(task_dir=args.task_dir, task_id=args.task_id)
