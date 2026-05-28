"""
harness_agent.py — Agentic harness generation.

Multi-round agent that explores a project, generates harnesses,
validates them with coverage feedback, and refines iteratively.

Works in both modes:
- Standalone: generates the initial entry-point harness from scratch
- OSS-CRS: generates additional targeted harnesses alongside OSS-Fuzz's

Round 1: EXPLORE — read project structure, build system, APIs
Round 2: GENERATE — write harness based on exploration
Round 3: COMPILE — compile with error feedback loop
Round 4: VALIDATE — run briefly, parse coverage
Round 5: REFINE — generate more harnesses for uncovered code
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from llm_client import VLLMClient
from code_analysis import find_include_dirs, find_static_lib, CallGraph

logger = logging.getLogger("gemma-fuzzer.harness-agent")


@dataclass
class HarnessResult:
    binary_path: str | None
    source_path: str | None
    coverage_funcs: int      # functions covered in validation
    total_execs: int         # executions during validation


# ══════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════

EXPLORE_PROMPT = """\
You are analyzing a C/C++ project to understand its API.

Given the project's file listing, headers, and source samples, determine:

1. PROJECT TYPE: What does this library do? (parser, crypto, compression, etc.)
2. INPUT FORMAT: What kind of data does it process? (text, binary, structured)
3. MAIN API FUNCTIONS: List the top 3-5 public functions that process input.
   For each: name, header file, what it does, parameters.
4. INITIALIZATION: What setup is needed before calling the API?
   (init functions, context creation, configuration)
5. CLEANUP: What teardown is needed after? (free, destroy, close)
6. BUILD INFO: What libraries does it link against? (-lz, -lssl, etc.)

Respond with ONLY a JSON object:
{"project_type": "JSON parser",
 "input_format": "text/string (null-terminated)",
 "api_functions": [
   {"name": "cJSON_Parse", "header": "cJSON.h",
    "description": "parses JSON string into tree",
    "params": "const char *value",
    "returns": "cJSON *"}
 ],
 "init_code": "",
 "cleanup_code": "cJSON_Delete(result);",
 "link_libs": ["-lm"],
 "notes": "also call cJSON_Print for output coverage"}"""

GENERATE_PROMPT = """\
Write a LibFuzzer harness for this library based on the analysis.

Project analysis:
{analysis}

RULES:
1. Include the necessary headers (found in the analysis)
2. Implement int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
3. For text input: null-terminate the fuzz data
4. For binary input: pass data and size directly
5. Call the main API function(s) from the analysis
6. Also call output/serialization functions for extra coverage
7. Clean up all resources
8. Return 0

If the analysis mentions multiple API functions, call ALL of them
to maximize code coverage.

Output ONLY C code. No markdown. Start with #include."""

REFINE_PROMPT = """\
The harness ran for {seconds}s and achieved {coverage} function coverage.

These functions were NOT reached by the harness:
{uncovered}

The current harness code:
```c
{current_code}
```

Write an ADDITIONAL harness (separate from the first one) that targets
the uncovered functions. Try different API entry points, different
options/flags, or different input formats.

Output ONLY C code. No markdown. Start with #include."""


# ══════════════════════════════════════════════════════════════════
# THE AGENT
# ══════════════════════════════════════════════════════════════════

class HarnessAgent:
    def __init__(self, llm: VLLMClient, src_dir: str, output_dir: str):
        self.llm = llm
        self.src_dir = src_dir
        self.output_dir = output_dir
        self.harness_dir = Path(output_dir) / "generated_harnesses"
        self.harness_dir.mkdir(parents=True, exist_ok=True)
        self.generated: list[HarnessResult] = []

    def run(self, call_graph: CallGraph = None, max_harnesses: int = 3) -> list[str]:
        """Run the full agentic loop. Returns list of compiled binary paths."""
        if not self.llm.is_available():
            return []

        # ── Round 1: EXPLORE ──
        logger.info("[harness-agent] Round 1: Exploring project...")
        analysis = self._explore()
        if not analysis:
            logger.warning("[harness-agent] Exploration failed.")
            return []

        logger.info("[harness-agent] Project: %s, Input: %s",
                    analysis.get("project_type", "?"),
                    analysis.get("input_format", "?"))

        # ── Round 2-3: GENERATE + COMPILE ──
        logger.info("[harness-agent] Round 2: Generating initial harness...")
        binary, source_code = self._generate_and_compile(analysis, "entry")
        if not binary:
            logger.warning("[harness-agent] Initial harness failed to compile.")
            return []

        binaries = [binary]

        # ── Round 4: VALIDATE ──
        logger.info("[harness-agent] Round 4: Validating with 30s fuzz run...")
        coverage_info = self._validate(binary, seconds=30)

        # ── Round 5: REFINE (generate more harnesses) ──
        if call_graph and len(call_graph.functions) > 0:
            # Find functions not covered by the initial harness
            all_funcs = set(call_graph.functions.keys())
            covered = set(coverage_info.get("covered_funcs", []))
            uncovered = all_funcs - covered

            # Filter to important uncovered functions
            important_uncovered = []
            for fname in uncovered:
                fdef = call_graph.functions.get(fname)
                if fdef and len(fdef.called_by) >= 2:
                    important_uncovered.append(fname)

            if important_uncovered and len(binaries) < max_harnesses:
                logger.info("[harness-agent] Round 5: %d important functions uncovered. Generating more harnesses...",
                           len(important_uncovered))

                refined_binary, _ = self._refine(
                    source_code, important_uncovered[:20],
                    coverage_info, analysis,
                )
                if refined_binary:
                    binaries.append(refined_binary)

        # ── Generate API-specific harnesses from analysis ──
        api_funcs = analysis.get("api_functions", [])
        for api in api_funcs[1:max_harnesses]:  # skip first (already in entry harness)
            if len(binaries) >= max_harnesses:
                break

            logger.info("[harness-agent] Generating harness for API: %s", api.get("name", "?"))
            api_binary, _ = self._generate_api_harness(api, analysis)
            if api_binary:
                binaries.append(api_binary)

        logger.info("[harness-agent] Generated %d harnesses total.", len(binaries))
        return binaries

    # ── EXPLORE ──────────────────────────────────────────────────

    def _explore(self) -> dict:
        """Read project structure and understand the API."""
        src_path = Path(self.src_dir)
        context = ""

        # File listing
        all_files = []
        for ext in ["*.h", "*.c"]:
            for f in sorted(src_path.rglob(ext)):
                if any(s in str(f).lower() for s in [".git", "test", "aflplusplus", "honggfuzz"]):
                    continue
                try:
                    all_files.append(str(f.relative_to(src_path)))
                except ValueError:
                    pass

        context += "Files in project:\n" + "\n".join(all_files[:30]) + "\n\n"

        # Headers (API surface)
        for h in sorted(src_path.rglob("*.h"))[:5]:
            if any(s in str(h).lower() for s in [".git", "test", "aflplusplus"]):
                continue
            try:
                content = h.read_text(errors="replace")
                rel = str(h.relative_to(src_path))
                context += f"// === {rel} ===\n{content[:4000]}\n\n"
            except Exception:
                pass

        # Main source file (first 2000 chars)
        for c in sorted(src_path.rglob("*.c"))[:2]:
            if any(s in str(c).lower() for s in [".git", "test", "fuzz", "aflplusplus"]):
                continue
            try:
                content = c.read_text(errors="replace")
                rel = str(c.relative_to(src_path))
                context += f"// === {rel} (first 2000 chars) ===\n{content[:2000]}\n\n"
            except Exception:
                pass

        # Build system clues
        for bf in ["Makefile", "CMakeLists.txt", "configure.ac", "meson.build"]:
            bpath = src_path / bf
            if bpath.exists():
                try:
                    content = bpath.read_text(errors="replace")
                    context += f"// === {bf} (first 1000 chars) ===\n{content[:1000]}\n\n"
                except Exception:
                    pass

        response = self.llm.chat(
            system="You are a code analysis expert. Respond ONLY with JSON.",
            user=EXPLORE_PROMPT + f"\n\nProject files and source:\n{context[:12000]}",
            max_tokens=1500, temperature=0.1,
        )

        if not response:
            return {}

        # Parse JSON
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        s, e = clean.find("{"), clean.rfind("}")
        if s != -1 and e != -1:
            clean = clean[s:e + 1]
        try:
            import json
            return json.loads(clean)
        except Exception:
            return {}

    # ── GENERATE + COMPILE ───────────────────────────────────────

    def _generate_and_compile(self, analysis: dict, name: str) -> tuple[str | None, str]:
        """Generate a harness and compile it with retry."""
        import json
        response = self.llm.chat(
            system="You are a fuzzing engineer. Output ONLY C code.",
            user=GENERATE_PROMPT.format(analysis=json.dumps(analysis, indent=2)),
            max_tokens=2000, temperature=0.2,
        )

        if not response:
            return None, ""

        code = _clean_code(response)

        # Try to compile (up to 3 attempts with error feedback)
        for attempt in range(3):
            binary, error = _compile(code, name, self.src_dir, str(self.harness_dir))

            if binary:
                return binary, code

            if attempt < 2 and error:
                logger.info("[harness-agent] Compile failed (attempt %d), fixing...", attempt + 1)
                fix_response = self.llm.chat(
                    system="Fix the C code. Output ONLY C code.",
                    user=f"Code:\n```c\n{code}\n```\nError:\n```\n{error[:800]}\n```\nFix it.",
                    max_tokens=2000, temperature=0.1,
                )
                if fix_response:
                    code = _clean_code(fix_response)

        return None, code

    # ── VALIDATE ─────────────────────────────────────────────────

    def _validate(self, binary: str, seconds: int = 30) -> dict:
        """Run harness briefly and extract coverage info."""
        corpus_dir = Path(self.output_dir) / "validation_corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["ASAN_OPTIONS"] = "abort_on_error=1:detect_leaks=0"

        cmd = [
            binary, str(corpus_dir),
            f"-max_total_time={seconds}",
            "-max_len=4096",
            "-print_final_stats=1",
            "-detect_leaks=0",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=seconds + 15, env=env,
            )
            output = (result.stderr or b"").decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            output = ""

        # Parse coverage from LibFuzzer output
        info = {"total_execs": 0, "covered_funcs": [], "edges": 0}

        for line in output.split("\n"):
            if "execs_done" in line or "stat::number_of_executed_units" in line:
                nums = re.findall(r'\d+', line)
                if nums:
                    info["total_execs"] = int(nums[-1])
            if "cov:" in line:
                nums = re.findall(r'\d+', line)
                if nums:
                    info["edges"] = max(info["edges"], int(nums[0]))

        logger.info("[harness-agent] Validation: %d execs, %d edge coverage.",
                    info["total_execs"], info["edges"])

        return info

    # ── REFINE ───────────────────────────────────────────────────

    def _refine(self, current_code: str, uncovered: list[str],
                coverage_info: dict, analysis: dict) -> tuple[str | None, str]:
        """Generate a refined harness targeting uncovered functions."""
        response = self.llm.chat(
            system="You are a fuzzing engineer. Output ONLY C code.",
            user=REFINE_PROMPT.format(
                seconds=30,
                coverage=coverage_info.get("edges", 0),
                uncovered="\n".join(f"  - {f}" for f in uncovered[:20]),
                current_code=current_code[:3000],
            ),
            max_tokens=2000, temperature=0.3,
        )

        if not response:
            return None, ""

        code = _clean_code(response)
        binary, error = _compile(code, "refined", self.src_dir, str(self.harness_dir))

        if binary:
            return binary, code

        # One retry
        if error:
            fix = self.llm.chat(
                system="Fix the C code. Output ONLY C code.",
                user=f"Code:\n```c\n{code}\n```\nError:\n```\n{error[:600]}\n```",
                max_tokens=2000, temperature=0.1,
            )
            if fix:
                code = _clean_code(fix)
                binary, _ = _compile(code, "refined_v2", self.src_dir, str(self.harness_dir))
                if binary:
                    return binary, code

        return None, ""

    # ── API-SPECIFIC HARNESS ─────────────────────────────────────

    def _generate_api_harness(self, api_func: dict, analysis: dict) -> tuple[str | None, str]:
        """Generate a harness for a specific API function."""
        import json
        prompt = f"""\
Write a LibFuzzer harness that specifically exercises the function `{api_func.get('name', '?')}`.

Function info:
{json.dumps(api_func, indent=2)}

Project info:
- Type: {analysis.get('project_type', '?')}
- Input format: {analysis.get('input_format', '?')}
- Init: {analysis.get('init_code', 'none')}
- Cleanup: {analysis.get('cleanup_code', 'none')}
- Headers: {', '.join(f.get('header', '') for f in analysis.get('api_functions', []))}

Output ONLY C code. Start with #include."""

        response = self.llm.chat(
            system="You are a fuzzing engineer. Output ONLY C code.",
            user=prompt, max_tokens=2000, temperature=0.2,
        )

        if not response:
            return None, ""

        code = _clean_code(response)
        name = api_func.get("name", "api").replace(" ", "_")
        binary, _ = _compile(code, f"api_{name}", self.src_dir, str(self.harness_dir))
        return (binary, code) if binary else (None, "")


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _compile(code, name, src_dir, output_dir):
    """Compile with auto-retry for missing macros."""
    import hashlib

    harness_dir = Path(output_dir)
    harness_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(code.encode()).hexdigest()[:8]
    src_path = harness_dir / f"{name}_{h}.c"
    bin_path = harness_dir / f"{name}_{h}"
    src_path.write_text(code)

    inc_dirs = find_include_dirs(src_dir)
    lib = find_static_lib(src_dir)

    def _build(extra_defines=None, extra_libs=None):
        _lld = ["-fuse-ld=lld"] if (shutil.which("ld.lld") or shutil.which("lld")) else []
        cmd = ["clang", "-g", "-O1", "-fsanitize=address,fuzzer", *_lld,
               "-Wno-macro-redefined", "-Wno-implicit-function-declaration",
               "-Wno-int-conversion", "-Wno-incompatible-pointer-types"]
        config_h = Path(src_dir) / "config.h"
        if config_h.exists():
            cmd.extend(["-DHAVE_CONFIG_H", "-include", str(config_h)])
        if extra_defines:
            cmd.extend(extra_defines)
        for d in inc_dirs:
            cmd.extend(["-I", d])
        cmd.append(str(src_path))
        if lib:
            cmd.append(lib)
        cmd.extend(["-lm", "-Wl,--allow-multiple-definition"])
        if extra_libs:
            cmd.extend(extra_libs)
        cmd.extend(["-o", str(bin_path)])
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    # Attempt 1
    try:
        result = _build()
        if result.returncode == 0:
            return str(bin_path), ""

        # Auto-fix: add missing libs
        stderr = result.stderr
        extra_libs = []
        if "undefined reference" in stderr:
            if "lzma" in stderr: extra_libs.append("-llzma")
            if "compress" in stderr or "inflate" in stderr: extra_libs.append("-lz")
            if "pthread" in stderr: extra_libs.append("-lpthread")
        if extra_libs:
            result = _build(extra_libs=extra_libs)
            if result.returncode == 0:
                return str(bin_path), ""
            stderr = result.stderr

        # Auto-fix: define unknown macros
        unknown = set()
        for pat in [r"unknown type name '(\w+)'", r"use of undeclared identifier '(\w+)'"]:
            for m in re.finditer(pat, stderr):
                macro = m.group(1)
                if macro.isupper() or macro.endswith("_HIDDEN") or \
                   macro.endswith("_EXPORT") or macro.startswith("__"):
                    unknown.add(macro)
        if unknown:
            defines = [f"-D{m}=" for m in unknown]
            result = _build(extra_defines=defines, extra_libs=extra_libs)
            if result.returncode == 0:
                return str(bin_path), ""
            stderr = result.stderr

        return None, stderr

    except Exception as exc:
        return None, str(exc)


def _clean_code(response):
    code = response.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1] if "\n" in code else code[3:]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    return code.strip()
