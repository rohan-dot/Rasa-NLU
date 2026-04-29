"""
agents.py — Multi-agent vulnerability discovery system.

Four specialized agents, each with a narrow job and focused prompt:

1. SCANNER — Reads code + call graph, outputs suspicious locations with
   confidence scores. Does NOT try to generate PoCs.
   
2. EXPLOITER — Takes ONE suspicious location from the Scanner, crafts a
   PoC input, runs it, iterates up to 3 times based on feedback.

3. HARNESS_BUILDER — Takes a suspicious function from the Scanner, generates
   a LibFuzzer harness that calls it DIRECTLY (not through xmlReadMemory).
   
4. VERIFIER — Takes a crash, confirms it's real, classifies severity and CWE.

Key insight from AISLE: a 3.6B model found real CVEs because the scaffolding
gave it one clear task at a time. We do the same here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from callgraph import CallGraph
from llm_client import VLLMClient

logger = logging.getLogger("gemma-fuzzer.agents")


@dataclass
class ScanFinding:
    """Output from the Scanner agent."""
    file: str
    function: str
    bug_type: str
    confidence: float  # 0.0 - 1.0
    description: str
    data_flow: str
    trigger_hint: str


@dataclass
class ExploitResult:
    """Output from the Exploiter agent."""
    finding: ScanFinding
    crashed: bool
    poc_path: str | None
    crash_output: str
    attempts: int


# ══════════════════════════════════════════════════════════════════
# AGENT 1: SCANNER
# ══════════════════════════════════════════════════════════════════

SCANNER_SYSTEM = """\
You are a security vulnerability scanner. Your ONLY job is to identify
suspicious code locations that might contain vulnerabilities.

You will receive:
1. Source code of a function
2. A call graph showing how external input reaches this function
3. The function's callers and callees

For each suspicious location, output:
- file, function, bug_type, confidence (0.0-1.0), description
- data_flow: how input reaches the vulnerable code
- trigger_hint: what kind of input would trigger the bug

IMPORTANT: Only report findings with confidence >= 0.5.
Be specific about the exact line/operation that is vulnerable.
Do NOT suggest fixes. Do NOT generate PoC inputs. Just find the bugs.

Respond with ONLY a JSON array:
[{"file": "dict.c", "function": "xmlDictAddQString", "bug_type": "integer-overflow",
  "confidence": 0.85,
  "description": "len + plen addition at line 487 can overflow when both are near INT_MAX",
  "data_flow": "xmlParseStartTag → xmlDictLookup → xmlDictAddQString(name, plen, len)",
  "trigger_hint": "XML element with extremely long namespace prefix and local name"}]

Your response must start with [ and end with ] — nothing else."""


class ScannerAgent:
    """Scans code for vulnerabilities. Narrow focus, high precision."""

    def __init__(self, llm: VLLMClient):
        self.llm = llm

    def scan_function(
        self,
        func_name: str,
        call_graph: CallGraph,
        src_dir: str,
    ) -> list[ScanFinding]:
        """Scan a specific function with its call context."""
        if not self.llm.is_available():
            return []

        # Get function source + caller context from call graph
        context = call_graph.get_function_context(func_name, src_dir)
        if not context:
            return []

        # Get call paths for data flow understanding
        fdef = call_graph.functions.get(func_name)
        callers = call_graph.get_callers(func_name, depth=3)

        user_msg = f"Function to scan: {func_name}\n"
        if fdef:
            user_msg += f"Defined in: {fdef.file}:{fdef.line}\n"
            user_msg += f"Called by: {', '.join(fdef.called_by[:5])}\n"
            user_msg += f"Calls: {', '.join(fdef.calls[:5])}\n"
        if callers:
            user_msg += f"\nCall chains reaching this function:\n"
            for path in callers[:3]:
                user_msg += f"  {' → '.join(path)}\n"
        user_msg += f"\nSource code:\n```c\n{context[:6000]}\n```"

        response = self.llm.chat(
            system=SCANNER_SYSTEM,
            user=user_msg,
            max_tokens=1500,
            temperature=0.1,  # low temp for precision
        )

        if not response:
            return []

        findings = _parse_json_array(response)
        results = []
        for f in findings:
            conf = f.get("confidence", 0)
            if conf < 0.5:
                continue
            results.append(ScanFinding(
                file=f.get("file", ""),
                function=f.get("function", func_name),
                bug_type=f.get("bug_type", "unknown"),
                confidence=conf,
                description=f.get("description", ""),
                data_flow=f.get("data_flow", ""),
                trigger_hint=f.get("trigger_hint", ""),
            ))

        return results

    def scan_top_targets(
        self,
        call_graph: CallGraph,
        src_dir: str,
        risky_files: list[tuple[str, str, int]],
        max_targets: int = 10,
    ) -> list[ScanFinding]:
        """Scan the most dangerous functions based on call graph + risk scores."""
        # Prioritize functions that:
        # 1. Are in high-risk files
        # 2. Have many callers (important internal APIs)
        # 3. Handle memory operations

        scored_funcs: list[tuple[str, float]] = []
        risky_file_names = {name for name, _, _ in risky_files}

        for name, fdef in call_graph.functions.items():
            score = 0.0
            # Boost if in a risky file
            if fdef.file in risky_file_names:
                risk_score = next(
                    (s for n, _, s in risky_files if n == fdef.file), 0
                )
                score += risk_score * 0.1

            # Boost if many callers (important API)
            score += len(fdef.called_by) * 2

            # Boost if function name suggests danger
            danger_names = [
                "add", "copy", "parse", "read", "decode", "alloc",
                "create", "append", "concat", "string", "buf",
                "dict", "encode", "grow", "resize",
            ]
            name_lower = name.lower()
            for dn in danger_names:
                if dn in name_lower:
                    score += 5
                    break

            # Boost if body contains dangerous operations
            if fdef.body:
                body_lower = fdef.body.lower()
                for pattern in ["memcpy", "realloc", "malloc", "strcpy", "strlen"]:
                    if pattern in body_lower:
                        score += 3

            if score > 0:
                scored_funcs.append((name, score))

        # Sort by score, scan the top targets
        scored_funcs.sort(key=lambda x: -x[1])
        all_findings: list[ScanFinding] = []

        for func_name, score in scored_funcs[:max_targets]:
            logger.info("[scanner] Scanning %s (score=%.1f)", func_name, score)
            findings = self.scan_function(func_name, call_graph, src_dir)
            for f in findings:
                logger.info(
                    "[scanner] FINDING: %s in %s (conf=%.2f) — %s",
                    f.bug_type, f.function, f.confidence, f.description[:60],
                )
            all_findings.extend(findings)

        # Sort by confidence
        all_findings.sort(key=lambda f: -f.confidence)
        return all_findings


# ══════════════════════════════════════════════════════════════════
# AGENT 2: EXPLOITER
# ══════════════════════════════════════════════════════════════════

EXPLOITER_SYSTEM = """\
You are a security exploit developer. You receive a SPECIFIC vulnerability
finding from a scanner. Your job is to write a Python script that GENERATES
a malicious input file designed to trigger this exact bug.

CRITICAL: Do NOT output raw hex bytes. Write a Python script that constructs
the malicious input programmatically. This is how real exploit developers work.

You will receive:
1. The vulnerability details (function, bug type, data flow)
2. The source code of the vulnerable function

Think step by step:
1. What input format does the program accept? (e.g., XML)
2. How does the input reach the vulnerable function? (follow the data_flow)
3. What properties must the input have to trigger the bug?
   - For integer overflow: what sizes cause the arithmetic to wrap?
   - For buffer overflow: how many bytes overflow the buffer?
   - For null deref: what causes the null pointer?
4. Write Python that constructs those exact bytes.

The script must:
- Write the malicious input to a file called "/tmp/poc_input"
- Use only standard library (no pip packages)
- Be self-contained and runnable with `python3 script.py`

Example for an integer overflow in a size calculation (prefix_len + name_len):
```
# Need prefix_len + name_len to wrap around INT_MAX (2^31 - 1)
prefix = b"A" * 65536
name = b"B" * 65536
xml = b"<" + prefix + b":" + name + b"/>"
with open("/tmp/poc_input", "wb") as f:
    f.write(xml)
```

Example for a stack buffer overflow via deep nesting:
```
depth = 10000
xml = b"<a>" * depth + b"x" + b"</a>" * depth
with open("/tmp/poc_input", "wb") as f:
    f.write(xml)
```

Respond with ONLY the Python script. No markdown fences, no explanation.
Start directly with a comment or import."""

EXPLOITER_RETRY_SYSTEM = """\
Your previous exploit script did NOT crash the target. Here is what happened:
{feedback}

Analyze why and write a COMPLETELY DIFFERENT Python script.
Consider:
- Was the input too small to trigger the overflow?
- Did the parser reject the input before reaching the vulnerable code?
- Do you need a different input structure entirely?

Write a new Python script that generates "/tmp/poc_input".
No markdown fences, no explanation. Start with a comment or import."""


class ExploiterAgent:
    """Takes a scan finding and writes Python scripts to generate exploits."""

    def __init__(self, llm: VLLMClient, binary_path: str, output_dir: str):
        self.llm = llm
        self.binary_path = binary_path
        self.output_dir = output_dir
        self.pov_dir = Path(output_dir) / "povs"
        self.scripts_dir = Path(output_dir) / "exploit_scripts"
        self.pov_dir.mkdir(parents=True, exist_ok=True)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

    def exploit(
        self,
        finding: ScanFinding,
        call_graph: CallGraph,
        src_dir: str,
        max_attempts: int = 3,
    ) -> ExploitResult:
        """Try to exploit a specific finding using LLM-generated scripts."""
        if not self.llm.is_available():
            return ExploitResult(
                finding=finding, crashed=False,
                poc_path=None, crash_output="", attempts=0,
            )

        context = call_graph.get_function_context(finding.function, src_dir)

        vuln_desc = (
            f"Vulnerability: {finding.bug_type}\n"
            f"Function: {finding.function} in {finding.file}\n"
            f"Description: {finding.description}\n"
            f"Data flow: {finding.data_flow}\n"
            f"Trigger hint: {finding.trigger_hint}\n"
        )

        feedback = ""
        for attempt in range(max_attempts):
            logger.info(
                "[exploiter] Attempt %d/%d for %s in %s",
                attempt + 1, max_attempts, finding.bug_type, finding.function,
            )

            if attempt == 0:
                response = self.llm.chat(
                    system=EXPLOITER_SYSTEM,
                    user=f"{vuln_desc}\nSource code:\n```c\n{context[:5000]}\n```",
                    max_tokens=2000,
                    temperature=0.3 + (attempt * 0.15),
                )
            else:
                response = self.llm.chat(
                    system=EXPLOITER_RETRY_SYSTEM.format(feedback=feedback),
                    user=f"{vuln_desc}\nSource code:\n```c\n{context[:5000]}\n```",
                    max_tokens=2000,
                    temperature=0.3 + (attempt * 0.15),
                )

            if not response:
                continue

            # Clean up the script
            script = response.strip()
            if script.startswith("```"):
                script = script.split("\n", 1)[1] if "\n" in script else script[3:]
            if script.endswith("```"):
                script = script.rsplit("```", 1)[0]
            script = script.strip()

            # Save the script
            script_hash = hashlib.sha256(script.encode()).hexdigest()[:8]
            script_path = self.scripts_dir / f"exploit_{finding.function}_{script_hash}.py"
            script_path.write_text(script)
            logger.info("[exploiter] Generated script: %s", script_path.name)

            # Run the script to generate the PoC input
            poc_input = self._run_exploit_script(str(script_path))
            if poc_input is None:
                feedback = f"Script failed to run or didn't create /tmp/poc_input"
                logger.warning("[exploiter] Script failed to generate input.")
                continue

            logger.info("[exploiter] Script generated %d byte input.", len(poc_input))

            # Test the generated input against the binary
            crashed, output = _run_poc(self.binary_path, poc_input)

            if crashed:
                poc_hash = hashlib.sha256(poc_input).hexdigest()[:12]
                poc_path = self.pov_dir / f"exploit-{finding.function}-{poc_hash}"
                poc_path.write_bytes(poc_input)
                logger.info(
                    "[exploiter] *** CRASH *** %s via %s (%d bytes)",
                    finding.bug_type, finding.function, len(poc_input),
                )
                return ExploitResult(
                    finding=finding, crashed=True,
                    poc_path=str(poc_path),
                    crash_output=output,
                    attempts=attempt + 1,
                )
            else:
                # Truncate output for feedback
                short_output = output[:300] if output else "no output"
                feedback = (
                    f"Script ran and generated {len(poc_input)} bytes, "
                    f"but NO crash occurred.\n"
                    f"Binary output: {short_output}\n"
                    f"The input may not be reaching the vulnerable function, "
                    f"or the sizes may not be large enough to trigger the overflow."
                )
                logger.info("[exploiter] No crash. Retrying with feedback.")

        return ExploitResult(
            finding=finding, crashed=False,
            poc_path=None, crash_output=feedback,
            attempts=max_attempts,
        )

    def _run_exploit_script(self, script_path: str) -> bytes | None:
        """Run a Python exploit script and return the generated input."""
        # Clean up any previous output
        poc_path = "/tmp/poc_input"
        try:
            os.unlink(poc_path)
        except FileNotFoundError:
            pass

        try:
            result = subprocess.run(
                ["python3", script_path],
                capture_output=True,
                timeout=30,
                text=True,
            )

            if result.returncode != 0:
                logger.warning(
                    "[exploiter] Script error: %s",
                    result.stderr[:300],
                )
                return None

            if not os.path.exists(poc_path):
                logger.warning("[exploiter] Script didn't create /tmp/poc_input")
                return None

            data = Path(poc_path).read_bytes()
            if len(data) == 0:
                logger.warning("[exploiter] Script created empty file")
                return None

            return data

        except subprocess.TimeoutExpired:
            logger.warning("[exploiter] Script timed out")
            return None
        except Exception as exc:
            logger.error("[exploiter] Script execution failed: %s", exc)
            return None


# ══════════════════════════════════════════════════════════════════
# AGENT 3: HARNESS BUILDER
# ══════════════════════════════════════════════════════════════════

TARGETED_HARNESS_SYSTEM = """\
You are a fuzzing engineer. Generate a LibFuzzer harness that calls
a SPECIFIC vulnerable function as DIRECTLY as possible.

You will receive:
1. The vulnerable function's source code
2. The call chain showing how to reach it
3. Required headers and setup

The harness must:
1. #include the correct headers
2. Implement `int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)`
3. Set up minimal state needed to call the target function
4. Use the fuzz input to exercise the vulnerable code path
5. Clean up resources
6. Return 0

CRITICAL: Call the vulnerable function as DIRECTLY as possible.
Do NOT just call xmlReadMemory — instead, set up the specific context
needed to call the function directly.

If the function requires internal types/state that can't be created
from the public API, use the CLOSEST public API that reaches it.

Respond with ONLY the C source code. No markdown fences, no explanation.
Start directly with #include."""


class HarnessBuilderAgent:
    """Builds targeted harnesses for specific vulnerable functions."""

    def __init__(self, llm: VLLMClient, src_dir: str, output_dir: str):
        self.llm = llm
        self.src_dir = src_dir
        self.output_dir = output_dir
        self.harness_dir = Path(output_dir) / "generated_harnesses"
        self.harness_dir.mkdir(parents=True, exist_ok=True)

    def build_for_finding(
        self,
        finding: ScanFinding,
        call_graph: CallGraph,
    ) -> str | None:
        """Build and compile a harness targeting a specific finding."""
        if not self.llm.is_available():
            return None

        context = call_graph.get_function_context(finding.function, self.src_dir)
        call_paths = call_graph.get_callers(finding.function, depth=4)

        user_msg = (
            f"Target function: {finding.function} in {finding.file}\n"
            f"Bug type: {finding.bug_type}\n"
            f"Description: {finding.description}\n"
            f"Data flow: {finding.data_flow}\n"
        )
        if call_paths:
            user_msg += "\nCall chains to reach this function:\n"
            for path in call_paths[:5]:
                user_msg += f"  {' → '.join(path)}\n"
        user_msg += f"\nSource code:\n```c\n{context[:5000]}\n```"

        # Try up to 2 times (generate + fix)
        for attempt in range(2):
            if attempt == 0:
                response = self.llm.chat(
                    system=TARGETED_HARNESS_SYSTEM,
                    user=user_msg,
                    max_tokens=2000,
                    temperature=0.3,
                )
            else:
                response = self.llm.chat(
                    system=(
                        "Fix the compilation error in this C fuzzer harness. "
                        "Respond with ONLY the corrected C code. Start with #include."
                    ),
                    user=f"Code:\n{code}\n\nError:\n{compile_error[:500]}",
                    max_tokens=2000,
                    temperature=0.1,
                )

            if not response:
                return None

            code = response.strip()
            if code.startswith("```"):
                code = code.split("\n", 1)[1]
            if code.endswith("```"):
                code = code.rsplit("```", 1)[0]

            # Write and compile
            harness_hash = hashlib.sha256(code.encode()).hexdigest()[:8]
            harness_src = self.harness_dir / f"targeted_{finding.function}_{harness_hash}.c"
            harness_bin = self.harness_dir / f"targeted_{finding.function}_{harness_hash}"
            harness_src.write_text(code)

            compile_cmd = [
                "clang", "-g", "-O1", "-fsanitize=address,fuzzer",
                f"-I{self.src_dir}/include",
                f"-I{self.src_dir}",
                str(harness_src),
            ]
            # Try linking against static lib
            lib_path = _find_static_lib(self.src_dir)
            if lib_path:
                compile_cmd.append(lib_path)
            compile_cmd.extend(["-lz", "-llzma", "-lm"])
            compile_cmd.extend(["-o", str(harness_bin)])

            result = subprocess.run(
                compile_cmd, capture_output=True, text=True, timeout=30,
            )

            if result.returncode == 0:
                logger.info(
                    "[harness-builder] SUCCESS: %s targeting %s",
                    harness_bin.name, finding.function,
                )
                return str(harness_bin)
            else:
                compile_error = result.stderr
                logger.warning(
                    "[harness-builder] Compile failed (attempt %d): %s",
                    attempt + 1, compile_error[:200],
                )

        return None


# ══════════════════════════════════════════════════════════════════
# AGENT 4: VERIFIER
# ══════════════════════════════════════════════════════════════════

VERIFIER_SYSTEM = """\
You are a vulnerability verifier. Given a crash report (ASAN output) and
the source code, determine:

1. Is this a REAL security vulnerability or a benign crash (e.g., assertion,
   intentional abort, test-only code)?
2. What is the exact CWE classification?
3. What is the CVSS severity (critical/high/medium/low)?
4. Is this exploitable (can an attacker leverage it)?

Respond with ONLY a JSON object:
{"real_vulnerability": true, "cwe": "CWE-122", "cwe_name": "Heap-based Buffer Overflow",
 "severity": "high", "exploitable": true,
 "root_cause": "one sentence",
 "impact": "one sentence about what an attacker could do"}

Your response must start with { and end with } — nothing else."""


class VerifierAgent:
    """Confirms crashes are real vulnerabilities, not false positives."""

    def __init__(self, llm: VLLMClient):
        self.llm = llm

    def verify(
        self,
        crash_output: str,
        finding: ScanFinding | None,
        call_graph: CallGraph | None,
        src_dir: str,
    ) -> dict:
        """Verify a crash and classify it."""
        if not self.llm.is_available():
            return {"real_vulnerability": True, "severity": "unknown", "verified": False}

        user_msg = f"## Crash Output\n```\n{crash_output[:3000]}\n```\n"

        if finding:
            user_msg += (
                f"\n## Scanner Finding\n"
                f"Function: {finding.function}\n"
                f"Bug type: {finding.bug_type}\n"
                f"Description: {finding.description}\n"
            )

        # Get source context
        if finding and call_graph:
            context = call_graph.get_function_context(finding.function, src_dir)
            if context:
                user_msg += f"\n## Source Code\n```c\n{context[:3000]}\n```"

        response = self.llm.chat(
            system=VERIFIER_SYSTEM,
            user=user_msg,
            max_tokens=500,
            temperature=0.1,
        )

        if not response:
            return {"real_vulnerability": True, "severity": "unknown", "verified": False}

        # Parse JSON
        clean = response.strip()
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1:
            clean = clean[start:end + 1]
        try:
            result = json.loads(clean)
            result["verified"] = True
            return result
        except json.JSONDecodeError:
            return {"real_vulnerability": True, "severity": "unknown", "verified": False}


# ══════════════════════════════════════════════════════════════════
# AGENT ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

class AgentOrchestrator:
    """Coordinates the four agents in a pipeline."""

    def __init__(
        self,
        llm: VLLMClient,
        call_graph: CallGraph,
        binary_path: str,
        src_dir: str,
        output_dir: str,
    ):
        self.scanner = ScannerAgent(llm)
        self.exploiter = ExploiterAgent(llm, binary_path, output_dir)
        self.harness_builder = HarnessBuilderAgent(llm, src_dir, output_dir)
        self.verifier = VerifierAgent(llm)
        self.call_graph = call_graph
        self.src_dir = src_dir
        self.output_dir = output_dir
        self.all_findings: list[ScanFinding] = []
        self.all_exploits: list[ExploitResult] = []
        self.generated_harnesses: list[str] = []
        self.bugs_dir = Path(output_dir) / "bugs"
        self.bugs_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline(
        self,
        risky_files: list[tuple[str, str, int]],
        max_scan_targets: int = 8,
        max_exploit_targets: int = 3,
    ) -> dict:
        """Run the full scan → exploit → verify pipeline."""
        logger.info("╔══════════════════════════════════════════════════╗")
        logger.info("║  Multi-Agent Pipeline                            ║")
        logger.info("╚══════════════════════════════════════════════════╝")

        # ── Phase 1: Scanner scans top targets ──
        logger.info("[pipeline] Phase 1: Scanning top %d targets...", max_scan_targets)
        findings = self.scanner.scan_top_targets(
            self.call_graph, self.src_dir, risky_files, max_scan_targets,
        )
        self.all_findings.extend(findings)
        logger.info("[pipeline] Scanner found %d suspicious locations.", len(findings))

        # ── Phase 2: Exploiter attacks top findings ──
        logger.info("[pipeline] Phase 2: Exploiting top %d findings...", max_exploit_targets)
        top_findings = sorted(findings, key=lambda f: -f.confidence)[:max_exploit_targets]

        for finding in top_findings:
            logger.info(
                "[pipeline] Exploiting: %s in %s (conf=%.2f)",
                finding.bug_type, finding.function, finding.confidence,
            )
            result = self.exploiter.exploit(
                finding, self.call_graph, self.src_dir,
            )
            self.all_exploits.append(result)

            if result.crashed:
                # ── Phase 3: Verify the crash ──
                logger.info("[pipeline] Phase 3: Verifying crash...")
                verification = self.verifier.verify(
                    result.crash_output, finding, self.call_graph, self.src_dir,
                )

                # Write verified bug report
                report = {
                    "function": finding.function,
                    "file": finding.file,
                    "bug_type": finding.bug_type,
                    "confidence": finding.confidence,
                    "description": finding.description,
                    "data_flow": finding.data_flow,
                    "poc_path": result.poc_path,
                    "verification": verification,
                    "strategy": "multi_agent_pipeline",
                    "timestamp": time.time(),
                }
                rhash = hashlib.sha256(
                    json.dumps(report, sort_keys=True, default=str).encode()
                ).hexdigest()[:12]
                bug_path = self.bugs_dir / f"verified-{rhash}.json"
                bug_path.write_text(json.dumps(report, indent=2, default=str))

                logger.info(
                    "[pipeline] *** VERIFIED BUG: %s in %s — %s (CWE: %s) ***",
                    finding.bug_type, finding.function,
                    verification.get("severity", "?"),
                    verification.get("cwe", "?"),
                )

        # ── Phase 4: Build targeted harnesses for unfixed findings ──
        uncrashed = [
            r.finding for r in self.all_exploits if not r.crashed
        ]
        if uncrashed:
            logger.info(
                "[pipeline] Phase 4: Building %d targeted harnesses...",
                min(len(uncrashed), 2),
            )
            for finding in uncrashed[:2]:
                harness = self.harness_builder.build_for_finding(
                    finding, self.call_graph,
                )
                if harness:
                    self.generated_harnesses.append(harness)

        # ── Summary ──
        crashes = sum(1 for r in self.all_exploits if r.crashed)
        summary = {
            "scan_findings": len(findings),
            "exploit_attempts": len(self.all_exploits),
            "crashes_found": crashes,
            "harnesses_built": len(self.generated_harnesses),
        }
        logger.info("[pipeline] Results: %s", json.dumps(summary))
        return summary


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _run_poc(binary: str, data: bytes) -> tuple[bool, str]:
    """Run a PoC input against the fuzzer binary."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".poc") as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        env = os.environ.copy()
        env["ASAN_OPTIONS"] = "abort_on_error=1:detect_leaks=0"

        result = subprocess.run(
            [binary, tmp_path],
            capture_output=True,
            timeout=10,
            env=env,
        )

        output = result.stderr.decode("utf-8", errors="replace")
        crashed = result.returncode != 0 and (
            "AddressSanitizer" in output or
            "SUMMARY:" in output or
            result.returncode < 0
        )
        return crashed, output[:1000]

    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:
        return False, str(exc)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _find_static_lib(src_dir: str) -> str | None:
    """Find libxml2.a or similar static library."""
    for pattern in ["**/*.a", ".libs/*.a"]:
        matches = list(Path(src_dir).glob(pattern))
        if matches:
            return str(matches[0])
    return None


def _parse_json_array(response: str) -> list[dict]:
    """Robustly parse a JSON array from LLM output."""
    clean = response.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
    if clean.endswith("```"):
        clean = clean.rsplit("```", 1)[0]
    clean = clean.strip()
    start = clean.find("[")
    end = clean.rfind("]")
    if start != -1 and end != -1 and end > start:
        clean = clean[start:end + 1]
    else:
        return []
    try:
        result = json.loads(clean)
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
        return []
    except json.JSONDecodeError:
        return []
