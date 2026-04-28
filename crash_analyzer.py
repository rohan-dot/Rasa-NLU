"""
crash_analyzer.py — LLM-powered crash analysis and seed generation.

Uses Gemma (via vLLM) to:
1. Analyze ASAN crash traces and produce bug-candidate reports
2. Suggest seed mutations that might trigger deeper bugs
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

from llm_client import VLLMClient

logger = logging.getLogger("gemma-fuzzer.analyzer")

# ── System prompts ────────────────────────────────────────────────

CRASH_ANALYSIS_SYSTEM = """\
You are a security researcher analyzing AddressSanitizer crash reports.
Given a crash stack trace and optionally source code context, produce a
concise JSON bug report with these fields:

{
  "crash_type": "heap-buffer-overflow | use-after-free | stack-overflow | ...",
  "severity": "high | medium | low",
  "root_cause": "one-sentence explanation of the likely root cause",
  "affected_function": "function name where the bug occurs",
  "cwe": "CWE-ID if identifiable, else null",
  "summary": "2-3 sentence summary suitable for a bug report"
}

Respond ONLY with the JSON object, no markdown fences or extra text."""

SEED_GENERATION_SYSTEM = """\
You are a fuzzing expert. Given information about a target program and
any crashes found so far, suggest 3 to 5 raw input byte sequences as hex
strings that might trigger new crashes in unexplored code paths.

Focus on:
- Boundary values (0, -1, INT_MAX, large sizes)
- Format-specific magic bytes and malformed headers
- Inputs that exercise error-handling code paths

You MUST respond with ONLY a JSON array. No explanation, no markdown, no
code fences, no text before or after. Example of the EXACT format:

[{"hex": "3c3f786d6c", "rationale": "XML declaration header"}, {"hex": "00000000ffffffff", "rationale": "boundary integer values"}]

Your response must start with [ and end with ] — nothing else."""


class CrashAnalyzer:
    """Uses the LLM to analyze crashes and generate seed inputs."""

    def __init__(self, llm: VLLMClient, src_dir: str, output_dir: str):
        self.llm = llm
        self.src_dir = Path(src_dir)
        self.output_dir = Path(output_dir)
        self.bugs_dir = self.output_dir / "bugs"
        self.seeds_dir = self.output_dir / "seeds"
        self.bugs_dir.mkdir(parents=True, exist_ok=True)
        self.seeds_dir.mkdir(parents=True, exist_ok=True)

    # ── Crash analysis ────────────────────────────────────────────

    def analyze_crash(self, crash_file: str, stack_trace: str) -> dict | None:
        """Analyze a crash with the LLM and write a bug-candidate report."""
        if not self.llm.is_available():
            logger.info("LLM unavailable — writing raw crash report.")
            return self._write_raw_report(crash_file, stack_trace)

        # Try to find relevant source context
        source_context = self._extract_source_context(stack_trace)

        user_msg = f"## Stack Trace\n```\n{stack_trace[:3000]}\n```\n"
        if source_context:
            user_msg += f"\n## Source Context\n```c\n{source_context[:2000]}\n```\n"

        response = self.llm.chat(
            system=CRASH_ANALYSIS_SYSTEM,
            user=user_msg,
            max_tokens=512,
            temperature=0.1,
        )

        if not response:
            return self._write_raw_report(crash_file, stack_trace)

        # Parse LLM response
        try:
            # Strip markdown fences if the model added them anyway
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1]
            if clean.endswith("```"):
                clean = clean.rsplit("```", 1)[0]
            clean = clean.strip()
            # Try to find a JSON object anywhere in the response
            start_idx = clean.find("{")
            end_idx = clean.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                clean = clean[start_idx:end_idx + 1]
            report = json.loads(clean)
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON — using raw report.")
            report = {
                "crash_type": "unknown",
                "severity": "unknown",
                "root_cause": response[:200],
                "summary": response[:500],
            }

        # Enrich with metadata
        report["crash_file"] = os.path.basename(crash_file)
        report["stack_trace"] = stack_trace[:5000]
        report["timestamp"] = time.time()

        # Write bug-candidate file (picked up by register-submit-dir)
        crash_hash = hashlib.sha256(stack_trace.encode()).hexdigest()[:12]
        bug_path = self.bugs_dir / f"bug-{crash_hash}.json"
        bug_path.write_text(json.dumps(report, indent=2))
        logger.info("Bug report written: %s", bug_path)

        return report

    def _write_raw_report(self, crash_file: str, stack_trace: str) -> dict:
        """Fallback: write a raw crash report without LLM analysis."""
        report = {
            "crash_file": os.path.basename(crash_file),
            "crash_type": "see-stack-trace",
            "severity": "unknown",
            "root_cause": "LLM analysis unavailable",
            "stack_trace": stack_trace[:5000],
            "timestamp": time.time(),
        }
        crash_hash = hashlib.sha256(stack_trace.encode()).hexdigest()[:12]
        bug_path = self.bugs_dir / f"bug-{crash_hash}.json"
        bug_path.write_text(json.dumps(report, indent=2))
        return report

    def _extract_source_context(self, stack_trace: str) -> str:
        """Try to extract source code around the crash location."""
        # Parse file:line from stack trace frames
        context_lines = []
        for line in stack_trace.split("\n"):
            if " in " not in line or ":" not in line:
                continue
            try:
                # Format: "#N 0xADDR in func_name file.c:42:13"
                after_in = line.split(" in ", 1)[1]
                parts = after_in.strip().split()
                if len(parts) < 2:
                    continue
                file_loc = parts[1]  # "file.c:42" or "file.c:42:13"
                file_parts = file_loc.split(":")
                filename = file_parts[0]
                lineno = int(file_parts[1])
            except (IndexError, ValueError):
                continue

            # Search for the file in src_dir
            matches = list(self.src_dir.rglob(os.path.basename(filename)))
            if not matches:
                continue

            src_file = matches[0]
            try:
                src_lines = src_file.read_text(errors="replace").split("\n")
                start = max(0, lineno - 10)
                end = min(len(src_lines), lineno + 10)
                snippet = "\n".join(
                    f"{i+1:4d} | {src_lines[i]}"
                    for i in range(start, end)
                )
                context_lines.append(f"// {src_file.name}:{lineno}\n{snippet}")
            except Exception:
                continue

            if len(context_lines) >= 2:
                break

        return "\n\n".join(context_lines)

    # ── Seed generation ───────────────────────────────────────────

    def generate_seeds(
        self,
        harness_name: str,
        crash_summaries: list[str],
    ) -> int:
        """Ask the LLM to suggest seed inputs. Returns count generated."""
        if not self.llm.is_available():
            return 0

        user_msg = f"Target harness: {harness_name}\n\n"
        if crash_summaries:
            user_msg += "Crashes found so far:\n"
            for s in crash_summaries[:5]:
                user_msg += f"- {s}\n"
        else:
            user_msg += "No crashes found yet. Suggest diverse initial seeds.\n"

        response = self.llm.chat(
            system=SEED_GENERATION_SYSTEM,
            user=user_msg,
            max_tokens=512,
            temperature=0.7,
        )

        if not response:
            return 0

        try:
            clean = response.strip()
            # Strip markdown fences
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1]
            if clean.endswith("```"):
                clean = clean.rsplit("```", 1)[0]
            clean = clean.strip()
            # Try to find a JSON array anywhere in the response
            start_idx = clean.find("[")
            end_idx = clean.rfind("]")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                clean = clean[start_idx:end_idx + 1]
            seeds = json.loads(clean)
            if not isinstance(seeds, list):
                seeds = [seeds]
        except json.JSONDecodeError:
            logger.warning(
                "LLM returned invalid JSON for seeds. Response: %s",
                response[:200],
            )
            return 0

        count = 0
        for i, seed_info in enumerate(seeds):
            hex_str = seed_info.get("hex", "")
            if not hex_str:
                continue
            try:
                data = bytes.fromhex(hex_str)
            except ValueError:
                continue

            seed_hash = hashlib.sha256(data).hexdigest()[:12]
            seed_path = self.seeds_dir / f"llm-seed-{seed_hash}"
            seed_path.write_bytes(data)
            count += 1
            logger.debug(
                "Generated seed %s (%d bytes): %s",
                seed_path.name, len(data),
                seed_info.get("rationale", ""),
            )

        logger.info("LLM generated %d seed inputs.", count)
        return count
