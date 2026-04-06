"""
crs/vuln_discovery.py — Vulnerability discovery for Level 0 tasks.

When no vulnerability description is provided, this module:
  1. LLM scans source code for vulnerability candidates (static analysis)
  2. Ranks candidates by exploitability
  3. Generates a synthetic description for each candidate
  4. Feeds candidates into the existing byte_strategies pipeline

Inspired by OSS-CRS / AIxCC Team Atlanta approach:
  - LLM reads code and identifies suspicious patterns
  - Short fuzzing burst confirms reachability
  - LLM crafts targeted inputs for confirmed candidates

This plugs into main.py: if task has no description, run discovery first.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from crs.code_intelligence import (
    CodeContext, classify_vulnerability, rank_files_by_relevance,
    get_source_files, detect_build_system,
)
from crs.config import cfg
from crs.data_loader import CyberGymTask
from crs.llm_router import LLMRouter

logger = logging.getLogger(__name__)


@dataclass
class VulnCandidate:
    """A potential vulnerability identified by the LLM."""
    function_name: str
    file_path: str
    line_number: int
    vuln_type: str
    description: str          # synthetic description for the byte pipeline
    confidence: float
    code_snippet: str
    attack_input_hint: str    # LLM's guess at what input triggers it


# ── Prompts ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_VULN_SCANNER = """\
You are an expert C/C++ vulnerability researcher performing a security audit.

You will be given source code from a project. Your task is to identify
REAL, EXPLOITABLE vulnerabilities — not style issues or theoretical concerns.

Focus on these high-value vulnerability classes:
1. BUFFER OVERFLOW: memcpy/strcpy/sprintf with unchecked lengths,
   array indexing without bounds checks, integer overflow in size calculations
2. HEAP CORRUPTION: use-after-free, double-free, heap overflow
3. NULL DEREFERENCE: unchecked return values from malloc/calloc/fopen
4. INTEGER OVERFLOW: arithmetic on sizes/lengths that can wrap
5. FORMAT STRING: user-controlled printf format arguments
6. UNINITIALIZED MEMORY: variables read before being written
7. OFF-BY-ONE: loop bounds, string termination, fence-post errors
8. MISSING VALIDATION: length/size fields from untrusted input not checked
   before use (e.g., "if (length > 0)" but code reads 5 bytes)

For each vulnerability found, output a JSON array. Each entry must have:
{
  "function": "name of the vulnerable function",
  "file": "filename.c",
  "line": approximate line number,
  "type": "buffer_overflow|use_after_free|null_deref|integer_overflow|...",
  "description": "One paragraph describing the bug precisely. Name the
    function, what input reaches it, what check is missing, and what
    memory corruption occurs. This description will be used by another
    system to generate a triggering input.",
  "confidence": 0.0 to 1.0,
  "input_hint": "What kind of input triggers this? e.g., 'A MNG file with
    a LOOP chunk shorter than 5 bytes' or 'A MQTT PUBLISH packet with
    topic length 0xFFFF'"
}

IMPORTANT:
- Only report bugs you can see evidence for in the provided code
- The description must name the SPECIFIC function and what check is missing
- Include the FORMAT of the input (what file type, protocol, etc.)
- Output ONLY the JSON array, no other text
- Maximum 5 candidates, ranked by confidence
"""

SYSTEM_PROMPT_DEEP_SCAN = """\
You are an expert C/C++ vulnerability researcher. You are given a specific
function from a project's source code.

Analyze this function for security vulnerabilities. Look at:
1. Every buffer access — is the index/length validated?
2. Every pointer — is NULL checked after allocation/lookup?
3. Every integer arithmetic — can it overflow/underflow?
4. Every loop — can it go out of bounds?
5. Every memcpy/read — is the source length validated against dest size?
6. Every field read from input — is the field length checked first?

If you find a vulnerability, describe it precisely:
- What line is it on?
- What is the exact check that is missing?
- What input value would trigger it?
- What is the consequence (OOB read, OOB write, crash)?

Output a JSON object with:
{
  "vulnerable": true/false,
  "function": "function name",
  "line": line number,
  "type": "vuln type",
  "description": "precise description for exploit generation",
  "input_hint": "what input triggers it"
}

Output ONLY the JSON, no other text.
"""


# ── Scanner ────────────────────────────────────────────────────────────────

def discover_vulnerabilities(
    task: CyberGymTask,
    router: LLMRouter,
    max_candidates: int = 5,
) -> List[VulnCandidate]:
    """
    Scan project source code for vulnerabilities using LLM analysis.

    Returns a list of VulnCandidate objects, each with a synthetic
    description that can be fed into the existing byte strategies pipeline.
    """
    repo = Path(task.repo_path).resolve()
    all_files = get_source_files(task)

    if not all_files:
        print("[discovery] No source files found")
        return []

    print(f"[discovery] Scanning {len(all_files)} source files...")

    # Phase 1: Broad scan — LLM reviews top files for vuln candidates
    candidates = _broad_scan(repo, all_files, router)
    print(f"[discovery] Broad scan found {len(candidates)} candidate(s)")

    # Phase 2: Deep scan — LLM analyzes each candidate function in detail
    confirmed = _deep_scan(repo, candidates, router)
    print(f"[discovery] Deep scan confirmed {len(confirmed)} candidate(s)")

    # Sort by confidence, return top N
    confirmed.sort(key=lambda c: c.confidence, reverse=True)
    return confirmed[:max_candidates]


def _broad_scan(
    repo: Path,
    all_files: List[Path],
    router: LLMRouter,
) -> List[VulnCandidate]:
    """Phase 1: LLM reviews high-risk source files for vulnerability patterns."""

    # Identify high-risk files: parsers, decoders, network handlers
    high_risk_patterns = [
        "parse", "read", "decode", "process", "handle", "load",
        "recv", "input", "packet", "message", "request", "coder",
        "codec", "format", "deserializ",
    ]

    # Score files by risk
    scored_files: list[tuple[Path, float]] = []
    for f in all_files:
        score = 0.0
        fname = f.name.lower()
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
            content_lower = content.lower()

            # File name risk
            for pat in high_risk_patterns:
                if pat in fname:
                    score += 3.0

            # Content risk: unsafe APIs
            for api in ["memcpy", "strcpy", "sprintf", "gets", "scanf",
                        "strcat", "malloc", "realloc", "free"]:
                score += content_lower.count(api) * 0.2

            # Content risk: parsing patterns
            for pat in ["length", "size", "offset", "chunk", "header",
                        "buffer", "packet"]:
                score += content_lower.count(pat) * 0.05

            # File size bonus (larger files = more attack surface)
            score += min(len(content) / 10000, 3.0)

        except Exception:
            pass
        scored_files.append((f, score))

    scored_files.sort(key=lambda x: x[1], reverse=True)

    # Send top files to LLM for analysis
    candidates: list[VulnCandidate] = []
    budget = 12000  # chars to send to LLM

    snippets_parts: list[str] = []
    total = 0
    for path, score in scored_files[:15]:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            # For large files, extract functions with risky patterns
            if len(content) > 3000:
                content = _extract_risky_functions(content, path.name)
            chunk = min(len(content), budget - total)
            if chunk <= 0:
                break
            header = f"\n// === {path.name} (risk={score:.1f}) ===\n"
            snippets_parts.append(header + content[:chunk])
            total += chunk + len(header)
        except Exception:
            pass

    if not snippets_parts:
        return candidates

    code_block = "".join(snippets_parts)

    prompt = (
        f"## Source Code to Audit\n```c\n{code_block}\n```\n\n"
        f"Identify up to 5 exploitable vulnerabilities in this code.\n"
        f"Output ONLY a JSON array as specified in your instructions.\n"
    )

    raw = router.chat(
        system_prompt=SYSTEM_PROMPT_VULN_SCANNER,
        user_prompt=prompt,
        max_tokens=3000,
        temperature=0.1,
    )

    if raw:
        candidates = _parse_candidates(raw)

    return candidates


def _deep_scan(
    repo: Path,
    candidates: List[VulnCandidate],
    router: LLMRouter,
) -> List[VulnCandidate]:
    """Phase 2: LLM does detailed analysis of each candidate function."""

    confirmed: list[VulnCandidate] = []

    for cand in candidates:
        # Find the actual function in the source
        func_code = _find_function_source(repo, cand.function_name, cand.file_path)
        if not func_code:
            # Still include it if broad scan was confident
            if cand.confidence >= 0.7:
                confirmed.append(cand)
            continue

        prompt = (
            f"## Function to Analyze: {cand.function_name}\n"
            f"## File: {cand.file_path}\n"
            f"## Suspected vulnerability: {cand.vuln_type}\n\n"
            f"```c\n{func_code[:4000]}\n```\n\n"
            f"Analyze this function for the suspected vulnerability.\n"
            f"Output ONLY a JSON object as specified.\n"
        )

        raw = router.chat(
            system_prompt=SYSTEM_PROMPT_DEEP_SCAN,
            user_prompt=prompt,
            max_tokens=1500,
            temperature=0.1,
        )

        if raw:
            result = _parse_deep_scan(raw)
            if result and result.get("vulnerable"):
                # Update candidate with refined info
                cand.description = result.get("description", cand.description)
                cand.attack_input_hint = result.get("input_hint", cand.attack_input_hint)
                cand.confidence = min(cand.confidence + 0.1, 1.0)
                confirmed.append(cand)

    return confirmed


# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_risky_functions(content: str, filename: str) -> str:
    """Extract functions that contain risky patterns from a large file."""
    lines = content.splitlines()
    risky_keywords = [
        "memcpy", "strcpy", "sprintf", "malloc", "realloc", "free",
        "length", "size", "chunk", "buffer", "offset", "read",
        "parse", "decode", "packet", "header",
    ]

    # Find lines with risky keywords
    risky_lines: set[int] = set()
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for kw in risky_keywords:
            if kw in line_lower:
                risky_lines.add(i)
                break

    if not risky_lines:
        return content[:3000]

    # Extract context around risky lines
    parts: list[str] = []
    total = 0
    for line_num in sorted(risky_lines):
        start = max(0, line_num - 15)
        end = min(len(lines), line_num + 15)
        section = "\n".join(f"{i+1}: {lines[i]}" for i in range(start, end))
        if total + len(section) > 6000:
            break
        parts.append(section)
        total += len(section)

    return "\n...\n".join(parts)


def _parse_candidates(raw: str) -> List[VulnCandidate]:
    """Parse LLM output into VulnCandidate objects."""
    import json

    # Strip markdown fences
    cleaned = re.sub(r"```json\s*", "", raw)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    candidates: list[VulnCandidate] = []
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data = [data]
        for item in data:
            if not isinstance(item, dict):
                continue
            candidates.append(VulnCandidate(
                function_name=item.get("function", "unknown"),
                file_path=item.get("file", "unknown"),
                line_number=item.get("line", 0),
                vuln_type=item.get("type", "other"),
                description=item.get("description", ""),
                confidence=float(item.get("confidence", 0.5)),
                code_snippet="",
                attack_input_hint=item.get("input_hint", ""),
            ))
    except json.JSONDecodeError:
        # Try to extract individual JSON objects
        for m in re.finditer(r'\{[^{}]+\}', raw, re.DOTALL):
            try:
                item = json.loads(m.group())
                candidates.append(VulnCandidate(
                    function_name=item.get("function", "unknown"),
                    file_path=item.get("file", "unknown"),
                    line_number=item.get("line", 0),
                    vuln_type=item.get("type", "other"),
                    description=item.get("description", ""),
                    confidence=float(item.get("confidence", 0.5)),
                    code_snippet="",
                    attack_input_hint=item.get("input_hint", ""),
                ))
            except (json.JSONDecodeError, ValueError):
                pass

    return candidates


def _parse_deep_scan(raw: str) -> Optional[dict]:
    """Parse deep scan JSON response."""
    import json
    cleaned = re.sub(r"```json\s*", "", raw)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def _find_function_source(
    repo: Path, func_name: str, file_hint: str
) -> Optional[str]:
    """Find and extract a function's source code from the repo."""
    # Try to find the file
    target_files: list[Path] = []
    for f in repo.rglob("*"):
        if f.is_file() and f.name == file_hint:
            target_files.append(f)
    if not target_files:
        # Try partial match
        for f in repo.rglob("*.c"):
            if file_hint.replace(".c", "") in f.name:
                target_files.append(f)
        for f in repo.rglob("*.cc"):
            if file_hint.replace(".cc", "") in f.name:
                target_files.append(f)

    for tf in target_files:
        try:
            content = tf.read_text(encoding="utf-8", errors="replace")
            # Find the function
            pattern = rf'(?:static\s+)?(?:[\w*]+\s+)+{re.escape(func_name)}\s*\('
            m = re.search(pattern, content)
            if not m:
                continue
            start = m.start()
            # Find opening brace
            brace_pos = content.find("{", m.end())
            if brace_pos == -1:
                continue
            # Match braces
            depth = 0
            pos = brace_pos
            while pos < len(content):
                if content[pos] == "{":
                    depth += 1
                elif content[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        return content[max(0, start - 200):pos + 1]
                pos += 1
        except Exception:
            continue
    return None


# ── Integration with main pipeline ─────────────────────────────────────────

def build_context_from_candidate(
    task: CyberGymTask,
    candidate: VulnCandidate,
) -> CodeContext:
    """
    Build a CodeContext from a VulnCandidate, so the existing byte
    strategies pipeline can work with it.

    The candidate's description becomes the task's vulnerability_description.
    """
    from crs.code_intelligence import (
        extract_relevant_snippets, build_context,
    )

    # Temporarily set the description to the candidate's synthetic one
    original_desc = task.vulnerability_description
    task.vulnerability_description = candidate.description

    # Build context using the normal pipeline
    ctx = build_context(task)

    # Restore original (empty) description
    task.vulnerability_description = original_desc

    # Override with candidate-specific info
    ctx.description = candidate.description
    ctx.vuln_type = classify_vulnerability(candidate.description)

    return ctx


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from crs.data_loader import load_task_from_local

    if len(sys.argv) < 2:
        print("Usage: python -m crs.vuln_discovery <task_dir>")
        sys.exit(1)

    task = load_task_from_local(sys.argv[1])
    router = LLMRouter()
    candidates = discover_vulnerabilities(task, router)

    print(f"\n{'='*60}")
    print(f"  Found {len(candidates)} vulnerability candidate(s)")
    print(f"{'='*60}")
    for i, c in enumerate(candidates):
        print(f"\n  [{i}] {c.function_name} in {c.file_path}:{c.line_number}")
        print(f"      Type: {c.vuln_type}  Confidence: {c.confidence:.2f}")
        print(f"      {c.description[:120]}...")
        print(f"      Input hint: {c.attack_input_hint}")
