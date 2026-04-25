"""
crs/strategy_smart.py — Agentic constraint-solving PoC generation.

Instead of asking the LLM to "read code and craft bytes" in one shot,
this strategy makes the LLM think like a security researcher:

Phase 1 — CRASH ANALYSIS
  "What exact operation at the crash site causes the bug?
   What variable has the bad value? What should it have been?"

Phase 2 — BACKWARD TAINT TRACE (multi-step, one function at a time)
  For each function in the call chain, from crash site upward:
  "Where does variable X get its value in this function?
   What conditions must be true for execution to reach this point?
   What are the constraints on the input so far?"

  The LLM accumulates a CONSTRAINT LIST across calls:
    constraint 1: bytes[0:4] == 0x89504E47  (PNG magic)
    constraint 2: bytes[8:12] == "IHDR"     (chunk type)
    constraint 3: bytes[16:20] > 0x7FFFFFFF (width field, triggers overflow)
    ...

Phase 3 — INPUT CONSTRUCTION
  Given the full constraint list, generate a Python script that
  constructs bytes satisfying ALL constraints simultaneously.

Phase 4 — VERIFICATION & ADJUSTMENT
  If the PoC doesn't crash, feed the error back and ask:
  "Which constraint was wrong? What did the binary actually check?"

This is fundamentally different from one-shot generation because:
- Each LLM call is FOCUSED on one small piece of reasoning
- Context ACCUMULATES across calls (the constraint list grows)
- The final generation has EXPLICIT constraints to satisfy
- Failures produce SPECIFIC feedback ("constraint 3 was wrong")
"""
from __future__ import annotations
import re, os, subprocess, tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from crs.config import cfg
from crs.byte_gen import PoCCandidate, _save_poc, _run_script, _extract_python


# ── Source helpers ─────────────────────────────────────────────────────────

def _find_and_read(repo: Path, filename: str, center_line: int = 0,
                    context: int = 80) -> str:
    """Find a source file in the repo and read around a specific line."""
    basename = Path(filename).name
    candidates = list(repo.rglob(basename))
    if not candidates:
        return ""
    target = min(candidates, key=lambda p: len(p.parts))
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").split("\n")
        if center_line > 0:
            start = max(0, center_line - context // 2)
            end = min(len(lines), center_line + context // 2)
        else:
            start, end = 0, min(len(lines), 200)
        return "\n".join(f"{i+1:4d}| {lines[i]}" for i in range(start, end))
    except Exception:
        return ""


def _parse_stack(description: str) -> list[dict]:
    """Extract call chain frames from ASAN output."""
    frames = re.findall(
        r"#(\d+)\s+\S+\s+in\s+(\w+)\s+(\S+?):(\d+)",
        description,
    )
    return [
        {"frame": int(f[0]), "func": f[1], "file": f[2], "line": int(f[3])}
        for f in frames
    ]


def _extract_crash_from_text(description: str, repo: Path) -> list[dict]:
    """
    Fallback: extract function and file names from a plain-text CVE
    description when no ASAN stack trace is available.

    Handles descriptions like:
      "heap-buffer-overflow in stszin in mp4read.c"
      "NULL dereference in ic_predict of libfaad/ic_predict.c"
      "buffer overflow in calculate_gain function in libfaad/filtbank.c"
    """
    desc_lower = description.lower()

    # Collect all .c/.cpp/.h filenames mentioned in the description
    mentioned_files = re.findall(r'[\w/]+\.(?:c|cpp|cc|h|hpp)\b', description)

    # Collect function-like words near keywords like "in", "function", "of"
    # Pattern: "in <func_name>" or "function <func_name>"
    func_candidates = []
    for m in re.finditer(r'(?:in|function|of)\s+(\w{3,})', description, re.IGNORECASE):
        word = m.group(1)
        # Skip common English words
        skip = {"the", "this", "that", "from", "with", "which", "before",
                "after", "when", "where", "faad2", "version", "allows",
                "attacker", "cause", "code", "execution", "denial",
                "service", "issue", "discovered", "exists", "located",
                "freeware", "advanced", "audio", "third", "instance"}
        if word.lower() not in skip and not word[0].isupper():
            func_candidates.append(word)

    # Also look for C-style identifiers that look like function names
    # (contain underscores or are all lowercase with no spaces)
    for m in re.finditer(r'\b([a-z_]\w*(?:_\w+)+)\b', description):
        word = m.group(1)
        if len(word) >= 4 and word not in func_candidates:
            func_candidates.append(word)

    if not func_candidates and not mentioned_files:
        return []

    # Try to find these functions in the actual source code
    chain = []
    src_exts = {".c", ".cpp", ".cc", ".h", ".hpp"}

    for func_name in func_candidates:
        # Search repo for this function definition
        for f in sorted(repo.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in src_exts:
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                # Look for function definition (not just call)
                # Pattern: return_type func_name(
                pattern = rf'\b{re.escape(func_name)}\s*\('
                m = re.search(pattern, content)
                if m:
                    # Find line number
                    line_num = content[:m.start()].count('\n') + 1
                    chain.append({
                        "frame": len(chain),
                        "func": func_name,
                        "file": str(f.relative_to(repo)) if f.is_relative_to(repo) else f.name,
                        "line": line_num,
                    })
                    break
            except Exception:
                continue

    # If we found functions in mentioned files, prioritize those
    if mentioned_files and not chain:
        for mf in mentioned_files:
            basename = Path(mf).name
            candidates = list(repo.rglob(basename))
            if candidates:
                target = min(candidates, key=lambda p: len(p.parts))
                chain.append({
                    "frame": 0,
                    "func": "unknown",
                    "file": str(target.relative_to(repo)) if target.is_relative_to(repo) else target.name,
                    "line": 1,
                })

    return chain


def _parse_crash_type(description: str) -> tuple[str, str, str]:
    """Extract crash type, operation, and size."""
    crash_type = ""
    m = re.search(r"ERROR: AddressSanitizer: (\S+)", description)
    if m: crash_type = m.group(1)

    operation = ""
    m = re.search(r"(READ|WRITE) of size (\d+)", description)
    if m: operation = f"{m.group(1)} of size {m.group(2)}"

    return crash_type, operation, ""


# ── Phase 1: Crash analysis ───────────────────────────────────────────────

PROMPT_CRASH_ANALYSIS = """\
You are a vulnerability analyst. Look at the crash site and answer
ONLY these questions. Be precise — name exact variables and values.

1. WHAT OPERATION crashes? (memcpy? array access? pointer deref?)
2. WHAT VARIABLE has the bad value? (an index? a length? a pointer?)
3. WHY is the value bad? (too large? points to freed memory? NULL?)
4. WHERE does that variable get its value? (parameter? struct field?
   computed from what?)

Answer in this EXACT format:
OPERATION: <the crashing operation>
BAD_VARIABLE: <variable name>
WHY_BAD: <why the value causes the crash>
SOURCE_OF_VALUE: <where the variable gets its value from>
"""


def _phase1_crash_analysis(router, crash_func, crash_file, crash_line,
                            crash_type, operation, repo) -> dict:
    """Analyze the crash site to identify what variable causes the bug."""
    source = _find_and_read(repo, crash_file, crash_line, context=60)

    prompt = (
        f"## Crash Site\n"
        f"Function: {crash_func}()\n"
        f"File: {crash_file}, line {crash_line}\n"
        f"Crash type: {crash_type}\n"
        f"Operation: {operation}\n\n"
        f"## Source Code Around Crash\n```c\n{source}\n```\n\n"
        f"Analyze this crash site.\n"
    )

    print(f"    Phase 1: Analyzing crash in {crash_func}()...")
    response = router.chat(PROMPT_CRASH_ANALYSIS, prompt, max_tokens=1024, temperature=0.05)

    # Parse structured response
    analysis = {
        "operation": "",
        "bad_variable": "",
        "why_bad": "",
        "source_of_value": "",
        "raw": response,
    }
    for key, pattern in [
        ("operation", r"OPERATION:\s*(.+)"),
        ("bad_variable", r"BAD_VARIABLE:\s*(.+)"),
        ("why_bad", r"WHY_BAD:\s*(.+)"),
        ("source_of_value", r"SOURCE_OF_VALUE:\s*(.+)"),
    ]:
        m = re.search(pattern, response)
        if m:
            analysis[key] = m.group(1).strip()

    print(f"      Bad variable: {analysis['bad_variable']}")
    print(f"      Why bad: {analysis['why_bad']}")
    print(f"      Source: {analysis['source_of_value']}")
    return analysis


# ── Phase 2: Backward taint trace ─────────────────────────────────────────

PROMPT_TAINT_TRACE = """\
You are tracing a variable BACKWARDS through a function to find where
its value comes from and what conditions must be true.

You are given:
- The VARIABLE to trace
- The FUNCTION's source code
- CONSTRAINTS accumulated so far from deeper in the call chain

Answer in this EXACT format:

VARIABLE_ORIGIN: <where does the variable get its value in THIS function?>
  (e.g., "parsed from bytes[6:10] as uint32_t little-endian")
  (e.g., "parameter passed from caller: specrec_decode(data, len)")
  (e.g., "computed as header->width * header->height")

NEW_CONSTRAINTS:
- <constraint on input bytes, e.g., "bytes[0:4] must be 0x89504E47">
- <another constraint, e.g., "bytes[4:8] must be chunk length > 0">
- <branch condition, e.g., "bytes[12] must be 0x02 to enter PLTE path">

NEXT_VARIABLE: <if the value comes from a parameter or another variable,
  what should we trace next in the CALLER function?>
  (write "NONE" if we've reached raw input bytes)
"""


def _phase2_taint_step(router, variable, func_name, file_name, line_num,
                        constraints_so_far, repo) -> dict:
    """One step of backward taint tracing through a single function."""
    source = _find_and_read(repo, file_name, line_num, context=100)

    constraint_text = "\n".join(f"  - {c}" for c in constraints_so_far) if constraints_so_far else "  (none yet)"

    prompt = (
        f"## Variable to Trace\n{variable}\n\n"
        f"## Function: {func_name}() in {file_name}\n"
        f"```c\n{source}\n```\n\n"
        f"## Constraints Accumulated So Far\n{constraint_text}\n\n"
        f"Trace '{variable}' backwards in {func_name}(). Where does it "
        f"get its value? What conditions must be true?\n"
    )

    response = router.chat(PROMPT_TAINT_TRACE, prompt, max_tokens=1024, temperature=0.05)

    result = {
        "variable_origin": "",
        "new_constraints": [],
        "next_variable": "",
        "raw": response,
    }

    m = re.search(r"VARIABLE_ORIGIN:\s*(.+?)(?:\n\n|\nNEW_)", response, re.DOTALL)
    if m:
        result["variable_origin"] = m.group(1).strip()

    # Extract constraints
    constraints_section = re.search(r"NEW_CONSTRAINTS:\s*\n((?:- .+\n?)+)", response)
    if constraints_section:
        for line in constraints_section.group(1).split("\n"):
            line = line.strip().lstrip("- ").strip()
            if line and len(line) > 5:
                result["new_constraints"].append(line)

    m = re.search(r"NEXT_VARIABLE:\s*(.+)", response)
    if m:
        nv = m.group(1).strip()
        result["next_variable"] = "" if "NONE" in nv.upper() else nv

    return result


def _phase2_full_trace(router, crash_analysis, call_chain, repo,
                        max_depth: int = 5) -> list[str]:
    """
    Walk backwards through the call chain, tracing the bad variable
    at each level and accumulating constraints.
    """
    constraints: list[str] = []
    current_var = crash_analysis.get("bad_variable", "")
    if not current_var:
        current_var = crash_analysis.get("source_of_value", "unknown")

    print(f"    Phase 2: Tracing '{current_var}' backwards through call chain...")

    for i, frame in enumerate(call_chain[:max_depth]):
        print(f"      Step {i+1}: {frame['func']}() — tracing '{current_var}'")

        step = _phase2_taint_step(
            router, current_var, frame["func"], frame["file"],
            frame["line"], constraints, repo,
        )

        # Accumulate constraints
        for c in step["new_constraints"]:
            if c not in constraints:
                constraints.append(c)
                print(f"        + constraint: {c}")

        # Move to next variable/function
        if step["next_variable"]:
            current_var = step["next_variable"]
        else:
            print(f"      Reached input bytes — trace complete")
            break

    print(f"    Total constraints: {len(constraints)}")
    return constraints


# ── Phase 3: Constraint-based input construction ──────────────────────────

PROMPT_CONSTRUCT = """\
You are constructing a crafted input that satisfies a set of constraints
derived from backward analysis of a vulnerability.

You MUST satisfy ALL constraints. The input must navigate through all
the parser's checks to reach the crash site.

Write a Python script that constructs these exact bytes.

RULES:
1. Output ONLY a Python script inside ```python ... ```
2. sys.stdout.buffer.write(...) for output
3. ONLY Python standard library (struct, sys, os)
4. Address EVERY constraint — comment which constraint each byte range satisfies
5. If a constraint says "must be valid X format header", generate the
   correct magic bytes and structure
6. If constraints conflict, prioritize the ones closer to the crash
   (later in the list = closer to crash = higher priority)
"""


def _phase3_construct(router, constraints, description, crash_info,
                       binary_name="") -> Optional[str]:
    """Generate a Python script that satisfies all accumulated constraints."""
    constraint_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(constraints))

    prompt = (
        f"## Vulnerability\n{description[:500]}\n\n"
        f"## Crash Site\n"
        f"Function: {crash_info.get('crash_func', 'unknown')}()\n"
        f"Type: {crash_info.get('crash_type', 'unknown')}\n"
        f"Bad variable: {crash_info.get('bad_variable', 'unknown')}\n"
        f"Why: {crash_info.get('why_bad', 'unknown')}\n\n"
        f"## CONSTRAINTS (satisfy ALL of these)\n{constraint_text}\n\n"
    )
    if binary_name:
        prompt += f"## Target binary: {binary_name} (receives input as file)\n\n"

    prompt += (
        f"Write a Python script that constructs input bytes satisfying "
        f"every constraint above. Comment which constraint each byte "
        f"range addresses.\n"
    )

    print(f"    Phase 3: Constructing input from {len(constraints)} constraints...")
    response = router.chat(PROMPT_CONSTRUCT, prompt, max_tokens=cfg.MAX_TOKENS, temperature=0.15)
    return _extract_python(response) if response else None


# ── Phase 4: Verification & adjustment ────────────────────────────────────

PROMPT_ADJUST = """\
Your crafted input was tested against the target binary but didn't
trigger the expected vulnerability. Analyze what went wrong.

You will see:
- The constraints you were trying to satisfy
- The Python script that generated the input
- The binary's output (error messages, parser rejections, etc.)

Figure out WHICH CONSTRAINT was wrong or which check you missed.
Then write a FIXED script.

Focus on the binary's error message — it tells you exactly what
validation check the input failed.

RULES:
1. ONLY a Python script inside ```python ... ```
2. sys.stdout.buffer.write(...)
3. ONLY Python standard library
4. Comment: "# FIX: ..." for every change you made
"""


def _phase4_adjust(router, constraints, script, binary_output,
                    return_code, description) -> Optional[str]:
    """Adjust the PoC based on the binary's error output."""
    constraint_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(constraints))

    prompt = (
        f"## Constraints\n{constraint_text}\n\n"
        f"## Script That Generated the Input\n```python\n{script}\n```\n\n"
        f"## Binary Output (exit code {return_code})\n```\n{binary_output[:2500]}\n```\n\n"
        f"## Vulnerability Description\n{description[:300]}\n\n"
        f"Which constraint was wrong? What check did we miss? Fix the script.\n"
    )

    print(f"    Phase 4: Adjusting based on binary feedback...")
    response = router.chat(PROMPT_ADJUST, prompt, max_tokens=cfg.MAX_TOKENS, temperature=0.2)
    return _extract_python(response) if response else None


# ── Main strategy entry point ─────────────────────────────────────────────

def strategy_constraint_solver(
    task_id: str,
    description: str,
    repo_path: Path,
    router,
    binary_name: str = "",
    binary: Path = None,
    run_func = None,
    max_refine: int = 3,
) -> List[PoCCandidate]:
    """
    The smart strategy: backward constraint solving.

    Phase 1 → Analyze crash site
    Phase 2 → Trace variable backwards, accumulate constraints
    Phase 3 → Generate input satisfying all constraints
    Phase 4 → Test, adjust, repeat
    """
    results: list[PoCCandidate] = []
    repo = Path(repo_path)

    # Parse ASAN stack from description
    call_chain = _parse_stack(description)
    crash_type, operation, _ = _parse_crash_type(description)

    # Fallback: extract function/file names from plain text description
    if not call_chain:
        print("    No ASAN stack trace — extracting from description text...")
        call_chain = _extract_crash_from_text(description, repo)
        if not crash_type:
            # Infer crash type from description keywords
            desc_lower = description.lower()
            if "heap-buffer-overflow" in desc_lower or "heap buffer overflow" in desc_lower:
                crash_type = "heap-buffer-overflow"
            elif "stack-buffer-overflow" in desc_lower or "stack buffer" in desc_lower:
                crash_type = "stack-buffer-overflow"
            elif "use-after-free" in desc_lower or "use after free" in desc_lower:
                crash_type = "heap-use-after-free"
            elif "null" in desc_lower and ("dereference" in desc_lower or "pointer" in desc_lower):
                crash_type = "null-dereference"
            elif "buffer overflow" in desc_lower or "buffer-overflow" in desc_lower:
                crash_type = "buffer-overflow"
            elif "integer overflow" in desc_lower:
                crash_type = "integer-overflow"
            else:
                crash_type = "memory-corruption"

    if not call_chain:
        print("    No crash info found in description — cannot run constraint solver")
        return results

    crash_func = call_chain[0]["func"]
    crash_file = call_chain[0]["file"]
    crash_line = call_chain[0]["line"]

    print(f"    Target: {crash_type} in {crash_func}() at {crash_file}:{crash_line}")
    print(f"    Call chain depth: {len(call_chain)} frames")

    # Phase 1: Analyze the crash
    crash_analysis = _phase1_crash_analysis(
        router, crash_func, crash_file, crash_line,
        crash_type, operation, repo,
    )

    # Phase 2: Backward taint trace
    constraints = _phase2_full_trace(router, crash_analysis, call_chain, repo)

    if not constraints:
        print("    No constraints extracted — falling back to crash analysis only")
        constraints = [
            f"Input must reach {crash_func}()",
            f"Must trigger {crash_type}",
            crash_analysis.get("why_bad", ""),
        ]

    # Phase 3: Construct input
    crash_info = {
        "crash_func": crash_func,
        "crash_type": crash_type,
        "bad_variable": crash_analysis.get("bad_variable", ""),
        "why_bad": crash_analysis.get("why_bad", ""),
    }

    script = _phase3_construct(router, constraints, description, crash_info, binary_name)
    if not script:
        print("    Phase 3 failed — no script generated")
        return results

    data = _run_script(script)
    if not data or len(data) == 0:
        print("    Phase 3 script produced no output")
        return results

    path = _save_poc(data, task_id, "constraint_solver")
    poc = PoCCandidate(
        "constraint_solver", data, path, 0.9,
        f"Constraint-solved: {len(data)}B, {len(constraints)} constraints",
        script,
    )
    results.append(poc)
    print(f"    Phase 3: Generated {len(data)} bytes from {len(constraints)} constraints")

    # Phase 4: Test and refine
    if binary and run_func:
        current_script = script
        current_poc = poc

        for attempt in range(1, max_refine + 1):
            result = run_func(current_poc, binary)
            if result.triggered:
                current_poc.confidence = 0.98
                current_poc.name = f"constraint_solved_v{attempt}"
                print(f"    Phase 4: TRIGGERED on attempt {attempt}!")
                return results

            print(f"    Phase 4: Attempt {attempt} — not triggered (rc={result.return_code})")

            adjusted_script = _phase4_adjust(
                router, constraints, current_script,
                result.sanitizer_output, result.return_code, description,
            )
            if not adjusted_script:
                break

            adjusted_data = _run_script(adjusted_script)
            if not adjusted_data or len(adjusted_data) == 0:
                break

            path = _save_poc(adjusted_data, task_id, f"constraint_v{attempt+1}")
            current_poc = PoCCandidate(
                f"constraint_v{attempt+1}", adjusted_data, path,
                0.9 + attempt * 0.02,
                f"Constraint-adjusted v{attempt+1}: {len(adjusted_data)}B",
                adjusted_script,
            )
            current_script = adjusted_script
            results.append(current_poc)

    return results
