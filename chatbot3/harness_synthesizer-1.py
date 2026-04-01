"""
crs/harness_synthesizer.py — Synthesize a fuzz harness when none exists.

Most CyberGym tarballs only contain the project source. The actual
LLVMFuzzerTestOneInput harness lives in google/oss-fuzz, which isn't
included. This module has the LLM write a minimal harness that:

  1. Includes the project's headers
  2. Parses the raw input bytes using the project's API
  3. Calls the vulnerable function (or the function that leads to it)

The synthesized harness is then compiled with the project source
and used as the fuzz target.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from crs.code_intelligence import CodeContext
from crs.config import cfg
from crs.data_loader import CyberGymTask
from crs.harness_finder import HarnessInfo
from crs.llm_router import LLMRouter


# ── Prompt ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_HARNESS_WRITER = """\
You are an expert at writing fuzz harnesses for C/C++ projects.

Your task is to write a LLVMFuzzerTestOneInput function that:
1. Takes raw bytes (const uint8_t *data, size_t size)
2. Passes them to the project's parsing/processing functions
3. Reaches the vulnerable code path described in the vulnerability description

CRITICAL RULES:
1. Include the project's ACTUAL headers — this harness will be compiled
   against the real project source code, so project headers ARE available.
2. The function signature MUST be:
   extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
   (use extern "C" even for .c files — it's harmless and ensures linkage)
3. Call the project's real parsing/API functions — do NOT reproduce logic inline.
4. Handle NULL returns and error codes gracefully (return 0, don't crash in the harness itself).
5. Free any allocated resources before returning.
6. Keep it SHORT — under 50 lines. The simpler the better.
7. Output ONLY the C/C++ code inside a single ```c ... ``` block.

=== EXAMPLE for a JSON parser library ===
```c
#include <stdint.h>
#include <stdlib.h>
#include "json_parser.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    // Create a null-terminated copy for the parser
    char *input = (char *)malloc(size + 1);
    if (!input) return 0;
    memcpy(input, data, size);
    input[size] = '\\0';

    // Parse using the project's API
    json_value *val = json_parse(input, size);
    if (val) {
        json_free(val);
    }

    free(input);
    return 0;
}
```

=== EXAMPLE for an MQTT broker (like Mosquitto) ===
```c
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "mosquitto_internal.h"
#include "packet_mosq.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    struct mosquitto *mosq = mosquitto__calloc(1, sizeof(struct mosquitto));
    if (!mosq) return 0;

    // Set up a fake packet from the fuzz input
    mosq->in_packet.payload = mosquitto__malloc(size);
    if (!mosq->in_packet.payload) { mosquitto__free(mosq); return 0; }
    memcpy(mosq->in_packet.payload, data, size);
    mosq->in_packet.remaining_length = size;
    mosq->in_packet.pos = 0;

    // Process the packet — this reaches the vulnerable code
    handle__publish(mosq);

    mosquitto__free(mosq->in_packet.payload);
    mosquitto__free(mosq);
    return 0;
}
```
"""


def synthesize_harness(
    task: CyberGymTask,
    context: CodeContext,
    router: LLMRouter,
) -> Optional[HarnessInfo]:
    """
    Ask the LLM to write a fuzz harness for the project.

    The LLM gets:
      - Vulnerability description
      - Top source file snippets (including headers and API)
      - List of available header files
      - Vulnerability type

    Returns a HarnessInfo with the synthesized harness saved to disk,
    or None if synthesis fails.
    """
    repo = Path(task.repo_path).resolve()

    # Gather available headers
    headers: list[str] = []
    for h in sorted(repo.rglob("*.h")):
        rel = h.relative_to(repo)
        headers.append(str(rel))
    headers = headers[:50]  # cap

    # Get top snippets
    snippets = context.top_snippets[:6000] if isinstance(context.top_snippets, str) else ""

    # Find key API functions (exported functions from headers)
    api_functions = _find_api_functions(repo)

    user_prompt = (
        f"## Vulnerability Description\n{context.description}\n\n"
        f"## Vulnerability Type\n{context.vuln_type}\n\n"
        f"## Available Project Headers\n{chr(10).join(headers[:30])}\n\n"
        f"## Key API Functions Found in Headers\n{chr(10).join(api_functions[:30])}\n\n"
        f"## Relevant Source Code\n```c\n{snippets}\n```\n\n"
        f"Write a LLVMFuzzerTestOneInput harness that:\n"
        f"1. Includes the necessary project headers from the list above\n"
        f"2. Takes raw bytes and passes them to the project's parsing functions\n"
        f"3. Reaches the code path where the '{context.vuln_type}' vulnerability exists\n"
        f"4. Is as simple as possible — just enough to get the bytes to the vulnerable function\n"
    )

    print(f"[synth] Asking LLM to write a fuzz harness...")

    raw = router.chat(
        system_prompt=SYSTEM_PROMPT_HARNESS_WRITER,
        user_prompt=user_prompt,
        max_tokens=cfg.MAX_TOKENS,
        temperature=0.2,
    )

    if not raw:
        print(f"[synth] LLM returned empty")
        return None

    # Extract code block
    code = _extract_code_block(raw)
    if not code:
        print(f"[synth] No code block found in LLM response")
        return None

    # Validate it has the right entry point
    if "LLVMFuzzerTestOneInput" not in code:
        print(f"[synth] Generated code doesn't contain LLVMFuzzerTestOneInput")
        return None

    # Save to disk
    work = cfg.task_work_dir(task.task_id)
    harness_path = work / "synthesized_harness.c"
    harness_path.write_text(code, encoding="utf-8")
    print(f"[synth] Saved synthesized harness to {harness_path}")
    print(f"[synth] Harness code ({len(code)} chars):")
    print(code[:2000])

    # Extract metadata
    includes = re.findall(r'#include\s*[<"]([^>"]+)[>"]', code)
    called = _extract_called_functions(code)

    std_headers = {
        "stdio.h", "stdlib.h", "string.h", "stdint.h", "stdbool.h",
        "stddef.h", "unistd.h", "math.h",
    }
    project_headers = [h for h in includes if h not in std_headers]

    return HarnessInfo(
        harness_path=harness_path,
        harness_code=code,
        harness_function_code=code,  # whole file is the harness
        called_functions=called,
        includes=includes,
        has_init=False,
        project_headers=project_headers,
        all_harness_files=[harness_path],
    )


def refine_harness(
    harness: HarnessInfo,
    build_errors: str,
    context: CodeContext,
    router: LLMRouter,
    task: CyberGymTask,
    iteration: int = 1,
) -> Optional[HarnessInfo]:
    """
    If the synthesized harness fails to compile, feed the errors back
    to the LLM and ask it to fix the harness.
    """
    print(f"[synth] Refining harness (iteration {iteration})...")

    user_prompt = (
        f"## Vulnerability Description\n{context.description}\n\n"
        f"## Current Harness Code\n```c\n{harness.harness_code}\n```\n\n"
        f"## Compilation Errors\n```\n{build_errors[:3000]}\n```\n\n"
        f"Fix the harness so it compiles. Common issues:\n"
        f"- Wrong header names — check the available headers listed below\n"
        f"- Wrong function signatures — check the source snippets\n"
        f"- Wrong struct field names — copy exact names from the source\n"
        f"- Missing includes\n\n"
    )

    # Re-provide available headers
    repo = Path(task.repo_path).resolve()
    headers = []
    for h in sorted(repo.rglob("*.h")):
        headers.append(str(h.relative_to(repo)))
    user_prompt += f"## Available Headers\n{chr(10).join(headers[:30])}\n\n"

    snippets = context.top_snippets[:4000] if isinstance(context.top_snippets, str) else ""
    user_prompt += f"## Source Snippets\n```c\n{snippets}\n```\n"

    raw = router.chat(
        system_prompt=SYSTEM_PROMPT_HARNESS_WRITER,
        user_prompt=user_prompt,
        max_tokens=cfg.MAX_TOKENS,
        temperature=0.1,
    )

    if not raw:
        return None

    code = _extract_code_block(raw)
    if not code or "LLVMFuzzerTestOneInput" not in code:
        return None

    work = cfg.task_work_dir(task.task_id)
    harness_path = work / f"synthesized_harness_v{iteration + 1}.c"
    harness_path.write_text(code, encoding="utf-8")
    print(f"[synth] Saved refined harness to {harness_path}")

    includes = re.findall(r'#include\s*[<"]([^>"]+)[>"]', code)
    called = _extract_called_functions(code)
    std_headers = {
        "stdio.h", "stdlib.h", "string.h", "stdint.h", "stdbool.h",
        "stddef.h", "unistd.h", "math.h",
    }
    project_headers = [h for h in includes if h not in std_headers]

    return HarnessInfo(
        harness_path=harness_path,
        harness_code=code,
        harness_function_code=code,
        called_functions=called,
        includes=includes,
        has_init=False,
        project_headers=project_headers,
        all_harness_files=[harness_path],
    )


# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_code_block(text: str) -> Optional[str]:
    m = re.search(r"```(?:c\+\+|cpp|c)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _extract_called_functions(code: str) -> list[str]:
    calls = re.findall(r"\b([a-zA-Z_]\w+)\s*\(", code)
    skip = {
        "if", "for", "while", "switch", "return", "sizeof", "typeof",
        "malloc", "calloc", "realloc", "free", "memcpy", "memset",
        "memmove", "printf", "fprintf", "sprintf", "snprintf",
        "fopen", "fclose", "fread", "fwrite",
        "LLVMFuzzerTestOneInput", "LLVMFuzzerInitialize",
        "int", "void", "char", "unsigned", "extern",
    }
    return list(dict.fromkeys(c for c in calls if c not in skip))


def _find_api_functions(repo: Path) -> list[str]:
    """
    Scan header files for function declarations — these are the
    project's public API that the harness should call.
    """
    funcs: list[str] = []
    for h in sorted(repo.rglob("*.h")):
        try:
            content = h.read_text(encoding="utf-8", errors="replace")
            # Match function declarations (not definitions — no opening brace on same line)
            matches = re.findall(
                r"(?:extern\s+)?(?:[\w*]+\s+)+(\w+)\s*\([^)]*\)\s*;",
                content,
            )
            for m in matches:
                if m not in funcs and not m.startswith("_"):
                    funcs.append(m)
        except Exception:
            pass
    return funcs[:100]
