"""
crs/llm_router.py — LLM interface for the Cyber Reasoning System.

Uses the raw openai client directly (NOT LangChain) to avoid tool_choice
injection issues with vLLM-served models.

Changes from original:
  - SYSTEM_PROMPT_POC_GENERATOR: added explicit WRONG/RIGHT examples
  - SYSTEM_PROMPT_POC_REFINER: added anti-hallucination rules for undefined refs
  - SYSTEM_PROMPT_ANALYST: added reminder not to suggest project-specific APIs
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

import openai

from crs.config import cfg as _cfg

logger = logging.getLogger(__name__)

# ── System prompts ─────────────────────────────────────────────────────────

SYSTEM_PROMPT_POC_GENERATOR = """\
You are an expert C/C++ vulnerability researcher and exploit developer.

Your task is to write a complete, compilable C/C++ Proof-of-Concept (PoC) program that triggers a specific vulnerability.

CRITICAL RULES — NEVER VIOLATE:
1. ONLY use function names, struct names, and type names that appear VERBATIM in the code snippets provided to you.
2. NEVER invent, guess, or hallucinate function names, struct fields, or header files.
3. NEVER include project-specific headers like "mosquitto.h", "db.h", "jq.h", "config.h", or similar — even if you see them referenced in the snippets.
4. If you need a struct, copy its definition DIRECTLY from the provided snippets into your PoC.
5. The PoC must be ENTIRELY SELF-CONTAINED using only standard C headers:
   #include <stdio.h>, #include <stdlib.h>, #include <string.h>,
   #include <stdint.h>, #include <stdbool.h>, #include <unistd.h>
6. Reproduce the vulnerable logic INLINE in main() using only what you see in the provided code snippets. COPY the vulnerable code into your PoC — do NOT call it via project APIs.
7. Do NOT mock, stub, or simulate vulnerable functions — reproduce their logic directly.
8. Output ONLY the C/C++ code inside a single ```c ... ``` fenced block. No explanation.

=== WRONG (will NOT compile — never do this) ===
    #include "mosquitto.h"           // NEVER include project headers
    #include "db.h"                  // NEVER include project headers
    #include "mosquitto_internal.h"  // NEVER include project headers
    mosquitto_broker_init(...);      // NEVER call project functions
    retain_init();                   // NEVER call project functions
    db__message_store(...);          // NEVER call project functions

=== RIGHT (self-contained — always do this) ===
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    // Copy the vulnerable struct definition from the snippets:
    typedef struct { char *topic; void *payload; int payloadlen; } my_msg;

    // Reproduce the vulnerable logic inline:
    int main(void) {
        my_msg *msg = malloc(sizeof(my_msg));
        // ... trigger the bug using copied logic ...
        free(msg);
        return 0;
    }
"""

SYSTEM_PROMPT_POC_REFINER = """\
You are an expert C/C++ compiler error fixer.

You will be given a PoC program that failed to compile, along with the compiler errors.
Your task is to fix ALL compilation errors and return the corrected, complete program.

CRITICAL RULES:
1. Fix syntax errors, type mismatches, and missing includes.
2. Do NOT remove functionality — only fix compilation issues.
3. If a function does not exist in standard headers, reproduce its logic inline using basic C code.
4. NEVER add project-specific headers (mosquitto.h, db.h, jq.h, config.h, etc.) — these will NOT be available at compile time.
5. If an error says "undefined reference to `some_project_func`":
   - Do NOT add an #include for the project header.
   - Instead, REMOVE the call entirely and replace it with a direct inline reproduction
     of what the function does, using only standard C.
   - If you cannot reproduce it, replace the call with a simpler trigger for the same
     vulnerability class (e.g. a direct memcpy overflow, a direct use-after-free, etc.).
6. Output ONLY the corrected C/C++ code inside a single ```c ... ``` fenced block.

=== WRONG way to fix "undefined reference to `retain_init`" ===
    #include "mosquitto.h"   // NEVER DO THIS
    retain_init();           // still calls the undefined function

=== RIGHT way to fix "undefined reference to `retain_init`" ===
    // Removed retain_init() — not available as standalone.
    // Reproduce the relevant initialization logic inline:
    memset(&retain_tree, 0, sizeof(retain_tree));
"""

SYSTEM_PROMPT_ANALYST = """\
You are an expert C/C++ vulnerability analyst.

Analyze the provided vulnerable code and vulnerability description carefully.
Identify:
1. The exact function(s) where the vulnerability occurs
2. The data flow that leads to the vulnerability
3. What input or state is needed to trigger it
4. The type of memory corruption or undefined behavior

Be precise and technical. Reference actual function names and line numbers from the code.

IMPORTANT: When describing how to trigger the vulnerability, focus on the DATA and LOGIC
needed — not on calling project-specific API functions. The PoC will need to reproduce
the vulnerable logic inline using only standard C, so describe what the code DOES rather
than which project functions to call.
"""


# ── Router ─────────────────────────────────────────────────────────────────

class LLMRouter:
    """
    Thin wrapper around openai.OpenAI that adds retry logic and logging.
    Always uses .chat() — never .query().
    """

    def __init__(
        self,
        primary_model: str = _cfg.LLM_MODEL,
        primary_base_url: str = _cfg.LLM_BASE_URL,
        primary_api_key: str = _cfg.LLM_API_KEY,
    ) -> None:
        self.primary_model = primary_model
        self._calls = 0
        self._failures = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

        self._primary_client = openai.OpenAI(
            base_url=primary_base_url,
            api_key=primary_api_key,
            timeout=300.0,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = _cfg.MAX_TOKENS,
        temperature: float = 0.2,
        attempt_json: bool = False,
    ) -> str:
        """
        Send a chat completion request. Returns the response text.
        Raises RuntimeError if all retries fail.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        last_error: Optional[Exception] = None
        for attempt in range(1, _cfg.MAX_RETRIES + 1):
            try:
                response = self._primary_client.chat.completions.create(
                    model=self.primary_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                self._calls += 1
                usage = response.usage
                if usage:
                    self._prompt_tokens     += usage.prompt_tokens or 0
                    self._completion_tokens += usage.completion_tokens or 0

                content = response.choices[0].message.content or ""
                return content

            except openai.APITimeoutError as e:
                last_error = e
                wait = 2 ** (attempt - 1)
                print(f"Retryable error on {self.primary_model} (attempt {attempt}/{_cfg.MAX_RETRIES}): APITimeoutError — backing off {wait:.1f}s")
                time.sleep(wait)

            except openai.APIConnectionError as e:
                last_error = e
                wait = 2 ** (attempt - 1)
                print(f"Retryable error on {self.primary_model} (attempt {attempt}/{_cfg.MAX_RETRIES}): APIConnectionError — backing off {wait:.1f}s")
                time.sleep(wait)

            except openai.RateLimitError as e:
                last_error = e
                wait = 2 ** attempt
                print(f"Retryable error on {self.primary_model} (attempt {attempt}/{_cfg.MAX_RETRIES}): RateLimitError — backing off {wait:.1f}s")
                time.sleep(wait)

            except openai.APIStatusError as e:
                # 404 model not found, 400 context too large — not retryable
                self._failures += 1
                print(f"Non-retryable API error on {self.primary_model}: Error code: {e.status_code} - {e.body}")
                raise RuntimeError(
                    f"LLMRouter: all models exhausted after retries. Last error: Error code: {e.status_code} - {e.body}"
                )

            except Exception as e:
                last_error = e
                wait = 2 ** (attempt - 1)
                print(f"Retryable error on {self.primary_model} (attempt {attempt}/{_cfg.MAX_RETRIES}): {type(e).__name__} — backing off {wait:.1f}s")
                time.sleep(wait)

        self._failures += 1
        raise RuntimeError(
            f"LLMRouter: all models exhausted after retries. Last error: {last_error}"
        )

    # ── Parsing helpers ─────────────────────────────────────────────────────

    def extract_code_block(self, response: str, language: str = "c") -> Optional[str]:
        """Extract the first fenced code block from a response."""
        # Try language-specific fence first
        pattern = rf"```{language}\s*\n(.*?)```"
        m = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # Fall back to any fenced block
        m = re.search(r"```[a-z+]*\s*\n(.*?)```", response, re.DOTALL)
        if m:
            return m.group(1).strip()
        return None

    def extract_json(self, response: str) -> Optional[dict]:
        """Extract JSON from a response, handling markdown fences."""
        # Strip json fences
        cleaned = re.sub(r"```json\s*", "", response)
        cleaned = re.sub(r"```\s*", "", cleaned)
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find a JSON object anywhere in the response
            m = re.search(r"\{.*\}", response, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        return None

    # ── Stats ───────────────────────────────────────────────────────────────

    def log_stats(self) -> None:
        total = self._prompt_tokens + self._completion_tokens
        print(
            f"[LLMRouter] calls={self._calls} failed={self._failures} "
            f"prompt_tok={self._prompt_tokens} "
            f"completion_tok={self._completion_tokens} "
            f"total_tok={total}"
        )

    @property
    def stats(self) -> dict[str, int]:
        return {
            "calls": self._calls,
            "failed": self._failures,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._prompt_tokens + self._completion_tokens,
        }
