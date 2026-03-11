"""
crs/llm_router.py — Resilient, retry-aware LLM gateway for the Cyber Reasoning System.

This is the ONLY module in the CRS that talks to an LLM.  Every other module
imports LLMRouter (or the module-level SYSTEM_PROMPT_* constants) from here.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Optional

import openai

from crs import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level system prompts (imported by poc_strategies.py and others)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_POC_GENERATOR = """\
You are an expert C/C++ security researcher. Your task is to write a Proof-of-Concept (PoC) \
program that triggers a specific vulnerability in a target codebase.

Rules:
1. Output ONLY a single C or C++ source file. No prose before or after.
2. Wrap your code in a fenced code block: ```c ... ``` or ```cpp ... ```
3. The PoC must be a standalone program with a main() function.
4. It must call into the vulnerable library or binary in a way that triggers the described bug.
5. Use compile-time includes for the target's own headers if available (use relative paths from repo root).
6. Keep it minimal — 30-150 lines. Shorter is better if it still triggers the bug.
7. Do not use external fuzzing libraries. Plain C stdlib is fine.
8. If writing a file-based input, write the input bytes to a temp file then pass it as argv[1].
"""

SYSTEM_PROMPT_POC_REFINER = """\
You are an expert C/C++ security researcher reviewing a PoC that did not work.
Given the original PoC code, the build/run error, and the vulnerability description,
produce a corrected PoC. Apply exactly one focused fix per iteration.
Output ONLY the corrected C/C++ source file in a fenced code block.
"""

SYSTEM_PROMPT_ANALYST = """\
You are a senior vulnerability analyst. Given a vulnerability description and relevant \
code snippets, identify: (1) the exact function(s) most likely to contain the bug, \
(2) what input or call sequence triggers it, (3) what memory safety violation results.
Be concise and precise. Use JSON output format.
"""

# ---------------------------------------------------------------------------
# Retry / back-off configuration
# ---------------------------------------------------------------------------
MAX_RETRIES: int = 4
BACKOFF_CAP_SECONDS: float = 30.0

# Exceptions that warrant a retry (transient network / rate-limit issues)
_RETRYABLE_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)


class LLMRouter:
    """Wraps one or more OpenAI-compatible endpoints with retry, fallback,
    token tracking, and structured-output helpers."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        primary_model: str = config.LLM_MODEL,
        fallback_model: str | None = None,
        primary_base_url: str = config.LLM_BASE_URL,
        primary_api_key: str = config.LLM_API_KEY,
    ):
        self.primary_model = primary_model
        self.fallback_model = fallback_model

        # Primary client
        self._primary_client = openai.OpenAI(
            base_url=primary_base_url,
            api_key=primary_api_key,
            timeout=config.TIMEOUT if hasattr(config, "TIMEOUT") else 120.0,
        )

        # Fallback client (same endpoint unless a separate one is configured)
        self._fallback_client: openai.OpenAI | None = None
        if fallback_model:
            fallback_base_url = getattr(config, "FALLBACK_BASE_URL", primary_base_url)
            fallback_api_key = getattr(config, "FALLBACK_API_KEY", primary_api_key)
            self._fallback_client = openai.OpenAI(
                base_url=fallback_base_url,
                api_key=fallback_api_key,
                timeout=config.TIMEOUT if hasattr(config, "TIMEOUT") else 120.0,
            )

        # Thread-safe cumulative token counters
        self._lock = threading.Lock()
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._total_calls: int = 0
        self._failed_calls: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = config.MAX_TOKENS,
        temperature: float = 0.2,
        attempt_json: bool = False,
    ) -> str:
        """Send a chat-completion request with automatic retry and fallback.

        Parameters
        ----------
        system_prompt : str
            The system-level instruction.
        user_prompt : str
            The user-level content / question.
        max_tokens : int
            Maximum tokens to generate.
        temperature : float
            Sampling temperature.
        attempt_json : bool
            If *True*, append a JSON-only instruction to the system prompt
            so the model is more likely to return parseable JSON.

        Returns
        -------
        str  — raw assistant message content.

        Raises
        ------
        RuntimeError  — if both primary and fallback models are exhausted.
        """
        if attempt_json:
            system_prompt = (
                system_prompt.rstrip()
                + "\n\nIMPORTANT: Respond with valid JSON only. "
                "No markdown fences, no commentary outside the JSON object."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # --- Try primary model ---
        last_exc = self._try_model(
            client=self._primary_client,
            model=self.primary_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if isinstance(last_exc, str):
            return last_exc  # success — returned content string

        # --- Try fallback model (if configured) ---
        if self._fallback_client and self.fallback_model:
            logger.warning(
                "Primary model %s exhausted retries. Falling back to %s.",
                self.primary_model,
                self.fallback_model,
            )
            fallback_result = self._try_model(
                client=self._fallback_client,
                model=self.fallback_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if isinstance(fallback_result, str):
                return fallback_result

            last_exc = fallback_result  # keep the latest exception

        # All attempts exhausted
        with self._lock:
            self._failed_calls += 1
        raise RuntimeError(
            f"LLMRouter: all models exhausted after retries. Last error: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------
    def extract_code_block(self, response: str, language: str = "c") -> str | None:
        """Return the first fenced code block matching *language* (c, cpp, etc.).

        Falls back to content between the outermost ``{`` … ``}`` pair.
        Returns *None* if nothing code-like is found.
        """
        # Try explicit fenced blocks first (```c, ```cpp, ```C, etc.)
        patterns = [
            rf"```{language}\s*\n(.*?)```",
            r"```(?:c|cpp|c\+\+|C|CPP)\s*\n(.*?)```",
            r"```\s*\n(.*?)```",  # bare fence
        ]
        for pat in patterns:
            m = re.search(pat, response, re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()

        # Fallback: outermost { … }
        first_brace = response.find("{")
        last_brace = response.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            # Walk backwards to capture any preceding function signature / includes
            pre = response[:first_brace].rsplit("\n\n", 1)[-1]
            return (pre + response[first_brace : last_brace + 1]).strip()

        return None

    def extract_json(self, response: str) -> dict | None:
        """Best-effort JSON extraction from a potentially messy LLM reply."""
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?\s*", "", response)
        cleaned = cleaned.replace("```", "")

        # Try parsing the whole cleaned string first
        try:
            return json.loads(cleaned.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # Locate outermost { … }
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first != -1 and last > first:
            try:
                return json.loads(cleaned[first : last + 1])
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    def log_stats(self) -> None:
        """Print cumulative token usage and call counts."""
        with self._lock:
            prompt = self._prompt_tokens
            completion = self._completion_tokens
            total = self._total_calls
            failed = self._failed_calls
        logger.info(
            "LLMRouter stats — calls: %d | failed: %d | "
            "prompt_tokens: %d | completion_tokens: %d | total_tokens: %d",
            total,
            failed,
            prompt,
            completion,
            prompt + completion,
        )
        print(
            f"[LLMRouter] calls={total}  failed={failed}  "
            f"prompt_tok={prompt}  completion_tok={completion}  "
            f"total_tok={prompt + completion}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _try_model(
        self,
        client: openai.OpenAI,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str | Exception:
        """Attempt up to MAX_RETRIES calls.  Returns the content string on
        success, or the last exception on failure."""
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content or ""

                # Track tokens
                usage = response.usage
                with self._lock:
                    self._total_calls += 1
                    if usage:
                        self._prompt_tokens += usage.prompt_tokens or 0
                        self._completion_tokens += usage.completion_tokens or 0

                logger.debug(
                    "LLM call success (model=%s, attempt=%d, tokens=%s)",
                    model,
                    attempt,
                    usage.total_tokens if usage else "?",
                )
                return content

            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                wait = min(2**attempt, BACKOFF_CAP_SECONDS)
                logger.warning(
                    "Retryable error on %s (attempt %d/%d): %s — backing off %.1fs",
                    model,
                    attempt + 1,
                    MAX_RETRIES,
                    type(exc).__name__,
                    wait,
                )
                time.sleep(wait)

            except openai.APIError as exc:
                # Non-retryable API errors (auth, bad request, etc.)
                last_exc = exc
                logger.error("Non-retryable API error on %s: %s", model, exc)
                break

        return last_exc  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Quick connectivity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    print("=== LLMRouter connectivity check ===")
    router = LLMRouter()

    try:
        reply = router.chat(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say hello in exactly five words.",
            max_tokens=32,
            temperature=0.0,
        )
        print(f"Model replied: {reply}")
    except RuntimeError as e:
        print(f"Connection check FAILED: {e}")
    finally:
        router.log_stats()
