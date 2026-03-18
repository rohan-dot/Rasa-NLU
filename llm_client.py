"""
BioASQ 14b - LLM Client
========================
Raw OpenAI-compatible client for vLLM-served Gemma.

IMPORTANT: Do NOT use LangChain's ChatOpenAI wrapper for generation-heavy
tasks — it silently injects `tool_choice` parameters that cause Gemma on
vLLM to return empty content. Always use the raw openai.OpenAI client.
"""

import logging
import time
from typing import Optional

from openai import OpenAI

from config import (
    VLLM_BASE_URL,
    VLLM_MODEL,
    VLLM_API_KEY,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    MAX_RETRIES,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around OpenAI client for vLLM/Gemma inference."""

    def __init__(
        self,
        base_url: str = VLLM_BASE_URL,
        model: str = VLLM_MODEL,
        api_key: str = VLLM_API_KEY,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        logger.info(f"LLM client initialized: {base_url} / {model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Generate a completion from the LLM.

        Args:
            prompt: User prompt / question.
            system_prompt: System-level instructions.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            stop: Optional stop sequences.

        Returns:
            Generated text string.

        Raises:
            RuntimeError: After MAX_RETRIES failures.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "user", "content": system_prompt})
            messages.append({"role": "assistant", "content": "Understood. I'll follow these instructions."})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    timeout=300,  # Gemma can be slow on large prompts
                )

                content = response.choices[0].message.content
                if not content or content.strip() == "":
                    logger.warning(f"Empty response on attempt {attempt}")
                    last_error = RuntimeError("Empty LLM response")
                    time.sleep(2)
                    continue

                return content.strip()

            except Exception as e:
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt}/{MAX_RETRIES}): {e}")
                time.sleep(2 * attempt)

        raise RuntimeError(f"LLM generation failed after {MAX_RETRIES} attempts: {last_error}")

    def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = MAX_TOKENS,
    ) -> str:
        """Generate and attempt to extract structured output.

        Same as generate() but with lower temperature for deterministic output.
        """
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.1,
        )
