"""
BioASQ 14b - LLM Client
========================
Raw OpenAI-compatible client for vLLM-served Gemma.
Do NOT use LangChain's ChatOpenAI — it injects tool_choice causing empty responses.
"""

import logging
import time
from typing import Optional

from openai import OpenAI

from config import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY, MAX_TOKENS, TEMPERATURE, TOP_P, MAX_RETRIES

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, base_url=VLLM_BASE_URL, model=VLLM_MODEL, api_key=VLLM_API_KEY):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        logger.info(f"LLM client: {base_url} / {model}")

    def generate(self, prompt, system_prompt=None, max_tokens=MAX_TOKENS,
                 temperature=TEMPERATURE, top_p=TOP_P, stop=None):
        messages = []
        if system_prompt:
            messages.append({"role": "user", "content": system_prompt})
            messages.append({"role": "assistant", "content": "Understood. I'll follow these instructions."})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature,
                    top_p=top_p, stop=stop, timeout=300,
                )
                content = response.choices[0].message.content
                if not content or content.strip() == "":
                    last_error = RuntimeError("Empty LLM response")
                    time.sleep(2)
                    continue
                return content.strip()
            except Exception as e:
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt}): {e}")
                time.sleep(2 * attempt)

        raise RuntimeError(f"LLM failed after {MAX_RETRIES} attempts: {last_error}")
