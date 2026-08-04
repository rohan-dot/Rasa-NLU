"""Async LLM client for the LiteLLM proxy (OpenAI-compatible).

Design points, matching what has worked in the FCG codebase:
- temperature 0, hard max_tokens cap
- async with a shared semaphore
- retries with linear backoff
- optional response_format json mode; if the proxy/model rejects it once,
  we remember and stop sending it (prompt-enforced JSON + lenient parser
  carry the load instead)
"""

import asyncio
import logging
from typing import Any, Optional

from . import config
from .json_utils import parse_json_lenient

log = logging.getLogger("fcg.llm")

_client = None  # lazy: openai only needed when an LLM call actually happens
_sem = asyncio.Semaphore(config.CONCURRENCY)
_json_mode_ok: Optional[bool] = None  # None = untested


def _get_client():
    global _client
    if _client is None:
        import httpx
        from openai import AsyncOpenAI  # deferred import
        http_client = (None if config.TLS_VERIFY
                       else httpx.AsyncClient(verify=False))
        _client = AsyncOpenAI(base_url=config.LLM_BASE_URL,
                              api_key=config.LLM_API_KEY,
                              http_client=http_client)
    return _client


async def chat(system: str, user: str, *, json_mode: bool = True,
               max_tokens: int = None) -> str:
    """Single chat completion. Returns raw text ('' on total failure)."""
    global _json_mode_ok
    max_tokens = max_tokens or config.MAX_TOKENS

    async with _sem:
        for attempt in range(1, config.RETRIES + 1):
            kwargs: dict[str, Any] = dict(
                model=config.LLM_MODEL,
                temperature=config.TEMPERATURE,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            use_json = (json_mode and config.TRY_JSON_MODE
                        and _json_mode_ok is not False)
            if use_json:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = await _get_client().chat.completions.create(**kwargs)
                if use_json and _json_mode_ok is None:
                    _json_mode_ok = True
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                if use_json and ("response_format" in msg or "json_object" in msg
                                 or "not supported" in msg):
                    log.warning("json mode rejected by endpoint; disabling for run")
                    _json_mode_ok = False
                    continue  # retry immediately without json mode
                log.warning("LLM call failed (attempt %d/%d): %s",
                            attempt, config.RETRIES, e)
                if attempt < config.RETRIES:
                    await asyncio.sleep(config.RETRY_BACKOFF_S * attempt)
    return ""


async def chat_json(system: str, user: str, *, max_tokens: int = None) -> Optional[Any]:
    """Chat completion parsed through the three-tier JSON parser."""
    text = await chat(system, user, json_mode=True, max_tokens=max_tokens)
    parsed = parse_json_lenient(text)
    if parsed is None and text:
        log.warning("unparseable JSON (first 200 chars): %r", text[:200])
    return parsed
