#!/usr/bin/env python3
"""
agent_core.py — a small, dependency-light agent runtime over any OpenAI-compatible
chat endpoint (vLLM, OpenAI, Anthropic's OpenAI-compat layer, LiteLLM, Ollama, ...).

Why not a framework: the raw `openai.OpenAI` client is the only hard dependency, so
this runs unchanged on a restricted HPC node pointed at a local vLLM server.

Two tool-calling transports are supported and chosen automatically:
  * native — OpenAI function calling (`tools=` / `tool_calls`). Needs vLLM started with
             `--enable-auto-tool-choice --tool-call-parser <parser>`.
  * json   — a text protocol: the model answers with one ```json {"tool": ..., "args": ...}```
             block; the runtime parses it and feeds the result back as a user message.
             Works with ANY chat model, no server flags needed.
  `tool_mode="auto"` tries native first and falls back to json if the server rejects tools.

Other things handled here: context trimming (old tool outputs are evicted, the agent's
saved notes are not), retries, a JSONL trace of every step, and a long-output helper
that keeps asking the model to continue when it hits max_tokens.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# --------------------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------------------


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    finish_reason: str | None
    raw: Any = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema for the arguments object
    fn: Callable[..., str]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class AgentResult:
    final_text: str
    notes: list[str]
    turns: int
    tool_calls: int
    usage: dict


class NativeToolsUnsupported(Exception):
    """Raised internally when the server rejects OpenAI-style tool calling."""


class ToolError(Exception):
    """Raise inside a tool to return a clean error string to the model."""


# --------------------------------------------------------------------------------------
# LLM client
# --------------------------------------------------------------------------------------

RETRYABLE = {
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
    "InternalServerError",
    "ServiceUnavailableError",
    "ConflictError",
}


@dataclass
class LLMConfig:
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = 0.2
    max_tokens: int = 2048
    tool_mode: str = "auto"  # auto | native | json
    timeout: float = 300.0
    system_role: bool = True  # False -> fold the system prompt into the first user turn (Gemma-2 style templates)
    extra_body: dict | None = None  # passed straight through, e.g. {"chat_template_kwargs": {...}} for vLLM
    ca_bundle: str | None = None  # CA bundle for TLS (corporate / lab proxies, self-hosted LiteLLM behind HTTPS)
    insecure: bool = False  # disable TLS verification (last resort)

    @classmethod
    def from_env(cls, **overrides: Any) -> "LLMConfig":
        env = os.environ

        def first(*names: str) -> str | None:
            for n in names:
                v = env.get(n)
                if v:
                    return v
            return None

        cfg: dict[str, Any] = {
            "model": first("LLM_MODEL", "AGENT_MODEL", "LITELLM_MODEL", "OPENAI_MODEL") or "",
            "base_url": first("LLM_BASE_URL", "LITELLM_BASE_URL", "LITELLM_PROXY_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE"),
            "api_key": first("LLM_API_KEY", "LITELLM_API_KEY", "LITELLM_KEY", "OPENAI_API_KEY"),
            "tool_mode": env.get("LLM_TOOL_MODE", "auto"),
            # httpx (used by the openai SDK) only reads SSL_CERT_FILE/SSL_CERT_DIR on its own;
            # pick up the other common names too so a lab CA bundle is never silently ignored.
            "ca_bundle": first("LLM_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS"),
            "insecure": env.get("LLM_INSECURE", "").strip().lower() in {"1", "true", "yes"}
            or env.get("LLM_SSL_VERIFY", "").strip().lower() in {"0", "false", "no"},
        }
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        if not cfg.get("model"):
            raise SystemExit(
                "No model given. Pass --model or set LLM_MODEL (e.g. google/gemma-3-27b-it on vLLM, gpt-4.1, claude-sonnet-4-5)."
            )
        if not cfg.get("api_key"):
            if cfg.get("base_url"):
                cfg["api_key"] = "EMPTY"  # vLLM & friends ignore the key but the SDK wants one
            else:
                raise SystemExit("No API key. Set LLM_API_KEY / OPENAI_API_KEY, or pass --base-url for a local server.")
        return cls(**cfg)


class LLMClient:
    """Thin wrapper around `openai.OpenAI().chat.completions.create` with fallbacks."""

    def __init__(self, cfg: LLMConfig, client: Any | None = None):
        self.cfg = cfg
        self.mode = cfg.tool_mode  # "auto" behaves as native until the server says otherwise
        if client is None:
            try:
                from openai import OpenAI  # lazy: only needed for real runs
            except ImportError as e:  # pragma: no cover
                raise SystemExit("The `openai` package is required: pip install openai") from e
            kwargs: dict[str, Any] = {"api_key": cfg.api_key or "EMPTY", "timeout": cfg.timeout, "max_retries": 2}
            if cfg.base_url:
                kwargs["base_url"] = cfg.base_url
            if cfg.insecure or cfg.ca_bundle:
                import httpx  # dependency of the openai SDK

                if cfg.insecure:
                    print("[llm] WARNING: TLS verification disabled", file=sys.stderr)
                    verify: Any = False
                else:
                    if not os.path.exists(cfg.ca_bundle):  # type: ignore[arg-type]
                        raise SystemExit(f"CA bundle not found: {cfg.ca_bundle}")
                    verify = cfg.ca_bundle
                # trust_env=True keeps HTTPS_PROXY / NO_PROXY behaviour
                kwargs["http_client"] = httpx.Client(verify=verify, timeout=cfg.timeout, trust_env=True)
            client = OpenAI(**kwargs)
        self.client = client
        self._dropped_params: set[str] = set()
        self._use_max_completion_tokens = False
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.transport_used: str | None = None  # "native" or "json", set on the first tool-enabled call

    # -- public -------------------------------------------------------------------------
    @property
    def native_tools(self) -> bool:
        return self.mode != "json"

    def chat(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int | None = None) -> LLMResponse:
        use_native = bool(tools) and self.native_tools
        kwargs: dict[str, Any] = {"model": self.cfg.model, "messages": messages}
        if self.cfg.temperature is not None and "temperature" not in self._dropped_params:
            kwargs["temperature"] = self.cfg.temperature
        kwargs["max_completion_tokens" if self._use_max_completion_tokens else "max_tokens"] = max_tokens or self.cfg.max_tokens
        if use_native:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.cfg.extra_body:
            kwargs["extra_body"] = self.cfg.extra_body

        resp = self._create(kwargs)
        self._account(resp)
        if tools:
            self.transport_used = "native" if use_native else "json"
        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""
        if isinstance(text, list):  # some servers return content parts
            text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
        calls: list[ToolCall] = []
        if use_native:
            for i, tc in enumerate(getattr(msg, "tool_calls", None) or []):
                fn = tc.function
                raw_args = fn.arguments
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else (raw_args or {})
                except json.JSONDecodeError:
                    args = {"_raw": raw_args}
                if not isinstance(args, dict):
                    args = {"_raw": args}
                calls.append(ToolCall(id=tc.id or f"call_{i}", name=fn.name, args=args))
        elif tools:  # json transport: look for a tool call inside the text
            parsed = parse_json_tool_call(text)
            if parsed is not None:
                calls.append(parsed)
        return LLMResponse(text=text, tool_calls=calls, finish_reason=getattr(choice, "finish_reason", None), raw=resp)

    # -- internals ----------------------------------------------------------------------
    def _create(self, kwargs: dict) -> Any:
        attempt = 0
        while True:
            attempt += 1
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001 — we classify by name to avoid importing openai here
                name = type(e).__name__
                msg = str(e)
                if name == "BadRequestError":
                    if "max_tokens" in kwargs and "max_completion_tokens" in msg:
                        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                        self._use_max_completion_tokens = True
                        continue
                    if "temperature" in kwargs and "temperature" in msg:
                        kwargs.pop("temperature")
                        self._dropped_params.add("temperature")
                        continue
                    if kwargs.get("tools") and self.mode == "auto" and re.search(r"tool", msg, re.I):
                        self.mode = "json"
                        raise NativeToolsUnsupported(msg) from e
                    raise
                if name in RETRYABLE and attempt <= 3:
                    time.sleep(min(2 ** attempt, 20))
                    continue
                raise

    def _account(self, resp: Any) -> None:
        self.usage["calls"] += 1
        u = getattr(resp, "usage", None)
        if u is not None:
            self.usage["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
            self.usage["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)


# --------------------------------------------------------------------------------------
# JSON tool protocol helpers
# --------------------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _balanced_json_objects(text: str) -> list[str]:
    """Return substrings that are balanced {...} objects starting at a '{"tool"' marker."""
    out = []
    for m in re.finditer(r'\{\s*"tool"', text):
        depth, in_str, esc = 0, False, False
        for j in range(m.start(), len(text)):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        out.append(text[m.start() : j + 1])
                        break
    return out


def parse_json_tool_call(text: str) -> ToolCall | None:
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)] or _balanced_json_objects(text)
    for cand in reversed(candidates):  # the last block is the actionable one
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
            args = obj.get("args", obj.get("arguments", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            if not isinstance(args, dict):
                args = {}
            return ToolCall(id=f"json_{int(time.time() * 1000)}", name=obj["tool"], args=args)
    return None


def _schema_signature(schema: dict) -> str:
    props = (schema or {}).get("properties", {}) or {}
    required = set((schema or {}).get("required", []) or [])
    parts = []
    for k, v in props.items():
        t = v.get("type", "any") if isinstance(v, dict) else "any"
        parts.append(f"{k}{'' if k in required else '?'}: {t}")
    return ", ".join(parts)


def json_protocol_text(tools: list[Tool]) -> str:
    lines = [
        "",
        "## TOOL PROTOCOL (this server has no native function calling)",
        "To call a tool, reply with ONLY one fenced json block, nothing else:",
        "```json",
        '{"tool": "<tool_name>", "args": {"<arg>": <value>}}',
        "```",
        "Exactly one tool call per reply. The result arrives in the next message as `[TOOL RESULT <tool_name>]`.",
        "When you are finished, reply with plain text (no json block).",
        "",
        "Available tools:",
    ]
    for t in tools:
        lines.append(f"- {t.name}({_schema_signature(t.parameters)}): {t.description}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Tool registry
# --------------------------------------------------------------------------------------


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None, max_result_chars: int = 14_000):
        self._tools: dict[str, Tool] = {}
        self.max_result_chars = max_result_chars
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_specs(self, specs: list[dict]) -> None:
        for s in specs:
            self.register(Tool(name=s["name"], description=s["description"], parameters=s["parameters"], fn=s["fn"]))

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def call(self, name: str, args: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool '{name}'. Available: {', '.join(self._tools)}"
        try:
            out = tool.fn(**(args or {}))
        except ToolError as e:
            return f"ERROR: {e}"
        except TypeError as e:
            return f"ERROR: bad arguments for {name}: {e}. Expected: {_schema_signature(tool.parameters)}"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        out = "" if out is None else str(out)
        if len(out) > self.max_result_chars:
            out = out[: self.max_result_chars] + f"\n...[truncated {len(out) - self.max_result_chars} chars — narrow the request]"
        return out if out.strip() else "(empty result)"


# --------------------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------------------

STEP_LIMIT_MSG = (
    "You have reached the exploration step limit. Do NOT call any more tools. "
    "Reply now with a plain-text summary of what you learned and what you could not verify."
)


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        system_prompt: str,
        *,
        name: str = "agent",
        max_turns: int = 40,
        context_char_budget: int = 240_000,
        keep_recent_results: int = 4,
        trace_path: str | Path | None = None,
        verbose: bool = True,
    ):
        self.llm = llm
        self.registry = registry
        self.base_system_prompt = system_prompt
        self.name = name
        self.max_turns = max_turns
        self.context_char_budget = context_char_budget
        self.keep_recent_results = keep_recent_results
        self.trace_path = Path(trace_path) if trace_path else None
        self.verbose = verbose
        self.notes: list[str] = []
        self.messages: list[dict] = []
        self._tool_call_count = 0
        self.registry.register(
            Tool(
                name="save_note",
                description=(
                    "Save a durable finding. Older tool outputs get evicted from context; notes never do. "
                    "Cite file paths in every note."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "short label, e.g. 'LoRA config', 'eval metric'"},
                        "note": {"type": "string", "description": "the finding, with file path citations"},
                    },
                    "required": ["topic", "note"],
                },
                fn=self._save_note,
            )
        )
        self.registry.register(
            Tool(
                name="list_notes",
                description="Show all notes saved so far.",
                parameters={"type": "object", "properties": {}},
                fn=self._list_notes,
            )
        )

    # -- notes --------------------------------------------------------------------------
    def _save_note(self, topic: str, note: str) -> str:
        self.notes.append(f"[{topic}] {note.strip()}")
        return f"saved note #{len(self.notes)}"

    def _list_notes(self) -> str:
        if not self.notes:
            return "(no notes yet)"
        return "\n".join(f"{i + 1}. {n}" for i, n in enumerate(self.notes))

    def notes_markdown(self) -> str:
        return "\n".join(f"- {n}" for n in self.notes) if self.notes else "- (no notes were saved)"

    # -- messages -----------------------------------------------------------------------
    def _system_text(self) -> str:
        if self.llm.native_tools:
            return self.base_system_prompt
        return self.base_system_prompt + "\n" + json_protocol_text(self.registry.tools())

    def _api_messages(self) -> list[dict]:
        clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in self.messages]
        system = self._system_text()
        if self.llm.cfg.system_role:
            return [{"role": "system", "content": system}] + clean
        # Fold the system prompt into the first user message (templates without a system role).
        if clean and clean[0]["role"] == "user":
            clean[0] = dict(clean[0], content=f"{system}\n\n---\n\n{clean[0]['content']}")
            return clean
        return [{"role": "user", "content": system}] + clean

    def _append_user(self, text: str) -> None:
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages[-1]["content"] = f"{self.messages[-1]['content']}\n\n{text}"
        else:
            self.messages.append({"role": "user", "content": text})

    def _append_assistant(self, resp: LLMResponse, calls: list[ToolCall]) -> None:
        if self.llm.native_tools:
            m: dict[str, Any] = {"role": "assistant", "content": resp.text or ""}
            if calls:
                m["tool_calls"] = [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                    for c in calls
                ]
            self.messages.append(m)
        else:
            self.messages.append({"role": "assistant", "content": resp.text or "(empty reply)"})

    def _append_tool_result(self, call: ToolCall, result: str) -> None:
        if self.llm.native_tools:
            self.messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result, "_kind": "tool_result", "_tool": call.name}
            )
        else:
            self.messages.append(
                {
                    "role": "user",
                    "content": f"[TOOL RESULT {call.name}]\n{result}",
                    "_kind": "tool_result",
                    "_tool": call.name,
                }
            )

    def _convert_to_json_mode(self) -> None:
        """Rewrite any native tool messages into the json text protocol (used after a mid-run fallback)."""
        new: list[dict] = []
        for m in self.messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                blocks = [m.get("content") or ""]
                for tc in m["tool_calls"]:
                    fn = tc["function"]
                    try:
                        args = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    blocks.append("```json\n" + json.dumps({"tool": fn["name"], "args": args}) + "\n```")
                new.append({"role": "assistant", "content": "\n".join(b for b in blocks if b)})
            elif m["role"] == "tool":
                content = f"[TOOL RESULT {m.get('_tool', 'tool')}]\n{m['content']}"
                if new and new[-1]["role"] == "user":
                    new[-1]["content"] += "\n\n" + content
                else:
                    new.append({"role": "user", "content": content, "_kind": "tool_result", "_tool": m.get("_tool")})
            else:
                new.append(m)
        self.messages = new

    # -- context management -------------------------------------------------------------
    def _total_chars(self) -> int:
        return len(self._system_text()) + sum(len(str(m.get("content") or "")) + 100 for m in self.messages)

    def _trim_context(self) -> None:
        while self._total_chars() > self.context_char_budget:
            idx = [i for i, m in enumerate(self.messages) if m.get("_kind") == "tool_result" and not m.get("_evicted")]
            if len(idx) <= self.keep_recent_results:
                break
            i = idx[0]
            m = self.messages[i]
            n = len(m["content"])
            stub = (
                f"[tool result of {m.get('_tool')} removed to free context ({n} chars). "
                "Call the tool again if you still need it; your saved notes are preserved.]"
            )
            m["content"] = stub if self.llm.native_tools else f"[TOOL RESULT {m.get('_tool')}]\n{stub}"
            m["_evicted"] = True
            self._log(f"context trim: evicted result of {m.get('_tool')} ({n} chars)")

    # -- logging ------------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.name}] {msg}", file=sys.stderr, flush=True)

    def _trace(self, event: dict) -> None:
        if not self.trace_path:
            return
        event = {"ts": round(time.time(), 3), "agent": self.name, **event}
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    # -- main loop ----------------------------------------------------------------------
    def run(self, user_prompt: str, *, max_tokens: int | None = None) -> AgentResult:
        self.messages = [{"role": "user", "content": user_prompt}]
        self._trace({"type": "start", "system": self._system_text(), "user": user_prompt})
        turn = 0
        empties = 0
        final = ""
        while True:
            turn += 1
            self._trim_context()
            at_limit = turn > self.max_turns
            if at_limit:
                self._append_user(STEP_LIMIT_MSG)
            tools = None if at_limit else self.registry.schemas()
            try:
                resp = self.llm.chat(self._api_messages(), tools=tools, max_tokens=max_tokens)
            except NativeToolsUnsupported:
                self._log("server rejected native tool calling -> switching to the json tool protocol")
                self._trace({"type": "fallback", "to": "json"})
                self._convert_to_json_mode()
                turn -= 1
                continue
            calls = [] if at_limit else resp.tool_calls
            text = (resp.text or "").strip()
            if not calls:
                if not text and empties < 2 and not at_limit:
                    empties += 1
                    self._append_assistant(resp, [])
                    self._append_user("(Your reply was empty. Either call a tool or write your final summary.)")
                    continue
                self._append_assistant(resp, [])
                final = _strip_tool_blocks(text) if not self.llm.native_tools else text
                self._trace({"type": "final", "turn": turn, "text": final})
                self._log(f"turn {turn}: final answer ({len(final)} chars)")
                break
            self._append_assistant(resp, calls)
            for call in calls:
                self._tool_call_count += 1
                result = self.registry.call(call.name, call.args)
                self._append_tool_result(call, result)
                self._trace({"type": "tool", "turn": turn, "name": call.name, "args": call.args, "result_chars": len(result), "result_head": result[:300]})
                self._log(f"turn {turn}: {call.name}({_short_args(call.args)}) -> {len(result)} chars")
        return AgentResult(
            final_text=final,
            notes=list(self.notes),
            turns=turn,
            tool_calls=self._tool_call_count,
            usage=dict(self.llm.usage),
        )

    # -- one-shot generation (no tools), with continuation on max_tokens --------------
    def generate_long(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 8192, max_continues: int = 3) -> str:
        msgs: list[dict] = [{"role": "user", "content": user_prompt}]
        if self.llm.cfg.system_role:
            api = [{"role": "system", "content": system_prompt}] + msgs
        else:
            api = [{"role": "user", "content": f"{system_prompt}\n\n---\n\n{user_prompt}"}]
        pieces: list[str] = []
        for i in range(max_continues + 1):
            resp = self.llm.chat(api, tools=None, max_tokens=max_tokens)
            pieces.append(resp.text or "")
            self._trace({"type": "generate", "part": i, "chars": len(resp.text or ""), "finish": resp.finish_reason})
            if resp.finish_reason != "length":
                break
            self._log(f"output hit max_tokens, asking the model to continue (part {i + 2})")
            api = api + [
                {"role": "assistant", "content": resp.text or ""},
                {"role": "user", "content": "Continue exactly where you stopped. Do not repeat anything already written."},
            ]
        return "".join(pieces).strip()


def _strip_tool_blocks(text: str) -> str:
    """Remove fenced json blocks that look like tool calls (json transport only)."""

    def _drop(m: re.Match) -> str:
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            return m.group(0)
        return "" if isinstance(obj, dict) and "tool" in obj else m.group(0)

    return _FENCE_RE.sub(_drop, text).strip()


def _short_args(args: dict, limit: int = 90) -> str:
    s = ", ".join(f"{k}={json.dumps(v)[:40]}" for k, v in (args or {}).items())
    return s if len(s) <= limit else s[:limit] + "..."


# --------------------------------------------------------------------------------------
# Shared CLI plumbing
# --------------------------------------------------------------------------------------


def add_llm_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("LLM backend (any OpenAI-compatible endpoint)")
    g.add_argument("--model", default=None, help="model name (env LLM_MODEL). e.g. google/gemma-3-27b-it, gpt-4.1, claude-sonnet-4-5")
    g.add_argument("--base-url", default=None, help="e.g. http://localhost:8000/v1 for vLLM (env LLM_BASE_URL); omit for api.openai.com")
    g.add_argument("--api-key", default=None, help="env LLM_API_KEY / OPENAI_API_KEY; any string works for vLLM")
    g.add_argument("--tool-mode", choices=["auto", "native", "json"], default=None, help="auto (default): native function calling, fallback to the json text protocol")
    g.add_argument("--temperature", type=float, default=0.2, help="sampling temperature; pass -1 to omit the parameter (some models only accept the default)")
    g.add_argument("--max-tokens", type=int, default=2048, help="per exploration turn")
    g.add_argument("--synthesis-max-tokens", type=int, default=8192, help="for the final document(s)")
    g.add_argument("--timeout", type=float, default=300.0)
    g.add_argument("--no-system-role", action="store_true", help="fold the system prompt into the first user turn (chat templates without a system role)")
    g.add_argument("--extra-body", default=None, help='JSON passed through to the API, e.g. \'{"chat_template_kwargs":{"enable_thinking":false}}\'')
    g.add_argument("--ca-bundle", default=None, help="PEM CA bundle for TLS (env LLM_CA_BUNDLE / SSL_CERT_FILE / REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE)")
    g.add_argument("--insecure", action="store_true", help="disable TLS verification (env LLM_INSECURE=1)")


def llm_from_args(args: argparse.Namespace) -> LLMClient:
    extra = json.loads(args.extra_body) if getattr(args, "extra_body", None) else None
    cfg = LLMConfig.from_env(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        tool_mode=args.tool_mode,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        system_role=not getattr(args, "no_system_role", False),
        extra_body=extra,
        ca_bundle=getattr(args, "ca_bundle", None),
        insecure=True if getattr(args, "insecure", False) else None,
    )
    if args.temperature is not None and args.temperature < 0:
        cfg.temperature = None
    return LLMClient(cfg)


def add_agent_args(p: argparse.ArgumentParser, default_max_turns: int) -> None:
    g = p.add_argument_group("agent loop")
    g.add_argument("--max-turns", type=int, default=default_max_turns, help="exploration turns before forcing a summary")
    g.add_argument("--context-chars", type=int, default=240_000, help="rough context budget in characters (~4 chars/token)")
    g.add_argument("--quiet", action="store_true")
