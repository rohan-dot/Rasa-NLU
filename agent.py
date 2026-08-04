"""Standalone agent driver for the FCG checklist pipeline.

No Claude Code required: this script IS the agent harness. It runs an
interactive loop against Claude (Opus 4.8) via the LiteLLM proxy using
OpenAI-compatible tool calling. The model gets CLAUDE.md + the skill file
as its operating brief, and a small whitelisted toolset:

  run_pipeline    -- run one orchestrator command (status/preprocess/
                     stage1/stage2/stage3); stage3 requires HUMAN y/N
                     confirmation enforced in Python, not by the model
  read_artifact   -- read a known pipeline artifact (whitelisted keys)
  read_file       -- read any file INSIDE the repo (path-confined)
  update_schema   -- validate + write stage2_schema.json (auto-backup)

Guardrails live in code, not prompts: arbitrary shell is impossible,
paths outside the repo are rejected, and gate-skipping flags are refused.

Usage:
  pip install openai
  export FCG_LLM_BASE_URL=http://localhost:4000/v1
  python agent.py
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pipeline import config

ROOT = Path(__file__).resolve().parent
TOOL_OUTPUT_CAP = 8000
HISTORY_MAX_MESSAGES = 60          # crude context control for long sessions

ALLOWED_COMMANDS = {"status", "preprocess", "stage1", "stage2", "stage3"}
CONFIRM_COMMANDS = {"stage3"}      # human y/N enforced in Python

ARTIFACTS = {
    "checklist_items": config.CHECKLIST_ITEMS_JSONL,
    "stage1_flagged": config.STAGE1_FLAGGED_JSON,
    "stage2_verified": config.STAGE2_VERIFIED_JSON,
    "stage2_schema": config.STAGE2_SCHEMA_JSON,
    "stage3_checkpoint": config.STAGE3_CHECKPOINT_JSONL,
    "final_csv": config.FINAL_CSV,
}

TOOLS = [
    {"type": "function", "function": {
        "name": "run_pipeline",
        "description": ("Run one pipeline orchestrator command. stage3 will "
                        "ask the human for y/N confirmation in the terminal."),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "enum": sorted(ALLOWED_COMMANDS)}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_artifact",
        "description": "Read a known pipeline artifact by key.",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string", "enum": sorted(ARTIFACTS)},
            "head_chars": {"type": "integer",
                           "description": "optional char cap (default 8000)"}},
            "required": ["key"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": ("Read a file inside the repo (relative path). Use "
                        "head_chars to sample large data files first."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "head_chars": {"type": "integer",
                           "description": "optional char cap (default 8000)"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "update_schema",
        "description": ("Replace stage2_schema.json with the provided JSON "
                        "array. Validated; previous version backed up."),
        "parameters": {"type": "object", "properties": {
            "schema_json": {"type": "string",
                            "description": "full JSON array as a string"}},
            "required": ["schema_json"]}}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List a directory inside the repo (relative path).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "default: repo root"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": ("Create or overwrite a file inside the repo with the "
                        "given content. ALWAYS write complete files, never "
                        "fragments. Overwrites keep a .bak backup."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_python",
        "description": ("Run a Python file that exists inside the repo, "
                        "with optional args, cwd=repo root. Use for testing "
                        "code you wrote. 15-minute timeout; output capped."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "e.g. scratch/probe.py"},
            "args": {"type": "array", "items": {"type": "string"}}},
            "required": ["path"]}}},
]


# ----------------------------------------------------------------------------
# Tool implementations (all guardrails live here)
# ----------------------------------------------------------------------------

def _cap(text: str, cap: int = TOOL_OUTPUT_CAP) -> str:
    if len(text) <= cap:
        return text
    half = cap // 2
    return text[:half] + f"\n...[{len(text) - cap} chars omitted]...\n" + text[-half:]


def tool_run_pipeline(command: str) -> str:
    if command not in ALLOWED_COMMANDS:
        return f"REFUSED: '{command}' is not an allowed command."
    if command in CONFIRM_COMMANDS:
        answer = input(f"\n>>> Agent wants to run '{command}' (full LLM pass "
                       f"over all countries). Proceed? [y/N] ").strip().lower()
        if answer != "y":
            return "HUMAN DECLINED: stage3 not run. Ask what they want first."
    proc = subprocess.run(
        [sys.executable, "-m", "pipeline.run_pipeline", command],
        cwd=ROOT, capture_output=True, text=True, timeout=6 * 3600)
    out = (proc.stdout or "") + (("\nSTDERR:\n" + proc.stderr) if proc.stderr else "")
    return _cap(f"exit={proc.returncode}\n{out}")


def tool_read_artifact(key: str, head_chars: int = TOOL_OUTPUT_CAP) -> str:
    path = ARTIFACTS.get(key)
    if path is None:
        return f"Unknown artifact key '{key}'."
    if not path.exists():
        return f"Artifact '{key}' does not exist yet ({path})."
    return _cap(path.read_text(encoding="utf-8", errors="replace"),
                min(head_chars or TOOL_OUTPUT_CAP, 30000))


def _confine(rel_path: str):
    """Resolve a relative path and refuse anything escaping the repo."""
    target = (ROOT / rel_path).resolve()
    if ROOT not in target.parents and target != ROOT:
        return None
    return target


def tool_read_file(rel_path: str, head_chars: int = TOOL_OUTPUT_CAP) -> str:
    target = _confine(rel_path)
    if target is None:
        return "REFUSED: path escapes the repo."
    if not target.is_file():
        return f"Not a file: {rel_path}"
    return _cap(target.read_text(encoding="utf-8", errors="replace"),
                min(head_chars or TOOL_OUTPUT_CAP, 30000))


def tool_list_dir(rel_path: str) -> str:
    target = _confine(rel_path or ".")
    if target is None:
        return "REFUSED: path escapes the repo."
    if not target.is_dir():
        return f"Not a directory: {rel_path}"
    lines = []
    for p in sorted(target.iterdir()):
        if p.name.startswith((".git", "__pycache__")):
            continue
        size = p.stat().st_size if p.is_file() else 0
        lines.append(f"{'d' if p.is_dir() else 'f'} {size:>10}  {p.name}")
    return _cap("\n".join(lines) or "(empty)")


PROTECTED_FILES = {"agent.py"}   # the agent must not rewrite its own harness


def tool_write_file(rel_path: str, content: str) -> str:
    target = _confine(rel_path)
    if target is None:
        return "REFUSED: path escapes the repo."
    if target.name in PROTECTED_FILES:
        return "REFUSED: this file is protected; ask the human to change it."
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.copy2(target, target.with_suffix(target.suffix + ".bak"))
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {rel_path}" \
           + (" (backup kept)" if target.with_suffix(target.suffix + ".bak").exists() else "")


def tool_run_python(rel_path: str, args: list) -> str:
    target = _confine(rel_path)
    if target is None:
        return "REFUSED: path escapes the repo."
    if not target.is_file() or target.suffix != ".py":
        return f"Not a Python file in the repo: {rel_path}"
    argv = [str(a) for a in (args or [])]
    try:
        proc = subprocess.run(
            [sys.executable, str(target)] + argv,
            cwd=ROOT, capture_output=True, text=True, timeout=15 * 60)
    except subprocess.TimeoutExpired:
        return "TIMEOUT: script exceeded 15 minutes."
    out = (proc.stdout or "") + (("\nSTDERR:\n" + proc.stderr) if proc.stderr else "")
    return _cap(f"exit={proc.returncode}\n{out}")


def tool_update_schema(schema_json: str) -> str:
    try:
        schema = json.loads(schema_json)
    except json.JSONDecodeError as e:
        return f"REFUSED: invalid JSON ({e})."
    if not isinstance(schema, list) or not schema:
        return "REFUSED: schema must be a non-empty JSON array."
    for i, field in enumerate(schema):
        if not isinstance(field, dict) or not field.get("field") \
                or not field.get("question"):
            return f"REFUSED: entry {i} missing 'field' or 'question'."
    names = [f["field"] for f in schema]
    if len(names) != len(set(names)):
        return "REFUSED: duplicate field names."
    path = config.STAGE2_SCHEMA_JSON
    if path.exists():
        shutil.copy2(path, path.with_suffix(f".bak.{int(time.time())}.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return f"Wrote {len(schema)} fields to {path.name} (backup kept)."


DISPATCH = {
    "run_pipeline": lambda a: tool_run_pipeline(a.get("command", "")),
    "read_artifact": lambda a: tool_read_artifact(a.get("key", ""),
                                                  a.get("head_chars") or TOOL_OUTPUT_CAP),
    "read_file": lambda a: tool_read_file(a.get("path", ""),
                                          a.get("head_chars") or TOOL_OUTPUT_CAP),
    "update_schema": lambda a: tool_update_schema(a.get("schema_json", "")),
    "list_dir": lambda a: tool_list_dir(a.get("path", ".")),
    "write_file": lambda a: tool_write_file(a.get("path", ""),
                                            a.get("content", "")),
    "run_python": lambda a: tool_run_python(a.get("path", ""),
                                            a.get("args") or []),
}


# ----------------------------------------------------------------------------
# Agent loop
# ----------------------------------------------------------------------------

def _system_prompt() -> str:
    brief = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    skill_path = ROOT / ".claude/skills/fcg-checklist-pipeline/SKILL.md"
    skill = skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""
    return f"""You are the coding + pipeline agent for Jo's FCG route-planning
project, working via the tools provided. Your operating brief follows; obey
its architecture principles.

CODING CONDUCT:
- EXPLORE BEFORE CODING: use list_dir and read_file (with head_chars) to
  sample real data files before writing code that parses them. Never guess
  file structure.
- Write COMPLETE files with write_file (Jo's standing preference — never
  fragments or diffs). Put throwaway probes in scratch/.
- After writing code, TEST it with run_python on a small sample before
  declaring it done. Show Jo the test output.
- Propose a short plan and get Jo's go-ahead before large builds; small
  fixes can just be done.
- Keep the extraction principles: closed-world (dossier text only, else
  NA), verbatim raw quotes, one fact per question, batched fields
  (~20/call), per-country resume checkpoints.

CURRENT MISSION CONTEXT: 94-item FCG checklist extraction per country.
Dossier sources per country: the country's FCG JSON, plus relevant slices
of meis_airfield_details (GIANT airfield JSONs, likely ICAO-keyed),
SVC_RMK (enroute support text), and udl_notifications (old NOTAMs text).
First task when asked: explore those source files, determine how each maps
to a country, and report the keying scheme to Jo before extending
fcg_extract_simple.py.

PIPELINE RULES:
- Work ONE stage at a time; stop at review gates and WAIT for Jo.
- stage3 / full extractions trigger a terminal confirmation for Jo.
- If a tool refuses an action, tell Jo plainly; do not retry variants.
- Keep responses tight: findings, then a single proposed next action.

=== CLAUDE.md ===
{brief}

=== SKILL.md ===
{skill}"""


def main() -> None:
    import httpx
    from openai import OpenAI
    http_client = None if config.TLS_VERIFY else httpx.Client(verify=False)
    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY,
                    http_client=http_client)

    messages = [{"role": "system", "content": _system_prompt()}]
    print(f"FCG pipeline agent -- model={config.LLM_MODEL} "
          f"endpoint={config.LLM_BASE_URL}")
    print("Type your instruction ('quit' to exit). "
          "Suggested opener: run status and tell me where we are.\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("quit", "exit"):
            break
        messages.append({"role": "user", "content": user})

        # tool loop
        for _ in range(20):  # hard cap on tool rounds per turn
            resp = client.chat.completions.create(
                model=config.LLM_MODEL, temperature=0, max_tokens=4000,
                messages=messages, tools=TOOLS)
            msg = resp.choices[0].message
            messages.append({"role": "assistant",
                             "content": msg.content or "",
                             "tool_calls": [tc.model_dump() for tc in msg.tool_calls]
                             if msg.tool_calls else None})
            if not msg.tool_calls:
                print(f"\nagent> {msg.content}\n")
                break
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                print(f"  [tool] {name}({json.dumps(args)[:120]})")
                result = DISPATCH.get(name, lambda a: f"Unknown tool {name}")(args)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": result})
        else:
            print("\nagent> (stopped: tool-round cap reached)\n")

        # crude history trim: keep system + recent tail
        if len(messages) > HISTORY_MAX_MESSAGES:
            messages = [messages[0]] + messages[-(HISTORY_MAX_MESSAGES - 1):]


if __name__ == "__main__":
    main()
