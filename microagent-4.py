#!/usr/bin/env python3
"""
microagent.py — minimal agentic coding assistant for any local repo, talking to
an OpenAI-compatible endpoint (LiteLLM proxy, OpenAI, vLLM, Ollama, ...).

Setup (once, in your shell profile):
    export LITELLM_BASE_URL=https://llai-proxy.llan.ll.mit.edu/v1
    export LITELLM_API_KEY=sk-...
    export AGENT_MODEL=claude-opus-4-8            # any model your proxy serves

TLS (internal CA): pick ONE
    export AGENT_CA_BUNDLE=/path/to/internal-ca.pem   # preferred
    export AGENT_TLS_VERIFY=false                     # stopgap only

Run (from inside the repo, or point at one with --repo):
    microagent "add type hints to utils.py"      # one-shot
    microagent                                   # interactive REPL
    microagent --verify                          # health-check the endpoint
    microagent --init                            # write a starter AGENT.md
    microagent --approve "refactor the parser"   # confirm each bash/write

Project context loaded into the system prompt if present at <repo>:
    AGENT.md  or  CLAUDE.md          project instructions
    skills/*.md  or  .agent/*.md     specialist instructions
    ~/.config/microagent/AGENT.md    your personal, cross-project instructions

Optional self-management (--self-manage or AGENT_SELF_MANAGE=1): after each
task the agent appends to CHANGELOG.md, refreshes "## Agent Status" in
README.md, and makes one git commit. Off by default — the agent never commits
unless you ask.
"""

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import httpx
from openai import OpenAI

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

MODEL = os.environ.get("AGENT_MODEL", "claude-opus-4-8")
BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1")
API_KEY = os.environ.get("LITELLM_API_KEY", "")
MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "80"))
MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "8192"))
TEMPERATURE = float(os.environ.get("AGENT_TEMPERATURE", "0.2"))
MAX_TOOL_OUTPUT = 40_000
BASH_TIMEOUT = int(os.environ.get("AGENT_BASH_TIMEOUT", "300"))
SELF_MANAGE = os.environ.get("AGENT_SELF_MANAGE", "").lower() in {"1", "true", "yes"}

CA_BUNDLE = os.environ.get("AGENT_CA_BUNDLE")
TLS_VERIFY = os.environ.get("AGENT_TLS_VERIFY", "true").lower() != "false"

USER_CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "microagent"

IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env",
               ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
               "dist", "build", "target", ".idea", ".vscode"}

if CA_BUNDLE:
    _verify = CA_BUNDLE
elif not TLS_VERIFY:
    _verify = False
    print("[warn] TLS verification DISABLED (AGENT_TLS_VERIFY=false). "
          "Set AGENT_CA_BUNDLE to the internal CA for real use.", file=sys.stderr)
else:
    _verify = True

if not API_KEY:
    print("[warn] LITELLM_API_KEY is not set — requests will likely be rejected.",
          file=sys.stderr)

client = OpenAI(base_url=BASE_URL, api_key=API_KEY or "dummy",
                http_client=httpx.Client(verify=_verify, timeout=120.0))

# ----------------------------------------------------------------------------
# Tools (executed locally, sandboxed to the repo root)
# ----------------------------------------------------------------------------

class Toolbox:
    def __init__(self, repo: Path, approve: bool = False):
        self.repo = repo.resolve()
        self.approve = approve

    def _resolve(self, rel: str) -> Path:
        p = (self.repo / rel).resolve()
        if p != self.repo and self.repo not in p.parents:
            raise ValueError(f"Path escapes repo root: {rel}")
        return p

    def _confirm(self, what: str) -> bool:
        if not self.approve:
            return True
        try:
            ans = input(f"  ? {what}  [y/N] ").strip().lower()
        except EOFError:
            return False
        return ans in {"y", "yes"}

    def read_file(self, path: str, start_line: int = 1, end_line: int = -1) -> str:
        p = self._resolve(path)
        lines = p.read_text(errors="replace").splitlines()
        if end_line == -1:
            end_line = len(lines)
        window = lines[start_line - 1:end_line]
        return "\n".join(f"{i}\t{l}" for i, l in
                         enumerate(window, start=start_line)) or "(empty file)"

    def write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        if not self._confirm(f"write {path} ({len(content)} chars)"):
            return "DENIED by user: write_file not executed."
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} chars to {path}"

    def append_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        if not self._confirm(f"append to {path} ({len(content)} chars)"):
            return "DENIED by user: append_file not executed."
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(content if content.endswith("\n") else content + "\n")
        return f"Appended {len(content)} chars to {path}"

    def str_replace(self, path: str, old_str: str, new_str: str) -> str:
        p = self._resolve(path)
        text = p.read_text(errors="replace")
        n = text.count(old_str)
        if n == 0:
            return "ERROR: old_str not found. Re-read the file — it may have changed."
        if n > 1:
            return f"ERROR: old_str matches {n} places. Include more surrounding context."
        if not self._confirm(f"edit {path}"):
            return "DENIED by user: str_replace not executed."
        p.write_text(text.replace(old_str, new_str, 1))
        return f"Edited {path}"

    def list_files(self, path: str = ".", pattern: str = "*") -> str:
        root = self._resolve(path)
        out = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            for f in filenames:
                rel = os.path.relpath(os.path.join(dirpath, f), self.repo)
                if fnmatch.fnmatch(f, pattern):
                    out.append(rel)
            if len(out) > 500:
                out.append("... (truncated at 500 entries)")
                break
        return "\n".join(out) or "(no matches)"

    def grep(self, pattern: str, path: str = ".", glob: str = "*") -> str:
        root = self._resolve(path)
        rx = re.compile(pattern)
        hits = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            for f in filenames:
                if not fnmatch.fnmatch(f, glob):
                    continue
                fp = Path(dirpath) / f
                try:
                    for i, line in enumerate(
                            fp.read_text(errors="replace").splitlines(), 1):
                        if rx.search(line):
                            rel = os.path.relpath(fp, self.repo)
                            hits.append(f"{rel}:{i}: {line.strip()}")
                            if len(hits) >= 300:
                                hits.append("... (truncated at 300 hits)")
                                return "\n".join(hits)
                except (OSError, UnicodeDecodeError):
                    continue
        return "\n".join(hits) or "(no matches)"

    def bash(self, command: str) -> str:
        if not self._confirm(f"run: {command}"):
            return "DENIED by user: command not executed."
        try:
            r = subprocess.run(command, shell=True, cwd=self.repo,
                               capture_output=True, text=True,
                               timeout=BASH_TIMEOUT)
            out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
            return f"[exit {r.returncode}]\n{out.strip() or '(no output)'}"
        except subprocess.TimeoutExpired:
            return f"ERROR: command timed out after {BASH_TIMEOUT}s"


TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file (line-numbered). Optionally a line range.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "default": 1},
            "end_line": {"type": "integer", "default": -1}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a file with full content. Prefer "
                       "str_replace for edits to existing files.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "append_file",
        "description": "Append content to the end of a file (creates it if missing).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "str_replace",
        "description": "Surgical edit: replace one unique occurrence of old_str "
                       "with new_str. old_str must match file text exactly "
                       "(without the line-number prefixes from read_file).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old_str": {"type": "string"},
            "new_str": {"type": "string"}},
            "required": ["path", "old_str", "new_str"]}}},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "Recursively list files under a directory, optional glob.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "default": "."},
            "pattern": {"type": "string", "default": "*"}}}}},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Regex search across files. Returns file:line: match.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
            "glob": {"type": "string", "default": "*"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command in the repo root (tests, build, git "
                       f"status/diff, etc). {BASH_TIMEOUT}s timeout.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
]

# ----------------------------------------------------------------------------
# System prompt
# ----------------------------------------------------------------------------

BASE_PROMPT = """\
You are an expert coding assistant operating on a local repository at {repo}.
Use the provided tools to explore, edit, and verify code.

Working rules:
- Read before you edit. Never guess at file contents.
- Prefer surgical str_replace edits over full-file rewrites.
- Match the project's existing style, language conventions, and tooling.
- After changes, run the relevant tests, linter, or at minimum a syntax/compile
  check via bash. Report the actual result — never claim something passed if
  you didn't run it.
- Do not run git commit, git push, or any destructive git command unless the
  user explicitly asks. Reading git state (status, diff, log) is fine.
- Do not delete files or run destructive shell commands (rm -rf, force-push,
  dropping tables, etc.) without being asked.
- If the task is ambiguous, ask a brief clarifying question instead of guessing.
- When done, reply in plain text with a concise summary: what changed, why,
  and how it was verified.
"""

SELF_MANAGEMENT = """
# Self-management contract (enabled for this session)

You maintain your own record of work. At the END of every task, before your
final summary:

1. **CHANGELOG.md** — append a dated entry: the task, files changed, what you
   did and why, and how you verified it (commands run + result). Use
   append_file. Newest entries at the bottom. One entry per task.

2. **README.md** — keep a section titled "## Agent Status" current: what the
   project does now, what you last changed, and known gaps / next steps. Create
   it if missing; otherwise update it with str_replace. Do not rewrite the rest
   of the README.

3. **Living docs** — if you discovered that AGENT.md or a skills file is wrong
   or incomplete, fix it in the same task and note the doc change in
   CHANGELOG.md. Treat drift as a bug.

4. **Checkpoint** — git add -A && git commit -m "<concise task summary>". Never
   git push. Each task = one revertible commit including the doc updates above.
   (This overrides the general no-commit rule.)

If a task fails or you stop early, still write a CHANGELOG.md entry saying what
was attempted, what blocked it, and what to try next. Report failures as
failures.
"""

PROJECT_FILES = ["AGENT.md", "CLAUDE.md"]
SKILL_DIRS = ["skills", ".agent"]

AGENT_MD_TEMPLATE = """\
# AGENT.md — project instructions for the coding agent

## What this project is
<one or two sentences: purpose, main language, entry points>

## How to build / test
<exact commands, e.g. `make test`, `pytest -q`, `npm test`>

## Conventions
- <formatter / linter, e.g. black + ruff, prettier>
- <naming, module layout, anything non-obvious>

## Do not touch
- <generated files, vendored dirs, config that must stay as-is>

## Gotchas
- <things that have bitten people before>
"""


def _read_if_exists(p: Path) -> str | None:
    try:
        return p.read_text() if p.is_file() else None
    except OSError:
        return None


def build_system_prompt(repo: Path, self_manage: bool) -> str:
    parts = [BASE_PROMPT.format(repo=repo)]
    if self_manage:
        parts.append(SELF_MANAGEMENT)

    # Personal, cross-project instructions.
    personal = _read_if_exists(USER_CONFIG_DIR / "AGENT.md")
    if personal:
        parts.append(f"# User instructions (~/.config/microagent/AGENT.md)\n{personal}")

    # Project instructions: first match wins.
    for name in PROJECT_FILES:
        text = _read_if_exists(repo / name)
        if text:
            parts.append(f"# Project instructions ({name})\n{text}")
            break

    # Skills.
    for d in SKILL_DIRS:
        skills_dir = repo / d
        if skills_dir.is_dir():
            for f in sorted(skills_dir.glob("*.md")):
                parts.append(f"# Skill: {f.name}\n{f.read_text()}")
    return "\n\n".join(parts)

# ----------------------------------------------------------------------------
# Agent loop
# ----------------------------------------------------------------------------

def run_task(task: str, messages: list, toolbox: Toolbox, model: str,
             max_turns: int) -> list:
    messages.append({"role": "user", "content": task})
    for _ in range(max_turns):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_SPECS,
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        msg = resp.choices[0].message
        dumped = msg.model_dump(exclude_none=True)
        # Some proxy->Anthropic paths reject an assistant turn that has
        # tool_calls but no content key; keep an empty string.
        if msg.tool_calls and "content" not in dumped:
            dumped["content"] = ""
        messages.append(dumped)

        if not msg.tool_calls:
            print(f"\n{msg.content}\n")
            return messages

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args, result = {}, "ERROR: could not parse tool arguments"
            else:
                print(f"  \u2192 {name}({json.dumps(args)[:160]})")
                fn = getattr(toolbox, name, None)
                if fn is None or name.startswith("_"):
                    result = f"ERROR: unknown tool {name}"
                else:
                    try:
                        result = fn(**args)
                    except Exception as e:
                        result = f"ERROR: {type(e).__name__}: {e}"
            if len(result) > MAX_TOOL_OUTPUT:
                result = result[:MAX_TOOL_OUTPUT] + "\n... (truncated)"
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": result})
    print(f"\n[stopped: hit max turns ({max_turns}) \u2014 say 'continue' to keep going]\n")
    return messages


def verify_connection(model: str) -> bool:
    """One cheap call to confirm base URL + key + model + TLS all work."""
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "reply with the single word: ok"}],
            max_tokens=5, temperature=0)
        print(f"[verify] OK \u2014 model={model} replied: "
              f"{r.choices[0].message.content!r}")
        return True
    except Exception as e:
        print(f"[verify] FAILED: {type(e).__name__}: {e}")
        print("  Check LITELLM_BASE_URL, LITELLM_API_KEY, AGENT_MODEL, and TLS "
              "(AGENT_CA_BUNDLE or AGENT_TLS_VERIFY=false).")
        return False


REPL_HELP = """\
Commands:
  /help      show this
  /reset     clear conversation history (keeps system prompt)
  /status    show model, repo, turn count
  exit|quit  leave
Ctrl-C during a task aborts that task and returns to the prompt.
"""


def main():
    ap = argparse.ArgumentParser(
        description="Minimal agentic coding assistant for a local repo.",
        epilog="Env: LITELLM_BASE_URL, LITELLM_API_KEY, AGENT_MODEL, AGENT_MAX_TURNS, "
               "AGENT_SELF_MANAGE, AGENT_CA_BUNDLE, AGENT_TLS_VERIFY")
    ap.add_argument("task", nargs="?", help="Task to run once (omit for REPL).")
    ap.add_argument("--repo", default=".",
                    help="Repo root the agent works in (default: current dir).")
    ap.add_argument("--model", default=MODEL,
                    help=f"Model name (default: $AGENT_MODEL or {MODEL}).")
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS,
                    help=f"Max model calls per task (default {MAX_TURNS}).")
    ap.add_argument("--approve", action="store_true",
                    help="Ask before every bash command and file write/edit.")
    ap.add_argument("--self-manage", action="store_true", default=SELF_MANAGE,
                    help="Have the agent update CHANGELOG.md/README.md and "
                         "commit after each task.")
    ap.add_argument("--verify", action="store_true",
                    help="Run one cheap health-check call and exit.")
    ap.add_argument("--init", action="store_true",
                    help="Write a starter AGENT.md into the repo and exit.")
    ap.add_argument("--show-prompt", action="store_true",
                    help="Print the assembled system prompt and exit.")
    args = ap.parse_args()

    if args.verify:
        sys.exit(0 if verify_connection(args.model) else 1)

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        sys.exit(f"Not a directory: {repo}")

    if args.init:
        target = repo / "AGENT.md"
        if target.exists():
            sys.exit(f"{target} already exists — not overwriting.")
        target.write_text(AGENT_MD_TEMPLATE)
        print(f"Wrote {target}. Fill it in, then run microagent again.")
        return

    system_prompt = build_system_prompt(repo, args.self_manage)
    if args.show_prompt:
        print(system_prompt)
        return

    toolbox = Toolbox(repo, approve=args.approve)
    messages = [{"role": "system", "content": system_prompt}]
    flags = " \u00b7 ".join(f for f, on in [("approve", args.approve),
                                          ("self-manage", args.self_manage)] if on)
    print(f"microagent \u00b7 model={args.model} \u00b7 repo={repo}"
          + (f" \u00b7 {flags}" if flags else ""))

    if args.task:
        run_task(args.task, messages, toolbox, args.model, args.max_turns)
        return

    print("Type a task, or /help.")
    while True:
        try:
            task = input("task> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not task:
            continue
        if task in {"exit", "quit"}:
            break
        if task == "/help":
            print(REPL_HELP)
            continue
        if task == "/reset":
            messages = messages[:1]
            print("[history cleared]")
            continue
        if task == "/status":
            print(f"model={args.model} repo={repo} messages={len(messages) - 1}")
            continue
        try:
            messages = run_task(task, messages, toolbox, args.model, args.max_turns)
        except KeyboardInterrupt:
            print("\n[task aborted]")
            # Drop any dangling assistant/tool messages so the next call is
            # well-formed: keep everything up to and including the last user turn.
            while messages and messages[-1]["role"] != "user":
                messages.pop()
            if messages and messages[-1]["role"] == "user":
                messages.pop()
        except Exception as e:
            print(f"\n[error] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
