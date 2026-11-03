#!/usr/bin/env python3
"""
microagent.py — minimal self-managing agentic coder for a LiteLLM-hosted model.

Setup:
    export LITELLM_BASE_URL=https://llai-proxy.llan.ll.mit.edu/v1
    export LITELLM_API_KEY=sk-...
    export AGENT_MODEL=claude-opus-4-8

TLS (internal MIT LL cert): pick ONE
    export AGENT_CA_BUNDLE=/path/to/mit-ll-ca.pem   # preferred: verify properly
    export AGENT_TLS_VERIFY=false                     # stopgap: skip verification

Run:
    python microagent.py --repo /path/to/discver "task ..."   # one-shot
    python microagent.py --repo /path/to/discver              # interactive REPL

Approval gate: every write_file, append_file, str_replace and bash call is shown to you
first and runs only if you answer y (once), a (always, this tool, this session) or it
matches AGENT_AUTO_ALLOW. n sends a refusal back to the model as the tool result.
    export AGENT_AUTO_ALLOW="python -m pytest,python -m py_compile,git status,git diff,ls"
    python microagent.py --repo ... --yes "task"   # no prompts: throwaway folders only

Context files loaded into the system prompt if present at <repo>:
    AGENT.md              project instructions
    skills/*.md           specialist skills

Self-management: after each task the agent appends to CHANGELOG.md and refreshes
the "## Agent Status" section of README.md, then commits. See AGENT.md.
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
API_KEY = os.environ.get("LITELLM_API_KEY", "dummy")
MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "80"))
MAX_TOOL_OUTPUT = 40_000
BASH_TIMEOUT = 300

CA_BUNDLE = os.environ.get("AGENT_CA_BUNDLE")            # path to internal CA
TLS_VERIFY = os.environ.get("AGENT_TLS_VERIFY", "true").lower() != "false"

# Commands that run without asking (prefix match). Everything else prompts.
AUTO_ALLOW = tuple(c.strip() for c in os.environ.get("AGENT_AUTO_ALLOW", "").split(",") if c.strip())

IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
               ".mypy_cache", ".pytest_cache", "dist", "build", ".idea"}

# TLS: prefer a real CA bundle; fall back to verify=False only if asked.
if CA_BUNDLE:
    _verify = CA_BUNDLE
elif not TLS_VERIFY:
    _verify = False
    print("[warn] TLS verification DISABLED (AGENT_TLS_VERIFY=false). "
          "Set AGENT_CA_BUNDLE to the internal CA and remove this for real use.")
else:
    _verify = True

client = OpenAI(base_url=BASE_URL, api_key=API_KEY,
                http_client=httpx.Client(verify=_verify, timeout=120.0))

# ----------------------------------------------------------------------------
# Tools (executed locally, sandboxed to --repo)
# ----------------------------------------------------------------------------

class Toolbox:
    def __init__(self, repo: Path, auto_yes: bool = False):
        self.repo = repo.resolve()
        self.auto_yes = auto_yes
        self.always: set[str] = set()      # tools approved for the rest of the session

    def _resolve(self, rel: str) -> Path:
        p = (self.repo / rel).resolve()
        if p != self.repo and self.repo not in p.parents:   # not a string prefix: /repo-other must fail
            raise ValueError(f"Path escapes repo root: {rel}")
        return p

    def _confirm(self, tool: str, what: str) -> str | None:
        """Ask before a tool acts. Returns None when allowed, else the refusal text for the model."""
        if self.auto_yes or tool in self.always:
            return None
        if tool == "bash" and what.startswith(AUTO_ALLOW):
            return None
        print(f"\n  [{tool}] {what[:300]}")
        while True:
            try:
                ans = input("  allow? [y]es once / [a]lways this tool / [n]o: ").strip().lower()
            except EOFError:                       # no terminal (background job, scheduler): fail closed
                return (f"{tool} needs interactive approval and no terminal is attached. "
                        "Stop and report; the operator can rerun with --yes in a throwaway folder.")
            if ans in ("y", "yes"):
                return None
            if ans in ("a", "always"):
                self.always.add(tool)
                return None
            if ans in ("n", "no", ""):
                return (f"User declined this {tool} call. Do not retry it unchanged; "
                        "explain what you wanted to do and ask, or take a different approach.")

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
        verb = "overwrite" if p.exists() else "create"
        if refused := self._confirm("write_file", f"{verb} {path} ({len(content.splitlines())} lines)"):
            return refused
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} chars to {path}"

    def append_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        if refused := self._confirm("append_file", f"append {len(content)} chars to {path}"):
            return refused
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
        if refused := self._confirm("str_replace", f"edit {path}: -{len(old_str.splitlines())} +{len(new_str.splitlines())} lines\n"
                                    f"    - {old_str.strip()[:120]!r}\n    + {new_str.strip()[:120]!r}"):
            return refused
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
        if refused := self._confirm("bash", command.strip()):
            return refused
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
        "description": "Create or overwrite a file with full content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "append_file",
        "description": "Append content to the end of a file (creates it if "
                       "missing). Use for CHANGELOG.md entries.",
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
        "description": "Run a shell command in the repo root (tests, git, etc). "
                       f"{BASH_TIMEOUT}s timeout.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
]

# ----------------------------------------------------------------------------
# System prompt: AGENT.md + skills + self-management contract
# ----------------------------------------------------------------------------

SELF_MANAGEMENT = """
# Self-management contract (do this every task, without being told)

You maintain your own record of work. At the END of every task, before you give
your final summary:

1. **CHANGELOG.md** — append a dated entry with: the task, the files you changed,
   what you did and why, and how you verified it (tests/commands run + result).
   Use append_file. Newest entries go at the bottom. One entry per task.

2. **README.md** — keep a section titled "## Agent Status" current: what the
   pipeline does now, what you last changed, and known gaps / next steps. If the
   section doesn't exist, create it with write_file; otherwise update it with
   str_replace. Do not rewrite the rest of the README.

3. **Living docs** — if during the task you discovered that AGENT.md or a
   skills/*.md file is now wrong or incomplete (stale module name, a rule that
   didn't hold), fix it in the same task and note the doc change in CHANGELOG.md.
   These docs are yours to keep accurate; treat drift as a bug.

4. **Checkpoint** — git add -A && git commit -m "<concise task summary>". Never
   git push. Each task = one revertible commit including the doc updates above.

If a task fails or you stop early, still write a CHANGELOG.md entry saying what
was attempted, what blocked it, and what the next agent should try. The record
must never lie about what happened — report failures as failures.
"""


def build_system_prompt(repo: Path) -> str:
    parts = [
        "You are an expert agentic coding assistant operating on a local "
        f"repository at {repo}. Use the provided tools to explore, edit, and "
        "verify code. Read before you edit. Prefer surgical str_replace edits "
        "over full rewrites. After changes, run the relevant tests or at least "
        "a syntax check via bash. When done, reply in plain text with a concise "
        "summary of what changed and why.",
        SELF_MANAGEMENT,
    ]
    agent_md = repo / "AGENT.md"
    if agent_md.exists():
        parts.append(f"# Project instructions (AGENT.md)\n{agent_md.read_text()}")
    skills_dir = repo / "skills"
    if skills_dir.is_dir():
        for f in sorted(skills_dir.glob("*.md")):
            parts.append(f"# Skill: {f.name}\n{f.read_text()}")
    return "\n\n".join(parts)

# ----------------------------------------------------------------------------
# Agent loop
# ----------------------------------------------------------------------------

def run_task(task: str, messages: list, toolbox: Toolbox) -> list:
    messages.append({"role": "user", "content": task})
    for _ in range(MAX_TURNS):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SPECS,
            max_tokens=8192, temperature=0.2)
        msg = resp.choices[0].message
        dumped = msg.model_dump(exclude_none=True)
        # Some LiteLLM->Anthropic paths reject an assistant turn that has
        # tool_calls but no content key. Keep an empty string so the pairing
        # is always well-formed.
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
                try:
                    result = getattr(toolbox, name)(**args)
                except Exception as e:
                    result = f"ERROR: {type(e).__name__}: {e}"
            if len(result) > MAX_TOOL_OUTPUT:
                result = result[:MAX_TOOL_OUTPUT] + "\n... (truncated)"
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": result})
    print("\n[stopped: hit MAX_TURNS \u2014 say 'continue' to keep going]\n")
    return messages


def verify_connection() -> bool:
    """One cheap call to confirm base URL + key + model + TLS all work,
    before spending credits on a full agent run."""
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "reply with the single word: ok"}],
            max_tokens=5, temperature=0)
        print(f"[verify] OK \u2014 model={MODEL} replied: "
              f"{r.choices[0].message.content!r}")
        return True
    except Exception as e:
        print(f"[verify] FAILED: {type(e).__name__}: {e}")
        print("  Check LITELLM_BASE_URL, LITELLM_API_KEY, AGENT_MODEL, and TLS "
              "(AGENT_CA_BUNDLE or AGENT_TLS_VERIFY=false).")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--verify", action="store_true",
                    help="Run one cheap health-check call and exit.")
    ap.add_argument("--yes", action="store_true",
                    help="Approve every write, edit and command without asking. Throwaway folders only.")
    ap.add_argument("task", nargs="?")
    args = ap.parse_args()

    if args.verify:
        sys.exit(0 if verify_connection() else 1)

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        sys.exit(f"Not a directory: {repo}")

    toolbox = Toolbox(repo, auto_yes=args.yes)
    messages = [{"role": "system", "content": build_system_prompt(repo)}]
    print(f"microagent \u00b7 model={MODEL} \u00b7 repo={repo} \u00b7 "
          f"approval={'OFF (--yes)' if args.yes else 'on; auto-allow: ' + (', '.join(AUTO_ALLOW) or 'none')}")

    if args.task:
        run_task(args.task, messages, toolbox)
        return

    while True:
        try:
            task = input("task> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not task or task in {"exit", "quit"}:
            break
        messages = run_task(task, messages, toolbox)


if __name__ == "__main__":
    main()
