#!/usr/bin/env python3
"""
fix_loop.py — run a Python script, and when it fails, have a gateway model
(LiteLLM, e.g. GPT 5.6 or Opus) fix the file and retry. Repeats until the
command succeeds or the round limit is hit.

    python fix_loop.py --file site_recon_agent.py --rounds 5 -- \\
        python site_recon_agent.py --origin KSUU --dest FKKD --aircraft C-17 --no-extract --max-pages 20

Everything after `--` is the command to run. Each round:
  1. runs the command, captures stdout/stderr and the exit code
  2. if exit code != 0: sends the FULL current file + the last 6000 chars of
     output + your note to the model, asks for the complete corrected file
  3. backs up the current file to <file>.bak.<round>, writes the fix, shows a
     unified diff, and (unless --yes) asks you before applying
  4. reruns

Safety
  - --must-keep "text" (repeatable): the fix is REJECTED if any of these
    substrings disappear from the file. Defaults protect the recon agent's
    never-click list and stop checks.
  - Environment variable VALUES are never sent to the model (only the error
    text and the code). Redact anything secret from the error output before
    it is sent: a regex blanks things that look like keys/passwords.
  - Ctrl-C at any time; the last applied file stays, backups are kept.

Env: LITELLM_BASE_URL, LITELLM_API_KEY, AGENT_MODEL (override with --model),
     AGENT_TLS_VERIFY=false for self-signed gateway certs.
"""

import argparse
import difflib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SYSTEM = """You are a senior Python engineer fixing ONE script so that a command runs
without error. You receive the full file, the failing command, its output, and
the author's note.

Rules:
- Return the COMPLETE corrected file inside one ```python fenced block, nothing
  else before or after the block. Not a diff, not a snippet.
- Make the smallest change that fixes the reported error; do not refactor or
  rename things, do not remove safety checks, guards, logging, or docstrings.
- Keep every existing command-line flag and environment variable name.
- If the error is environmental (missing package, no network, missing env var,
  wrong selector for a specific website) and cannot be fixed in code, add a
  clear, actionable error message at the right place instead of guessing, and
  say so in a comment at the top of the file: `# FIX NOTE: ...`.
- Never add code that deletes files, sends data anywhere new, or disables TLS
  verification by default.
"""

DEFAULT_KEEP = ["NEVER = re.compile", "def check_stop", "StopRequested",
                "ALLOW_CALCULATE", "delete|remove"]

SECRET_RE = re.compile(
    r"((?:api[_-]?key|token|password|passwd|secret|authorization|bearer)\s*[:=]\s*)\S+",
    re.I)


def make_client(model_override):
    import httpx
    from openai import OpenAI
    base = os.environ.get("LITELLM_BASE_URL")
    key = os.environ.get("LITELLM_API_KEY")
    model = model_override or os.environ.get("AGENT_MODEL")
    if not (base and key and model):
        sys.exit("set LITELLM_BASE_URL, LITELLM_API_KEY and AGENT_MODEL (or --model)")
    verify = os.environ.get("AGENT_TLS_VERIFY", "true").lower() not in ("0", "false", "no")
    return OpenAI(base_url=base, api_key=key,
                  http_client=httpx.Client(verify=verify, timeout=600)), model


def run(cmd, timeout):
    print(f"\n[run] {' '.join(cmd)}")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + ("\n--- STDERR ---\n" + p.stderr if p.stderr else "")
        code = p.returncode
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"").decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")) \
            + f"\n--- TIMEOUT after {timeout}s ---"
        code = 124
    print(f"[run] exit {code} in {time.time()-t0:.0f}s")
    tail = out[-3000:]
    print(tail if tail.strip() else "(no output)")
    return code, out


def redact(text):
    return SECRET_RE.sub(r"\1<redacted>", text)


def extract_code(reply):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", reply, re.S)
    return (m.group(1) if m else reply).rstrip() + "\n"


def ask_fix(client, model, path, code_text, cmd, output, note, history):
    user = (f"FILE: {path}\n\nFAILING COMMAND:\n{' '.join(cmd)}\n\n"
            f"OUTPUT (tail):\n{redact(output[-6000:])}\n\n"
            f"AUTHOR NOTE: {note or '(none)'}\n\n"
            + (f"PREVIOUS ATTEMPTS THIS SESSION (what was tried, still failing):\n"
               + "\n".join(history) + "\n\n" if history else "")
            + f"CURRENT FILE CONTENTS:\n```python\n{code_text}\n```")
    r = client.chat.completions.create(
        model=model, temperature=0, max_tokens=16000,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user}])
    return r.choices[0].message.content or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="the script to fix")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per run")
    ap.add_argument("--model", default=None)
    ap.add_argument("--note", default="", help="anything you know about the error")
    ap.add_argument("--must-keep", action="append", default=None,
                    help="substring that must survive every fix (repeatable)")
    ap.add_argument("--yes", action="store_true", help="apply fixes without asking")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command to run")
    args = ap.parse_args()
    cmd = [c for c in args.cmd if c != "--"]
    if not cmd:
        sys.exit("give the command after `--`")
    path = Path(args.file)
    if not path.exists():
        sys.exit(f"{path} not found")
    keep = args.must_keep if args.must_keep is not None else \
        (DEFAULT_KEEP if "site_recon" in path.name else [])

    client, model = make_client(args.model)
    history = []
    for rnd in range(1, args.rounds + 1):
        code, out = run(cmd, args.timeout)
        if code == 0:
            print(f"\n[ok] command succeeded after {rnd-1} fix(es).")
            return
        print(f"\n[round {rnd}/{args.rounds}] asking {model} for a fix ...")
        current = path.read_text(encoding="utf-8")
        try:
            reply = ask_fix(client, model, path.name, current, cmd, out, args.note, history)
        except Exception as e:
            sys.exit(f"[llm] call failed: {e}")
        new = extract_code(reply)
        if len(new) < 0.5 * len(current):
            print("[reject] model returned a file less than half the original size "
                  "(probably a snippet, not the whole file). Not applying.")
            history.append(f"round {rnd}: model returned a partial file; rejected")
            continue
        missing = [k for k in keep if k in current and k not in new]
        if missing:
            print(f"[reject] fix removed protected text: {missing}. Not applying.")
            history.append(f"round {rnd}: fix removed protected text {missing}; rejected")
            continue
        try:
            compile(new, path.name, "exec")
        except SyntaxError as e:
            print(f"[reject] fix has a syntax error: {e}")
            history.append(f"round {rnd}: fix had SyntaxError {e}; rejected")
            continue
        diff = list(difflib.unified_diff(current.splitlines(), new.splitlines(),
                                         "current", "proposed", lineterm="", n=2))
        print("\n".join(diff[:200]) if diff else "(no textual change)")
        if len(diff) > 200:
            print(f"... ({len(diff)-200} more diff lines)")
        if not diff:
            history.append(f"round {rnd}: model returned an identical file")
            continue
        note_m = re.search(r"#\s*FIX NOTE:(.*)", new)
        if note_m:
            print(f"\n[model note] {note_m.group(1).strip()}")
        if not args.yes:
            ans = input("\nApply this fix and rerun? [y/N/q] ").strip().lower()
            if ans == "q":
                print("[quit] file unchanged"); return
            if ans != "y":
                history.append(f"round {rnd}: user declined the proposed fix")
                continue
        bak = path.with_suffix(path.suffix + f".bak.{rnd}")
        bak.write_text(current, encoding="utf-8")
        path.write_text(new, encoding="utf-8")
        changed = sum(1 for l in diff if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))
        history.append(f"round {rnd}: applied a fix touching ~{changed} lines "
                       f"(backup {bak.name}); error was: {out.strip().splitlines()[-1][:160] if out.strip() else '?'}")
        print(f"[applied] backup at {bak}")
    code, _ = run(cmd, args.timeout)
    print("\n[done] " + ("succeeded on final run." if code == 0 else
                        f"still failing after {args.rounds} rounds. Backups: "
                        f"{', '.join(str(p) for p in path.parent.glob(path.name + '.bak.*'))}"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[stop] interrupted; last applied file is in place, backups kept.")
