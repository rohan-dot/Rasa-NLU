#!/usr/bin/env python3
"""
requests_task_writer.py — have your gateway model (GPT 5.6 / Opus via LiteLLM)
WRITE a requests-based portal script from the probe output, TEST it, and FIX it.

Inputs it reads (from portal_probe.py):
  probe_report.md        the human-readable site map
  probe_out/*.html       raw pages (scripts stripped, trimmed) so the model sees
                         real form names, link hrefs, table layouts

Output:
  portal_requests_tasks.py   a readable script for THIS portal, using only
                             safe_session.SafeSession (guarded GET/POST)
  portal_results_<ts>.md/.json  when it runs clean

Loop: write -> run -> on failure, send script + output + the responses it
saved -> corrected script -> run ... up to --rounds. Every version backed up.

Env: LITELLM_BASE_URL, LITELLM_API_KEY, AGENT_MODEL, AGENT_TLS_VERIFY
     SITE_URL, SITE_USER, SITE_PASS

Usage:
  python requests_task_writer.py --mission "KSUU to FKKD C-17 2026-10-20" \\
     --task "1) Open Enter/Confirm/Update Standard Departure Procedures and capture
             every leg's SDP text and status. 2) For each Alternate BO, trigger its
             calculator and capture the result; then trigger the Main BO calculator
             and capture fuel and weights. Never save." --rounds 5
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime

GEN = "portal_requests_tasks.py"
PROBE_DIR = "probe_out"
PROBE_REPORT = "probe_report.md"


def make_client():
    import httpx
    from openai import OpenAI
    base, key, model = (os.environ.get("LITELLM_BASE_URL"), os.environ.get("LITELLM_API_KEY"),
                        os.environ.get("AGENT_MODEL"))
    if not (base and key and model):
        sys.exit("[FATAL] set LITELLM_BASE_URL, LITELLM_API_KEY, AGENT_MODEL")
    verify = os.environ.get("AGENT_TLS_VERIFY", "true").lower() not in ("0", "false", "no")
    return OpenAI(base_url=base, api_key=key,
                  http_client=httpx.Client(verify=verify, timeout=900)), model


def chat(client, model, system, user, max_tokens=16000):
    r = client.chat.completions.create(model=model, max_tokens=max_tokens,
                                       messages=[{"role": "system", "content": system},
                                                 {"role": "user", "content": user}])
    return r.choices[0].message.content or ""


def strip_html(html, limit):
    html = re.sub(r"<script.*?</script>", "", html, flags=re.S | re.I)
    html = re.sub(r"<style.*?</style>", "", html, flags=re.S | re.I)
    html = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    html = re.sub(r"\s+", " ", html)
    return html[:limit]


def probe_digest(max_total=90000, per_file=9000):
    parts = []
    if os.path.exists(PROBE_REPORT):
        parts.append("=== probe_report.md ===\n" + open(PROBE_REPORT, encoding="utf-8").read()[:15000])
    if os.path.isdir(PROBE_DIR):
        for fn in sorted(os.listdir(PROBE_DIR)):
            if not fn.endswith(".html"):
                continue
            try:
                raw = open(f"{PROBE_DIR}/{fn}", encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            # credentials never appear in saved pages, but blank any value= that
            # equals the env username just in case
            u = os.environ.get("SITE_USER", "")
            if u:
                raw = raw.replace(u, "<user>")
            parts.append(f"=== {PROBE_DIR}/{fn} (scripts stripped, trimmed) ===\n{strip_html(raw, per_file)}")
    d = "\n\n".join(parts)
    return d[:max_total]


WRITE_SYS = """You write a Python script that automates specific READ-ONLY tasks on one web
portal WITHOUT a browser, using only the provided SafeSession API (module
safe_session). You are given the portal's probe output: a report plus the raw
HTML of the pages reached after login (forms, links, frames, tables).

API (the only network access allowed):
  s = SafeSession(site_url, run_dir)     # created for you as `s`
  s.login() -> "ok" | message
  r = s.get(url_or_href, label="...", **query_params)   # relative hrefs OK
  p = s.parse(r)   # p.forms: [{action, method, fields{name:value}, inputs[...]}],
                   # p.links: [{href,title,id,onclick}], p.frames: [src], p.title, p.text
  s.text(r); s.find(regex, r) -> snippets
  r = s.post_form(form, overrides={...}, submit_name="btnName", label="...")
  r = s.post(url, data={...}, label="...")            # calculate/view/search endpoints only
  s.abs(href, base_url)
  Every response is saved automatically under run_dir; s.log lists them.

Site facts to use: the home page is a FRAMESET; the application lives in the
frames (see frame src values) — navigate into those pages, not the frameset.
Older portals (ASP.NET / JSP) often carry hidden state fields (__VIEWSTATE,
__EVENTTARGET, etc.): always start from a freshly parsed form and override
only the fields you need; set __EVENTTARGET to the control id when a link's
onclick is __doPostBack('id', '').

Hard rules:
  * import only: os, re, json, time, sys, datetime, urllib.parse, safe_session.
    NO requests, playwright, selenium, subprocess, bs4, lxml.
  * Never POST to anything that saves/submits/deletes/approves (the API refuses).
  * Define `def run(s, mission: str, out: dict) -> None`. Fill `out` with
    structured results: for SDPs a list of {leg_or_airfield, text, status, url};
    for calculators a list of {which, inputs_used, results{label:value}, url};
    plus out["issues"] for anything not found. Quote values exactly as shown.
  * If the needed page is not among the probe pages, navigate there via the
    links/frames you can see, and record what you tried in out["issues"].
  * Be defensive: if an element is missing, record it and continue.
Return the COMPLETE file in one ```python block, nothing else."""

FIX_SYS = WRITE_SYS + """

You are now FIXING a previous version. You get the script, the test run output,
and the text of the responses the script received. Make the smallest change
that fixes it and return the complete corrected file."""

HARNESS = '''
# ---- harness (appended by requests_task_writer; do not edit) ----
if __name__ == "__main__":
    import os, json, sys, traceback
    from datetime import datetime
    from safe_session import SafeSession
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"portal_req_run_{stamp}"
    s = SafeSession(os.environ["SITE_URL"], run_dir)
    out = {"issues": []}
    rc = 0
    try:
        msg = s.login()
        print("[login]", msg)
        if msg != "ok":
            out["issues"].append("login: " + msg); rc = 1
        else:
            run(s, os.environ.get("MISSION", ""), out)
    except Exception:
        out["issues"].append(traceback.format_exc()); traceback.print_exc(); rc = 1
    finally:
        s.save_log()
    json.dump(out, open(f"portal_results_{stamp}.json", "w"), indent=1)
    md = [f"# Portal results {stamp}", "", f"Mission: {os.environ.get('MISSION','')}", "",
          "> Read-only run: GET requests plus calculate/view POSTs only; nothing saved on the portal.", "",
          "```json", json.dumps(out, indent=1), "```", "", "## Requests", ""]
    md += [f"- {e['method']} {e['status']} {e['url']}  ({e['label']})" for e in s.log]
    open(f"portal_results_{stamp}.md", "w").write("\\n".join(md))
    print(f"[done] portal_results_{stamp}.md  (responses in {run_dir}/)")
    sys.exit(rc)
'''

FORBIDDEN = re.compile(r"^\s*(import|from)\s+(requests|playwright|selenium|subprocess|httpx|bs4|lxml|urllib\.request)\b", re.M)


def extract_code(reply):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", reply, re.S)
    return (m.group(1) if m else reply).rstrip() + "\n"


def validate(code):
    if FORBIDDEN.search(code):
        return "forbidden import"
    if "def run(" not in code:
        return "missing def run(s, mission, out)"
    if re.search(r"\b_s\b\.(post|put|delete|patch)", code):
        return "bypasses SafeSession guards via _s"
    try:
        compile(code, GEN, "exec")
    except SyntaxError as e:
        return f"syntax error: {e}"
    return None


def latest_run_texts(limit=14000):
    runs = sorted(d for d in os.listdir(".") if d.startswith("portal_req_run_"))
    if not runs:
        return ""
    d = runs[-1]
    files = sorted(f for f in os.listdir(d) if f.endswith((".html", ".json")))[-5:]
    out = []
    for f in files:
        raw = open(f"{d}/{f}", encoding="utf-8", errors="ignore").read()
        out.append(f"--- {d}/{f} ---\n{strip_html(raw, 2500)}")
    return "\n\n".join(out)[:limit]


def test(mission, timeout):
    env = dict(os.environ, MISSION=mission)
    print(f"\n[test] python {GEN}")
    try:
        p = subprocess.run([sys.executable, GEN], capture_output=True, text=True, timeout=timeout, env=env)
        out = (p.stdout or "") + ("\n--- STDERR ---\n" + p.stderr if p.stderr else "")
        rc = p.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = f"TIMEOUT after {timeout}s", 124
    print(out[-3000:])
    return rc, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--skip-write", action="store_true", help="test/fix the existing generated file")
    args = ap.parse_args()
    for v in ("SITE_URL", "SITE_USER", "SITE_PASS"):
        if not os.environ.get(v):
            sys.exit(f"[FATAL] set {v}")
    if not os.path.exists(PROBE_REPORT):
        sys.exit("[FATAL] probe_report.md not found — run portal_probe.py first")
    client, model = make_client()
    digest = probe_digest()
    print(f"[probe] digest {len(digest):,} chars from {PROBE_REPORT} + {PROBE_DIR}/")

    if not (args.skip_write and os.path.exists(GEN)):
        print(f"[write] asking {model} ...")
        code = extract_code(chat(client, model, WRITE_SYS,
                                 f"MISSION: {args.mission}\nTASKS:\n{args.task}\n\nPROBE OUTPUT:\n{digest}"))
        err = validate(code)
        if err:
            print(f"[write] rejected ({err}); retrying once")
            code = extract_code(chat(client, model, WRITE_SYS,
                                     f"MISSION: {args.mission}\nTASKS:\n{args.task}\n"
                                     f"PREVIOUS ATTEMPT REJECTED: {err}\n\nPROBE OUTPUT:\n{digest}"))
            err = validate(code)
            if err:
                sys.exit(f"[FATAL] generated code rejected again: {err}")
        open(GEN, "w", encoding="utf-8").write(code + HARNESS)
        print(f"[write] wrote {GEN}")

    for rnd in range(1, args.rounds + 1):
        rc, out = test(args.mission, args.timeout)
        results = sorted(f for f in os.listdir(".") if f.startswith("portal_results_") and f.endswith(".json"))
        issues = []
        if results:
            try:
                issues = json.load(open(results[-1])).get("issues", [])
            except Exception:
                pass
        if rc == 0 and not issues:
            print(f"\n[ok] {GEN} ran clean -> {results[-1] if results else '?'}")
            return
        print(f"\n[fix {rnd}/{args.rounds}] rc={rc} issues={len(issues)} — asking {model}")
        code = open(GEN, encoding="utf-8").read().split("# ---- harness")[0]
        new = extract_code(chat(client, model, FIX_SYS,
                                f"MISSION: {args.mission}\nTASKS:\n{args.task}\n\nCURRENT SCRIPT:\n```python\n{code}\n```\n\n"
                                f"TEST OUTPUT (tail):\n{out[-5000:]}\nISSUES:\n{json.dumps(issues)[:3000]}\n\n"
                                f"RESPONSES RECEIVED (stripped):\n{latest_run_texts()}\n\n"
                                f"PROBE OUTPUT (abridged):\n{digest[:40000]}"))
        err = validate(new)
        if err:
            print(f"[fix] rejected: {err}"); continue
        os.replace(GEN, f"{GEN}.bak.{rnd}")
        open(GEN, "w", encoding="utf-8").write(new + HARNESS)
        print(f"[fix] applied (backup {GEN}.bak.{rnd})")
    print(f"\n[done] not clean after {args.rounds} rounds — read {GEN}, the latest portal_results_*.json "
          f"'issues', and portal_req_run_*/ responses.")


if __name__ == "__main__":
    main()
