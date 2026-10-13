#!/usr/bin/env python3
"""
site_learner.py — LEARN the portal, WRITE a site-specific script, TEST it, FIX it.

  Phase 1  LEARN   logs in (safe_browser), explores the site read-only (links,
                   tabs, searches for the mission), saving each page's text,
                   element inventory and screenshot -> site_map/ + site_map.json
  Phase 2  WRITE   the gateway model (Opus / GPT 5.6) reads the site map and
                   your task description and writes portal_tasks_generated.py,
                   which may ONLY use safe_browser.SafeBrowser (guarded clicks:
                   Save/Submit/Delete are impossible even if the code tries).
  Phase 3  TEST    runs the generated script; captures its output.
  Phase 4  FIX     on failure, sends the script + error + latest screenshots'
                   page text back to the model for a corrected version.
                   Repeats up to --rounds.

The result is a readable, editable script for YOUR portal — rerun it any time
with: python portal_tasks_generated.py

Env: LITELLM_BASE_URL, LITELLM_API_KEY, AGENT_MODEL, AGENT_TLS_VERIFY
     SITE_URL, SITE_USER, SITE_PASS  (+ optional *_SEL overrides)

Usage:
  python site_learner.py --mission "KSUU to FKKD C-17 2026-10-20" \\
      --task "1) open Enter/Confirm/Update Standard Departure Procedures and capture
              every leg's SDP text and status. 2) click the calculator icon next to
              each Alternate BO, capture the result, then click the Main BO
              calculator and capture fuel and weights. Never save." \\
      --explore-pages 40 --rounds 4
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from urllib.parse import urljoin, urldefrag, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_browser import SafeBrowser, NEVER   # noqa: E402

GEN_FILE = "portal_tasks_generated.py"
MAP_DIR = "site_map"

# ----------------------------------------------------------------------------
# LLM
# ----------------------------------------------------------------------------

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


def json_of(txt):
    s, e = txt.find("{"), txt.rfind("}")
    try:
        return json.loads(txt[s:e+1]) if s >= 0 else {}
    except json.JSONDecodeError:
        return {}

# ----------------------------------------------------------------------------
# Phase 1: learn
# ----------------------------------------------------------------------------

EXPLORE_SYS = """You guide a READ-ONLY exploration of a flight-mission web portal so that a
script can later be written for it. Given the current page text and element
inventory, pick up to 6 elements worth opening next to understand the site:
navigation tabs, Missions/Search lists, the specific mission, and anything
about Standard Departure Procedures, Alternates, BO, fuel, weights, calculators.
If a search box is visible and the mission is not yet open, you may fill it.
Return ONLY JSON: {"actions": [{"type": "click"|"fill", "idx": <int>,
 "value": "<for fill>", "enter": true/false, "why": "..."}],
 "mission_page_open": true/false, "notes": "<what this page is, 1-2 sentences>"}
Never pick elements whose clues contain save, submit, delete, approve, remove,
cancel, logout, reset, send, create, new."""


def learn(br, client, model, mission, max_pages):
    os.makedirs(MAP_DIR, exist_ok=True)
    pages, seen = [], set()
    queue = deque()

    def record(label):
        base = br.snapshot(label)
        items = br.elements()
        key = hashlib.sha1((br.url + "|" + label).encode()).hexdigest()[:10]
        open(f"{MAP_DIR}/{key}_elements.txt", "w", encoding="utf-8").write(SafeBrowser.describe(items))
        rec = {"id": key, "label": label, "url": br.url, "snapshot": base,
               "text_head": br.text()[:1500], "n_elements": len(items),
               "elements_file": f"{MAP_DIR}/{key}_elements.txt"}
        pages.append(rec)
        return items, rec

    items, rec = record("after login")
    mission_open = False
    steps = 0
    while steps < max_pages:
        steps += 1
        user = (f"MISSION OF INTEREST: {mission}\nPAGE: {br.url}\n\nTEXT:\n{br.text()[:5000]}\n\n"
                f"ELEMENTS:\n{SafeBrowser.describe(items)[:14000]}")
        a = json_of(chat(client, model, EXPLORE_SYS, user, 2000))
        rec["notes"] = a.get("notes", "")
        mission_open = mission_open or bool(a.get("mission_page_open"))
        acts = a.get("actions", [])[:6]
        print(f"[learn {steps}] {rec['label'][:40]!r}: {len(acts)} actions; {a.get('notes','')[:80]}")
        if not acts:
            # nothing more here: fall back to breadth-first over unseen same-domain links
            for it in items:
                h = it.get("href")
                if h and it["tag"] == "a":
                    full = urldefrag(urljoin(br.url, h))[0]
                    if urlparse(full).netloc.split(":")[0].endswith(br.domain) and full not in seen \
                            and not NEVER.search(full):
                        queue.append(full)
            if not queue:
                break
            nxt = queue.popleft(); seen.add(nxt)
            if br.goto(nxt) != "ok":
                continue
            items, rec = record(f"link {len(pages)}")
            continue
        progressed = False
        for act in acts:
            idx = act.get("idx")
            if not isinstance(idx, int) or not (0 <= idx < len(items)):
                continue
            label = items[idx].get("text") or items[idx].get("title") or items[idx].get("href") or str(idx)
            if act.get("type") == "fill":
                r = br.fill(idx, act.get("value", mission), bool(act.get("enter")))
            else:
                r = br.click(idx, act.get("why", ""))
            print(f"    {act.get('type')} [{idx}] {label[:40]!r} -> {r}")
            if r == "ok":
                items, rec = record(f"{act.get('type')} {label[:30]}")
                progressed = True
                if br.url not in seen:
                    seen.add(br.url)
                if act.get("type") == "click" and len(acts) > 1:
                    br.back(); items = br.elements()
        if not progressed:
            break
    json.dump({"mission": mission, "site": br.site_url, "mission_page_seen": mission_open,
               "pages": pages, "actions": br.actions},
              open(f"{MAP_DIR}/site_map.json", "w"), indent=1)
    print(f"[learn] {len(pages)} page states recorded in {MAP_DIR}/ (mission page seen: {mission_open})")
    return pages

# ----------------------------------------------------------------------------
# Phase 2/4: write or fix the script
# ----------------------------------------------------------------------------

WRITE_SYS = """You write a Python script that automates specific tasks on ONE web portal,
using ONLY the provided SafeBrowser API (module safe_browser). You are given a
map of the site: for each page state, its URL, a text excerpt, and the clickable
elements with their text/title/alt/icon filenames/ids/hrefs.

API (the only browser access allowed):
  br = SafeBrowser(site_url, run_dir, headless)      # already created for you as `br`
  br.login() -> "ok" | message
  br.goto(url); br.back()
  br.text(); br.html(); br.find("phrase") -> list of snippets
  items = br.elements(); SafeBrowser.describe(items)  # inventory with idx
  br.click(idx_or_item, why="")  -> "ok" | "refused: ..." | "click failed: ..."
  br.click_text("visible text")  -> "ok" | "not found" | "refused"
  br.fill(idx_or_item, value, enter=False)
  br.snapshot("label")           # screenshot + text into run_dir
Choosing elements: call br.elements() on the live page and select by matching
title/alt/src/text/ctx (row context) with Python, e.g. all items whose
src or title contains 'calc' and whose ctx contains 'Alternate'. Never hardcode
idx numbers from the site map — they change; hardcode the CLUES.

Hard rules:
  * import only: os, re, json, time, sys, datetime, safe_browser. NO playwright,
    requests, selenium, subprocess.
  * Never attempt Save/Submit/Delete/Approve (the API refuses them anyway).
  * Define `def run(br, mission: str, out: dict) -> None` that performs the tasks
    and fills `out` with structured results (quoted values + the URL they came
    from). The harness creates `br`, calls run(), and writes out to JSON/MD.
  * Call br.snapshot() before and after every important click.
  * Be defensive: if an element is not found, record that in out["issues"] and
    continue to the next task instead of crashing.
Return the COMPLETE file in one ```python block, nothing else."""

FIX_SYS = WRITE_SYS + """

You are now FIXING a previous version: you get the script, the error/output of
the test run, and the text of the pages the browser was on. Make the smallest
change that fixes it and return the complete corrected file."""

HARNESS = '''
# ---- harness (appended by site_learner; do not edit) ----
if __name__ == "__main__":
    import os, json, sys, traceback
    from datetime import datetime
    from safe_browser import SafeBrowser
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"portal_run_{stamp}"
    br = SafeBrowser(os.environ["SITE_URL"], run_dir, headless=os.environ.get("HEADLESS", "0") == "1")
    out = {"issues": []}
    try:
        print("[login]", br.login())
        run(br, os.environ.get("MISSION", ""), out)
    except Exception:
        out["issues"].append(traceback.format_exc())
        traceback.print_exc()
        rc = 1
    else:
        rc = 0
    finally:
        br.snapshot("final")
        br.save_log()
        br.close()
    json.dump(out, open(f"portal_results_{stamp}.json", "w"), indent=1)
    md = [f"# Portal results {stamp}", "", f"Mission: {os.environ.get('MISSION','')}", "",
          "```json", json.dumps(out, indent=1), "```"]
    open(f"portal_results_{stamp}.md", "w").write("\\n".join(md))
    print(f"[done] portal_results_{stamp}.md  (screenshots in {run_dir}/)")
    sys.exit(rc if out.get("issues") and rc else 0)
'''

FORBIDDEN_IMPORTS = re.compile(r"^\s*(import|from)\s+(playwright|requests|selenium|subprocess|httpx|urllib)",
                               re.M)


def site_map_digest(pages, max_chars=60000):
    parts = []
    for p in pages:
        try:
            els = open(p["elements_file"], encoding="utf-8").read()
        except Exception:
            els = ""
        parts.append(f"=== PAGE STATE: {p['label']} ===\nURL: {p['url']}\nNOTES: {p.get('notes','')}\n"
                     f"TEXT HEAD:\n{p['text_head'][:1200]}\nELEMENTS:\n{els[:4000]}\n")
    d = "\n".join(parts)
    return d[:max_chars]


def extract_code(reply):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", reply, re.S)
    return (m.group(1) if m else reply).rstrip() + "\n"


def validate(code):
    if FORBIDDEN_IMPORTS.search(code):
        return "forbidden import (playwright/requests/selenium/subprocess/httpx/urllib)"
    if "def run(" not in code:
        return "missing def run(br, mission, out)"
    if re.search(r"\b_page\b|\b_browser\b|\b_ctx\b", code):
        return "accesses private browser internals"
    try:
        compile(code, GEN_FILE, "exec")
    except SyntaxError as e:
        return f"syntax error: {e}"
    return None


def write_script(client, model, pages, mission, task):
    user = (f"MISSION: {mission}\nTASKS:\n{task}\n\nSITE MAP:\n{site_map_digest(pages)}")
    return extract_code(chat(client, model, WRITE_SYS, user))


def fix_script(client, model, code, output, pages, mission, task, latest_texts):
    user = (f"MISSION: {mission}\nTASKS:\n{task}\n\nCURRENT SCRIPT:\n```python\n{code}\n```\n\n"
            f"TEST RUN OUTPUT (tail):\n{output[-6000:]}\n\n"
            f"PAGE TEXT FROM THE LAST SNAPSHOTS:\n{latest_texts[:12000]}\n\n"
            f"SITE MAP (abridged):\n{site_map_digest(pages, 30000)}")
    return extract_code(chat(client, model, FIX_SYS, user))


def test_script(mission, timeout, headless):
    env = dict(os.environ, MISSION=mission, HEADLESS="1" if headless else "0")
    print(f"\n[test] python {GEN_FILE}")
    try:
        p = subprocess.run([sys.executable, GEN_FILE], capture_output=True, text=True,
                           timeout=timeout, env=env)
        out = (p.stdout or "") + ("\n--- STDERR ---\n" + p.stderr if p.stderr else "")
        rc = p.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = f"TIMEOUT after {timeout}s\n{e.stdout or ''}", 124
    print(out[-3000:])
    return rc, out


def latest_run_texts():
    runs = sorted([d for d in os.listdir(".") if d.startswith("portal_run_")])
    if not runs:
        return ""
    d = runs[-1]
    txts = sorted(f for f in os.listdir(d) if f.endswith(".txt"))[-4:]
    return "\n\n".join(f"--- {f} ---\n" + open(f"{d}/{f}", encoding="utf-8").read()[:3000] for f in txts)

# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", required=True)
    ap.add_argument("--task", required=True, help="what the generated script must do")
    ap.add_argument("--explore-pages", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=4, help="write/test/fix rounds")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--skip-learn", action="store_true", help="reuse existing site_map/")
    ap.add_argument("--skip-write", action="store_true", help="test/fix the existing generated file")
    args = ap.parse_args()
    if not os.environ.get("SITE_URL"):
        sys.exit("[FATAL] set SITE_URL (+ SITE_USER / SITE_PASS)")
    client, model = make_client()

    # Phase 1
    if args.skip_learn and os.path.exists(f"{MAP_DIR}/site_map.json"):
        pages = json.load(open(f"{MAP_DIR}/site_map.json"))["pages"]
        print(f"[learn] reusing {len(pages)} page states from {MAP_DIR}/")
    else:
        br = SafeBrowser(os.environ["SITE_URL"], f"{MAP_DIR}/shots", headless=args.headless)
        try:
            msg = br.login(); print(f"[login] {msg}")
            if msg.startswith("login blocked"):
                sys.exit("[FATAL] " + msg + " — automation cannot pass MFA/CAC")
            pages = learn(br, client, model, args.mission, args.explore_pages)
        finally:
            br.close()

    # Phase 2
    if not (args.skip_write and os.path.exists(GEN_FILE)):
        print(f"\n[write] asking {model} to write {GEN_FILE} ...")
        code = write_script(client, model, pages, args.mission, args.task)
        err = validate(code)
        if err:
            print(f"[write] rejected: {err}; asking once more")
            code = write_script(client, model, pages, args.mission,
                                args.task + f"\n\nPREVIOUS ATTEMPT WAS REJECTED: {err}")
            err = validate(code)
            if err:
                sys.exit(f"[FATAL] generated code rejected again: {err}")
        open(GEN_FILE, "w", encoding="utf-8").write(code + HARNESS)
        print(f"[write] wrote {GEN_FILE}")

    # Phase 3/4
    for rnd in range(1, args.rounds + 1):
        rc, out = test_script(args.mission, args.timeout, args.headless)
        results = sorted(f for f in os.listdir(".") if f.startswith("portal_results_") and f.endswith(".json"))
        issues = []
        if results:
            try:
                issues = json.load(open(results[-1])).get("issues", [])
            except Exception:
                pass
        if rc == 0 and not issues:
            print(f"\n[ok] {GEN_FILE} ran clean. Results: {results[-1] if results else '?'}")
            return
        print(f"\n[fix {rnd}/{args.rounds}] rc={rc}, issues={len(issues)} — asking {model} for a fix")
        code = open(GEN_FILE, encoding="utf-8").read().split("# ---- harness")[0]
        new = fix_script(client, model, code, out + "\nISSUES:\n" + json.dumps(issues)[:3000],
                         pages, args.mission, args.task, latest_run_texts())
        err = validate(new)
        if err:
            print(f"[fix] rejected: {err}"); continue
        bak = f"{GEN_FILE}.bak.{rnd}"
        os.replace(GEN_FILE, bak)
        open(GEN_FILE, "w", encoding="utf-8").write(new + HARNESS)
        print(f"[fix] applied (backup {bak})")
    print(f"\n[done] still not clean after {args.rounds} rounds — open {GEN_FILE}, the latest "
          f"portal_run_*/ screenshots, and portal_results_*.json 'issues' to see what remains.")


if __name__ == "__main__":
    main()
