#!/usr/bin/env python3
"""
airfield_lookup_agent.py — read-only browsing agent that enriches a planner run
with information from a live website.

What it does
  1. Reads the most recent planner run (last_run_blocks.json): stops, countries,
     aircraft, mission date.
  2. Logs into one website with credentials from environment variables.
  3. Lets an LLM (via your LiteLLM gateway, e.g. Opus) navigate the site with a
     READ-ONLY tool set: open, read page, find text, click a link, go back,
     use the site's search. No form submission except the login step.
  4. For each stop airport (and optionally each overflown country) it asks the
     site for the information you specify (--ask), e.g. runway length,
     standard procedures, PPR, fuel, customs hours.
  5. Writes airfield_info.json + airfield_info.md with every fact quoted
     verbatim and its source URL, plus a full visit log.

Environment
  LITELLM_BASE_URL, LITELLM_API_KEY, AGENT_MODEL   (already set for your gateway)
  AGENT_TLS_VERIFY=false                           (if your gateway uses a self-signed cert)
  SITE_URL          login page URL
  SITE_USER         username
  SITE_PASS         password
  SITE_USER_SEL     optional CSS selector of the username field (auto-detected if unset)
  SITE_PASS_SEL     optional CSS selector of the password field
  SITE_SUBMIT_SEL   optional CSS selector of the login button
  SITE_SEARCH_SEL   optional CSS selector of the site's search box (enables the search tool)

Install
  pip install --user playwright openai httpx
  playwright install chromium

Usage
  python airfield_lookup_agent.py --run last_run_blocks.json \\
      --ask "runway length and surface, standard arrival/departure procedures, \\
             PPR requirement, fuel availability, customs hours" \\
      --max-steps 60 --headless
  Add --countries to also look up each overflown country.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from urllib.parse import urlparse

# ----------------------------------------------------------------------------
# LLM client (LiteLLM gateway, OpenAI-compatible)
# ----------------------------------------------------------------------------

def make_client():
    import httpx
    from openai import OpenAI
    base = os.environ.get("LITELLM_BASE_URL")
    key = os.environ.get("LITELLM_API_KEY")
    model = os.environ.get("AGENT_MODEL")
    if not (base and key and model):
        sys.exit("[FATAL] set LITELLM_BASE_URL, LITELLM_API_KEY, AGENT_MODEL")
    verify = os.environ.get("AGENT_TLS_VERIFY", "true").lower() not in ("0", "false", "no")
    return OpenAI(base_url=base, api_key=key,
                  http_client=httpx.Client(verify=verify, timeout=300)), model

# ----------------------------------------------------------------------------
# Read-only browser
# ----------------------------------------------------------------------------

class ReadOnlyBrowser:
    """Playwright wrapper exposing only navigation/reading. The ONLY form
    interaction is login(), which uses env credentials and never exposes them."""

    def __init__(self, site_url, headless=True, max_chars=6000):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless)
        self.page = self._browser.new_page()
        self.site_url = site_url
        self.domain = urlparse(site_url).netloc
        self.max_chars = max_chars
        self.visit_log = []
        self.pages_visited = 0

    def close(self):
        try:
            self._browser.close()
            self._pw.stop()
        except Exception:
            pass

    def _same_domain(self, url):
        return urlparse(url).netloc.endswith(self.domain.split(":")[0])

    def _log(self, note):
        self.visit_log.append({"url": self.page.url, "note": note,
                               "time": datetime.now().isoformat(timespec="seconds")})

    # ---- login (the one write action, credentials never leave this function)
    def login(self):
        user = os.environ.get("SITE_USER")
        pw = os.environ.get("SITE_PASS")
        if not (user and pw):
            return "login skipped: SITE_USER / SITE_PASS not set"
        self.page.goto(self.site_url, wait_until="domcontentloaded")
        self.pages_visited += 1
        usel = os.environ.get("SITE_USER_SEL")
        psel = os.environ.get("SITE_PASS_SEL") or "input[type=password]"
        ssel = os.environ.get("SITE_SUBMIT_SEL")
        try:
            if not usel:
                # username = the text/email input closest before the password field
                usel = ("input[type=email], input[type=text][name*=user i], "
                        "input[name*=user i], input[id*=user i], input[name*=login i], "
                        "input[type=text]")
            self.page.locator(usel).first.fill(user)
            self.page.locator(psel).first.fill(pw)
            if ssel:
                self.page.locator(ssel).first.click()
            else:
                self.page.locator(psel).first.press("Enter")
            self.page.wait_for_load_state("networkidle", timeout=20000)
        except Exception as e:
            self._log(f"login attempt error: {e.__class__.__name__}")
            return f"login failed: {e.__class__.__name__}: {str(e)[:200]}"
        txt = self.page.inner_text("body")[:1500].lower()
        if any(k in txt for k in ("mfa", "multi-factor", "one-time code", "authenticator",
                                  "captcha", "smart card", "cac")):
            self._log("login blocked by MFA/CAC/CAPTCHA")
            return "login blocked: the site requires MFA/CAC/CAPTCHA — stop and report"
        self._log("logged in")
        return f"login submitted; now at {self.page.url}"

    # ---- read-only tools
    def open(self, url):
        if not url.startswith("http"):
            url = self.site_url.rstrip("/") + "/" + url.lstrip("/")
        if not self._same_domain(url):
            return f"refused: {url} is outside {self.domain}"
        self.page.goto(url, wait_until="domcontentloaded")
        self.pages_visited += 1
        self._log("opened")
        return self.read_page()

    def read_page(self):
        try:
            body = self.page.inner_text("body")
        except Exception as e:
            return f"could not read page: {e}"
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        links = []
        for a in self.page.locator("a[href]").all()[:120]:
            try:
                t = a.inner_text().strip()
                h = a.get_attribute("href") or ""
                if t and len(t) < 90:
                    links.append(f"[{t}] -> {h}")
            except Exception:
                continue
        out = (f"URL: {self.page.url}\nTITLE: {self.page.title()}\n\n"
               f"TEXT (first {self.max_chars} chars):\n{body[:self.max_chars]}")
        if len(body) > self.max_chars:
            out += f"\n...[{len(body) - self.max_chars} more chars; use find_in_page]"
        out += "\n\nLINKS (first 120):\n" + "\n".join(links)
        return out

    def find_in_page(self, query, context=400):
        body = self.page.inner_text("body")
        hits, low = [], body.lower()
        i = low.find(query.lower())
        while i != -1 and len(hits) < 8:
            s, e = max(0, i - context), min(len(body), i + len(query) + context)
            hits.append("…" + body[s:e].replace("\n", " ") + "…")
            i = low.find(query.lower(), i + 1)
        self._log(f"searched page for '{query}' ({len(hits)} hits)")
        return "\n---\n".join(hits) if hits else f"'{query}' not found on this page"

    def click_link(self, text):
        loc = self.page.get_by_role("link", name=text, exact=False)
        if loc.count() == 0:
            loc = self.page.get_by_text(text, exact=False)
        if loc.count() == 0:
            return f"no link containing '{text}'"
        href = loc.first.get_attribute("href") or ""
        if href.startswith("http") and not self._same_domain(href):
            return f"refused: link leaves {self.domain}"
        loc.first.click()
        self.page.wait_for_load_state("domcontentloaded")
        self.pages_visited += 1
        self._log(f"clicked '{text}'")
        return self.read_page()

    def go_back(self):
        self.page.go_back(wait_until="domcontentloaded")
        self._log("back")
        return self.read_page()

    def site_search(self, query):
        sel = os.environ.get("SITE_SEARCH_SEL")
        if not sel:
            return "site search not configured (set SITE_SEARCH_SEL)"
        self.page.locator(sel).first.fill(query)
        self.page.locator(sel).first.press("Enter")
        self.page.wait_for_load_state("domcontentloaded")
        self.pages_visited += 1
        self._log(f"site search '{query}'")
        return self.read_page()

# ----------------------------------------------------------------------------
# Tool schema for the LLM
# ----------------------------------------------------------------------------

TOOLS = [
    {"type": "function", "function": {"name": "open", "description":
        "Open a URL on the site (absolute or path) and return its text + links.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}},
                       "required": ["url"]}}},
    {"type": "function", "function": {"name": "read_page", "description":
        "Re-read the current page (text + links).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "find_in_page", "description":
        "Find a word/phrase on the current page; returns surrounding text.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {"name": "click_link", "description":
        "Click a link by its visible text (partial match). Navigation only.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                       "required": ["text"]}}},
    {"type": "function", "function": {"name": "go_back", "description": "Browser back.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "site_search", "description":
        "Use the site's own search box, if configured.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {"name": "record_finding", "description":
        "Record one fact you found. Quote the page text verbatim.",
        "parameters": {"type": "object", "properties": {
            "entity": {"type": "string", "description": "airport ICAO or country code"},
            "item": {"type": "string", "description": "what the fact is about, e.g. runway length"},
            "value": {"type": "string", "description": "the fact, as stated on the site"},
            "quote": {"type": "string", "description": "verbatim text from the page"},
            "url": {"type": "string"}},
            "required": ["entity", "item", "value", "quote", "url"]}}},
    {"type": "function", "function": {"name": "finish", "description":
        "Call when done or blocked. Summarize coverage and anything unresolved.",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"},
                                                         "unresolved": {"type": "array",
                                                                        "items": {"type": "string"}}},
                       "required": ["summary"]}}},
]

SYSTEM = """You are a READ-ONLY research agent browsing one website on behalf of a
flight planner. Rules:
- You may only navigate, read, search and record. Never submit forms, save,
  request, approve, upload, or change settings. If a page needs such an action
  to proceed, stop and report it via finish().
- Stay on the site's domain.
- Record facts ONLY with record_finding(), quoting the page verbatim and giving
  the URL. Never state a fact from memory. If the site does not have it, say
  "not found on site" in the finish() summary's unresolved list.
- Be efficient: use find_in_page before opening more pages; you have a hard
  step budget.
- Credentials are handled by the tool layer; you never see or type them.
"""

# ----------------------------------------------------------------------------
# main loop
# ----------------------------------------------------------------------------

def load_run(path):
    b = json.load(open(path))
    return {"query": b.get("query"), "date": b.get("date"),
            "aircraft": (b.get("aircraft") or {}).get("type"),
            "stops": b.get("stops", []),
            "stop_countries": b.get("stop_countries", []),
            "overflown": sorted({c for cor in b.get("corridors", [])
                                 for c in cor["chosen"]["clearances"]}),
            "auto_stops": b.get("auto_stops", [])}


def run_agent(args):
    client, model = make_client()
    run = load_run(args.run)
    targets = [s for s in run["stops"] if not str(s).startswith("CENTROID")]
    if args.countries:
        targets += run["overflown"]
    print(f"[run] {run['query']} on {run['date']}, aircraft {run['aircraft']}")
    print(f"[run] targets: {targets}")

    br = ReadOnlyBrowser(os.environ["SITE_URL"], headless=args.headless)
    findings, visit_cap = [], args.max_pages
    try:
        login_msg = br.login()
        print(f"[browser] {login_msg}")
        if login_msg.startswith("login blocked") or login_msg.startswith("login failed"):
            summary = {"summary": login_msg, "unresolved": targets}
            write_outputs(run, findings, br.visit_log, summary, args)
            return
        task = (f"Mission: {run['query']} on {run['date']}, aircraft {run['aircraft']}.\n"
                f"Stops (airports): {run['stops']}; auto-inserted: {run['auto_stops']}.\n"
                f"Countries overflown: {run['overflown']}.\n"
                f"Look up, for EACH of these targets: {targets}\n"
                f"the following information: {args.ask}\n"
                f"Start from the current page after login. Budget: {args.max_steps} "
                f"tool calls, {visit_cap} page visits. Record every fact with "
                f"record_finding; call finish when done.")
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": task + "\n\nCURRENT PAGE:\n" + br.read_page()}]
        for step in range(args.max_steps):
            if br.pages_visited >= visit_cap:
                messages.append({"role": "user", "content":
                                 "Page-visit budget reached. Call finish now."})
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS, tool_choice="auto",
                max_tokens=1500, temperature=0)
            msg = resp.choices[0].message
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])]})
            if not msg.tool_calls:
                messages.append({"role": "user", "content":
                                 "Use a tool, or call finish() if done."})
                continue
            done = False
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    a = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    a = {}
                print(f"[{step:02d}] {name} {json.dumps(a)[:120]}")
                if name == "record_finding":
                    findings.append(a)
                    result = f"recorded ({len(findings)} total)"
                elif name == "finish":
                    summary = a
                    done = True
                    result = "finishing"
                elif hasattr(br, name):
                    try:
                        result = getattr(br, name)(**a)
                    except Exception as e:
                        result = f"tool error: {e.__class__.__name__}: {str(e)[:200]}"
                else:
                    result = f"unknown tool {name}"
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": str(result)[:8000]})
            if done:
                break
        else:
            summary = {"summary": "step budget exhausted",
                       "unresolved": [t for t in targets
                                      if not any(f.get("entity") == t for f in findings)]}
    finally:
        br.close()
    write_outputs(run, findings, br.visit_log, summary, args)


def write_outputs(run, findings, visit_log, summary, args):
    out = {"run": run, "asked_for": args.ask, "findings": findings,
           "summary": summary, "visit_log": visit_log,
           "generated": datetime.now().isoformat(timespec="seconds")}
    json.dump(out, open(args.out_json, "w"), indent=1)
    lines = [f"# Airfield information for {run['query']} ({run['date']}, "
             f"{run['aircraft']})", "", f"Requested: {args.ask}", "",
             f"Summary: {summary.get('summary', '')}", ""]
    by_entity = {}
    for f in findings:
        by_entity.setdefault(f.get("entity", "?"), []).append(f)
    for ent in [s for s in run["stops"]] + run["overflown"]:
        if ent not in by_entity:
            continue
        lines.append(f"## {ent}")
        for f in by_entity[ent]:
            lines.append(f"- **{f['item']}**: {f['value']}")
            lines.append(f"  - source: {f['url']}")
            lines.append(f"  - quote: \"{f['quote'][:300]}\"")
        lines.append("")
    if summary.get("unresolved"):
        lines += ["## Not found / unresolved"] + [f"- {u}" for u in summary["unresolved"]]
    lines += ["", f"## Visit log ({len(visit_log)} pages)"] + \
             [f"- {v['time']} {v['url']} — {v['note']}" for v in visit_log]
    open(args.out_md, "w").write("\n".join(lines))
    print(f"[done] {len(findings)} findings -> {args.out_json}, {args.out_md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="last_run_blocks.json")
    ap.add_argument("--ask", required=True,
                    help="what to look up per airfield/country, in plain words")
    ap.add_argument("--countries", action="store_true",
                    help="also look up each overflown country")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--max-pages", type=int, default=40)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--out-json", default="airfield_info.json")
    ap.add_argument("--out-md", default="airfield_info.md")
    args = ap.parse_args()
    if not os.environ.get("SITE_URL"):
        sys.exit("[FATAL] set SITE_URL (and SITE_USER / SITE_PASS)")
    run_agent(args)
