#!/usr/bin/env python3
"""
site_recon_agent.py — standalone, READ-ONLY website research agent.

Give it a website + login + a mission (origin, destination, aircraft, optional
stops/date). It:

  PHASE 1  CRAWL   logs in, then systematically visits every reachable page/tab
                   on the site's domain (breadth-first, capped), saving each
                   page's text locally. Never clicks anything whose label or URL
                   suggests a write action (save, submit, calculate, update,
                   delete, approve, logout, ...).
  PHASE 2  EXTRACT the LLM (your LiteLLM gateway model, e.g. Opus) reads every
                   saved page in chunks and records every fact relevant to the
                   mission entities and topics — quoted verbatim with URL.
  PHASE 3  TARGET  the LLM browses with a read-only tool set to fill gaps
                   (e.g. use the site's search for an airfield code, open a
                   detail page the crawl could not reach).
  PHASE 4  REPORT  writes <out>.md (and .json): mission context, site map,
                   findings per topic / airfield / country, weather, unresolved
                   items, and a full visit log.

No dependency on the route planner. Optionally --run last_run_blocks.json
pre-fills stops/aircraft/date from a planner run.

Environment
  LITELLM_BASE_URL, LITELLM_API_KEY, AGENT_MODEL, AGENT_TLS_VERIFY (gateway)
  SITE_URL, SITE_USER, SITE_PASS                                    (website)
  optional: SITE_USER_SEL, SITE_PASS_SEL, SITE_SUBMIT_SEL, SITE_SEARCH_SEL

Install
  pip install --user playwright openai httpx
  playwright install chromium

Usage
  python site_recon_agent.py --origin KSUU --dest FKKD --aircraft C-17 \\
      --stops TJSJ,SBFZ --date 2026-10-20 --max-pages 150 --headless
  python site_recon_agent.py --run last_run_blocks.json --headless
  python site_recon_agent.py --origin KSUU --dest FKKD --aircraft C-17 \\
      --topics-file my_topics.txt         # one topic per line, replaces defaults
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import deque
from datetime import datetime
from urllib.parse import urljoin, urlparse, urldefrag

# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------

DEFAULT_TOPICS = [
    "Airfield data: runway length, width, surface, weight bearing capacity (PCN/LCN), elevation",
    "Standard Departure Procedures (SDPs) and standard arrival procedures",
    "Requirements, alternates, and alternate airfields",
    "Fuel: availability, type, fuel calculations, fuel/weight planning guidance",
    "C-17 Extended Range (ER) aircraft requirements",
    "Aircraft suitability and restrictions by type (C-17, C-130, KC-135, etc.)",
    "Prior Permission Required (PPR), slot times, operating hours, quiet hours",
    "Diplomatic clearance, lead times, overflight and landing permissions",
    "Customs, immigration, agriculture, and health requirements",
    "Weather: forecasts, climatology, seasonal hazards, METAR/TAF links",
    "NOTAMs, airfield closures, construction, hazards",
    "Ground handling, MOG, parking, towing, ARFF, services, points of contact",
    "HazMat, cargo, and special handling restrictions",
    "Payment: AIR Card acceptance, cash requirements, landing and handling fees",
    "Security, threats, force protection notes",
]

# NEVER clicked in any mode — destructive or persisting actions
NEVER = re.compile(
    r"(delete|remove|purge|clear all|save|submit|approve|reject|cancel request|"
    r"logout|log out|sign out|signout|upload|create|new request|modify|confirm|"
    r"send|post|reset|change password|unsubscribe|purchase|pay|order|checkout|"
    r"archive|discard|overwrite|finalize|release)", re.I)
# refused in read-only mode, allowed with --allow-calculate
CALC = re.compile(r"(calculat|compute|recalc|update|edit|enter)", re.I)
ALLOW_CALCULATE = False          # set by --allow-calculate
CONFIRM_EACH = False             # set by --confirm (y/n before every calculate click)
STOP_FILE = "STOP"               # create this file to stop gracefully


class StopRequested(Exception):
    pass


def check_stop():
    if os.path.exists(STOP_FILE):
        raise StopRequested("STOP file present")


def DANGER_search(text):
    """True if clicking/opening `text` is not allowed in the current mode."""
    if NEVER.search(text):
        return True
    if not ALLOW_CALCULATE and CALC.search(text):
        return True
    return False


class _D:                        # keeps existing DANGER.search(...) call sites working
    @staticmethod
    def search(t):
        return DANGER_search(t)
DANGER = _D()

# ----------------------------------------------------------------------------
# LLM client
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


def llm_json(client, model, system, user, max_tokens=4000):
    for attempt in (1, 2):
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        txt = r.choices[0].message.content or ""
        s, e = txt.find("{"), txt.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.loads(txt[s:e + 1])
            except json.JSONDecodeError:
                pass
        user = user + "\n\nReply with the JSON object only."
    return {}

# ----------------------------------------------------------------------------
# Read-only browser
# ----------------------------------------------------------------------------

class ReadOnlyBrowser:
    def __init__(self, site_url, headless=True, max_chars=7000):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        launch_kw = {"headless": headless}
        proxy = os.environ.get("SITE_PROXY") or os.environ.get("HTTPS_PROXY") \
            or os.environ.get("https_proxy")
        if proxy and os.environ.get("SITE_USE_PROXY", "false").lower() in ("1", "true", "yes"):
            launch_kw["proxy"] = {"server": proxy}
        self._browser = self._pw.chromium.launch(**launch_kw)
        self.ctx = self._browser.new_context(
            ignore_https_errors=os.environ.get("SITE_IGNORE_TLS", "true").lower()
            in ("1", "true", "yes"))
        self.page = self.ctx.new_page()
        self.page.set_default_timeout(int(os.environ.get("SITE_TIMEOUT_MS", "60000")))
        self.site_url = site_url
        self.domain = urlparse(site_url).netloc.split(":")[0]
        self.max_chars = max_chars
        self.visit_log = []
        self.pages_visited = 0

    def close(self):
        try:
            self._browser.close(); self._pw.stop()
        except Exception:
            pass

    def same_domain(self, url):
        return urlparse(url).netloc.split(":")[0].endswith(self.domain)

    def _log(self, note):
        self.visit_log.append({"url": self.page.url, "note": note,
                               "time": datetime.now().isoformat(timespec="seconds")})

    # ---- the single write action: login with env credentials
    def login(self):
        user, pw = os.environ.get("SITE_USER"), os.environ.get("SITE_PASS")
        try:
            self.page.goto(self.site_url, wait_until="commit",
                           timeout=int(os.environ.get("SITE_TIMEOUT_MS", "60000")))
            try:
                self.page.wait_for_load_state("domcontentloaded", timeout=30000)
            except Exception:
                pass   # some portals never fire it; proceed with what rendered
        except Exception as e:
            msg = str(e).split("\n")[0][:300]
            sys.exit(f"[FATAL] cannot open {self.site_url}: {msg}\n"
                     f"  - Can this machine reach the site in a normal browser?\n"
                     f"  - Behind a proxy? set SITE_USE_PROXY=true (uses HTTPS_PROXY)\n"
                     f"  - Slow site? raise SITE_TIMEOUT_MS (default 60000)\n"
                     f"  - Try without --headless to watch what happens")
        self.pages_visited += 1
        if not (user and pw):
            self._log("no credentials; continuing unauthenticated")
            return "no credentials set; continuing without login"
        usel = os.environ.get("SITE_USER_SEL") or (
            "input[type=email], input[name*=user i], input[id*=user i], "
            "input[name*=login i], input[id*=login i], input[type=text]")
        psel = os.environ.get("SITE_PASS_SEL") or "input[type=password]"
        ssel = os.environ.get("SITE_SUBMIT_SEL")
        try:
            if self.page.locator(psel).count() == 0:
                self._log("no password field on landing page")
                return f"no login form found at {self.page.url}; continuing"
            self.page.locator(usel).first.fill(user)
            self.page.locator(psel).first.fill(pw)
            if ssel:
                self.page.locator(ssel).first.click()
            else:
                self.page.locator(psel).first.press("Enter")
            try:
                self.page.wait_for_load_state("networkidle", timeout=20000)
            except Exception:
                pass
        except Exception as e:
            self._log(f"login error {e.__class__.__name__}")
            return f"login failed: {e.__class__.__name__}: {str(e)[:200]}"
        low = self.text()[:2000].lower()
        if any(k in low for k in ("multi-factor", "one-time code", "authenticator",
                                  "captcha", "smart card", "common access card")):
            self._log("blocked by MFA/CAC/CAPTCHA")
            return "login blocked: MFA/CAC/CAPTCHA required"
        if self.page.locator(psel).count() > 0 and "invalid" in low:
            return "login failed: credentials rejected"
        self._log("logged in")
        return f"logged in; now at {self.page.url}"

    # ---- reading
    def text(self):
        try:
            t = self.page.inner_text("body")
        except Exception:
            t = ""
        # include iframes' text when accessible
        for fr in self.page.frames[1:]:
            try:
                t += "\n\n[frame] " + fr.inner_text("body")
            except Exception:
                pass
        return re.sub(r"\n{3,}", "\n\n", t).strip()

    def links(self):
        out = []
        try:
            for a in self.page.locator("a[href]").all()[:300]:
                try:
                    h = a.get_attribute("href") or ""
                    t = (a.inner_text() or "").strip()[:90]
                    if h and not h.startswith(("javascript:", "mailto:", "tel:", "#")):
                        out.append((t, urldefrag(urljoin(self.page.url, h))[0]))
                except Exception:
                    continue
        except Exception:
            pass
        return out

    def tab_labels(self):
        """Clickable tab-like controls that are not links (JS tabs)."""
        labels = []
        try:
            for sel in ("[role=tab]", "button", "li.nav-item", ".tab", ".nav-link"):
                for el in self.page.locator(sel).all()[:80]:
                    try:
                        t = (el.inner_text() or "").strip()
                        if 1 < len(t) < 50 and not DANGER.search(t):
                            labels.append(t)
                    except Exception:
                        continue
        except Exception:
            pass
        seen, uniq = set(), []
        for l in labels:
            if l.lower() not in seen:
                seen.add(l.lower()); uniq.append(l)
        return uniq

    def snapshot(self):
        return {"url": self.page.url, "title": self.page.title(),
                "text": self.text(), "links": self.links()}

    # ---- navigation (read-only)
    def open(self, url):
        check_stop()
        if not url.startswith("http"):
            url = urljoin(self.site_url, url)
        if not self.same_domain(url):
            return f"refused: outside {self.domain}"
        if DANGER.search(urlparse(url).path + "?" + (urlparse(url).query or "")):
            return "refused: URL looks like a write action"
        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            return f"open failed: {e.__class__.__name__}"
        self.pages_visited += 1
        self._log("opened")
        return self.page_view()

    def page_view(self):
        s = self.snapshot()
        body = s["text"]
        out = (f"URL: {s['url']}\nTITLE: {s['title']}\n\nTEXT:\n{body[:self.max_chars]}")
        if len(body) > self.max_chars:
            out += f"\n...[{len(body)-self.max_chars} more chars — use find_in_page]"
        out += "\n\nLINKS:\n" + "\n".join(f"[{t}] -> {h}" for t, h in s["links"][:120])
        tabs = self.tab_labels()
        if tabs:
            out += "\n\nTABS/BUTTONS (click_text): " + " | ".join(tabs[:40])
        return out

    def find_in_page(self, query, context=400):
        body = self.text(); low = body.lower(); q = query.lower()
        hits, i = [], low.find(q)
        while i != -1 and len(hits) < 8:
            hits.append("…" + body[max(0, i-context):i+len(q)+context].replace("\n", " ") + "…")
            i = low.find(q, i + 1)
        self._log(f"find '{query}' ({len(hits)})")
        return "\n---\n".join(hits) if hits else f"'{query}' not found on this page"

    def click_text(self, text):
        check_stop()
        if DANGER.search(text):
            return f"refused: '{text}' looks like a write action"
        loc = self.page.get_by_role("link", name=text, exact=False)
        if loc.count() == 0:
            loc = self.page.get_by_role("tab", name=text, exact=False)
        if loc.count() == 0:
            loc = self.page.get_by_role("button", name=text, exact=False)
        if loc.count() == 0:
            loc = self.page.get_by_text(text, exact=False)
        if loc.count() == 0:
            return f"nothing clickable containing '{text}'"
        href = loc.first.get_attribute("href") or ""
        if href.startswith("http") and not self.same_domain(href):
            return "refused: leaves the site"
        try:
            loc.first.click(timeout=10000)
            self.page.wait_for_load_state("domcontentloaded")
            time.sleep(0.8)
        except Exception as e:
            return f"click failed: {e.__class__.__name__}"
        self.pages_visited += 1
        self._log(f"clicked '{text}'")
        return self.page_view()

    # ---- calculate mode only: fill a field, press a calculate button ----------
    def fill_field(self, label, value):
        if not ALLOW_CALCULATE:
            return "refused: fill_field needs --allow-calculate"
        check_stop()
        loc = self.page.get_by_label(label, exact=False)
        if loc.count() == 0:
            loc = self.page.get_by_placeholder(label, exact=False)
        if loc.count() == 0:
            loc = self.page.locator(f"input[name*='{label}' i], input[id*='{label}' i], "
                                    f"select[name*='{label}' i], textarea[name*='{label}' i]")
        if loc.count() == 0:
            return f"no field matching '{label}'"
        try:
            el = loc.first
            tag = el.evaluate("e => e.tagName.toLowerCase()")
            if tag == "select":
                el.select_option(label=value)
            else:
                el.fill(str(value))
        except Exception as e:
            return f"fill failed: {e.__class__.__name__}: {str(e)[:150]}"
        self._log(f"filled '{label}'")
        return f"filled '{label}' = {value}"

    def click_calculate(self, text):
        if not ALLOW_CALCULATE:
            return "refused: click_calculate needs --allow-calculate"
        check_stop()
        if NEVER.search(text):
            return f"refused: '{text}' is a persisting/destructive action (never allowed)"
        if not CALC.search(text):
            return f"refused: '{text}' is not a calculate/compute control; use click_text"
        if CONFIRM_EACH:
            ans = input(f"\n[confirm] click '{text}' on {self.page.url}? [y/N] ").strip().lower()
            if ans != "y":
                self._log(f"user declined '{text}'")
                return "user declined this click"
        loc = self.page.get_by_role("button", name=text, exact=False)
        if loc.count() == 0:
            loc = self.page.get_by_text(text, exact=False)
        if loc.count() == 0:
            loc = self.page.locator(f"[title*='{text}' i], [aria-label*='{text}' i], "
                                    f"img[alt*='{text}' i]")
        if loc.count() == 0:
            return f"nothing clickable containing '{text}'"
        try:
            loc.first.click(timeout=10000)
            try:
                self.page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
            time.sleep(1.0)
        except Exception as e:
            return f"click failed: {e.__class__.__name__}"
        self._log(f"CALCULATE '{text}'")
        return self.page_view()

    def go_back(self):
        try:
            self.page.go_back(wait_until="domcontentloaded")
        except Exception:
            pass
        self._log("back")
        return self.page_view()

    def site_search(self, query):
        sel = os.environ.get("SITE_SEARCH_SEL")
        if not sel:
            # try to auto-detect a search box
            for cand in ("input[type=search]", "input[name*=search i]",
                         "input[placeholder*=search i]", "input[id*=search i]"):
                if self.page.locator(cand).count():
                    sel = cand; break
        if not sel:
            return "no search box found on this page (set SITE_SEARCH_SEL)"
        try:
            self.page.locator(sel).first.fill(query)
            self.page.locator(sel).first.press("Enter")
            self.page.wait_for_load_state("domcontentloaded")
            time.sleep(0.8)
        except Exception as e:
            return f"search failed: {e.__class__.__name__}"
        self.pages_visited += 1
        self._log(f"search '{query}'")
        return self.page_view()

# ----------------------------------------------------------------------------
# PHASE 1: systematic crawl
# ----------------------------------------------------------------------------

def crawl(br, max_pages, cache_dir, skip_patterns):
    os.makedirs(cache_dir, exist_ok=True)
    start = urldefrag(br.page.url)[0]
    seen, queue, pages = set(), deque([start]), []
    skip_re = re.compile("|".join(skip_patterns), re.I) if skip_patterns else None

    def save(snap, via):
        key = hashlib.sha1(snap["url"].encode()).hexdigest()[:12]
        rec = {"id": key, "url": snap["url"], "title": snap["title"], "via": via,
               "chars": len(snap["text"]), "n_links": len(snap["links"])}
        with open(f"{cache_dir}/{key}.txt", "w", encoding="utf-8") as f:
            f.write(f"URL: {snap['url']}\nTITLE: {snap['title']}\n\n{snap['text']}")
        pages.append(rec)
        return rec

    # the page we landed on after login
    snap = br.snapshot()
    seen.add(start); save(snap, "login landing")
    for t, h in snap["links"]:
        if br.same_domain(h) and h not in seen:
            queue.append(h)
    # JS tabs on the landing page
    for lab in br.tab_labels()[:25]:
        if br.pages_visited >= max_pages:
            break
        r = br.click_text(lab)
        if r.startswith(("refused", "nothing", "click failed")):
            continue
        s2 = br.snapshot()
        key = urldefrag(s2["url"])[0] + "#tab=" + lab
        if key not in seen:
            seen.add(key); save(s2, f"tab '{lab}'")
            for t, h in s2["links"]:
                if br.same_domain(h) and h not in seen:
                    queue.append(h)
    while queue and br.pages_visited < max_pages:
        check_stop()
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        path = urlparse(url).path + "?" + (urlparse(url).query or "")
        if DANGER.search(path) or (skip_re and skip_re.search(url)):
            continue
        if re.search(r"\.(pdf|zip|xlsx?|docx?|pptx?|png|jpe?g|gif|csv)(\?|$)", url, re.I):
            pages.append({"id": None, "url": url, "title": "(file link, not opened)",
                          "via": "crawl", "chars": 0, "n_links": 0})
            continue
        r = br.open(url)
        if r.startswith(("refused", "open failed")):
            continue
        s = br.snapshot()
        save(s, "crawl")
        print(f"[crawl {br.pages_visited}/{max_pages}] {s['title'][:50]!r} "
              f"{len(s['text'])} chars, {len(s['links'])} links")
        for t, h in s["links"]:
            if br.same_domain(h) and h not in seen and len(queue) < 2000:
                queue.append(h)
    json.dump(pages, open(f"{cache_dir}/_index.json", "w"), indent=1)
    print(f"[crawl] saved {len([p for p in pages if p['id']])} pages to {cache_dir}/")
    return pages

# ----------------------------------------------------------------------------
# PHASE 2: extraction from cached pages
# ----------------------------------------------------------------------------

EXTRACT_SYS = """You extract facts from one web page for a military flight planner.
Return ONLY JSON:
{"page_summary": "<=25 words on what this page is",
 "relevant": true/false,
 "findings": [{"entity": "<ICAO code, ISO3 country code, or GENERAL>",
               "topic": "<one of the topic names given, or 'other'>",
               "item": "<what the fact is about>",
               "value": "<the fact exactly as the page states it>",
               "quote": "<verbatim text, <=300 chars>"}]}
Rules: only facts present in the text; quote verbatim; prefer facts about the
mission entities, but also record general procedures, weather, and planning
rules that would apply to any mission. Never invent."""


def extract(client, model, cache_dir, pages, mission, topics, chunk=12000):
    findings, summaries = [], {}
    ents = ", ".join(mission["entities"])
    topic_txt = "\n".join(f"- {t}" for t in topics)
    for p in pages:
        check_stop()
        if not p["id"]:
            continue
        txt = open(f"{cache_dir}/{p['id']}.txt", encoding="utf-8").read()
        body = txt.split("\n\n", 1)[1] if "\n\n" in txt else txt
        if len(body.strip()) < 80:
            continue
        for k in range(0, len(body), chunk):
            piece = body[k:k + chunk]
            user = (f"Mission entities (airports/countries/aircraft): {ents}\n"
                    f"Mission: {mission['desc']}\nTopics of interest:\n{topic_txt}\n\n"
                    f"PAGE URL: {p['url']}\nPAGE TITLE: {p['title']}\n"
                    f"PAGE TEXT (part {k//chunk+1}):\n{piece}")
            ans = llm_json(client, model, EXTRACT_SYS, user)
            if k == 0:
                summaries[p["url"]] = ans.get("page_summary", "")
            for f in ans.get("findings", []) or []:
                f["url"] = p["url"]; f["page_title"] = p["title"]
                findings.append(f)
        print(f"[extract] {p['title'][:50]!r}: {len(findings)} findings so far")
    return findings, summaries

# ----------------------------------------------------------------------------
# PHASE 3: targeted gap-filling with tools
# ----------------------------------------------------------------------------

TOOLS = [
    {"type": "function", "function": {"name": "open", "description": "Open a URL on the site.",
     "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "find_in_page", "description": "Find text on current page.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "click_text", "description":
     "Click a link, tab or button by visible text. Write-action labels are refused.",
     "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "go_back", "description": "Browser back.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "site_search", "description": "Use the site's search box.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "fill_field", "description":
     "(calculate mode only) Type/select a value into a form field found by label/placeholder/name.",
     "parameters": {"type": "object", "properties": {"label": {"type": "string"}, "value": {"type": "string"}},
                    "required": ["label", "value"]}}},
    {"type": "function", "function": {"name": "click_calculate", "description":
     "(calculate mode only) Press a Calculate/Compute control (button or calculator icon) and return the page. "
     "SAVE/SUBMIT/DELETE are always refused.",
     "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "record_finding", "description": "Record a fact (verbatim quote + URL).",
     "parameters": {"type": "object", "properties": {
         "entity": {"type": "string"}, "topic": {"type": "string"}, "item": {"type": "string"},
         "value": {"type": "string"}, "quote": {"type": "string"}, "url": {"type": "string"}},
         "required": ["entity", "topic", "item", "value", "quote", "url"]}}},
    {"type": "function", "function": {"name": "finish", "description": "Done or blocked.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"},
                                                     "unresolved": {"type": "array", "items": {"type": "string"}}},
                    "required": ["summary"]}}},
]

TARGET_SYS = """You are a READ-ONLY agent browsing one website. You may navigate, read,
search and record. Never submit forms, save, calculate, update, approve, upload
or change anything — if a page requires that, stop and report via finish().
Stay on the site's domain. Record facts only via record_finding with a verbatim
quote and URL. If the site lacks something, list it under unresolved. Be
efficient; you have a hard step budget."""


def target(client, model, br, mission, topics, findings, max_steps):
    covered = {}
    for f in findings:
        covered.setdefault(f.get("entity", "?"), set()).add(f.get("topic", "other"))
    gaps = []
    for e in mission["entities"]:
        have = covered.get(e, set())
        missing = [t.split(":")[0] for t in topics if t not in have]
        if missing:
            gaps.append(f"{e}: nothing yet on {', '.join(missing[:6])}")
    calc_txt = ""
    if ALLOW_CALCULATE:
        calc_txt = (f"\nCALCULATE MODE: you may fill requirement/alternate fields with the "
                    f"mission values (origin {mission['origin']}, destination {mission['dest']}, "
                    f"stops {mission['stops']}, aircraft {mission['aircraft']}, date "
                    f"{mission['date']}) using fill_field, and press Calculate controls "
                    f"(e.g. 'Calculate Fuels', calculator icons next to Alternate BO and "
                    f"Main BO) using click_calculate, then record the resulting numbers "
                    f"verbatim. NEVER press SAVE, Submit, Delete, Approve or anything that "
                    f"persists — those are refused by the tools, do not attempt them. "
                    f"Leave the form unsaved when done.\n")
    task = (calc_txt + f"Mission: {mission['desc']}\nEntities: {mission['entities']}\n"
            f"Topics: {[t.split(':')[0] for t in topics]}\n"
            f"A systematic crawl already recorded {len(findings)} facts. Gaps to fill:\n"
            + "\n".join(gaps[:30]) +
            f"\nUse the site's search for airport codes / country names, open detail "
            f"pages, record facts. Budget {max_steps} tool calls. Call finish when done.")
    br.open(br.site_url)
    messages = [{"role": "system", "content": TARGET_SYS},
                {"role": "user", "content": task + "\n\nCURRENT PAGE:\n" + br.page_view()}]
    summary = {"summary": "step budget exhausted", "unresolved": gaps}
    for step in range(max_steps):
        r = client.chat.completions.create(model=model, messages=messages, tools=TOOLS,
                                           tool_choice="auto", max_tokens=1500, temperature=0)
        m = r.choices[0].message
        messages.append({"role": "assistant", "content": m.content or "",
                         "tool_calls": [tc.model_dump() for tc in (m.tool_calls or [])]})
        if not m.tool_calls:
            messages.append({"role": "user", "content": "Use a tool or call finish()."})
            continue
        done = False
        for tc in m.tool_calls:
            name = tc.function.name
            try:
                a = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                a = {}
            print(f"[target {step:02d}] {name} {json.dumps(a)[:100]}")
            if name == "record_finding":
                a["page_title"] = br.page.title(); findings.append(a); res = "recorded"
            elif name == "finish":
                summary, done, res = a, True, "ok"
            elif hasattr(br, name):
                try:
                    res = getattr(br, name)(**a)
                except Exception as e:
                    res = f"tool error: {e.__class__.__name__}"
            else:
                res = "unknown tool"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(res)[:9000]})
        if done:
            break
    return summary

# ----------------------------------------------------------------------------
# PHASE 4: report
# ----------------------------------------------------------------------------

def report(client, model, mission, topics, pages, summaries, findings, tsum,
           visit_log, out_md, out_json, login_msg):
    by_topic, by_entity = {}, {}
    for f in findings:
        by_topic.setdefault(f.get("topic", "other"), []).append(f)
        by_entity.setdefault(f.get("entity", "GENERAL"), []).append(f)

    # LLM executive summary, grounded on the findings list
    brief = ""
    try:
        compact = [{k: f.get(k) for k in ("entity", "topic", "item", "value")}
                   for f in findings][:400]
        ans = llm_json(client, model,
            "Write an executive summary for a flight planner from these recorded "
            "facts. Use ONLY the facts given. Return JSON {\"summary\": \"<300-450 "
            "words, plain text, organised by: route-critical items, airfield notes, "
            "fuel/weights, procedures, weather, open questions>\"}",
            f"Mission: {mission['desc']}\nFacts:\n{json.dumps(compact)}", max_tokens=1500)
        brief = ans.get("summary", "")
    except Exception as e:
        brief = f"(summary unavailable: {e})"

    L = [f"# Site reconnaissance report — {mission['desc']}", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M}. Source site: {os.environ.get('SITE_URL')}",
         f"Login: {login_msg}", "",
         f"Pages crawled: {len([p for p in pages if p['id']])} · facts recorded: "
         f"{len(findings)} · targeted browsing: {tsum.get('summary','')}", "",
         "> READ-ONLY: this agent only navigated and read pages. Items such as "
         "'Calculate Fuels', 'Enter Requirements', 'SAVE' were read as content, never actioned.",
         "", "## Executive summary", "", brief, ""]

    L += ["## Findings by topic", ""]
    for t in topics + ["other"]:
        name = t.split(":")[0]
        items = by_topic.get(t, []) + (by_topic.get(name, []) if name != t else [])
        if not items:
            continue
        L.append(f"### {name}")
        for f in items:
            L.append(f"- **{f.get('entity','')}** — {f.get('item','')}: {f.get('value','')}")
            L.append(f"  - quote: \"{str(f.get('quote',''))[:300]}\"")
            L.append(f"  - source: {f.get('url','')}")
        L.append("")

    L += ["## Findings by airfield / country", ""]
    order = mission["entities"] + sorted(k for k in by_entity if k not in mission["entities"])
    for e in order:
        if e not in by_entity:
            L.append(f"### {e}\n- not found on site\n")
            continue
        L.append(f"### {e}")
        for f in by_entity[e]:
            L.append(f"- [{f.get('topic','other').split(':')[0]}] {f.get('item','')}: "
                     f"{f.get('value','')}  \n  source: {f.get('url','')}")
        L.append("")

    if tsum.get("unresolved"):
        L += ["## Not found / unresolved"] + [f"- {u}" for u in tsum["unresolved"]] + [""]

    L += ["## Site map (pages visited in crawl)", ""]
    for p in pages:
        L.append(f"- {p['title'] or '(untitled)'} — {p['url']}"
                 + (f"  \n  {summaries.get(p['url'],'')}" if summaries.get(p['url']) else ""))
    L += ["", f"## Visit log ({len(visit_log)} actions)", ""]
    L += [f"- {v['time']} {v['note']} — {v['url']}" for v in visit_log]
    open(out_md, "w", encoding="utf-8").write("\n".join(L))
    json.dump({"mission": mission, "topics": topics, "findings": findings,
               "page_summaries": summaries, "targeted": tsum, "pages": pages,
               "visit_log": visit_log}, open(out_json, "w"), indent=1)
    print(f"[report] {out_md} ({len(findings)} facts)")

# ----------------------------------------------------------------------------

def build_mission(args):
    stops, aircraft, date, origin, dest, countries = [], args.aircraft, args.date, args.origin, args.dest, []
    if args.run and os.path.exists(args.run):
        b = json.load(open(args.run))
        stops = [s for s in b.get("stops", []) if not str(s).startswith("CENTROID")]
        aircraft = aircraft or (b.get("aircraft") or {}).get("type")
        date = date or b.get("date")
        if stops:
            origin, dest = origin or stops[0], dest or stops[-1]
        countries = sorted({c for cor in b.get("corridors", []) for c in cor["chosen"]["clearances"]}
                           | set(b.get("stop_countries", [])))
    if args.stops:
        stops = [s.strip().upper() for s in args.stops.split(",")]
    if not (origin and dest):
        sys.exit("[FATAL] give --origin and --dest (or --run with a planner file)")
    ents = []
    for e in [origin] + stops + [dest] + countries + ([aircraft] if aircraft else []):
        if e and e not in ents:
            ents.append(e)
    desc = f"{origin} -> {' -> '.join(s for s in stops if s not in (origin, dest))}{' -> ' if stops else ''}{dest}"
    desc = re.sub(r"(-> )+->", "->", desc).replace("->  ->", "->")
    desc += f", aircraft {aircraft or 'unspecified'}, date {date or 'unspecified'}"
    return {"origin": origin, "dest": dest, "stops": stops, "aircraft": aircraft,
            "date": date, "countries": countries, "entities": ents, "desc": desc}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin"); ap.add_argument("--dest")
    ap.add_argument("--stops", help="comma-separated intermediate stops")
    ap.add_argument("--aircraft"); ap.add_argument("--date")
    ap.add_argument("--run", default=None, help="optional planner last_run_blocks.json")
    ap.add_argument("--topics-file", default=None)
    ap.add_argument("--max-pages", type=int, default=120)
    ap.add_argument("--max-target-steps", type=int, default=50)
    ap.add_argument("--skip", default="", help="comma-separated URL patterns to skip")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-extract", action="store_true", help="crawl only (no LLM)")
    ap.add_argument("--out", default=None, help="output .md path")
    ap.add_argument("--allow-calculate", action="store_true",
                    help="let the agent fill requirement fields and press Calculate "
                         "controls (never Save/Submit/Delete)")
    ap.add_argument("--confirm", action="store_true",
                    help="ask y/N in the terminal before every Calculate click")
    args = ap.parse_args()
    ALLOW_CALCULATE = args.allow_calculate
    CONFIRM_EACH = args.confirm
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)
    print("[mode] " + ("CALCULATE allowed (Save/Submit/Delete always refused)"
                       if ALLOW_CALCULATE else "READ-ONLY")
          + f" — stop anytime with Ctrl-C or by creating a file named '{STOP_FILE}'")
    if not os.environ.get("SITE_URL"):
        sys.exit("[FATAL] set SITE_URL")
    mission = build_mission(args)
    topics = DEFAULT_TOPICS
    if args.topics_file:
        topics = [l.strip() for l in open(args.topics_file) if l.strip()]
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_md = args.out or f"site_report_{mission['origin']}_{mission['dest']}_{stamp}.md"
    out_json = out_md.rsplit(".", 1)[0] + ".json"
    cache_dir = f"site_cache_{mission['origin']}_{mission['dest']}_{stamp}"
    print(f"[mission] {mission['desc']}\n[entities] {mission['entities']}")

    client = model = None
    if not args.no_extract:
        client, model = make_client()
    br = ReadOnlyBrowser(os.environ["SITE_URL"], headless=args.headless)
    findings, summaries, pages = [], {}, []
    tsum, login_msg = {"summary": "skipped"}, "not attempted"
    try:
        login_msg = br.login(); print(f"[login] {login_msg}")
        pages = crawl(br, args.max_pages, cache_dir,
                      [p for p in args.skip.split(",") if p])
        if client:
            findings, summaries = extract(client, model, cache_dir, pages, mission, topics)
            if args.max_target_steps > 0 and not login_msg.startswith("login blocked"):
                tsum = target(client, model, br, mission, topics, findings,
                              args.max_target_steps)
    except (KeyboardInterrupt, StopRequested) as e:
        tsum = {"summary": f"STOPPED by user ({e.__class__.__name__}); partial results",
                "unresolved": tsum.get("unresolved", [])}
        print(f"\n[stop] {tsum['summary']} — writing report with what was collected")
    finally:
        br.close()
    if client:
        try:
            report(client, model, mission, topics, pages, summaries, findings, tsum,
                   br.visit_log, out_md, out_json, login_msg)
        except KeyboardInterrupt:
            json.dump({"findings": findings, "pages": pages, "visit_log": br.visit_log},
                      open(out_json, "w"), indent=1)
            print(f"[stop] raw findings saved to {out_json}")
    else:
        print(f"[done] crawl only; pages in {cache_dir}/")
