#!/usr/bin/env python3
"""
portal_probe.py — no-browser probe of a portal using python-requests.

Answers one question: can the SDP section and the BO calculators be driven
without a browser (plain HTML forms / simple endpoints), or are they
JavaScript widgets that need a real browser?

What it does (READ-ONLY: GET requests, plus exactly one POST = the login form):
  1. GET the login page; find the login <form>, its action and hidden fields.
  2. POST credentials to it (the only POST). Save the post-login HTML.
  3. Follow same-domain links/menus up to --max-pages, saving each page.
  4. For every saved page, report: forms (action/method/inputs), links,
     calculator-looking controls, SDP-looking text, and how much of the page
     is JS-driven (script count, XHR/fetch hints, frameworks).
  5. Write probe_report.md with a verdict per feature: HTML-automatable or
     browser-needed.

Env: SITE_URL, SITE_USER, SITE_PASS  (HTTPS_PROXY is honoured automatically)
Usage:
  python portal_probe.py --mission "KSUU FKKD" --max-pages 25
"""

import argparse
import json
import os
import re
import sys
from collections import deque
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, urldefrag

import requests

NEVER = re.compile(r"(save|submit|delete|remove|approve|reject|cancel|logout|log ?out|"
                   r"sign ?out|upload|create|new request|discard|finalize|release|send|post|"
                   r"reset|change password|purge)", re.I)
CALC = re.compile(r"(calc|calculat|fuel|weight|\bBO\b|alternate|altn)", re.I)
SDP = re.compile(r"(standard departure|\bSDP\b|departure procedure)", re.I)


class Page(HTMLParser):
    """Minimal HTML parser: forms (with inputs), links, scripts, title, text."""
    def __init__(self):
        super().__init__()
        self.forms, self.links, self.scripts, self.title = [], [], [], ""
        self._form, self._in_title, self._text, self._skip = None, False, [], 0
        self.buttons, self.imgs, self.frames = [], [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "form":
            self._form = {"action": a.get("action", ""), "method": (a.get("method") or "get").upper(),
                          "id": a.get("id", ""), "name": a.get("name", ""), "inputs": []}
        elif tag in ("input", "select", "textarea", "button"):
            rec = {"tag": tag, "type": a.get("type", ""), "name": a.get("name", ""),
                   "id": a.get("id", ""), "value": (a.get("value") or "")[:60],
                   "onclick": (a.get("onclick") or "")[:120], "title": a.get("title", "")}
            (self._form["inputs"] if self._form is not None else self.buttons).append(rec)
        elif tag == "a":
            self.links.append({"href": a.get("href", ""), "onclick": (a.get("onclick") or "")[:120],
                               "title": a.get("title", ""), "id": a.get("id", "")})
        elif tag == "img":
            self.imgs.append({"src": (a.get("src") or "").split("/")[-1], "alt": a.get("alt", ""),
                              "title": a.get("title", ""), "onclick": (a.get("onclick") or "")[:120]})
        elif tag == "script":
            self.scripts.append((a.get("src") or "").split("/")[-1] or "(inline)")
            self._skip += 1
        elif tag in ("style",):
            self._skip += 1
        elif tag in ("iframe", "frame"):
            self.frames.append(a.get("src", ""))
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag == "form" and self._form is not None:
            self.forms.append(self._form); self._form = None
        elif tag in ("script", "style") and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data.strip()
        elif not self._skip:
            t = data.strip()
            if t:
                self._text.append(t)

    @property
    def text(self):
        return "\n".join(self._text)


def parse(html):
    p = Page()
    try:
        p.feed(html)
    except Exception:
        pass
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="", help="words to look for, e.g. mission id or airports")
    ap.add_argument("--max-pages", type=int, default=25)
    ap.add_argument("--out", default="probe_out")
    args = ap.parse_args()
    url, user, pw = os.environ.get("SITE_URL"), os.environ.get("SITE_USER"), os.environ.get("SITE_PASS")
    if not (url and user and pw):
        sys.exit("[FATAL] set SITE_URL, SITE_USER, SITE_PASS")
    os.makedirs(args.out, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "python-requests (CAITT read-only probe)"
    domain = urlparse(url).netloc

    # 1. login page
    r = s.get(url, timeout=60)
    open(f"{args.out}/00_login_page.html", "w", encoding="utf-8").write(r.text)
    lp = parse(r.text)
    print(f"[login page] {r.status_code} title={lp.title!r} forms={len(lp.forms)} scripts={len(lp.scripts)}")
    login_form = None
    for f in lp.forms:
        if any(i["type"] == "password" for i in f["inputs"]):
            login_form = f; break
    if not login_form:
        print("[login] no password form in the HTML — login is JavaScript-driven or on another URL.")
        write_report(args, lp, [], "Login form is not plain HTML; browser needed (or find the real login URL).")
        return
    action = urljoin(r.url, login_form["action"] or r.url)
    data = {}
    user_field = pass_field = None
    for i in login_form["inputs"]:
        if not i["name"]:
            continue
        if i["type"] == "password":
            pass_field = i["name"]
        elif i["type"] in ("text", "email", "") and user_field is None and i["tag"] == "input":
            user_field = i["name"]
        if i["type"] in ("hidden", "submit") or i["tag"] == "select":
            data[i["name"]] = i["value"]
    if not (user_field and pass_field):
        print(f"[login] could not identify fields: {login_form['inputs']}")
        write_report(args, lp, [], "Login form fields unclear; see 00_login_page.html.")
        return
    data[user_field], data[pass_field] = user, pw
    print(f"[login] POST {action}  fields={list(data)}  (user->{user_field}, pass->{pass_field})")

    # 2. the one POST
    r2 = s.post(action, data=data, timeout=60, allow_redirects=True)
    open(f"{args.out}/01_after_login.html", "w", encoding="utf-8").write(r2.text)
    al = parse(r2.text)
    logged_in = (r2.status_code == 200 and not any(i["type"] == "password" for f in al.forms for i in f["inputs"]))
    print(f"[after login] {r2.status_code} {r2.url} title={al.title!r} logged_in={logged_in}")
    if not logged_in:
        write_report(args, lp, [(r2.url, al)], "Login POST did not produce a logged-in page "
                     "(password form still present). Credentials, CSRF token, or JS login.")
        return

    # 3. crawl same-domain GET links
    pages, seen, queue = [(r2.url, al)], {r2.url}, deque()
    for l in al.links:
        h = urldefrag(urljoin(r2.url, l["href"]))[0]
        if l["href"] and not l["href"].startswith(("javascript:", "#", "mailto:")) and \
                urlparse(h).netloc == domain and not NEVER.search(h):
            queue.append(h)
    while queue and len(pages) < args.max_pages:
        h = queue.popleft()
        if h in seen:
            continue
        seen.add(h)
        try:
            rr = s.get(h, timeout=60)
        except Exception as e:
            print(f"[crawl] {h} -> {e.__class__.__name__}"); continue
        pg = parse(rr.text)
        n = len(pages)
        open(f"{args.out}/{n:02d}_{re.sub(r'[^a-z0-9]+', '_', pg.title.lower())[:40] or 'page'}.html",
             "w", encoding="utf-8").write(rr.text)
        pages.append((rr.url, pg))
        print(f"[crawl {n}] {rr.status_code} {pg.title[:50]!r} forms={len(pg.forms)} "
              f"links={len(pg.links)} scripts={len(pg.scripts)} frames={len(pg.frames)}")
        for l in pg.links:
            hh = urldefrag(urljoin(rr.url, l["href"]))[0]
            if l["href"] and not l["href"].startswith(("javascript:", "#", "mailto:")) and \
                    urlparse(hh).netloc == domain and hh not in seen and not NEVER.search(hh):
                queue.append(hh)
    write_report(args, lp, pages, None)


def write_report(args, login_page, pages, early_verdict):
    L = [f"# Portal probe (no browser) — {datetime.now():%Y-%m-%d %H:%M}", "",
         f"Site: {os.environ.get('SITE_URL')}  ·  proxy: {os.environ.get('HTTPS_PROXY', '(none)')}", ""]
    if early_verdict:
        L += ["## Verdict", early_verdict, ""]
    sdp_hits, calc_hits, js_heavy, mission_hits = [], [], [], []
    for url, pg in pages:
        js = len(pg.scripts)
        xhr = bool(re.search(r"(XMLHttpRequest|fetch\(|\$\.ajax|__doPostBack|angular|react|vue|knockout|dojo)",
                             open_html(args, pg)))
        if js > 15 or xhr:
            js_heavy.append(url)
        if SDP.search(pg.text) or any(SDP.search(l.get("title", "") + l.get("href", "")) for l in pg.links):
            sdp_hits.append(url)
        ctrls = [c for c in pg.imgs + pg.buttons + pg.links
                 if CALC.search(" ".join(str(v) for v in c.values()))]
        if ctrls:
            calc_hits.append((url, ctrls[:12]))
        if args.mission and any(w.lower() in pg.text.lower() for w in args.mission.split()):
            mission_hits.append(url)
    L += ["## Pages fetched", ""] + [f"- {pg.title or '(untitled)'} — {url}  (forms {len(pg.forms)}, "
                                      f"scripts {len(pg.scripts)}, frames {len(pg.frames)})"
                                      for url, pg in pages] + [""]
    L += ["## Mission mentions", ""] + ([f"- {u}" for u in mission_hits] or ["- none found in fetched pages"]) + [""]
    L += ["## SDP (Standard Departure Procedures)", ""] + ([f"- {u}" for u in sdp_hits] or ["- not found"]) + [""]
    L += ["## Calculator-like controls", ""]
    for url, ctrls in calc_hits:
        L.append(f"### {url}")
        for c in ctrls:
            L.append("- " + ", ".join(f"{k}={v!r}" for k, v in c.items() if v))
    if not calc_hits:
        L.append("- none found")
    L += ["", "## Forms (what a requests-based script could POST)", ""]
    for url, pg in pages:
        for f in pg.forms:
            names = [i["name"] for i in f["inputs"] if i["name"]]
            flag = "  <-- WRITE-LIKE, never automate" if NEVER.search(f["action"] + " " + f["id"] + " " + f["name"]) else ""
            L.append(f"- {f['method']} {f['action'] or '(self)'} on {url}: {names[:15]}{flag}")
    L += ["", "## JavaScript-heavy pages", ""] + ([f"- {u}" for u in js_heavy] or ["- none"]) + [""]
    # verdict
    if not early_verdict:
        v = []
        v.append("SDP section: " + ("found in plain HTML — automatable with requests" if sdp_hits
                                     else "not found in fetched HTML — may be behind JS or deeper navigation"))
        if calc_hits:
            jsc = any(c.get("onclick") for _, cs in calc_hits for c in cs)
            v.append("Calculators: controls found; " + ("they use onclick JavaScript — likely need a browser "
                     "(or replaying the XHR they trigger)" if jsc else "they look like plain links/forms — "
                     "automatable with requests"))
        else:
            v.append("Calculators: no calculator-like controls in fetched pages — deeper navigation or JS-rendered")
        v.append("Overall: " + ("site is JavaScript-heavy; expect browser to be needed for interactive parts"
                                if len(js_heavy) > len(pages) / 2 else
                                "site is mostly server-rendered; a requests-based script is feasible"))
        L += ["## Verdict", ""] + [f"- {x}" for x in v] + [""]
    open("probe_report.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n[report] probe_report.md  (raw HTML in " + args.out + "/)")


def open_html(args, pg):
    # find the saved html for this parsed page by title match (best effort)
    for fn in os.listdir(args.out):
        if fn.endswith(".html") and re.sub(r'[^a-z0-9]+', '_', pg.title.lower())[:40] in fn:
            try:
                return open(f"{args.out}/{fn}", encoding="utf-8").read()
            except Exception:
                pass
    return ""


if __name__ == "__main__":
    main()
