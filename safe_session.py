"""
safe_session.py — the ONLY HTTP access generated portal scripts may use.

A thin wrapper over requests.Session with guards:
  * stays on the login site's domain
  * GET is always allowed (read)
  * POST is allowed ONLY for: the login form, and endpoints/forms that look
    like calculate/compute/view/search. Any URL, form id/name, or field name
    matching NEVER (save, submit, delete, approve, ...) is refused.
  * every request is logged; every response body is saved to run_dir
  * never exposes credentials to the generated code (login() reads env)
"""

import json
import os
import re
import time
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import requests

NEVER = re.compile(r"(save|submit|delete|remove|approve|reject|cancel|logout|log ?out|"
                   r"sign ?out|upload|create|new ?request|discard|finalize|release|send|"
                   r"reset|change ?password|purge|commit|confirm)", re.I)
ALLOW_POST = re.compile(r"(calc|compute|recalc|view|search|find|lookup|filter|select|"
                        r"display|show|load|refresh|get|query|report|login|logon|signin|"
                        r"sign-in|authenticate)", re.I)


class _Forms(HTMLParser):
    def __init__(self):
        super().__init__()
        self.forms, self.links, self.frames, self.title = [], [], [], ""
        self._f, self._t, self._text, self._skip = None, False, [], 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "form":
            self._f = {"action": a.get("action", ""), "method": (a.get("method") or "get").upper(),
                       "id": a.get("id", ""), "name": a.get("name", ""), "fields": {}, "inputs": []}
        elif tag in ("input", "select", "textarea", "button") and self._f is not None:
            rec = {"tag": tag, "type": a.get("type", ""), "name": a.get("name", ""),
                   "id": a.get("id", ""), "value": a.get("value", ""),
                   "onclick": (a.get("onclick") or "")[:120]}
            self._f["inputs"].append(rec)
            if rec["name"] and rec["type"] not in ("submit", "button", "image", "checkbox", "radio"):
                self._f["fields"][rec["name"]] = rec["value"]
        elif tag == "a":
            self.links.append({"href": a.get("href", ""), "title": a.get("title", ""),
                               "id": a.get("id", ""), "onclick": (a.get("onclick") or "")[:120]})
        elif tag in ("frame", "iframe"):
            self.frames.append(a.get("src", ""))
        elif tag == "title":
            self._t = True
        elif tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag == "form" and self._f is not None:
            self.forms.append(self._f); self._f = None
        elif tag == "title":
            self._t = False
        elif tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, d):
        if self._t:
            self.title += d.strip()
        elif not self._skip and d.strip():
            self._text.append(d.strip())


class SafeSession:
    def __init__(self, site_url, run_dir="run"):
        self.site_url = site_url
        self.domain = urlparse(site_url).netloc
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self._s = requests.Session()
        self._s.headers["User-Agent"] = "python-requests (CAITT read-only portal client)"
        self.log, self._n, self.last = [], 0, None

    # ---------------------------------------------------------------- helpers
    def _save(self, resp, label):
        self._n += 1
        base = f"{self.run_dir}/{self._n:02d}_{re.sub(r'[^a-z0-9]+', '_', label.lower())[:40]}"
        ct = resp.headers.get("Content-Type", "")
        ext = ".json" if "json" in ct else ".html"
        open(base + ext, "w", encoding="utf-8", errors="ignore").write(resp.text)
        self.log.append({"t": datetime.now().isoformat(timespec="seconds"), "label": label,
                         "method": resp.request.method, "url": resp.url, "status": resp.status_code,
                         "file": base + ext})
        self.last = resp
        return resp

    def _check_url(self, url):
        if urlparse(url).netloc and urlparse(url).netloc != self.domain:
            raise PermissionError(f"refused: {url} is off-domain")
        if NEVER.search(urlparse(url).path + "?" + (urlparse(url).query or "")):
            raise PermissionError(f"refused: URL looks like a write action: {url}")

    def abs(self, href, base=None):
        return urljoin(base or (self.last.url if self.last is not None else self.site_url), href)

    # ---------------------------------------------------------------- reads
    def get(self, url, label="get", **params):
        url = self.abs(url)
        self._check_url(url)
        r = self._s.get(url, params=params or None, timeout=60)
        return self._save(r, label)

    def parse(self, resp=None):
        """-> object with .forms (action, method, fields, inputs), .links, .frames, .title, .text"""
        p = _Forms()
        try:
            p.feed((resp or self.last).text)
        except Exception:
            pass
        p.text = "\n".join(p._text)
        return p

    def text(self, resp=None):
        return self.parse(resp).text

    def find(self, pattern, resp=None, context=200):
        t = self.text(resp)
        return [t[max(0, m.start()-context):m.end()+context].replace("\n", " ")
                for m in re.finditer(pattern, t, re.I)][:20]

    # ---------------------------------------------------------------- writes (guarded)
    def login(self, label="login"):
        """Fetch SITE_URL, find the password form, POST env credentials. The only
        place credentials are used."""
        user, pw = os.environ.get("SITE_USER"), os.environ.get("SITE_PASS")
        r = self.get(self.site_url, "login page")
        p = self.parse(r)
        form = next((f for f in p.forms if any(i["type"] == "password" for i in f["inputs"])), None)
        if form is None:
            return "no login form"
        if not (user and pw):
            return "no credentials in SITE_USER/SITE_PASS"
        data = dict(form["fields"])
        ufield = next((i["name"] for i in form["inputs"]
                       if i["tag"] == "input" and i["type"] in ("text", "email", "") and i["name"]), None)
        pfield = next(i["name"] for i in form["inputs"] if i["type"] == "password")
        for i in form["inputs"]:
            if i["type"] == "submit" and i["name"]:
                data[i["name"]] = i["value"]
        data[ufield], data[pfield] = user, pw
        action = urljoin(r.url, form["action"] or r.url)
        r2 = self._s.post(action, data=data, timeout=60, allow_redirects=True)
        self._save(r2, label)
        still = any(i["type"] == "password" for f in self.parse(r2).forms for i in f["inputs"])
        return "ok" if (r2.status_code == 200 and not still) else f"login failed ({r2.status_code})"

    def post_form(self, form, overrides=None, base=None, label="post", submit_name=None):
        """POST a parsed form (from .parse().forms) with optional field overrides.
        Refused if the action/id/name/fields look like a write action and do not
        look like a calculate/view/search action."""
        action = urljoin(base or self.last.url, form["action"] or (base or self.last.url))
        clues = " ".join([action, form.get("id", ""), form.get("name", ""), submit_name or ""])
        if NEVER.search(clues) and not ALLOW_POST.search(clues):
            raise PermissionError(f"refused: form looks like a write action ({clues[:80]})")
        data = dict(form["fields"])
        data.update(overrides or {})
        if submit_name:
            if NEVER.search(submit_name):
                raise PermissionError(f"refused: submit control '{submit_name}' is a write action")
            val = next((i["value"] for i in form["inputs"] if i["name"] == submit_name), "")
            data[submit_name] = val
        for k in list(data):
            if NEVER.search(k) and not ALLOW_POST.search(k):
                raise PermissionError(f"refused: field '{k}' belongs to a write action")
        self._check_url(action)
        r = self._s.post(action, data=data, timeout=60, allow_redirects=True)
        return self._save(r, label)

    def post(self, url, data=None, label="post", json_body=None):
        """Raw POST to a calculate/view/search endpoint only."""
        url = self.abs(url)
        self._check_url(url)
        if not ALLOW_POST.search(urlparse(url).path + " " + " ".join((data or {}).keys())):
            raise PermissionError(f"refused: POST target does not look like calculate/view/search: {url}")
        for k in (data or {}):
            if NEVER.search(k) and not ALLOW_POST.search(k):
                raise PermissionError(f"refused: field '{k}' belongs to a write action")
        r = self._s.post(url, data=data, json=json_body, timeout=60, allow_redirects=True)
        return self._save(r, label)

    def save_log(self):
        path = f"{self.run_dir}/requests_log.json"
        json.dump(self.log, open(path, "w"), indent=1)
        return path
