"""
safe_browser.py — the ONLY browser API that generated portal scripts may use.

Every click and fill passes through guards:
  * labels/attrs matching NEVER are refused (save, submit, delete, approve, ...)
  * navigation is restricted to the login site's domain
  * every action is logged; screenshots+text saved on request
Generated code cannot reach the raw Playwright page (it is a private attribute);
site_learner.py also rejects generated code that imports playwright directly.
"""

import os
import re
import time
import json
from datetime import datetime
from urllib.parse import urlparse

NEVER = re.compile(
    r"(save|submit|delete|remove|approve|reject|cancel|logout|log out|sign out|"
    r"upload|create|new request|discard|finalize|release|send|post|reset|"
    r"change password|purge|clear all)", re.I)


class SafeBrowser:
    def __init__(self, site_url, run_dir="run", headless=True):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        # SITE_BROWSER=chrome  -> use the installed, org-managed Google Chrome
        #                         (inherits corporate CA/device certs, proxy auth,
        #                         policies) instead of Playwright's bare Chromium.
        # SITE_PROFILE_DIR=<dir> -> reuse a Chrome profile directory (keeps SSO
        #                         sessions). Use a COPY of your profile or a
        #                         dedicated dir; Chrome must not be running on it.
        channel = os.environ.get("SITE_BROWSER", "").lower() or None
        profile = os.environ.get("SITE_PROFILE_DIR")
        launch_kw = {"headless": headless}
        # SITE_USE_PROXY=true -> route the browser through HTTPS_PROXY (the same
        # proxy python/requests used), instead of hitting the network directly.
        proxy = (os.environ.get("SITE_PROXY") or os.environ.get("HTTPS_PROXY")
                 or os.environ.get("https_proxy"))
        if proxy and os.environ.get("SITE_USE_PROXY", "false").lower() in ("1", "true", "yes"):
            launch_kw["proxy"] = {"server": proxy}
            print(f"[browser] using proxy {proxy.split('@')[-1]}")
        if channel in ("chrome", "msedge", "chrome-beta", "msedge-beta"):
            launch_kw["channel"] = channel
        if profile:
            self._ctx = self._pw.chromium.launch_persistent_context(
                profile, ignore_https_errors=True,
                viewport={"width": 1500, "height": 950}, **launch_kw)
            self._browser = self._ctx
            self._page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        else:
            self._browser = self._pw.chromium.launch(**launch_kw)
            self._ctx = self._browser.new_context(ignore_https_errors=True,
                                                  viewport={"width": 1500, "height": 950})
            self._page = self._ctx.new_page()
        self._page.set_default_timeout(45000)
        self.site_url = site_url
        self.domain = urlparse(site_url).netloc.split(":")[0]
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.actions = []
        self._n = 0

    # ---------------------------------------------------------------- lifecycle
    def close(self):
        try:
            self._browser.close(); self._pw.stop()
        except Exception:
            pass

    def _log(self, kind, detail):
        self.actions.append({"t": datetime.now().isoformat(timespec="seconds"),
                             "kind": kind, "detail": str(detail)[:200], "url": self._page.url})

    # ---------------------------------------------------------------- reading
    @property
    def url(self):
        return self._page.url

    def text(self):
        try:
            t = self._page.inner_text("body")
        except Exception:
            t = ""
        for fr in self._page.frames[1:]:
            try:
                t += "\n\n[frame]\n" + fr.inner_text("body")
            except Exception:
                pass
        return re.sub(r"\n{3,}", "\n\n", t).strip()

    def html(self, max_chars=200000):
        try:
            return self._page.content()[:max_chars]
        except Exception:
            return ""

    def find(self, query, context=300):
        body, low, q = self.text(), self.text().lower(), query.lower()
        hits, i = [], low.find(q)
        while i != -1 and len(hits) < 10:
            hits.append(body[max(0, i-context):i+len(q)+context].replace("\n", " "))
            i = low.find(q, i+1)
        return hits

    def snapshot(self, label):
        self._n += 1
        base = f"{self.run_dir}/{self._n:02d}_{re.sub(r'[^a-z0-9]+', '_', label.lower())[:40]}"
        try:
            self._page.screenshot(path=base + ".png", full_page=True)
        except Exception:
            pass
        open(base + ".txt", "w", encoding="utf-8").write(f"URL: {self._page.url}\n\n{self.text()}")
        self._log("snapshot", label)
        return base

    def elements(self, max_items=300):
        """Clickable/fillable elements with clues; each has an 'idx' for click()/fill()."""
        js = """() => {
          const sel='a,button,input,select,textarea,img,[role=button],[role=tab],[onclick],i,span';
          const out=[]; for (const e of document.querySelectorAll(sel)) {
            const r=e.getBoundingClientRect(); if(r.width===0&&r.height===0) continue;
            const g=a=>e.getAttribute(a)||'';
            const t=(e.innerText||e.value||'').trim().slice(0,80);
            const rec={tag:e.tagName.toLowerCase(),text:t,title:g('title'),alt:g('alt'),aria:g('aria-label'),
              src:g('src').split('/').pop(),id:g('id'),name:g('name'),cls:g('class').slice(0,60),
              href:g('href').slice(0,100),onclick:g('onclick').slice(0,100),ph:g('placeholder'),type:g('type'),
              x:Math.round(r.x),y:Math.round(r.y)};
            if(!rec.text&&!rec.title&&!rec.alt&&!rec.aria&&!rec.src&&!rec.onclick&&!rec.href&&!rec.ph&&!rec.name) continue;
            if(rec.type==='hidden') continue;
            const row=e.closest('tr,li,.row,.form-group,.panel');
            rec.ctx=row?(row.innerText||'').trim().replace(/\\s+/g,' ').slice(0,140):'';
            out.push(rec);} return out; }"""
        items = []
        for fr in self._page.frames:
            try:
                for it in fr.evaluate(js):
                    it["frame"] = "main" if fr == self._page.main_frame else (fr.name or fr.url[:40])
                    items.append(it)
            except Exception:
                continue
        items = items[:max_items]
        for i, it in enumerate(items):
            it["idx"] = i
        self._last_items = items
        return items

    @staticmethod
    def describe(items):
        return "\n".join(
            f"[{it['idx']}] <{it['tag']}> " + " ".join(
                f"{k}={v!r}" for k, v in it.items()
                if k in ("text", "title", "alt", "aria", "src", "id", "name", "href", "onclick", "ph", "type") and v)
            + (f" | row: {it['ctx']}" if it.get("ctx") else "")
            + (f" | frame: {it['frame']}" if it.get("frame") != "main" else "")
            for it in items)

    # ---------------------------------------------------------------- locating
    def _frame_of(self, it):
        for fr in self._page.frames:
            name = "main" if fr == self._page.main_frame else (fr.name or fr.url[:40])
            if name == it.get("frame"):
                return fr
        return self._page.main_frame

    def _locate(self, it):
        fr = self._frame_of(it)
        cands = []
        if it.get("id") and re.match(r"^[A-Za-z][\w\-:.]*$", it["id"]):
            cands.append(fr.locator(f"#{it['id']}"))
        for attr, a in (("title", "title"), ("alt", "alt"), ("aria", "aria-label"), ("name", "name")):
            if it.get(attr):
                cands.append(fr.locator(f"[{a}=\"{it[attr]}\"]"))
        if it.get("src"):
            cands.append(fr.locator(f"img[src$=\"{it['src']}\"]"))
        if it.get("href"):
            cands.append(fr.locator(f"a[href=\"{it['href']}\"]"))
        if it.get("text"):
            cands.append(fr.get_by_text(it["text"], exact=False))
        for loc in cands:
            try:
                n = loc.count()
            except Exception:
                continue
            if n == 1:
                return loc.first
            if n > 1:                         # several matches: nearest to recorded position
                best, bestd = 0, 1e9
                for k in range(min(n, 40)):
                    bb = loc.nth(k).bounding_box()
                    if bb:
                        d = abs(bb["x"]-it["x"]) + abs(bb["y"]-it["y"])
                        if d < bestd:
                            best, bestd = k, d
                return loc.nth(best)
        return None

    # ---------------------------------------------------------------- actions
    def goto(self, url):
        if not url.startswith("http"):
            url = self.site_url.rstrip("/") + "/" + url.lstrip("/")
        if not urlparse(url).netloc.split(":")[0].endswith(self.domain):
            self._log("refused", f"goto off-domain {url}")
            return "refused: off-domain"
        if NEVER.search(urlparse(url).path + "?" + (urlparse(url).query or "")):
            self._log("refused", f"goto write-like url {url}")
            return "refused: URL looks like a write action"
        self._page.goto(url, wait_until="domcontentloaded")
        self._log("goto", url)
        return "ok"

    def click(self, idx_or_item, why=""):
        it = idx_or_item if isinstance(idx_or_item, dict) else self._last_items[idx_or_item]
        label = " ".join(str(it.get(k, "")) for k in ("text", "title", "alt", "aria", "src", "id", "onclick", "href"))
        if NEVER.search(label):
            self._log("refused", f"click {label[:80]}")
            return f"refused: '{label[:60]}' is a persisting/destructive action"
        loc = self._locate(it)
        try:
            if loc is not None:
                loc.scroll_into_view_if_needed()
                loc.click(timeout=15000)
            else:
                self._page.mouse.click(it["x"]+3, it["y"]+3)
            try:
                self._page.wait_for_load_state("networkidle", timeout=20000)
            except Exception:
                pass
            time.sleep(1.0)
        except Exception as e:
            self._log("click-failed", f"{label[:60]} {e.__class__.__name__}")
            return f"click failed: {e.__class__.__name__}: {str(e)[:120]}"
        self._log("click", f"{label[:80]} ({why})")
        return "ok"

    def click_text(self, text, why=""):
        if NEVER.search(text):
            return f"refused: '{text}'"
        for getter in (lambda: self._page.get_by_role("link", name=text, exact=False),
                       lambda: self._page.get_by_role("tab", name=text, exact=False),
                       lambda: self._page.get_by_role("button", name=text, exact=False),
                       lambda: self._page.get_by_text(text, exact=False)):
            loc = getter()
            if loc.count():
                try:
                    loc.first.click(timeout=15000)
                    try:
                        self._page.wait_for_load_state("networkidle", timeout=20000)
                    except Exception:
                        pass
                    time.sleep(1.0)
                    self._log("click", f"{text} ({why})")
                    return "ok"
                except Exception as e:
                    return f"click failed: {e.__class__.__name__}"
        return f"not found: '{text}'"

    def fill(self, idx_or_item, value, enter=False):
        it = idx_or_item if isinstance(idx_or_item, dict) else self._last_items[idx_or_item]
        if it["tag"] not in ("input", "select", "textarea"):
            return "not a field"
        if NEVER.search(" ".join(str(it.get(k, "")) for k in ("name", "id", "ph", "aria"))):
            return "refused: field belongs to a write action"
        loc = self._locate(it)
        if loc is None:
            return "field not found"
        try:
            if it["tag"] == "select":
                loc.select_option(label=str(value))
            else:
                loc.fill(str(value))
                if enter:
                    loc.press("Enter")
                    try:
                        self._page.wait_for_load_state("networkidle", timeout=20000)
                    except Exception:
                        pass
            time.sleep(0.6)
        except Exception as e:
            return f"fill failed: {e.__class__.__name__}"
        self._log("fill", f"{it.get('name') or it.get('id') or it.get('ph')} = {str(value)[:40]}")
        return "ok"

    def back(self):
        try:
            self._page.go_back(wait_until="domcontentloaded")
        except Exception:
            pass
        self._log("back", "")
        return "ok"

    def login(self):
        user, pw = os.environ.get("SITE_USER"), os.environ.get("SITE_PASS")
        self._page.goto(self.site_url, wait_until="commit", timeout=90000)
        try:
            self._page.wait_for_load_state("domcontentloaded", timeout=30000)
        except Exception:
            pass
        low0 = self.text()[:3000].lower()
        if any(k in low0 for k in ("blocked by", "access denied", "web filter", "proxy",
                                   "this site has been blocked", "policy violation")):
            return ("login blocked: the browser received a proxy/web-filter block page, not the "
                    "portal. Try SITE_BROWSER=chrome (managed Chrome) and/or run from a network "
                    "where the portal is allowed.")
        if not (user and pw):
            return "no credentials"
        usel = os.environ.get("SITE_USER_SEL") or ("input[type=email], input[name*=user i], "
                                                   "input[id*=user i], input[type=text]")
        psel = os.environ.get("SITE_PASS_SEL") or "input[type=password]"
        ssel = os.environ.get("SITE_SUBMIT_SEL")
        try:
            if self._page.locator(psel).count() == 0:
                return "no login form visible"
            self._page.locator(usel).first.fill(user)
            self._page.locator(psel).first.fill(pw)
            (self._page.locator(ssel).first.click() if ssel else self._page.locator(psel).first.press("Enter"))
            try:
                self._page.wait_for_load_state("networkidle", timeout=20000)
            except Exception:
                pass
        except Exception as e:
            return f"login failed: {e.__class__.__name__}"
        low = self.text()[:3000].lower()
        if any(k in low for k in ("multi-factor", "one-time code", "authenticator", "captcha",
                                  "smart card", "common access card")):
            return "login blocked: MFA/CAC/CAPTCHA"
        self._log("login", "ok")
        return "ok"

    def save_log(self, path=None):
        path = path or f"{self.run_dir}/actions.json"
        json.dump(self.actions, open(path, "w"), indent=1)
        return path
