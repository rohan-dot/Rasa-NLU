#!/usr/bin/env python3
"""
portal_reachability_check.py — document whether this machine can reach the
portal, for an IT allowlist request. Read-only: one GET per URL.

  python portal_reachability_check.py https://portal.example.mil/login \
      https://opensky-network.org/api/states/all https://aviationweather.gov

Writes reachability_report.md with: hostname, proxy used, HTTP status, final
URL after redirects, page title, and whether the body looks like a proxy /
web-filter block page. Attach it to the ticket.
"""

import os
import re
import sys
from datetime import datetime
from urllib.parse import urlparse

import requests

BLOCK_HINTS = ("blocked", "access denied", "web filter", "policy", "not permitted",
               "forbidden by", "proxy", "category")


def check(url):
    rec = {"url": url, "host": urlparse(url).netloc,
           "proxy": os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or "(none)"}
    try:
        r = requests.get(url, timeout=30, allow_redirects=True, verify=True)
        body = r.text[:20000]
        title = re.search(r"<title>(.*?)</title>", body, re.I | re.S)
        rec.update({"status": r.status_code, "final_url": r.url,
                    "title": (title.group(1).strip()[:120] if title else ""),
                    "via_header": r.headers.get("Via", ""),
                    "server": r.headers.get("Server", ""),
                    "looks_blocked": any(h in body.lower() for h in BLOCK_HINTS)
                                     and urlparse(r.url).netloc != urlparse(url).netloc
                                     or r.status_code in (403, 407, 451)})
    except requests.exceptions.SSLError as e:
        rec.update({"status": "SSL error", "error": str(e)[:200]})
    except Exception as e:
        rec.update({"status": f"{e.__class__.__name__}", "error": str(e)[:200]})
    return rec


def main():
    urls = sys.argv[1:] or [os.environ.get("SITE_URL", "")]
    urls = [u for u in urls if u]
    if not urls:
        sys.exit("give URLs as arguments or set SITE_URL")
    import socket
    lines = [f"# Portal reachability report", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M} on host `{socket.gethostname()}`",
             f"HTTPS_PROXY: `{os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy') or '(none)'}`",
             "", "| URL | status | final URL | title | blocked? |", "|---|---|---|---|---|"]
    for u in urls:
        r = check(u)
        print(f"{r['host']:40} status={r.get('status')} blocked={r.get('looks_blocked')} "
              f"title={r.get('title','')[:50]!r}")
        lines.append(f"| {u} | {r.get('status')} | {r.get('final_url','')} | "
                     f"{r.get('title','') or r.get('error','')} | {r.get('looks_blocked','')} |")
    lines += ["", "Request: allow outbound HTTPS from this host (via the proxy above) to the "
                  "portal hostname(s) marked blocked, for read-only automated retrieval."]
    open("reachability_report.md", "w").write("\n".join(lines))
    print("\n[report] reachability_report.md")


if __name__ == "__main__":
    main()
