#!/usr/bin/env python3
"""
fix_routes.py — Have Claude (via LiteLLM) rewrite the broken extraction routes
in checklist_enriched.json, using the real FCG key structure.

WHY: ~10 extract items route to key-names that don't exist in the real FCG JSON,
so they fall back to the whole document and extract poorly. This sends Claude the
real keys + the current routes and asks it to point each broken route at the key
path that actually holds that field's answer. Routes are DATA — no code changes.

Nothing is pasted through a chat; the keys stay on the cluster. One LLM call.

INPUTS (in the current dir unless overridden):
  fcg_keys.txt              produced by: python dump_keys.py data/fcg/FRA.json > fcg_keys.txt
  checklist_enriched.json   the file to fix
  data/work/dossiers/<C>_phase1.txt   (optional) its FALLBACK line names the broken items

OUTPUT:
  checklist_enriched.fixed.json   the rewritten checklist (review, then rename over the original)
  route_changelog.txt             what changed and why

ENV (same gateway you use for extraction):
  LITELLM_BASE_URL   e.g. https://llai-proxy.llan.ll.mit.edu/v1
  LITELLM_API_KEY
  AGENT_MODEL        e.g. claude-opus-4-8
  AGENT_TLS_VERIFY   "false" for the self-signed gateway cert

Usage:
  LITELLM_BASE_URL=... LITELLM_API_KEY=sk-... AGENT_MODEL=claude-opus-4-8 AGENT_TLS_VERIFY=false python fix_routes.py
  # optional flags: --keys fcg_keys.txt --checklist checklist_enriched.json
  #                 --dossier data/work/dossiers/FRA_phase1.txt --max-tokens 8000
"""

import argparse
import json
import os
import re
import sys
import time

import requests
import urllib3

BASE_URL = os.environ.get("LITELLM_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("LITELLM_API_KEY", "")
MODEL = os.environ.get("AGENT_MODEL", "claude-opus-4-8")
TLS_VERIFY = os.environ.get("AGENT_TLS_VERIFY", "true").lower() not in ("false", "0", "no")
if not TLS_VERIFY:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def die(m):
    print(f"FATAL: {m}", file=sys.stderr)
    sys.exit(1)


SYSTEM = (
    "You are a precise data-engineering assistant fixing routing metadata for an "
    "aviation-document extraction pipeline. You are given (1) the REAL key paths "
    "of the Foreign Clearance Guide (FCG) JSON, and (2) a checklist where each "
    "extract item has a 'route' array of key-name candidates that direct the "
    "extractor to the relevant subtree. Some routes match no real key, so those "
    "items fail. Your job: for each extract item, rewrite its 'route' (and 'scope' "
    "if needed) so the candidates MATCH real key names from the provided key list "
    "that plausibly hold that item's answer. A candidate ending in '$' means exact "
    "key match; otherwise it's a case-insensitive substring of a key name. Only use "
    "key names that actually appear in the provided key list. If an item's answer "
    "genuinely is not present anywhere in the FCG keys, set its 'type' to "
    "'reference' and leave a short note in a '_note' field instead of inventing a "
    "route. Never change 'workflow' items. Never change item 'id' or 'question'. "
    "Return ONLY a JSON object: {\"items\":[...all items in original order...], "
    "\"changelog\":[{\"id\":..., \"old_route\":[...], \"new_route\":[...], "
    "\"matched_key\":\"...\", \"reason\":\"...\"}]}. No prose, no code fences."
)


def call_claude(system, user, max_tokens):
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    for attempt in range(1, 5):
        try:
            r = requests.post(url, json=payload, headers=headers, verify=TLS_VERIFY, timeout=300)
            if r.status_code != 200:
                body = r.text[:300]
                if "budget" in body.lower():
                    die(f"LiteLLM budget cap hit — raise the user budget. {body}")
                raise RuntimeError(f"HTTP {r.status_code}: {body}")
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            wait = min(2 ** attempt, 20)
            print(f"  attempt {attempt}/4 failed ({e}); retry in {wait}s")
            time.sleep(wait)
    die("LLM call failed after retries")


def extract_json(txt):
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip(), flags=re.S).strip()
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if m:
        txt = m.group(0)
    return json.loads(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="fcg_keys.txt")
    ap.add_argument("--checklist", default="checklist_enriched.json")
    ap.add_argument("--dossier", default="data/work/dossiers/FRA_phase1.txt")
    ap.add_argument("--out", default="checklist_enriched.fixed.json")
    ap.add_argument("--max-tokens", type=int, default=8000)
    args = ap.parse_args()

    if not BASE_URL or not API_KEY:
        die("Set LITELLM_BASE_URL and LITELLM_API_KEY")
    for f in (args.keys, args.checklist):
        if not os.path.exists(f):
            die(f"Missing input: {f}")

    keys = open(args.keys, encoding="utf-8", errors="replace").read()
    checklist = json.load(open(args.checklist, encoding="utf-8"))

    fallback = ""
    if os.path.exists(args.dossier):
        for line in open(args.dossier, encoding="utf-8", errors="replace"):
            if "FALLBACK" in line:
                fallback = line.strip()
                break

    n_extract = sum(1 for it in checklist if it.get("type") == "extract")
    print(f"Checklist: {len(checklist)} items ({n_extract} extract). "
          f"Keys file: {len(keys)} chars. Asking {MODEL} to fix routes...")

    user = (
        f"REAL FCG KEY STRUCTURE:\n{keys}\n\n"
        f"CURRENT CHECKLIST (fix the 'route'/'scope' of extract items only):\n"
        f"{json.dumps(checklist, ensure_ascii=False)}\n\n"
        + (f"These items currently route to nothing (fix these first): {fallback}\n\n"
           if fallback else "")
        + "Return the JSON object described in the system message."
    )

    raw = call_claude(SYSTEM, user, args.max_tokens)
    try:
        obj = extract_json(raw)
    except Exception as e:
        open("fix_routes.raw.txt", "w", encoding="utf-8").write(raw)
        die(f"Could not parse Claude's JSON ({e}). Raw saved to fix_routes.raw.txt")

    items = obj.get("items")
    if not isinstance(items, list) or len(items) != len(checklist):
        open("fix_routes.raw.txt", "w", encoding="utf-8").write(raw)
        die(f"Returned {len(items) if isinstance(items,list) else '?'} items, "
            f"expected {len(checklist)}. Raw saved to fix_routes.raw.txt — inspect before using.")

    # safety: ids and workflow types must be unchanged
    orig = {it.get("id"): it for it in checklist}
    for it in items:
        o = orig.get(it.get("id"))
        if o is None:
            die(f"New/renamed id appeared: {it.get('id')!r} — refusing to write.")
        if o.get("type") == "workflow" and it.get("type") != "workflow":
            die(f"Workflow item changed type: {it.get('id')!r} — refusing to write.")

    json.dump(items, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    cl = obj.get("changelog", [])
    with open("route_changelog.txt", "w", encoding="utf-8") as f:
        f.write(f"Route fix via {MODEL}\n{'='*60}\n")
        for c in cl:
            f.write(f"\n[{c.get('id')}]\n"
                    f"  old: {c.get('old_route')}\n"
                    f"  new: {c.get('new_route')}\n"
                    f"  matched: {c.get('matched_key')}\n"
                    f"  reason: {c.get('reason')}\n")
    print(f"\nWrote {args.out} and route_changelog.txt")
    print(f"Changed {len(cl)} items. Review the changelog, then:")
    print(f"  mv {args.out} {args.checklist}   # accept the fix")
    print(f"  rm -f data/work/checkpoint.json  # then re-run extraction")


if __name__ == "__main__":
    main()
