#!/usr/bin/env python3
"""
opus_learn.py — STAGE 1: the strong model (Opus via LiteLLM) reads ALL the data,
answers every checklist item, saves answers to CSV, and writes LEARNINGS.md —
its own notes on where answers live and what strategies worked. Those learnings
are then consumed by local_extract.py (STAGE 2) to make a local model perform
near the strong model's level.

Reuses the loaders/chunker from fcg_extract.py (must be in the same folder).

Usage (one line):
  LITELLM_BASE_URL=https://llai-proxy.llan.ll.mit.edu/v1 LITELLM_API_KEY=sk-... \
  AGENT_MODEL=claude-opus-4-8 AGENT_TLS_VERIFY=false \
  python opus_learn.py --checklist checklist_enriched.json --fcg-dir data/fcg \
    --svc data/svc_rmk.txt --meis data/meis1.json data/meis2.json data/meis3.json \
    --airports data/airports.csv --codes data/countries_codes_and_coordinates.csv \
    --countries ITA --out opus_answers.csv

Outputs: opus_answers.csv (one row per country, value + __src columns)
         LEARNINGS.md     (Opus's strategies — feed to local_extract.py)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import fcg_extract as fx  # loaders, chunker, merge, llm call, config via env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checklist", required=True)
    ap.add_argument("--fcg-dir", required=True)
    ap.add_argument("--svc")
    ap.add_argument("--meis", nargs="*", default=[])
    ap.add_argument("--airports")
    ap.add_argument("--codes")
    ap.add_argument("--countries", nargs="*")
    ap.add_argument("--out", default="opus_answers.csv")
    ap.add_argument("--learnings", default="LEARNINGS.md")
    args = ap.parse_args()

    if not fx.BASE_URL or not fx.API_KEY:
        fx.die("Set LITELLM_BASE_URL and LITELLM_API_KEY")

    items = fx.load_checklist(Path(args.checklist))
    extract_items = [it for it in items if it["type"] == "extract"]
    fx.log(f"Checklist: {len(items)} items ({len(extract_items)} extract)")

    codes_map = fx.load_codes_csv(Path(args.codes)) if args.codes else {}
    airports_by_a2 = fx.load_airports_csv(Path(args.airports)) if args.airports else {}
    svc_by_icao = fx.load_svc(Path(args.svc)) if args.svc else {}
    meis_by_icao, meis_by_country = fx.load_meis(args.meis) if args.meis else ({}, {})

    fcg_files = sorted(Path(args.fcg_dir).glob("*.json"))
    if args.countries:
        want = {c.lower() for c in args.countries}
        fcg_files = [f for f in fcg_files if f.stem.lower() in want]
    if not fcg_files:
        fx.die("No FCG files matched")

    ck = fx.load_checkpoint()

    # Resume/append: finished countries are written immediately and skipped on re-run.
    done = set()
    out_path = Path(args.out)
    cols = (["country", "country_name", "alpha2", "alpha3"]
            + [it["col"] for it in items if it["type"] == "extract"])
    if out_path.exists():
        with out_path.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                done.add(r["country"])
        fx.log(f"Resume: {len(done)} countries already in {args.out}; skipping them")
    else:
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore").writeheader()

    rows = []
    failures = []
    evidence_notes = []  # (country, field, answer, which_source) for the learnings pass

    for f in fcg_files:
        country, a2, a3, disp = fx.resolve_country(f, codes_map)
        if country in done:
            continue
        fx.log(f"== {country} ({disp}) ==")
        row = {"country": country, "country_name": disp, "alpha2": a2 or "", "alpha3": a3 or ""}
        src = {}
        for it in items:
            if it["type"] != "extract":
                row[it["col"]] = "NA"
                src[it["col"]] = it["type"].upper()   # WORKFLOW / REFERENCE / etc.

        # Phase 1: chunked scan of full FCG
        fcg_txt = fx.cap(fx.fcg_text_for(f), fx.BUDGET_FCG, "FCG")
        p1 = fx.scan_chunked(country, "OP1", fcg_txt, extract_items, ck,
                             f"FOREIGN CLEARANCE GUIDE ({disp})")
        for it in extract_items:
            v = p1.get(it["col"], "NA")
            row[it["col"]] = v
            src[it["col"]] = "FCG" if v != "NA" else "NA"
            if v != "NA":
                evidence_notes.append((country, it["col"], v[:200], "FCG"))

        # Phase 2: ancillary for still-NA
        na = [it for it in extract_items if row[it["col"]] == "NA"]
        if na:
            icaos = airports_by_a2.get(a2, []) if a2 else []
            body = fx.build_ancillary_dossier(icaos, svc_by_icao, meis_by_icao,
                                              meis_by_country, disp)
            if body.strip():
                p2 = fx.scan_chunked(country, "OP2", body, na, ck,
                                     f"ANCILLARY (MEIS+SVC) — {disp}")
                for it in na:
                    v = p2.get(it["col"], "NA")
                    if v != "NA":
                        row[it["col"]] = v
                        src[it["col"]] = "ANCILLARY"
                        evidence_notes.append((country, it["col"], v[:200], "ANCILLARY"))
        filled = sum(1 for it in extract_items if row[it["col"]] != "NA")
        fx.log(f"  {country}: FILLED {filled}/{len(extract_items)}")
        for it in items:
            row[f"{it['col']}__src"] = src[it["col"]]
        rows.append(row)
        with out_path.open("a", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore").writerow(row)

    fx.log(f"Done: {len(rows)} new countries appended -> {args.out} "
           f"({len(done)} already present)")
    if failures:
        fx.log(f"FAILED countries ({len(failures)}): {', '.join(failures)} — "
               f"re-run the same command to retry them")

    # LEARNINGS pass: one call — Opus reflects on what it found and writes the
    # playbook a weaker model should follow.
    fx.log("Asking the model to write LEARNINGS.md ...")
    sample_struct = fx.cap(fx.fcg_text_for(fcg_files[0]), 12000, "FCG-SAMPLE")
    notes = "\n".join(f"- [{c}] {field} ({where}): {ans}"
                      for c, field, ans, where in evidence_notes[:120])
    fields_list = "\n".join(f"- {it['col']}: {it['question']}" for it in extract_items)
    user = (
        "You just extracted the answers below from Foreign Clearance Guide (FCG) "
        "documents and ancillary airfield sources (MEIS/SVC_RMK). Write a concise "
        "LEARNINGS.md playbook that a much weaker local model will be given as "
        "expert hints when doing the same extraction. Include: (1) for each field, "
        "WHERE its answer typically lives (key names/sections) and what phrasing "
        "to look for; (2) general strategies (terminology differs from field "
        "names; quote verbatim; NA only when truly absent; compound fields answer "
        "partially); (3) pitfalls you noticed. Be specific and practical, "
        "markdown, under 500 lines.\n\n"
        f"FIELDS:\n{fields_list}\n\n"
        f"ANSWERS FOUND (field, source, text):\n{notes}\n\n"
        f"SAMPLE DOCUMENT STRUCTURE:\n{sample_struct}"
    )
    md = fx.llm_extract.__globals__  # noqa - just to assert import worked
    content = _raw_call(user)
    Path(args.learnings).write_text(content, encoding="utf-8")
    fx.log(f"Wrote {args.learnings} ({len(content)} chars)")


def _raw_call(user_text: str) -> str:
    """Plain completion call (no JSON schema) for the learnings document."""
    import requests
    url = f"{fx.BASE_URL}/chat/completions"
    payload = {"model": fx.MODEL, "temperature": 0, "max_tokens": 8000,
               "messages": [{"role": "user", "content": user_text}]}
    headers = {"Authorization": f"Bearer {fx.API_KEY}", "Content-Type": "application/json"}
    r = requests.post(url, json=payload, headers=headers,
                      verify=fx.TLS_VERIFY, timeout=fx.HTTP_TIMEOUT)
    if r.status_code != 200:
        body = r.text[:300]
        if "budget" in body.lower():
            fx.die(f"Budget cap hit: {body}")
        fx.die(f"Learnings call failed HTTP {r.status_code}: {body}")
    msg = r.json()["choices"][0]["message"]
    return str(msg.get("content") or msg.get("reasoning_content") or "")


if __name__ == "__main__":
    main()
