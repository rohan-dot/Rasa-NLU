#!/usr/bin/env python3
"""
validate_extract.py — Validate fcg_extract.csv against dossier audit files.

Because extraction is closed-world (verbatim quotes or "NA"), every non-NA
answer MUST appear in the dossier the model saw. This script checks exactly
that, giving a mechanical hallucination test plus fill-rate statistics.

Checks per country:
  1. VERBATIM: each non-NA answer is found in the matching dossier
     (FCG src -> <C>_phase1.txt, ANCILLARY src -> <C>_phase2.txt).
     Matching is whitespace-normalized; multi-snippet answers (' | ') are
     checked snippet-by-snippet.
  2. Fill-rate stats per country and per field.
  3. Fields NA across ALL countries (candidate checklist-wording problems).
  4. Dossiers with truncation markers or suspiciously small size.

Usage:
  python validate_extract.py --csv fcg_extract_local.csv --dossiers data/work/dossiers
  python validate_extract.py --csv ... --dossiers ... --show-misses 20
"""

import argparse
import csv
import re
import sys
from pathlib import Path

META_COLS = {"country", "country_name", "alpha2", "alpha3"}


def norm(s: str) -> str:
    """Whitespace/punct-light normalization for verbatim matching."""
    s = s.lower()
    s = re.sub(r"[\u2018\u2019]", "'", s)
    s = re.sub(r"[\u201c\u201d]", '"', s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def load_dossier(dossier_dir: Path, country: str, phase: int) -> str:
    p = dossier_dir / f"{country}_phase{phase}.txt"
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dossiers", default="data/work/dossiers")
    ap.add_argument("--show-misses", type=int, default=10,
                    help="How many verbatim failures to print in full")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8", newline="")))
    if not rows:
        sys.exit("Empty CSV")
    dossier_dir = Path(args.dossiers)

    item_cols = [c for c in rows[0].keys()
                 if c not in META_COLS and not c.endswith("__src")]

    total = {"answers": 0, "verbatim_ok": 0, "verbatim_fail": 0}
    misses = []
    field_fill = {c: 0 for c in item_cols}
    print(f"Validating {len(rows)} countries x {len(item_cols)} fields\n")
    print(f"{'country':<10} {'filled':>10} {'FCG':>5} {'ANC':>5} "
          f"{'verbatim_ok':>12} {'fail':>5}  dossier_sizes")
    print("-" * 78)

    for row in rows:
        country = row["country"]
        d1, d2 = (load_dossier(dossier_dir, country, i) for i in (1, 2))
        n1, n2 = norm(d1), norm(d2)
        filled = fcg_n = anc_n = ok = fail = 0
        for col in item_cols:
            val = (row.get(col) or "").strip()
            src = (row.get(f"{col}__src") or "").strip()
            if val == "NA" or not val:
                continue
            filled += 1
            field_fill[col] += 1
            if src == "FCG":
                fcg_n += 1
                hay = n1
            elif src == "ANCILLARY":
                anc_n += 1
                hay = n2
            else:
                hay = n1 + "\n" + n2
            snippets = [s for s in (p.strip() for p in val.split(" | ")) if s]
            good = all(norm(s) in hay for s in snippets) if snippets else False
            total["answers"] += 1
            if good:
                ok += 1
                total["verbatim_ok"] += 1
            else:
                fail += 1
                total["verbatim_fail"] += 1
                misses.append((country, col, src, val))
        trunc1 = "TRUNC" if "truncated at" in d1 else ""
        trunc2 = "TRUNC" if "truncated at" in d2 else ""
        print(f"{country:<10} {filled:>4}/{len(item_cols):<5} {fcg_n:>5} {anc_n:>5} "
              f"{ok:>12} {fail:>5}  P1={len(d1)}ch{trunc1} P2={len(d2)}ch{trunc2}")
        if len(d1) < 2000:
            print(f"  WARNING {country}: phase1 dossier only {len(d1)} chars — "
                  f"FCG flattening produced almost nothing; inspect "
                  f"{dossier_dir}/{country}_phase1.txt")

    print("\n== Overall ==")
    a = total["answers"]
    if a:
        print(f"Non-NA answers: {a} | verbatim-verified: {total['verbatim_ok']} "
              f"({100*total['verbatim_ok']/a:.0f}%) | "
              f"FAILED verbatim (possible hallucination/paraphrase): {total['verbatim_fail']}")
    else:
        print("No non-NA answers at all — extraction produced nothing to validate. "
              "Inspect the phase1 dossier files first.")

    never = [c for c, n in field_fill.items() if n == 0]
    if never:
        print(f"\nFields NA in EVERY country ({len(never)}/{len(item_cols)}) — "
              f"likely checklist wording vs source vocabulary mismatch:")
        for c in never:
            print(f"  - {c}")

    if misses:
        print(f"\n== First {min(args.show_misses, len(misses))} verbatim failures ==")
        for country, col, src, val in misses[: args.show_misses]:
            print(f"[{country}] {col} (src={src}):\n    {val[:200]}")


if __name__ == "__main__":
    main()
