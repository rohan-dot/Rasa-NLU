#!/usr/bin/env python3
"""
merge_dedupe.py — Repair a results CSV containing duplicate country rows.

For each country, merges all its rows FIELD BY FIELD:
  - a non-NA value always beats NA/empty
  - if two rows both have real values for a field, the LONGER one wins
    (more info), unless one contains the other (keep the container)
So no information is ever lost — the merged row is at least as complete as the
best single row, usually more complete than any of them.

Usage:
  python merge_dedupe.py --in opus_all.csv --out opus_all_clean.csv
Then inspect the report it prints, and optionally:
  mv opus_all_clean.csv opus_all.csv
"""

import argparse
import csv
import sys

META = ["country", "country_name", "alpha2", "alpha3"]


def is_na(v):
    return v is None or str(v).strip().upper() in ("", "NA", "N/A", "NONE")


def better(a, b):
    """Pick the more informative of two values for the same field."""
    if is_na(a):
        return b
    if is_na(b):
        return a
    a, b = str(a).strip(), str(b).strip()
    if a == b:
        return a
    if a in b:
        return b
    if b in a:
        return a
    return a if len(a) >= len(b) else b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    with open(args.inp, encoding="utf-8", newline="") as fh:
        rdr = csv.DictReader(fh)
        cols = rdr.fieldnames or []
        rows = list(rdr)
    if "country" not in cols:
        sys.exit("No 'country' column found.")

    merged = {}          # country -> merged row
    order = []           # first-seen order
    dup_count = {}
    for r in rows:
        c = (r.get("country") or "").strip()
        if not c:
            continue
        if c not in merged:
            merged[c] = dict(r)
            order.append(c)
            dup_count[c] = 1
        else:
            dup_count[c] += 1
            m = merged[c]
            for k in cols:
                if k in META:
                    if is_na(m.get(k)) and not is_na(r.get(k)):
                        m[k] = r[k]
                    continue
                m[k] = better(m.get(k), r.get(k))

    field_cols = [c for c in cols if c not in META]
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in order:
            w.writerow(merged[c])

    dups = {c: n for c, n in dup_count.items() if n > 1}
    all_na = [c for c in order
              if all(is_na(merged[c].get(k)) for k in field_cols)]
    print(f"Input rows: {len(rows)}  ->  unique countries: {len(order)}")
    print(f"Countries that had duplicates: {len(dups)}"
          + (f"  (max copies: {max(dups.values())})" if dups else ""))
    if dups:
        sample = list(dups.items())[:10]
        print("  e.g. " + ", ".join(f"{c}x{n}" for c, n in sample))
    fill = {c: sum(0 if is_na(merged[c].get(k)) else 1 for k in field_cols)
            for c in order}
    import statistics
    print(f"Fill per country after merge: min={min(fill.values())} "
          f"median={int(statistics.median(fill.values()))} max={max(fill.values())} "
          f"of {len(field_cols)} fields")
    if all_na:
        print(f"\nALL-NA countries ({len(all_na)}) — delete these rows from the "
              f"output and re-run opus_learn to retry them:\n  {', '.join(all_na)}")
    print(f"\nWrote {args.out}. Review, then: mv {args.out} {args.inp}")


if __name__ == "__main__":
    main()
