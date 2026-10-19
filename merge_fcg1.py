#!/usr/bin/env python3
"""
merge_fcg1.py — fold the columns of fcg1.csv into fcg_extract.csv, one time,
so the planner needs a single extract file.

  python merge_fcg1.py                               # fcg_extract.csv + fcg1.csv -> fcg_extract.csv (backup kept)
  python merge_fcg1.py --extract fcg_extract.csv --supplement fcg1.csv --out fcg_extract_merged.csv

Keys:  fcg_extract.csv  -> 'alpha3'  (e.g. AND)
       fcg1.csv         -> 'airport' (e.g. AND_FCG2.0)   suffix is stripped before joining
Columns from fcg1 that already exist in the extract get the suffix '_fcg1' so nothing is
overwritten. Prints how many countries matched and lists any fcg1 codes that found no row.
"""

import argparse
import os
import re
import shutil
from datetime import datetime

import pandas as pd

KEY_CANDIDATES = ("alpha3", "airport", "country", "code", "iso3", "country_code")


def norm(s):
    return (s.astype(str).str.strip().str.upper()
             .str.replace(r"_FCG.*$", "", regex=True)
             .str.replace(r"\.JSON$", "", regex=True))


def find_key(df, label):
    for k in KEY_CANDIDATES:
        if k in df.columns:
            return k
    raise SystemExit(f"[FATAL] {label}: no key column among {KEY_CANDIDATES}; found {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", default="fcg_extract.csv")
    ap.add_argument("--supplement", default="fcg1.csv")
    ap.add_argument("--out", default=None, help="default: overwrite --extract (a timestamped backup is kept)")
    a = ap.parse_args()

    ex = pd.read_csv(a.extract)
    su = pd.read_csv(a.supplement)
    kx, ks = find_key(ex, a.extract), find_key(su, a.supplement)
    print(f"[keys] extract '{kx}' ({len(ex)} rows), supplement '{ks}' ({len(su)} rows)")

    ex["_k"] = norm(ex[kx])
    su["_k"] = norm(su[ks])
    dups = su["_k"].duplicated(keep=False)
    if dups.any():
        print(f"[warn] supplement has {dups.sum()} duplicate codes: "
              f"{sorted(su.loc[dups, '_k'].unique())[:10]} — keeping the first of each")
        su = su.drop_duplicates("_k", keep="first")

    # drop identity/duplicate columns from the supplement, rename collisions
    drop = [c for c in su.columns if c in (ks, "country", "country_name", "alpha2", "alpha3", "name")]
    su = su.drop(columns=[c for c in drop if c != "_k"])
    collide = [c for c in su.columns if c in ex.columns and c != "_k"]
    su = su.rename(columns={c: f"{c}_fcg1" for c in collide})
    if collide:
        print(f"[cols] {len(collide)} name collisions renamed with _fcg1 suffix: {collide[:8]}")
    new_cols = [c for c in su.columns if c != "_k"]

    merged = ex.merge(su, on="_k", how="left")
    matched = int(merged[new_cols].notna().any(axis=1).sum()) if new_cols else 0
    unmatched_sup = sorted(set(su["_k"]) - set(ex["_k"]))
    print(f"[merge] +{len(new_cols)} columns; {matched}/{len(ex)} extract rows received supplement data")
    if unmatched_sup:
        print(f"[merge] {len(unmatched_sup)} supplement codes have no extract row (ignored): {unmatched_sup[:20]}")
    merged = merged.drop(columns=["_k"])

    out = a.out or a.extract
    if out == a.extract:
        bak = f"{a.extract}.bak_{datetime.now():%Y%m%d_%H%M%S}"
        shutil.copy2(a.extract, bak)
        print(f"[backup] {bak}")
    merged.to_csv(out, index=False)
    print(f"[done] {out}: {len(merged)} rows x {len(merged.columns)} columns")
    print(f"[next] add the new column names you want the judge to read to FEASIBILITY_COLS, "
          f"or leave fcg1.csv next to the planner and it will keep merging at load time. "
          f"New columns: {new_cols}")


if __name__ == "__main__":
    main()
