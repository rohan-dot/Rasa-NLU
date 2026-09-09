#!/usr/bin/env python3
"""
eval_routes.py — quantitative evaluation harness for country_route_planner.py

Layers:
  1. GEOMETRY   Overflight-detection accuracy of the centroid-radius model vs
                polygon ground truth (Natural Earth admin-0). Precision/recall/
                F1 per route + per-country confusion. Needs: shapely + the
                Natural Earth file (GeoJSON recommended).
  2. ROUTING    Ablation benchmark on N random airport pairs: direct-only vs
                land-graph (distance) vs penalized (no stones) vs full system.
                Metrics: NM, clearances (polygon truth if available),
                risk-listed crossings, found-rate, runtime. Plus a penalty
                sweep for the NM-vs-clearances Pareto curve. No LLM involved.
  3. JUDGE      Emits a labeling sheet (country x role x dossier) with empty
                human columns; --judge also records the local LLM's verdict
                and lead-time number 3x for self-consistency. Then
                --score-judge <labeled.csv> computes accuracy / kappa /
                lead-time exact match.

Usage:
  python eval_routes.py --geometry --routing --n 100 --date 2026-10-20
  python eval_routes.py --judge-sheet --n-judge 120 --judge --date 2026-10-20
  python eval_routes.py --score-judge eval_results/judge_sheet_labeled.csv

Outputs land in ./eval_results/  (CSVs + summary.md)
"""

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict

import pandas as pd

import country_route_planner as P   # the planner is imported as a library

OUT = "eval_results"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# Natural Earth polygons (ground truth for overflight)
# ----------------------------------------------------------------------------
NE_FILE = "ne_110m_admin_0_countries.geojson"   # or .shp (needs pyshp)

_polys = None
def load_polygons(path=NE_FILE):
    """Return {iso3: prepared shapely geometry} or None if unavailable."""
    global _polys
    if _polys is not None:
        return _polys
    try:
        from shapely.geometry import shape
        from shapely.prepared import prep
    except ImportError:
        print("[geom] shapely not installed — polygon truth disabled "
              "(pip install shapely --break-system-packages)")
        _polys = {}
        return _polys
    feats = []
    if path.lower().endswith(".geojson") or path.lower().endswith(".json"):
        try:
            gj = json.load(open(path, encoding="utf-8"))
            feats = [(f["properties"], f["geometry"]) for f in gj["features"]]
        except FileNotFoundError:
            print(f"[geom] {path} not found — polygon truth disabled")
            _polys = {}
            return _polys
    else:
        try:
            import shapefile
            sf = shapefile.Reader(path)
            fields = [f[0] for f in sf.fields[1:]]
            for rec, shp in zip(sf.records(), sf.shapes()):
                feats.append((dict(zip(fields, rec)), shp.__geo_interface__))
        except Exception as e:
            print(f"[geom] cannot read {path}: {e}")
            _polys = {}
            return _polys
    polys = {}
    for props, geom in feats:
        iso = str(props.get("ISO_A3", "-99"))
        if iso in ("-99", "", "None"):           # NE quirk: FRA, NOR, ...
            iso = str(props.get("ADM0_A3", "-99"))
        if iso == "-99":
            continue
        polys[iso.upper()] = prep(shape(geom))
    print(f"[geom] {len(polys)} country polygons loaded")
    _polys = polys
    return _polys


def polygon_overflights(p1, p2, exclude, step_nm=10.0):
    """Ground truth: countries whose polygon contains any sampled track point."""
    polys = load_polygons()
    if not polys:
        return None
    from shapely.geometry import Point
    samples, _ = P.gc_sample(p1, p2, step_nm=step_nm)
    hit = []
    seen = set()
    for lat, lon in samples:
        pt = Point(lon, lat)
        for iso, g in polys.items():
            if iso in seen:
                continue
            if g.contains(pt):
                seen.add(iso)
                hit.append(iso)
    return [c for c in hit if c not in exclude]


def prf(pred, truth):
    pred, truth = set(pred), set(truth)
    tp = len(pred & truth)
    p = tp / len(pred) if pred else (1.0 if not truth else 0.0)
    r = tp / len(truth) if truth else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f

# ----------------------------------------------------------------------------
# Benchmark pairs
# ----------------------------------------------------------------------------

def sample_pairs(nodes, ourairports, n, seed=7):
    """Random pairs of large airports in countries with FCG coverage."""
    rng = random.Random(seed)
    big = ourairports[ourairports["type"] == "large_airport"] \
        if "type" in ourairports.columns else ourairports
    big = big[big["iso3"].isin(nodes.index)]
    idents = list(big.index)
    pairs, tries = [], 0
    while len(pairs) < n and tries < n * 50:
        tries += 1
        a, b = rng.sample(idents, 2)
        ca, cb = big.loc[a, "iso3"], big.loc[b, "iso3"]
        if ca == cb:
            continue
        pa = (float(big.loc[a, "latitude_deg"]), float(big.loc[a, "longitude_deg"]))
        pb = (float(big.loc[b, "latitude_deg"]), float(big.loc[b, "longitude_deg"]))
        if P.haversine_nm(*pa, *pb) < 500:          # skip trivial hops
            continue
        pairs.append((a, ca, pa, b, cb, pb))
    return pairs

# ----------------------------------------------------------------------------
# Layer 1: geometry accuracy
# ----------------------------------------------------------------------------

def eval_geometry(pairs, nodes):
    rows, conf_fn, conf_fp = [], Counter(), Counter()
    for a, ca, pa, b, cb, pb in pairs:
        truth = polygon_overflights(pa, pb, (ca, cb))
        if truth is None:
            print("[geom] no polygons — skipping layer 1")
            return None
        _, pred, nm = P.direct_corridor(pa, pb, nodes, (ca, cb))
        p, r, f = prf(pred, truth)
        for c in set(truth) - set(pred):
            conf_fn[c] += 1          # missed (dangerous: unflagged overflight)
        for c in set(pred) - set(truth):
            conf_fp[c] += 1          # over-flagged (costly: needless clearance)
        rows.append({"origin": a, "dest": b, "nm": round(nm),
                     "truth": " ".join(truth), "pred": " ".join(pred),
                     "precision": round(p, 3), "recall": round(r, 3),
                     "f1": round(f, 3)})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/geometry_per_route.csv", index=False)
    pd.DataFrame([{"country": c, "missed_count": n} for c, n in
                  conf_fn.most_common()]).to_csv(f"{OUT}/geometry_missed.csv",
                                                 index=False)
    pd.DataFrame([{"country": c, "overflagged_count": n} for c, n in
                  conf_fp.most_common()]).to_csv(
        f"{OUT}/geometry_overflagged.csv", index=False)
    summ = {"routes": len(df),
            "precision_mean": round(df.precision.mean(), 3),
            "recall_mean": round(df.recall.mean(), 3),
            "f1_mean": round(df.f1.mean(), 3),
            "f1_median": round(df.f1.median(), 3),
            "routes_with_missed_country": int((df.recall < 1).sum()),
            "top_missed": conf_fn.most_common(8),
            "top_overflagged": conf_fp.most_common(8)}
    print("[geom]", json.dumps(summ, indent=1))
    return summ

# ----------------------------------------------------------------------------
# Layer 2: routing ablation
# ----------------------------------------------------------------------------

def corridor_truth_count(corridor_countries, pa, pb, ca, cb):
    """If polygons exist, re-count clearances by polygon truth along the
    stone chain is not reconstructible; use predicted count + polygon truth
    of the DIRECT track as a reference column instead."""
    return len(corridor_countries)


def run_configs(pairs, nodes, graph, blocked, penalty):
    P.CLEARANCE_PENALTY_NM = penalty
    rows = []
    for a, ca, pa, b, cb, pb in pairs:
        base = {"origin": a, "dest": b, "penalty": penalty}
        # A: direct only
        t0 = time.time()
        corr, clear, nm = P.direct_corridor(pa, pb, nodes, (ca, cb))
        found = not (set(clear) & blocked)
        rows.append({**base, "config": "A_direct", "found": found,
                     "nm": round(nm), "clearances": len(clear),
                     "risk_crossed": len(set(clear) & set(P.RISK_EXTRA)),
                     "sec": round(time.time() - t0, 3)})
        # B: land graph, distance-only weights
        t0 = time.time()
        path, dnm = P.dijkstra(graph, ca, cb, blocked)
        if path:
            clear_b = [c for c in path if c not in (ca, cb)]
            rows.append({**base, "config": "B_land_distance", "found": True,
                         "nm": round(dnm), "clearances": len(clear_b),
                         "risk_crossed": len(set(clear_b) & set(P.RISK_EXTRA)),
                         "sec": round(time.time() - t0, 3)})
        else:
            rows.append({**base, "config": "B_land_distance", "found": False,
                         "nm": None, "clearances": None, "risk_crossed": None,
                         "sec": round(time.time() - t0, 3)})
        # C: penalized land graph + direct (no stepping stones)
        t0 = time.time()
        cands = []
        if found:
            cands.append((P.corridor_score(nm, clear), nm, clear))
        pgraph = {u: {v: w + penalty + P.RISK_EXTRA.get(v, 0.0)
                      for v, w in nb.items()} for u, nb in graph.items()}
        path, _ = P.dijkstra(pgraph, ca, cb, blocked)
        if path:
            clear_c = [c for c in path if c not in (ca, cb)]
            lnm = sum(graph[u][v] for u, v in zip(path, path[1:]))
            cands.append((P.corridor_score(lnm, clear_c), lnm, clear_c))
        if cands:
            cands.sort(key=lambda t: t[0])
            _, cnm, cclear = cands[0]
            rows.append({**base, "config": "C_penalized_nostones", "found": True,
                         "nm": round(cnm), "clearances": len(cclear),
                         "risk_crossed": len(set(cclear) & set(P.RISK_EXTRA)),
                         "sec": round(time.time() - t0, 3)})
        else:
            rows.append({**base, "config": "C_penalized_nostones",
                         "found": False, "nm": None, "clearances": None,
                         "risk_crossed": None, "sec": round(time.time() - t0, 3)})
        # D: full system (direct + penalized land + stepping stones)
        t0 = time.time()
        stones = P.stepping_stone_corridors(pa, pb, ca, cb, nodes, blocked, nm)
        for s in stones:
            cands.append((s[0], s[3], s[4]))
        if cands:
            cands.sort(key=lambda t: t[0])
            _, dnm2, dclear = cands[0]
            rows.append({**base, "config": "D_full", "found": True,
                         "nm": round(dnm2), "clearances": len(dclear),
                         "risk_crossed": len(set(dclear) & set(P.RISK_EXTRA)),
                         "sec": round(time.time() - t0, 3)})
        else:
            rows.append({**base, "config": "D_full", "found": False,
                         "nm": None, "clearances": None, "risk_crossed": None,
                         "sec": round(time.time() - t0, 3)})
    return rows


def eval_routing(pairs, nodes, graph, blocked, penalties):
    all_rows = []
    for pen in penalties:
        print(f"[routing] penalty {pen} NM ...")
        all_rows += run_configs(pairs, nodes, graph, blocked, pen)
    df = pd.DataFrame(all_rows)
    df.to_csv(f"{OUT}/routing_runs.csv", index=False)
    # Summary at the default penalty
    d = df[df.penalty == P_DEFAULT_PENALTY]
    summ = {}
    for cfg, g in d.groupby("config"):
        ok = g[g.found == True]
        summ[cfg] = {"found_rate": round(g.found.mean(), 3),
                     "nm_mean": round(ok.nm.mean(), 0) if len(ok) else None,
                     "clearances_mean": round(ok.clearances.mean(), 2) if len(ok) else None,
                     "risk_crossed_mean": round(ok.risk_crossed.mean(), 2) if len(ok) else None,
                     "sec_mean": round(g.sec.mean(), 3)}
    # Pareto sweep for D_full
    sweep = []
    for pen, g in df[(df.config == "D_full") & (df.found == True)].groupby("penalty"):
        sweep.append({"penalty": pen, "nm_mean": round(g.nm.mean()),
                      "clearances_mean": round(g.clearances.mean(), 2),
                      "risk_crossed_mean": round(g.risk_crossed.mean(), 2)})
    pd.DataFrame(sweep).to_csv(f"{OUT}/penalty_sweep.csv", index=False)
    print("[routing]", json.dumps(summ, indent=1))
    print("[routing] sweep:", sweep)
    return summ, sweep

# ----------------------------------------------------------------------------
# Layer 3: judge labeling sheet + scoring
# ----------------------------------------------------------------------------

def judge_sheet(nodes, fcg, n, date, run_llm, seed=11):
    rng = random.Random(seed)
    codes = [c for c in nodes.index if c in fcg.index]
    rows = []
    for code in rng.sample(codes, min(n, len(codes))):
        role = rng.choice(["overflight", "stop"])
        dossier = P.country_dossier(code, fcg)
        row = {"country": code, "name": nodes.loc[code, "country"],
               "role": role, "mission_date": date,
               "dossier": dossier[:6000],
               "human_allowed": "", "human_min_lead_days": "",
               "human_notes": ""}
        if run_llm:
            verdicts = [P.judge_country(code, nodes.loc[code, "country"],
                                        role, date, fcg) for _ in range(3)]
            for i, v in enumerate(verdicts, 1):
                row[f"llm{i}_allowed"] = v.get("allowed")
                row[f"llm{i}_lead"] = v.get("min_lead_days")
                row[f"llm{i}_reason"] = v.get("reason")
            row["llm_consistent"] = len({str(v.get("allowed")) for v in verdicts}) == 1
        rows.append(row)
        print(f"[judge] {code}/{role} done")
    pd.DataFrame(rows).to_csv(f"{OUT}/judge_sheet.csv", index=False)
    print(f"[judge] wrote {OUT}/judge_sheet.csv — fill human_* columns, "
          f"save as judge_sheet_labeled.csv, then --score-judge it")


def score_judge(path):
    df = pd.read_csv(path)
    df = df[df.human_allowed.notna()]
    if "llm1_allowed" not in df.columns:
        sys.exit("sheet has no LLM columns — regenerate with --judge")
    def tf(x):
        return str(x).strip().lower() in ("true", "1", "yes", "y", "allow")
    h = df.human_allowed.map(tf)
    m = df.llm1_allowed.map(tf)
    acc = (h == m).mean()
    # kappa
    po = acc
    pe = (h.mean() * m.mean()) + ((1 - h.mean()) * (1 - m.mean()))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    deny_h, deny_m = ~h, ~m
    tp = (deny_h & deny_m).sum()
    prec = tp / deny_m.sum() if deny_m.sum() else 0
    rec = tp / deny_h.sum() if deny_h.sum() else 0
    lead_rows = df[df.human_min_lead_days.notna()]
    lead_match = None
    if len(lead_rows):
        lead_match = (pd.to_numeric(lead_rows.human_min_lead_days, errors="coerce")
                      == pd.to_numeric(lead_rows.llm1_lead, errors="coerce")).mean()
    cons = df.llm_consistent.mean() if "llm_consistent" in df.columns else None
    out = {"n": int(len(df)), "accuracy": round(acc, 3),
           "cohen_kappa": round(kappa, 3),
           "deny_precision": round(prec, 3), "deny_recall": round(rec, 3),
           "lead_time_exact_match": None if lead_match is None else round(lead_match, 3),
           "self_consistency_3x": None if cons is None else round(cons, 3)}
    print("[judge]", json.dumps(out, indent=1))
    json.dump(out, open(f"{OUT}/judge_scores.json", "w"), indent=1)

# ----------------------------------------------------------------------------

P_DEFAULT_PENALTY = 300.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", action="store_true")
    ap.add_argument("--routing", action="store_true")
    ap.add_argument("--judge-sheet", action="store_true")
    ap.add_argument("--judge", action="store_true",
                    help="with --judge-sheet: also run the local LLM 3x")
    ap.add_argument("--score-judge", default=None)
    ap.add_argument("--n", type=int, default=60, help="benchmark pairs")
    ap.add_argument("--n-judge", type=int, default=100)
    ap.add_argument("--date", default="2026-10-20")
    ap.add_argument("--penalties", default="0,150,300,600,1200")
    ap.add_argument("--pairs", default=None,
                    help='specific routes to evaluate INSTEAD of random ones, '
                         'e.g. "KWRB:VTBD,KWRB:GVAC,KCHS:SCEL" (ICAO codes)')
    args = ap.parse_args()

    if args.score_judge:
        score_judge(args.score_judge)
        return

    nodes, fcg, mapping, a2c, candidates, ourairports = P.build_dataset()
    summary = {"date": args.date, "n_pairs": args.n}

    if args.geometry or args.routing:
        if args.pairs:
            pairs = []
            for spec in args.pairs.split(","):
                a, b = [s.strip().upper() for s in spec.split(":")]
                for code in (a, b):
                    if code not in ourairports.index:
                        sys.exit(f"[pairs] {code} not in OurAirports")
                ra, rb = ourairports.loc[a], ourairports.loc[b]
                pairs.append((a, str(ra["iso3"]),
                              (float(ra["latitude_deg"]), float(ra["longitude_deg"])),
                              b, str(rb["iso3"]),
                              (float(rb["latitude_deg"]), float(rb["longitude_deg"]))))
            print(f"[bench] using {len(pairs)} user-specified pairs")
        else:
            pairs = sample_pairs(nodes, ourairports, args.n)
        pd.DataFrame([{"origin": a, "o_ctry": ca, "dest": b, "d_ctry": cb}
                      for a, ca, _, b, cb, _ in pairs]).to_csv(
            f"{OUT}/benchmark_pairs.csv", index=False)
        print(f"[bench] {len(pairs)} airport pairs")

    if args.geometry:
        summary["geometry"] = eval_geometry(pairs, nodes)

    if args.routing:
        graph = P.build_graph(nodes)
        blocked, _ = P.apply_risk(nodes, mapping, use_feed=False)  # manual list only
        pens = [float(x) for x in args.penalties.split(",")]
        summ, sweep = eval_routing(pairs, nodes, graph, blocked, pens)
        summary["routing"] = summ
        summary["penalty_sweep"] = sweep

    if args.judge_sheet:
        judge_sheet(nodes, fcg, args.n_judge, args.date, args.judge)

    with open(f"{OUT}/summary.md", "w") as f:
        f.write("# Route planner evaluation summary\n\n```json\n"
                + json.dumps(summary, indent=1, default=str) + "\n```\n")
    print(f"[done] see {OUT}/summary.md")


if __name__ == "__main__":
    main()
