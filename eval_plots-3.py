#!/usr/bin/env python3
"""
eval_plots.py — turn eval_results/*.csv into figures, and compare the planner
against a REAL reference route (a real mission's stop sequence).

Usage:
  # figures from an existing eval run (routing_runs.csv, penalty_sweep.csv,
  # geometry_per_route.csv, judge_scores.json if present)
  python eval_plots.py

  # compare planner vs a real route with the same stops
  python eval_plots.py --reference "KCHS,SAZB,SCEL,SEGU"
  # optionally: what the real flight actually overflew per leg, if known,
  # as ';'-separated ISO3 lists per leg (same number of legs):
  python eval_plots.py --reference "KCHS,SAZB,SCEL,SEGU" \\
      --reference-overflights "BHS DOM VEN BRA;ARG;PER"

Outputs: eval_results/plots/*.png, eval_results/reference_compare.csv
"""

import argparse
import json
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "eval_results"
PLOTS = f"{OUT}/plots"
os.makedirs(PLOTS, exist_ok=True)

LABELS = {"A_direct": "Direct GC", "B_land_distance": "Land (distance)",
          "C_penalized_nostones": "Penalized (no stones)", "D_full": "Full system"}


def plot_ablation(default_penalty=300.0):
    p = f"{OUT}/routing_runs.csv"
    if not os.path.exists(p):
        print("[plots] no routing_runs.csv — run eval_routes.py --routing first")
        return
    df = pd.read_csv(p)
    d = df[(df.penalty == default_penalty) & (df.found == True)]
    g = d.groupby("config").agg(nm=("nm", "mean"), clear=("clearances", "mean"),
                                risk=("risk_crossed", "mean"))
    g = g.reindex([c for c in LABELS if c in g.index])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar([LABELS[c] for c in g.index], g.nm, color="#4c72b0")
    ax[0].set_ylabel("mean route length (NM)")
    ax[0].set_title("Distance by configuration")
    ax[0].tick_params(axis="x", rotation=15)
    ax[1].bar([LABELS[c] for c in g.index], g.clear, color="#dd8452",
              label="clearances")
    ax[1].bar([LABELS[c] for c in g.index], g.risk, color="#c44e52",
              label="risk-listed crossings")
    ax[1].set_ylabel("mean count per route")
    ax[1].set_title("Clearance burden by configuration")
    ax[1].legend()
    ax[1].tick_params(axis="x", rotation=15)
    fig.suptitle(f"Routing ablation, n={d.origin.nunique()} routes, "
                 f"penalty={default_penalty:.0f} NM/clearance")
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/ablation.png", dpi=150)
    print(f"[plots] ablation.png")
    # found-rate table
    fr = df[df.penalty == default_penalty].groupby("config").found.mean()
    print("[plots] found-rate:", fr.round(3).to_dict())


def plot_pareto():
    p = f"{OUT}/penalty_sweep.csv"
    if not os.path.exists(p):
        return
    s = pd.read_csv(p).sort_values("penalty")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(s.clearances_mean, s.nm_mean, "o-", color="#4c72b0")
    for _, r in s.iterrows():
        ax.annotate(f"penalty {r.penalty:.0f}", (r.clearances_mean, r.nm_mean),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("mean clearances per route")
    ax.set_ylabel("mean route length (NM)")
    ax.set_title("Miles vs. clearances trade-off (penalty sweep)")
    ax.invert_xaxis()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/pareto.png", dpi=150)
    print("[plots] pareto.png")


def plot_geometry():
    p = f"{OUT}/geometry_per_route.csv"
    if not os.path.exists(p):
        return
    g = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(g.f1, bins=10, range=(0, 1), color="#55a868")
    ax[0].set_xlabel("F1 per route")
    ax[0].set_ylabel("routes")
    ax[0].set_title(f"Overflight detection F1 (mean {g.f1.mean():.2f}, "
                    f"recall {g.recall.mean():.2f}, precision "
                    f"{g.precision.mean():.2f})")
    m = f"{OUT}/geometry_missed.csv"
    o = f"{OUT}/geometry_overflagged.csv"
    def safe_csv(path):
        try:
            return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    miss = safe_csv(m)
    over = safe_csv(o)
    top = pd.concat([
        miss.head(8).rename(columns={"missed_count": "n"}).assign(kind="missed"),
        over.head(8).rename(columns={"overflagged_count": "n"}).assign(kind="over-flagged"),
    ]) if len(miss) or len(over) else pd.DataFrame()
    if len(top):
        colors = top.kind.map({"missed": "#c44e52", "over-flagged": "#dd8452"})
        ax[1].barh(top.country + " (" + top.kind + ")", top.n, color=colors)
        ax[1].set_xlabel("routes affected")
        ax[1].set_title("Which countries the centroid model gets wrong")
        ax[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/geometry.png", dpi=150)
    print("[plots] geometry.png")


def plot_judge():
    p = f"{OUT}/judge_scores.json"
    if not os.path.exists(p):
        return
    s = json.load(open(p))
    keys = ["accuracy", "cohen_kappa", "deny_precision", "deny_recall",
            "lead_time_exact_match", "self_consistency_3x"]
    vals = [s.get(k) or 0 for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([k.replace("_", "\n") for k in keys], vals, color="#8172b3")
    ax.set_ylim(0, 1)
    for i, v in enumerate(vals):
        ax.text(i, v + .02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_title(f"Clearance judge vs human labels (n={s.get('n')})")
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/judge.png", dpi=150)
    print("[plots] judge.png")

# ----------------------------------------------------------------------------
# Reference-route comparison
# ----------------------------------------------------------------------------

def compare_reference(stops, ref_overflights=None, blocks_file=None):
    import country_route_planner as P
    try:
        from eval_routes import polygon_overflights
    except Exception:
        polygon_overflights = None
    nodes, fcg, mapping, a2c, candidates, ourairports = P.build_dataset()
    blocked, _ = P.apply_risk(nodes, mapping, use_feed=False)
    if blocks_file:
        fb = json.load(open(blocks_file))
        extra = set(fb.get("blocked_countries", []))
        blocked = blocked | extra
        print(f"[compare] + FCG-derived blocks from planner run "
              f"({fb.get('query')}, {fb.get('date')}): {sorted(extra)}")

    def pt(ap):
        r = ourairports.loc[ap]
        return (float(r["latitude_deg"]), float(r["longitude_deg"])), str(r["iso3"])

    rows = []
    for i, (a, b) in enumerate(zip(stops, stops[1:])):
        pa, ca = pt(a)
        pb, cb = pt(b)
        # reference leg = straight great circle between the real stops
        corr_c, clear_c, nm = P.direct_corridor(pa, pb, nodes, (ca, cb))
        truth = polygon_overflights(pa, pb, (ca, cb)) if polygon_overflights else None
        # planner's corridor for the same leg (its cheapest option)
        cands = []
        if not (set(clear_c) & blocked):
            cands.append((P.corridor_score(nm, clear_c), "DIRECT", nm, clear_c))
        for s in P.stepping_stone_corridors(pa, pb, ca, cb, nodes, blocked, nm):
            cands.append((s[0], s[1], s[3], s[4]))
        cands.sort(key=lambda t: t[0])
        best = cands[0] if cands else (None, "none", None, [])
        ref_ov = None
        if ref_overflights and i < len(ref_overflights):
            ref_ov = ref_overflights[i]
        jacc = None
        if ref_ov is not None:
            A, B = set(ref_ov), set(best[3])
            jacc = len(A & B) / len(A | B) if (A | B) else 1.0
        rows.append({
            "leg": f"{a}->{b}",
            "ref_gc_nm": round(nm),
            "ref_overflights_detected": " ".join(clear_c),
            "ref_overflights_polygon_truth": " ".join(truth) if truth is not None else "",
            "ref_overflights_actual(if given)": " ".join(ref_ov) if ref_ov else "",
            "planner_choice": best[1],
            "planner_nm": round(best[2]) if best[2] else None,
            "planner_overflights": " ".join(best[3]),
            "planner_clearances": len(best[3]),
            "ref_clearances_detected": len(clear_c),
            "jaccard_vs_actual": None if jacc is None else round(jacc, 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/reference_compare.csv", index=False)
    print(df.to_string(index=False))
    tot_ref = df.ref_gc_nm.sum()
    tot_pl = df.planner_nm.sum()
    print(f"\n[compare] total: reference {tot_ref:,.0f} NM / "
          f"{df.ref_clearances_detected.sum()} clearances  vs  planner "
          f"{tot_pl:,.0f} NM / {df.planner_clearances.sum()} clearances")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = range(len(df))
    ax[0].bar([i - .2 for i in x], df.ref_gc_nm, width=.4, label="reference (GC)")
    ax[0].bar([i + .2 for i in x], df.planner_nm, width=.4, label="planner")
    ax[0].set_xticks(list(x)); ax[0].set_xticklabels(df.leg, rotation=15)
    ax[0].set_ylabel("NM"); ax[0].legend(); ax[0].set_title("Leg distance")
    ax[1].bar([i - .2 for i in x], df.ref_clearances_detected, width=.4,
              label="reference")
    ax[1].bar([i + .2 for i in x], df.planner_clearances, width=.4,
              label="planner")
    ax[1].set_xticks(list(x)); ax[1].set_xticklabels(df.leg, rotation=15)
    ax[1].set_ylabel("overflight clearances"); ax[1].legend()
    ax[1].set_title("Clearances per leg")
    fig.suptitle("Planner vs reference route, same stops: " + " -> ".join(stops))
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/reference_compare.png", dpi=150)
    print("[plots] reference_compare.png")


def endpoints_compare(stops, aircraft, blocks_file=None):
    """ENDPOINTS-ONLY test: give the planner just first/last stop + aircraft,
    let it choose its own fuel stops, compare against the human's stops."""
    import country_route_planner as P
    nodes, fcg, mapping, a2c, candidates, ourairports = P.build_dataset()
    blocked, _ = P.apply_risk(nodes, mapping, use_feed=False)
    if blocks_file:
        blocked |= set(json.load(open(blocks_file)).get("blocked_countries", []))
    spec = P.load_aircraft(aircraft)
    if not spec or not spec.get("range_nm"):
        sys.exit(f"[endpoints] aircraft '{aircraft}' has no range in aircraft_specs.csv")
    rng = spec["range_nm"]

    def pt(ap):
        r = ourairports.loc[ap]
        return (float(r["latitude_deg"]), float(r["longitude_deg"])), str(r["iso3"])

    def chain_stats(chain, label):
        rows, tot_nm, clear_all, over_range = [], 0.0, [], 0
        for a, b in zip(chain, chain[1:]):
            pa, ca = pt(a); pb, cb = pt(b)
            _, clear, nm = P.direct_corridor(pa, pb, nodes, (ca, cb))
            tot_nm += nm
            leg_clear = list(clear) + ([cb] if cb != chain_country_end else [])
            clear_all += [c for c in leg_clear if c not in clear_all]
            if nm > 0.85 * rng:
                over_range += 1
            rows.append({"who": label, "leg": f"{a}->{b}", "nm": round(nm),
                         "overflights": " ".join(clear),
                         "lands_in": cb, "over_range": nm > 0.85 * rng})
        landings = len(chain) - 2
        score = tot_nm + P.CLEARANCE_PENALTY_NM * len(clear_all) \
            + P.LANDING_PENALTY_NM * landings
        return rows, {"who": label, "stops": " -> ".join(chain),
                      "total_nm": round(tot_nm), "landings": landings,
                      "clearances": len(clear_all),
                      "legs_over_range": over_range, "score": round(score)}

    src_ap, dst_ap = stops[0], stops[-1]
    chain_country_end = pt(dst_ap)[1]
    p_src, c_src = pt(src_ap); p_dst, c_dst = pt(dst_ap)
    auto = P.auto_stop_search(p_src, c_src, src_ap, p_dst, c_dst, dst_ap,
                              candidates, ourairports, nodes, blocked, set(), rng)
    if auto is None:
        sys.exit("[endpoints] planner found no feasible stop chain")
    planner_chain = [src_ap] + auto + [dst_ap]
    r1, s1 = chain_stats(stops, "reference")
    r2, s2 = chain_stats(planner_chain, "planner")
    pd.DataFrame(r1 + r2).to_csv(f"{OUT}/endpoints_legs.csv", index=False)
    pd.DataFrame([s1, s2]).to_csv(f"{OUT}/endpoints_compare.csv", index=False)
    print(pd.DataFrame([s1, s2]).to_string(index=False))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for k, (key, title) in enumerate([("total_nm", "Total NM"),
                                       ("clearances", "Clearances"),
                                       ("landings", "Landings")]):
        ax[k].bar(["reference", "planner"], [s1[key], s2[key]],
                  color=["#999999", "#4c72b0"])
        ax[k].set_title(title)
    fig.suptitle(f"Endpoints-only: {src_ap} -> {dst_ap}, {spec['type']} "
                 f"(range {rng:,.0f} NM)")
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/endpoints_compare.png", dpi=150)
    print("[plots] endpoints_compare.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=None,
                    help='comma-separated ICAO stops of a real route')
    ap.add_argument("--blocks-file", default=None,
                    help="last_run_blocks.json from a planner run, so the "
                         "comparison respects the LLM-judged FCG denials")
    ap.add_argument("--endpoints-only", action="store_true",
                    help="with --reference and --aircraft: give the planner "
                         "only first/last stop and let it choose fuel stops")
    ap.add_argument("--aircraft", default=None)
    ap.add_argument("--reference-overflights", default=None,
                    help="';'-separated per-leg ISO3 lists the real flight "
                         "actually overflew (space-separated within a leg)")
    args = ap.parse_args()
    plot_ablation()
    plot_pareto()
    plot_geometry()
    plot_judge()
    if args.reference:
        stops = [s.strip().upper() for s in args.reference.split(",")]
        ov = None
        if args.reference_overflights:
            ov = [[c.strip().upper() for c in leg.split()]
                  for leg in args.reference_overflights.split(";")]
        if args.endpoints_only:
            if not args.aircraft:
                sys.exit("--endpoints-only needs --aircraft")
            endpoints_compare(stops, args.aircraft, args.blocks_file)
        else:
            compare_reference(stops, ov, args.blocks_file)
    print(f"[done] figures in {PLOTS}/")
