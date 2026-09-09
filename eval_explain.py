#!/usr/bin/env python3
"""
eval_explain.py — read eval_results/ and explain, in plain English, what the
numbers and figures mean. Facts are computed in code; --llm only rewords them.

Usage:
  python eval_explain.py            # deterministic report
  python eval_explain.py --llm      # + narrative from the local vLLM model
"""

import argparse
import json
import os
import re
import sys

import pandas as pd

OUT = "eval_results"


def load_csv(name):
    p = f"{OUT}/{name}"
    try:
        return pd.read_csv(p) if os.path.exists(p) else None
    except pd.errors.EmptyDataError:
        return None


def load_json(name):
    p = f"{OUT}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


LABELS = {"A_direct": "direct great circle (ignores clearances)",
          "B_land_distance": "shortest land route by miles",
          "C_penalized_nostones": "clearance-penalized, no ocean hops",
          "D_full": "full system (penalized + stepping stones)"}


def explain_routing(facts, lines):
    df = load_csv("routing_runs.csv")
    if df is None:
        return
    pen = 300.0
    d = df[(df.penalty == pen)]
    n_routes = d.origin.nunique() if "origin" in d else 0
    lines.append("== ROUTING ABLATION (ablation.png) ==")
    lines.append(f"Tested {n_routes} route(s). Each was planned four ways, "
                 f"then compared at the default price of {pen:.0f} NM per "
                 f"overflight clearance.")
    per = {}
    for cfg, g in d.groupby("config"):
        ok = g[g.found == True]
        per[cfg] = {"found": round(g.found.mean(), 2),
                    "nm": round(ok.nm.mean()) if len(ok) else None,
                    "clear": round(ok.clearances.mean(), 2) if len(ok) else None,
                    "risk": round(ok.risk_crossed.mean(), 2) if len(ok) else None}
        lines.append(f"  - {LABELS.get(cfg, cfg)}: found a route "
                     f"{per[cfg]['found']*100:.0f}% of the time; avg "
                     f"{per[cfg]['nm']} NM, {per[cfg]['clear']} clearances, "
                     f"{per[cfg]['risk']} risk-listed countries crossed.")
    facts["routing"] = per
    a, dfull = per.get("A_direct"), per.get("D_full")
    b = per.get("B_land_distance")
    if a and dfull and a["nm"] and dfull["nm"]:
        dn = dfull["nm"] - a["nm"]
        dc = a["clear"] - dfull["clear"]
        lines.append(f"  -> Versus flying the pure great circle, the full "
                     f"system adds {dn:+,.0f} NM ({dn/a['nm']*100:+.1f}%) and "
                     f"removes {dc:.2f} clearances per route on average.")
        facts["full_vs_direct"] = {"delta_nm": dn, "delta_clear": dc}
    if b and dfull and b["nm"] and dfull["nm"]:
        lines.append(f"  -> Versus the shortest land route, the full system "
                     f"is {dfull['nm']-b['nm']:+,.0f} NM and "
                     f"{dfull['clear']-b['clear']:+.2f} clearances, and crosses "
                     f"{dfull['risk']-b['risk']:+.2f} risk-listed countries.")
    lines.append("  How to read the figure: left panel = miles, right panel = "
                 "clearance burden. A good system is not the shortest bar on "
                 "the left; it is the lowest bar on the right at an acceptable "
                 "cost on the left.")


def explain_pareto(facts, lines):
    s = load_csv("penalty_sweep.csv")
    if s is None or s.empty:
        return
    s = s.sort_values("penalty")
    lines.append("\n== MILES vs CLEARANCES TRADE-OFF (pareto.png) ==")
    lines.append("The 'penalty' is how many extra miles we are willing to fly "
                 "to avoid one diplomatic clearance. Sweeping it shows the "
                 "menu of options a planner can choose from:")
    prev = None
    for _, r in s.iterrows():
        extra = ""
        if prev is not None and prev.clearances_mean != r.clearances_mean:
            dn = r.nm_mean - prev.nm_mean
            dc = prev.clearances_mean - r.clearances_mean
            if dc > 0:
                extra = f"  (each clearance avoided here costs ~{dn/dc:,.0f} NM)"
        lines.append(f"  - penalty {r.penalty:>5.0f}: {r.nm_mean:,.0f} NM, "
                     f"{r.clearances_mean:.2f} clearances{extra}")
        prev = r
    first, last = s.iloc[0], s.iloc[-1]
    facts["pareto"] = {"min_nm": int(first.nm_mean),
                       "min_clear": float(last.clearances_mean),
                       "nm_range": int(last.nm_mean - first.nm_mean),
                       "clear_range": float(first.clearances_mean - last.clearances_mean)}
    lines.append(f"  -> Across the sweep, clearances drop from "
                 f"{first.clearances_mean:.2f} to {last.clearances_mean:.2f} "
                 f"while distance rises from {first.nm_mean:,.0f} to "
                 f"{last.nm_mean:,.0f} NM. The curve makes the exchange rate "
                 f"explicit instead of hiding it in a heuristic.")


def explain_geometry(facts, lines):
    g = load_csv("geometry_per_route.csv")
    if g is None or g.empty:
        return
    miss = load_csv("geometry_missed.csv")
    over = load_csv("geometry_overflagged.csv")
    lines.append("\n== OVERFLIGHT DETECTION ACCURACY (geometry.png) ==")
    lines.append(f"We checked our 'which countries does this track cross' "
                 f"detector against true country borders on {len(g)} route(s).")
    lines.append(f"  - recall {g.recall.mean():.2f}: fraction of truly "
                 f"overflown countries we flagged. 1.00 = we never missed one. "
                 f"This is the safety-critical number (a missed overflight = "
                 f"an unrequested clearance).")
    lines.append(f"  - precision {g.precision.mean():.2f}: fraction of flagged "
                 f"countries that were truly overflown. Below 1.00 means we "
                 f"sometimes flag a neighbour we merely pass near (a needless "
                 f"permit request, not a safety issue).")
    lines.append(f"  - F1 {g.f1.mean():.2f} (mean), {g.f1.median():.2f} (median).")
    nmiss = int((g.recall < 1).sum())
    lines.append(f"  - routes with at least one MISSED country: {nmiss} of {len(g)}.")
    if miss is not None and len(miss):
        lines.append("  - most-missed countries: " + ", ".join(
            f"{r.country} ({r.missed_count})" for _, r in miss.head(6).iterrows()))
    if over is not None and len(over):
        lines.append("  - most over-flagged countries: " + ", ".join(
            f"{r.country} ({r.overflagged_count})" for _, r in over.head(6).iterrows()))
    facts["geometry"] = {"routes": len(g), "recall": round(g.recall.mean(), 3),
                         "precision": round(g.precision.mean(), 3),
                         "f1": round(g.f1.mean(), 3), "routes_missed": nmiss,
                         "top_missed": miss.head(5).country.tolist() if miss is not None and len(miss) else [],
                         "top_overflagged": over.head(5).country.tolist() if over is not None and len(over) else []}
    lines.append("  -> Interpretation: the detector is conservative. Over-"
                 "flagging costs paperwork; missing costs an incident. Any "
                 "missed countries listed above are the priority to fix "
                 "(bigger detection radius, or the polygon upgrade).")


def explain_reference(facts, lines):
    r = load_csv("reference_compare.csv")
    if r is None or r.empty:
        return
    lines.append("\n== PLANNER vs REAL ROUTE (reference_compare.png) ==")
    lines.append("For a real mission's stop sequence, leg by leg: what the "
                 "straight-line reference overflies vs what the planner chose.")
    for _, row in r.iterrows():
        lines.append(f"  - {row.leg}: reference {row.ref_gc_nm:,.0f} NM over "
                     f"[{row.ref_overflights_detected or 'open water'}] "
                     f"(truth: [{row.ref_overflights_polygon_truth or 'open water'}]); "
                     f"planner chose {row.planner_choice} = "
                     f"{row.planner_nm:,.0f} NM over "
                     f"[{row.planner_overflights or 'open water'}]"
                     + (f"; overlap with actual flight {row.jaccard_vs_actual}"
                        if pd.notna(row.get("jaccard_vs_actual")) else ""))
    tr, tp = r.ref_gc_nm.sum(), r.planner_nm.sum()
    cr, cp = r.ref_clearances_detected.sum(), r.planner_clearances.sum()
    lines.append(f"  -> Totals: reference {tr:,.0f} NM / {cr} clearances; "
                 f"planner {tp:,.0f} NM / {cp} clearances "
                 f"({tp-tr:+,.0f} NM, {cp-cr:+d} clearances).")
    facts["reference"] = {"ref_nm": int(tr), "ref_clear": int(cr),
                          "planner_nm": int(tp), "planner_clear": int(cp)}
    if cp < cr:
        lines.append("  -> The planner reaches the same stops with fewer "
                     "clearances; the distance delta is the price of that.")
    elif cp == cr and abs(tp - tr) < 50:
        lines.append("  -> The planner reproduces the reference routing; "
                     "no cheaper compliant option exists under our cost model.")
    else:
        lines.append("  -> The planner did not beat the reference on "
                     "clearances; check whether a blocked country forced a "
                     "detour (see last_run_blocks.json).")


def explain_judge(facts, lines):
    j = load_json("judge_scores.json")
    if not j:
        return
    lines.append("\n== CLEARANCE JUDGE vs HUMAN LABELS (judge.png) ==")
    lines.append(f"On {j.get('n')} country/role cases labeled by a human:")
    lines.append(f"  - accuracy {j.get('accuracy')}: agreement on allow/deny.")
    lines.append(f"  - Cohen's kappa {j.get('cohen_kappa')}: agreement beyond "
                 f"chance (0.6-0.8 substantial, >0.8 near-perfect).")
    lines.append(f"  - deny precision {j.get('deny_precision')} / recall "
                 f"{j.get('deny_recall')}: when the model says 'deny', how "
                 f"often it is right / how many true denials it catches.")
    if j.get("lead_time_exact_match") is not None:
        lines.append(f"  - lead-time extraction exact match "
                     f"{j.get('lead_time_exact_match')}: the number of days it "
                     f"read from the text matched the human's reading.")
    if j.get("self_consistency_3x") is not None:
        lines.append(f"  - self-consistency {j.get('self_consistency_3x')}: same "
                     f"verdict on 3 repeated runs (stability at temperature 0).")
    facts["judge"] = j


def explain_blocks(facts, lines):
    if not os.path.exists("last_run_blocks.json"):
        return
    b = json.load(open("last_run_blocks.json"))
    lines.append("\n== FCG CONSTRAINTS FROM THE LAST PLANNER RUN ==")
    lines.append(f"Query '{b.get('query')}' on {b.get('date')}: the FCG judge "
                 f"denied overflight of {len(b.get('blocked_countries', []))} "
                 f"countries: {', '.join(b.get('blocked_countries', [])) or 'none'}.")
    reasons = [v.get("reason", "") for v in b.get("verdicts", {}).values()
               if v and not v.get("allowed", True)]
    lt = sum("LEAD TIME" in r for r in reasons)
    lines.append(f"  - {lt} of those were lead-time denials (mission date too "
                 f"soon), the rest are hard prohibitions in the FCG text.")
    facts["fcg_blocks"] = {"n": len(b.get("blocked_countries", [])),
                           "lead_time_denials": lt}


def llm_narrative(facts, report):
    try:
        import country_route_planner as P
    except Exception as e:
        return f"(LLM narrative unavailable: {e})"
    prompt = (
        "You are helping a researcher explain an evaluation to their advisor. "
        "Below are COMPUTED FACTS (json) and a plain report. Write 180-250 "
        "words, plain text, no markdown, for a technical advisor: what was "
        "measured, what the numbers show, the honest limitations, and the "
        "single most important next step. Use ONLY the numbers given; do not "
        "invent any.\n\nFACTS:\n" + json.dumps(facts, indent=1, default=str)
        + "\n\nREPORT:\n" + report)
    try:
        return P.llm_chat(prompt, system="You are a precise, plain-spoken "
                          "research writing assistant.")
    except Exception as e:
        return f"(LLM narrative failed: {e})"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()
    facts, lines = {}, []
    explain_routing(facts, lines)
    explain_pareto(facts, lines)
    explain_geometry(facts, lines)
    explain_reference(facts, lines)
    explain_judge(facts, lines)
    explain_blocks(facts, lines)
    if not lines:
        sys.exit("No eval_results found. Run eval_routes.py / eval_plots.py first.")
    report = "\n".join(lines)
    print(report)
    if args.llm:
        print("\n== NARRATIVE FOR YOUR ADVISOR (LLM, from the facts above) ==")
        print(llm_narrative(facts, report))
    with open(f"{OUT}/EXPLANATION.txt", "w") as f:
        f.write(report + "\n")
    print(f"\n[saved] {OUT}/EXPLANATION.txt")
