#!/usr/bin/env python3
"""
compare_routes.py — ONE command: the route the planner just produced
(last_run_blocks.json) vs the route humans actually flew.

  python compare_routes.py --human "KSUU,TJSJ,SBFZ,FKKD"

Does three things and writes them to eval_results/compare_<ts>.md (+ .png):
  1. NUMBERS   – per-leg and total: distance, countries overflown, landings,
                 legs over 85% of aircraft range, FCG-denied countries touched
  2. EXPLAIN   – plain-English reading of the numbers (deterministic, no LLM)
  3. JUDGE     – the LLM (Sol via gateway, or vLLM) sees the raw facts of both
                 routes + FCG text for every country touched and picks one,
                 N times, with agreement reported

Human route: comma-separated ICAO stops, in order. Planner route: read from
last_run_blocks.json ("stops"), or --planner "A,B,C" to override.
Aircraft/date come from the blocks file unless given.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import country_route_planner as P
from route_judge import chain_facts, fcg_evidence, run_judge

OUT = "eval_results"
os.makedirs(OUT, exist_ok=True)


def resolve_chain(chain, nodes, ourairports):
    """Every ICAO must have coordinates; use the planner's own resolver so
    unknown fields get the station lookup + cache instead of a KeyError."""
    out = []
    for ap in chain:
        ap = ap.strip().upper()
        if ap not in ourairports.index:
            P.resolve_waypoint(ap, {}, nodes, {}, use_llm=False, ourairports=ourairports)
        if ap not in ourairports.index:
            sys.exit(f"[compare] cannot locate airport {ap}")
        out.append(ap)
    return out


def explain(H, Pl, rng, blocked):
    L = []
    dn = Pl["total_nm"] - H["total_nm"]
    dl = Pl["landings_en_route"] - H["landings_en_route"]
    dc = Pl["n_clearances"] - H["n_clearances"]
    L.append(f"Humans flew {' -> '.join(H['stops'])}: {H['total_nm']:,} NM, "
             f"{H['landings_en_route']} en-route landings, {H['n_clearances']} countries needing "
             f"clearance [{' '.join(H['distinct_clearance_countries'])}].")
    L.append(f"Planner chose {' -> '.join(Pl['stops'])}: {Pl['total_nm']:,} NM, "
             f"{Pl['landings_en_route']} en-route landings, {Pl['n_clearances']} countries needing "
             f"clearance [{' '.join(Pl['distinct_clearance_countries'])}].")
    L.append(f"Difference (planner minus human): {dn:+,} NM, {dl:+d} landings, {dc:+d} clearances.")
    hb = sorted({c for l in H["legs"] for c in l["overflies_blocked"]} |
                {l["lands_in"] for l in H["legs"] if l["lands_in_blocked"]})
    pb = sorted({c for l in Pl["legs"] for c in l["overflies_blocked"]} |
                {l["lands_in"] for l in Pl["legs"] if l["lands_in_blocked"]})
    if hb:
        L.append(f"COMPLIANCE: the human route touches FCG-denied/blocked countries for this date: "
                 f"{', '.join(hb)}. (Denied by the FCG judge or the risk list — see last_run_blocks.json.)")
    if pb:
        L.append(f"COMPLIANCE: the planner route touches blocked countries: {', '.join(pb)} — this "
                 f"should not happen; check the blocks file.")
    if not hb and not pb:
        L.append("COMPLIANCE: neither route touches a denied or blocked country.")
    if rng:
        ho = [f"{l['from']}->{l['to']} ({l['pct_of_range']}%)" for l in H["legs"] if not l["within_85pct_range"]]
        po = [f"{l['from']}->{l['to']} ({l['pct_of_range']}%)" for l in Pl["legs"] if not l["within_85pct_range"]]
        if ho:
            L.append(f"FUEL: human legs over 85% of range: {', '.join(ho)} — likely a different aircraft, "
                     f"air refueling, or a tolerance the planner does not assume.")
        if po:
            L.append(f"FUEL: planner legs over 85% of range: {', '.join(po)}.")
        if not ho and not po:
            L.append(f"FUEL: every leg on both routes is within 85% of the aircraft's {rng:,.0f} NM range.")
    # the one-line take
    if hb and not pb:
        L.append("BOTTOM LINE: the human route is not fileable for this date under current FCG/risk "
                 "data; the planner route is.")
    elif dc < 0 and dn <= 0:
        L.append("BOTTOM LINE: planner route is shorter AND needs fewer clearances.")
    elif dc < 0:
        L.append(f"BOTTOM LINE: planner trades {dn:+,} NM for {-dc} fewer clearance(s) "
                 f"(~{dn/(-dc):,.0f} NM per clearance avoided).")
    elif dc > 0 and dn < 0:
        L.append(f"BOTTOM LINE: planner is {-dn:,} NM shorter but needs {dc} more clearance(s).")
    elif dc == 0 and abs(dn) < 50 and dl == 0:
        L.append("BOTTOM LINE: the planner reproduced the human route.")
    else:
        L.append("BOTTOM LINE: human route is at least as good on these numbers; the humans likely used "
                 "knowledge the planner lacks (fuel/MOG/support/agreements) — see the judge below.")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True, help="ICAO stops actually flown, comma-separated")
    ap.add_argument("--planner", default=None, help="override planner stops (else last_run_blocks.json)")
    ap.add_argument("--blocks-file", default="last_run_blocks.json")
    ap.add_argument("--aircraft", default=None)
    ap.add_argument("--date", default=None)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    blocks = json.load(open(args.blocks_file)) if os.path.exists(args.blocks_file) else {}
    if not blocks and not args.planner:
        sys.exit("no last_run_blocks.json — run country_route_planner.py first, or pass --planner")
    nodes, fcg, mapping, a2c, candidates, ourairports = P.build_dataset()
    blocked, _ = P.apply_risk(nodes, mapping, use_feed=False)
    blocked |= set(blocks.get("blocked_countries", []))
    date = args.date or blocks.get("date") or "unspecified"
    spec = P.load_aircraft(args.aircraft) if args.aircraft else blocks.get("aircraft")
    rng = (spec or {}).get("range_nm")

    human = resolve_chain(args.human.split(","), nodes, ourairports)
    planner = resolve_chain(args.planner.split(",") if args.planner else blocks["stops"], nodes, ourairports)
    if human[0] != planner[0] or human[-1] != planner[-1]:
        print(f"[compare] WARNING endpoints differ: human {human[0]}->{human[-1]}, "
              f"planner {planner[0]}->{planner[-1]} — comparison is apples to oranges")

    try:
        from eval_routes import polygon_overflights
    except Exception:
        polygon_overflights = None
    H = chain_facts(human, "human", nodes, ourairports, rng, blocked, polygon_overflights)
    Pl = chain_facts(planner, "planner", nodes, ourairports, rng, blocked, polygon_overflights)

    # ---- 1. numbers
    def leg_rows(F):
        return [f"| {l['from']} -> {l['to']} | {l['great_circle_nm']:,} | "
                f"{' '.join(l['overflies']) or 'water'} | {l['lands_in']} | "
                f"{str(l.get('pct_of_range', '')) + '%' if rng else ''} |" for l in F["legs"]]
    md = [f"# Planner vs human route — {human[0]} to {human[-1]}", "",
          f"Mission date {date}; aircraft {(spec or {}).get('type', '?')}"
          + (f" (range {rng:,.0f} NM, 85% = {0.85*rng:,.0f} NM)" if rng else ""),
          f"FCG/risk-blocked countries for this date: {', '.join(sorted(blocked & set(H['distinct_clearance_countries'] + Pl['distinct_clearance_countries']))) or 'none touched'}",
          "", "| | Human | Planner |", "|---|---|---|",
          f"| Stops | {' -> '.join(human)} | {' -> '.join(planner)} |",
          f"| Total NM | {H['total_nm']:,} | {Pl['total_nm']:,} |",
          f"| En-route landings | {H['landings_en_route']} | {Pl['landings_en_route']} |",
          f"| Countries needing clearance | {H['n_clearances']} | {Pl['n_clearances']} |",
          f"| Legs over 85% range | {sum(not l.get('within_85pct_range', True) for l in H['legs'])} | "
          f"{sum(not l.get('within_85pct_range', True) for l in Pl['legs'])} |",
          "", "## Human legs", "| leg | NM | overflies | lands | % range |", "|---|---|---|---|---|"] + leg_rows(H) + [
          "", "## Planner legs", "| leg | NM | overflies | lands | % range |", "|---|---|---|---|---|"] + leg_rows(Pl)

    # ---- 2. explain
    lines = explain(H, Pl, rng, blocked)
    md += ["", "## Explanation", ""] + [f"- {l}" for l in lines]
    print("\n".join(lines))

    # ---- chart
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for a, (k, t) in zip(axes, [("total_nm", "Total NM"), ("landings_en_route", "En-route landings"),
                                ("n_clearances", "Clearance countries")]):
        bars = a.bar(["human", "planner"], [H[k], Pl[k]], color=["#999999", "#4c72b0"])
        for b in bars:
            a.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():,}", ha="center", va="bottom")
        a.set_title(t)
    fig.suptitle(f"{human[0]} -> {human[-1]}: human vs planner ({date})")
    fig.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    png = f"{OUT}/compare_{ts}.png"
    fig.savefig(png, dpi=150)

    # ---- 3. judge
    if not args.no_judge:
        print(f"\n[judge] backend {P.LLM_BACKEND}, {args.runs} runs ...")
        codes = []
        for c in H["distinct_clearance_countries"] + Pl["distinct_clearance_countries"]:
            if c not in codes:
                codes.append(c)
        reasons = {}
        for k, v in blocks.get("verdicts", {}).items():
            if v and not v.get("allowed", True):
                reasons.setdefault(k.split("/")[0], str(v.get("reason", ""))[:160])
        payload = {"date": date,
                   "aircraft": (spec or {}).get("type", "unspecified") + (f" (range {rng:,.0f} NM)" if rng else ""),
                   "blocked": ", ".join(sorted(blocked & set(codes))) or "none on either route",
                   "block_reasons": json.dumps({c: reasons.get(c, "") for c in sorted(blocked & set(codes))}),
                   "human": json.dumps(H, indent=1), "planner": json.dumps(Pl, indent=1),
                   "evidence": json.dumps(fcg_evidence(codes, nodes, fcg), indent=1)}
        results = run_judge(payload, args.runs)
        from collections import Counter
        win, votes = Counter(r.get("winner") for r in results).most_common(1)[0]
        best = next(r for r in results if r.get("winner") == win)
        md += ["", f"## LLM judge: **{str(win).upper()}** ({votes}/{len(results)} runs agree, "
                   f"confidence {best.get('confidence')})", "",
               f"> {best.get('one_sentence_verdict', best.get('error', ''))}", "",
               f"Compliance — human: {best.get('compliance', {}).get('human')}, planner: "
               f"{best.get('compliance', {}).get('planner')}. {best.get('compliance', {}).get('notes', '')}", "",
               "**Key reasons**"] + [f"- {r}" for r in best.get("key_reasons", [])] + [
               "", "**What the humans may have known that the planner does not**"] + \
              [f"- {r}" for r in best.get("what_human_route_may_know", [])] + [
               "", "**Risks in the winning route**"] + [f"- {r}" for r in best.get("risks_in_winner", [])] + [
               "", "<sub>The judge saw raw facts and FCG text only — not the planner's score or penalties.</sub>"]
        json.dump({"human": H, "planner": Pl, "judge_runs": results, "majority": win},
                  open(f"{OUT}/compare_{ts}.json", "w"), indent=1, default=str)
        print(f"\n[judge] {win} ({votes}/{len(results)}): {best.get('one_sentence_verdict', '')}")

    md += ["", f"![chart]({os.path.basename(png)})"]
    path = f"{OUT}/compare_{ts}.md"
    open(path, "w").write("\n".join(md) + "\n")
    print(f"\n[compare] report -> {path}\n[compare] chart  -> {png}")


if __name__ == "__main__":
    main()
