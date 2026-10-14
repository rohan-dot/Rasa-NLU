#!/usr/bin/env python3
"""
route_judge.py — LLM judge: the route we ACTUALLY flew vs the route the planner
produced. Independent of the planner's own cost model: the model sees the raw
facts (legs, distances, overflights, landings, fuel margin, FCG text for every
country touched, FCG-judge denials for the mission date) and must say which
route it would file and why, with a confidence and the reasons the human route
might still be right.

Uses the same LLM backend as the planner (env LLM_BACKEND / AGENT_MODEL ->
LiteLLM gateway, e.g. GPT 5.6 "sol"; otherwise local vLLM).

  python route_judge.py --reference "KSUU,TJSJ,SBFZ,FKKD" --aircraft C-17 \
      --blocks-file last_run_blocks.json --date 2026-11-01 --runs 3

Reads (if present) eval_results/endpoints_compare.csv, endpoints_legs.csv,
reference_compare.csv from eval_plots.py; if the endpoints test has not been
run it computes the planner chain itself. Writes
  eval_results/llm_route_verdict.md   (human-readable)
  eval_results/llm_route_verdict.json (structured; one entry per run + agreement)
"""

import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd

import country_route_planner as P

OUT = "eval_results"
os.makedirs(OUT, exist_ok=True)


def chain_facts(chain, label, nodes, ourairports, rng, blocked, polygon_ov=None):
    """Per-leg facts for a stop chain, no scoring — the judge does the judging."""
    def pt(ap):
        r = ourairports.loc[ap]
        return (float(r["latitude_deg"]), float(r["longitude_deg"])), str(r["iso3"])
    legs, countries = [], []
    for a, b in zip(chain, chain[1:]):
        pa, ca = pt(a); pb, cb = pt(b)
        _, clear, nm = P.direct_corridor(pa, pb, nodes, (ca, cb))
        if polygon_ov:
            try:
                clear = polygon_ov(pa, pb, (ca, cb))
            except Exception:
                pass
        leg = {"from": a, "to": b, "lands_in": cb, "great_circle_nm": round(nm),
               "overflies": list(clear),
               "overflies_blocked": sorted(set(clear) & blocked),
               "lands_in_blocked": cb in blocked}
        if rng:
            leg["pct_of_range"] = round(100 * nm / rng)
            leg["within_85pct_range"] = nm <= 0.85 * rng
        legs.append(leg)
        countries += [c for c in list(clear) + [cb] if c not in countries]
    return {"who": label, "stops": chain, "legs": legs,
            "total_nm": sum(l["great_circle_nm"] for l in legs),
            "landings_en_route": max(0, len(chain) - 2),
            "distinct_clearance_countries": countries,
            "n_clearances": len(countries)}


def fcg_evidence(codes, nodes, fcg, cap=1200):
    ev = {}
    for c in codes:
        try:
            name = nodes.loc[c, "country"]
        except Exception:
            name = c
        try:
            d = P.country_dossier(c, fcg)
        except Exception as e:
            d = f"(no FCG dossier: {e})"
        ev[c] = {"name": str(name), "fcg_excerpt": str(d)[:cap]}
    return ev


JUDGE_SYSTEM = ("You are a senior USAF mission planner and foreign-clearance officer "
                "reviewing two candidate routings for the same mission. Be rigorous, "
                "use only the facts given, and output strict JSON only.")

JUDGE_PROMPT = """Two routings for the same mission: the route ACTUALLY FLOWN by the human
planners ("human") and the route produced by an automated planner ("planner").
Decide which one you would file today and why.

Mission date: {date}. Aircraft: {aircraft}.
FCG judge (automated, reading the Foreign Clearance Guide for this date) DENIED
overflight/landing in these countries: {blocked}. Reasons: {block_reasons}

HUMAN route facts:
{human}

PLANNER route facts:
{planner}

FCG excerpts for every country either route overflies or lands in:
{evidence}

Judge on, in this order of importance:
1. Compliance: does either route overfly or land in a denied country, or a
   country whose FCG text requires lead time / conditions the date cannot meet?
2. Operational risk: number of diplomatic clearances needed (each is a point of
   failure), number of landings, legs near the fuel limit.
3. Efficiency: total distance.
Also state what the human route may know that the planner does not (fuel
availability, MOG, en-route support, standing agreements) — be specific if the
FCG text suggests it.

Return ONLY this JSON:
{{"winner": "human" | "planner" | "tie",
  "confidence": 0.0-1.0,
  "compliance": {{"human": "ok|concern|violation", "planner": "ok|concern|violation",
                  "notes": "..."}},
  "key_reasons": ["...", "..."],
  "what_human_route_may_know": ["..."],
  "risks_in_winner": ["..."],
  "one_sentence_verdict": "..."}}"""


def run_judge(payload, runs):
    results = []
    for k in range(runs):
        try:
            txt = P.llm_chat(JUDGE_PROMPT.format(**payload), system=JUDGE_SYSTEM)
            import re
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            if not m:
                raise ValueError(f"no JSON in reply: {txt[:150]!r}")
            js = json.loads(m.group(0))
            if not isinstance(js, dict):
                raise ValueError("non-dict JSON")
        except Exception as e:
            js = {"winner": "error", "confidence": 0, "error": str(e)[:300]}
        js["run"] = k + 1
        results.append(js)
        print(f"[judge] run {k+1}/{runs}: {js.get('winner')} "
              f"(conf {js.get('confidence')}) — {js.get('one_sentence_verdict', js.get('error',''))}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="ICAO stops actually flown, comma-separated")
    ap.add_argument("--aircraft", default=None)
    ap.add_argument("--blocks-file", default="last_run_blocks.json")
    ap.add_argument("--date", default=None)
    ap.add_argument("--runs", type=int, default=3, help="repeat the judge N times; report agreement")
    ap.add_argument("--planner-stops", default=None,
                    help="override: ICAO stops the planner chose (else read from blocks-file / "
                         "endpoints_compare.csv / recomputed)")
    args = ap.parse_args()
    print(f"[llm] backend: {P.LLM_BACKEND}")

    human = [s.strip().upper() for s in args.reference.split(",") if s.strip()]
    nodes, fcg, mapping, a2c, candidates, ourairports = P.build_dataset()
    blocked, _ = P.apply_risk(nodes, mapping, use_feed=False)
    blocks, block_reasons = {}, {}
    if args.blocks_file and os.path.exists(args.blocks_file):
        blocks = json.load(open(args.blocks_file))
        blocked |= set(blocks.get("blocked_countries", []))
        for k, v in blocks.get("verdicts", {}).items():      # keys look like "CODE/role/airport"
            if v and not v.get("allowed", True):
                block_reasons.setdefault(k.split("/")[0], str(v.get("reason", ""))[:160])
    date = args.date or blocks.get("date") or "unspecified"
    spec = P.load_aircraft(args.aircraft) if args.aircraft else (blocks.get("aircraft") or None)
    rng = (spec or {}).get("range_nm")

    # planner chain: explicit > blocks file > endpoints test > recompute
    if args.planner_stops:
        planner = [s.strip().upper() for s in args.planner_stops.split(",")]
    elif blocks.get("stops") and blocks["stops"][0] == human[0] and blocks["stops"][-1] == human[-1]:
        planner = list(blocks["stops"])
    elif os.path.exists(f"{OUT}/endpoints_compare.csv"):
        s = pd.read_csv(f"{OUT}/endpoints_compare.csv")
        planner = s[s.who == "planner"].iloc[0]["stops"].split(" -> ")
    else:
        if not rng:
            sys.exit("need --aircraft to recompute the planner's stops")
        def pt(ap_):
            r = ourairports.loc[ap_]
            return (float(r["latitude_deg"]), float(r["longitude_deg"])), str(r["iso3"])
        p0, c0 = pt(human[0]); p1, c1 = pt(human[-1])
        auto = P.auto_stop_search(p0, c0, human[0], p1, c1, human[-1],
                                  candidates, ourairports, nodes, blocked, set(), rng)
        if auto is None:
            sys.exit("planner found no feasible chain")
        planner = [human[0]] + auto + [human[-1]]
    print(f"[judge] human:   {' -> '.join(human)}")
    print(f"[judge] planner: {' -> '.join(planner)}")

    try:
        from eval_routes import polygon_overflights
    except Exception:
        polygon_overflights = None
    H = chain_facts(human, "human", nodes, ourairports, rng, blocked, polygon_overflights)
    Pl = chain_facts(planner, "planner", nodes, ourairports, rng, blocked, polygon_overflights)
    codes = []
    for c in H["distinct_clearance_countries"] + Pl["distinct_clearance_countries"]:
        if c not in codes:
            codes.append(c)
    ev = fcg_evidence(codes, nodes, fcg)

    payload = {"date": date, "aircraft": (spec or {}).get("type", args.aircraft or "unspecified")
               + (f" (range {rng:,.0f} NM)" if rng else ""),
               "blocked": ", ".join(sorted(blocked & set(codes))) or "none on either route",
               "block_reasons": json.dumps({c: block_reasons.get(c, "") for c in sorted(blocked & set(codes))}),
               "human": json.dumps(H, indent=1), "planner": json.dumps(Pl, indent=1),
               "evidence": json.dumps(ev, indent=1)}
    results = run_judge(payload, args.runs)

    winners = Counter(r.get("winner") for r in results)
    majority, votes = winners.most_common(1)[0]
    agreement = votes / len(results)
    best = next(r for r in results if r.get("winner") == majority)
    out = {"human_stops": human, "planner_stops": planner, "date": date,
           "backend": P.LLM_BACKEND, "model": os.environ.get("AGENT_MODEL") if P.LLM_BACKEND == "gateway" else P.VLLM_MODEL,
           "human_facts": H, "planner_facts": Pl, "runs": results,
           "majority_winner": majority, "agreement": agreement}
    json.dump(out, open(f"{OUT}/llm_route_verdict.json", "w"), indent=1, default=str)

    md = [f"# LLM route judge — human vs planner", "",
          f"Model: `{out['model']}` ({P.LLM_BACKEND}); mission date {date}; aircraft {payload['aircraft']}", "",
          f"- **Human route:** {' -> '.join(human)} — {H['total_nm']:,} NM, {H['landings_en_route']} landings, "
          f"{H['n_clearances']} clearance countries [{' '.join(H['distinct_clearance_countries'])}]",
          f"- **Planner route:** {' -> '.join(planner)} — {Pl['total_nm']:,} NM, {Pl['landings_en_route']} landings, "
          f"{Pl['n_clearances']} clearance countries [{' '.join(Pl['distinct_clearance_countries'])}]",
          f"- FCG-denied countries touched: {payload['blocked']}", "",
          f"## Verdict: **{majority.upper()}** ({votes}/{len(results)} runs agree, "
          f"confidence {best.get('confidence')})", "",
          f"> {best.get('one_sentence_verdict', '')}", "",
          f"**Compliance:** human = {best.get('compliance', {}).get('human')}, "
          f"planner = {best.get('compliance', {}).get('planner')}. {best.get('compliance', {}).get('notes', '')}", "",
          "**Key reasons**"] + [f"- {r}" for r in best.get("key_reasons", [])] + [
          "", "**What the human route may know that the planner does not**"] + \
         [f"- {r}" for r in best.get("what_human_route_may_know", [])] + [
          "", "**Risks in the winning route**"] + [f"- {r}" for r in best.get("risks_in_winner", [])] + [
          "", "## All runs", ""] + [f"- run {r['run']}: {r.get('winner')} (conf {r.get('confidence')}) — "
                                     f"{r.get('one_sentence_verdict', r.get('error', ''))}" for r in results] + [
          "", "_The judge saw raw facts only (distances, overflights, landings, fuel margin, FCG text, "
          "FCG-judge denials); it did not see the planner's score or the 300 NM clearance penalty._"]
    open(f"{OUT}/llm_route_verdict.md", "w").write("\n".join(md) + "\n")
    print(f"\n[judge] majority: {majority} ({votes}/{len(results)}) -> {OUT}/llm_route_verdict.md")


if __name__ == "__main__":
    main()
