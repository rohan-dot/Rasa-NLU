#!/usr/bin/env python3
"""
compare_extracts.py — Grade a candidate extraction against a gold standard.

Workflow:
  1. Build the gold set with the strong model (Opus/Sonnet via LiteLLM gateway):
       ... AGENT_MODEL=claude-opus-4-8 python fcg_extract.py ... --out gold.csv
  2. Build the candidate with the weak model (Gemma-4 via vLLM):
       ... AGENT_MODEL=<gemma-id> python fcg_extract.py ... --out gemma.csv
     (delete data/work/checkpoint.json between the two so each is really its own
      model — the checkpoint does not key on model.)
  3. Grade:
       python compare_extracts.py --gold gold.csv --candidate gemma.csv

The gold set is a REFERENCE, not absolute truth — the strong model can miss or
over-quote too. So this tool shows you WHERE they differ (with both values), not
just a score, so you can judge disagreements yourself.

Per extractable field, the candidate vs gold pair falls into one of:
  - agree_both_na    : both said NA (correct rejection)
  - agree_answered   : both answered AND overlap substantially (match)
  - disagree_content : both answered but the text differs a lot (review these)
  - missed           : gold answered, candidate said NA (weak model missed it)
  - extra            : candidate answered, gold said NA (weak model over-reached
                       or found something gold missed — worth a look)

Workflow items (never sent to the model) are excluded from scoring automatically.
"""

import argparse
import csv
import json
import re
from pathlib import Path

META_COLS = {"country", "country_name", "alpha2", "alpha3"}


def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[\u2018\u2019]", "'", s)
    s = re.sub(r"[\u201c\u201d]", '"', s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tokens(s: str) -> set:
    return set(norm(s).split())


def similarity(a: str, b: str) -> float:
    """Token Jaccard — cheap, robust to reordering and minor wording diffs."""
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def load(path: Path):
    rows = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["country"]] = row
    return rows


def is_na(v: str) -> bool:
    return (v or "").strip().upper() in ("", "NA", "N/A", "NONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="Strong-model CSV (reference)")
    ap.add_argument("--candidate", required=True, help="Weak-model CSV to grade")
    ap.add_argument("--checklist", default="checklist_enriched.json",
                    help="Used to score only extract items, skip workflow items")
    ap.add_argument("--sim-threshold", type=float, default=0.5,
                    help="Token-overlap above this = content match (default 0.5)")
    ap.add_argument("--show", type=int, default=25,
                    help="How many disagreements/misses to print in detail")
    ap.add_argument("--out", help="Optional: write a per-field diff CSV here")
    args = ap.parse_args()

    gold = load(Path(args.gold))
    cand = load(Path(args.candidate))

    extract_cols = None
    clp = Path(args.checklist)
    if clp.exists():
        items = json.loads(clp.read_text(encoding="utf-8"))
        def col(it):
            name = it.get("id") or it.get("name") or it.get("question")
            c = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip()).strip("_")[:80]
            return c
        # replicate dedupe suffixing from the extractor
        seen, cols = {}, []
        for it in items:
            if not isinstance(it, dict):
                cols.append((col({"id": it}), "extract")); continue
            c = col(it); n = seen.get(c, 0); seen[c] = n + 1
            cols.append((f"{c}_{n+1}" if n else c, (it.get("type") or "extract").lower()))
        extract_cols = [c for c, t in cols if t == "extract"]

    countries = sorted(set(gold) & set(cand))
    if not countries:
        raise SystemExit("No overlapping countries between the two CSVs.")

    all_item_cols = [c for c in next(iter(gold.values())).keys()
                     if c not in META_COLS and not c.endswith("__src")]
    score_cols = extract_cols if extract_cols else all_item_cols

    tally = {k: 0 for k in
             ("agree_both_na", "agree_answered", "disagree_content", "missed", "extra")}
    details = []

    for country in countries:
        g, c = gold[country], cand.get(country, {})
        for col in score_cols:
            gv, cv = g.get(col, "NA"), c.get(col, "NA")
            gna, cna = is_na(gv), is_na(cv)
            if gna and cna:
                tally["agree_both_na"] += 1
            elif not gna and cna:
                tally["missed"] += 1
                details.append(("missed", country, col, gv, cv))
            elif gna and not cna:
                tally["extra"] += 1
                details.append(("extra", country, col, gv, cv))
            else:
                sim = similarity(gv, cv)
                if sim >= args.sim_threshold:
                    tally["agree_answered"] += 1
                else:
                    tally["disagree_content"] += 1
                    details.append(("disagree_content", country, col, gv, cv))

    total = sum(tally.values())
    answered_gold = tally["agree_answered"] + tally["disagree_content"] + tally["missed"]
    agree = tally["agree_both_na"] + tally["agree_answered"]

    print(f"\n=== Gemma vs Gold — {len(countries)} countries x {len(score_cols)} "
          f"extractable fields = {total} comparisons ===\n")
    print(f"  agree (both NA)      : {tally['agree_both_na']}")
    print(f"  agree (answered ~=)  : {tally['agree_answered']}")
    print(f"  DISAGREE (content)   : {tally['disagree_content']}")
    print(f"  MISSED (gold had it) : {tally['missed']}")
    print(f"  EXTRA (cand only)    : {tally['extra']}")
    print(f"\n  Overall agreement    : {agree}/{total} = {100*agree/total:.1f}%")
    if answered_gold:
        recall = tally["agree_answered"] / answered_gold
        print(f"  Recall on gold's answers (content-match / gold-answered): "
              f"{tally['agree_answered']}/{answered_gold} = {100*recall:.1f}%")
    print(f"\n  Read: MISSED = weak model failed to find what the strong model did.")
    print(f"        DISAGREE = both answered differently — eyeball these, either")
    print(f"        could be right. EXTRA = candidate found something gold didn't.")

    # per-field agreement, worst first — shows which checklist items are hardest
    per_field = {}
    for country in countries:
        g, c = gold[country], cand.get(country, {})
        for col in score_cols:
            gv, cv = g.get(col, "NA"), c.get(col, "NA")
            gna, cna = is_na(gv), is_na(cv)
            ok = (gna and cna) or (not gna and not cna and
                                   similarity(gv, cv) >= args.sim_threshold)
            d = per_field.setdefault(col, [0, 0])
            d[0] += 1 if ok else 0
            d[1] += 1
    worst = sorted(per_field.items(), key=lambda kv: kv[1][0] / kv[1][1])[:10]
    print("\n  Weakest fields (lowest agreement — candidates for route/prompt work):")
    for col, (ok, n) in worst:
        print(f"    {ok}/{n}  {col}")

    if args.show and details:
        order = {"missed": 0, "disagree_content": 1, "extra": 2}
        details.sort(key=lambda d: order.get(d[0], 9))
        print(f"\n=== First {min(args.show, len(details))} differences ===")
        for kind, country, col, gv, cv in details[: args.show]:
            print(f"\n[{kind}] {country} · {col}")
            print(f"   GOLD: {gv[:200]}")
            print(f"   CAND: {cv[:200]}")

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kind", "country", "field", "gold", "candidate"])
            for kind, country, col, gv, cv in details:
                w.writerow([kind, country, col, gv, cv])
        print(f"\nWrote per-field diffs -> {args.out}")


if __name__ == "__main__":
    main()
