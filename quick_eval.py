#!/usr/bin/env python3
"""
Quick comparison: your submission vs gold answers.
Prints a compact error report you can screenshot.

Usage:
    python quick_eval.py \
        --pred submission_phase13.json \
        --gold 13B4_golden_bioasq_taskb_format.json
"""
import json, sys, argparse

def normalize(s):
    return s.lower().strip().strip('"\'.,!?')

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--gold", required=True)
    args = p.parse_args()

    with open(args.pred) as f: preds = {q["id"]: q for q in json.load(f).get("questions", [])}
    with open(args.gold) as f: golds = {q["id"]: q for q in json.load(f).get("questions", [])}

    stats = {"factoid": [0,0], "yesno": [0,0], "list": [0,0], "summary": [0,0]}
    errors = []

    for qid, gold in golds.items():
        if qid not in preds: continue
        pred = preds[qid]
        qtype = gold.get("type", "summary")
        body = gold.get("body", "")[:60]

        if qtype == "yesno":
            stats["yesno"][1] += 1
            g = normalize(str(gold.get("exact_answer", "")))
            if isinstance(g, list): g = normalize(str(g[0])) if g else ""
            pr = normalize(str(pred.get("exact_answer", "")))
            if g == pr:
                stats["yesno"][0] += 1
            else:
                errors.append(f"  YESNO WRONG: {body}... gold={g} pred={pr}")

        elif qtype == "factoid":
            stats["factoid"][1] += 1
            # Gold can be list of lists (synonyms)
            gold_ans = gold.get("exact_answer", [])
            gold_set = set()
            if isinstance(gold_ans, list):
                for item in gold_ans:
                    if isinstance(item, list):
                        for s in item: gold_set.add(normalize(s))
                    else: gold_set.add(normalize(str(item)))
            else: gold_set.add(normalize(str(gold_ans)))

            pred_ans = pred.get("exact_answer", [])
            if isinstance(pred_ans, str): pred_ans = [pred_ans]
            if not isinstance(pred_ans, list): pred_ans = [str(pred_ans)]
            pred_first = normalize(pred_ans[0]) if pred_ans else ""

            # Check if pred matches any gold synonym
            match = any(pred_first in g or g in pred_first for g in gold_set if g)
            if match:
                stats["factoid"][0] += 1
            else:
                gold_show = list(gold_set)[:3]
                errors.append(f"  FACTOID WRONG: {body}... gold={gold_show} pred={pred_first}")

        elif qtype == "list":
            stats["list"][1] += 1
            # Just count overlap
            gold_items = set()
            for item in gold.get("exact_answer", []):
                if isinstance(item, list):
                    for s in item: gold_items.add(normalize(s))
                else: gold_items.add(normalize(str(item)))

            pred_items = set()
            for item in pred.get("exact_answer", []):
                if isinstance(item, list):
                    for s in item: pred_items.add(normalize(s))
                else: pred_items.add(normalize(str(item)))

            overlap = sum(1 for pi in pred_items if any(pi in g or g in pi for g in gold_items))
            if gold_items and overlap / len(gold_items) >= 0.3:
                stats["list"][0] += 1
            else:
                errors.append(f"  LIST LOW: {body}... gold={len(gold_items)} items, matched={overlap}")

        elif qtype == "summary":
            stats["summary"][1] += 1
            # Just check it's not empty
            ideal = pred.get("ideal_answer", "")
            if len(ideal) > 20:
                stats["summary"][0] += 1
            else:
                errors.append(f"  SUMMARY EMPTY: {body}...")

    print("=" * 60)
    print("BIOASQ QUICK EVAL")
    print("=" * 60)
    total_right = 0
    total_all = 0
    for qtype, (right, total) in stats.items():
        if total > 0:
            pct = right/total*100
            print(f"  {qtype:10s}: {right:3d}/{total:3d} correct ({pct:.0f}%)")
            total_right += right
            total_all += total
    if total_all:
        print(f"  {'OVERALL':10s}: {total_right:3d}/{total_all:3d} ({total_right/total_all*100:.0f}%)")
    print()
    print(f"ERRORS ({len(errors)} total):")
    for e in errors:
        print(e)
    print("=" * 60)

if __name__ == "__main__":
    main()
