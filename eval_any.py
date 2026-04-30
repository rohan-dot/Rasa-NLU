#!/usr/bin/env python3
"""
Universal BioASQ Evaluator
===========================
Handles ANY prediction format:
  - Standard BioASQ: {"questions": [{"id": ..., "exact_answer": ..., "ideal_answer": ...}]}
  - Flat dict:       {"question_id": "answer text", ...}
  - Coworker format: {"question_id": "Yes/No. explanation...", ...}

Usage:
    python eval_any.py --pred predictions.json --gold 13B1_golden.json
"""
import json, sys, argparse, re

def normalize(s):
    if isinstance(s, list):
        s = s[0] if s else ""
        if isinstance(s, list):
            s = s[0] if s else ""
    return str(s).lower().strip().strip('"\'.,!?')

def load_predictions(path):
    """Auto-detect prediction format and normalize to {id: {exact, ideal}}."""
    with open(path) as f:
        data = json.load(f)

    preds = {}

    # Format 1: Standard BioASQ {"questions": [...]}
    if "questions" in data and isinstance(data["questions"], list):
        for q in data["questions"]:
            qid = q.get("id", "")
            ea = q.get("exact_answer")
            ia = q.get("ideal_answer", "")
            # Flatten list-of-lists
            if isinstance(ea, list):
                flat = []
                for item in ea:
                    if isinstance(item, list):
                        flat.extend(item)
                    else:
                        flat.append(str(item))
                ea = flat
            preds[qid] = {"exact_answer": ea, "ideal_answer": ia}
        print(f"  Loaded {len(preds)} predictions (BioASQ format)")
        return preds

    # Format 2: Flat dict {"question_id": "answer text"}
    if isinstance(data, dict):
        for qid, answer in data.items():
            if qid in ("questions",):
                continue
            answer = str(answer).strip()

            # Try to detect yes/no from the answer text
            lower = answer.lower()
            if lower.startswith("yes") or lower.startswith("no"):
                exact = "yes" if lower.startswith("yes") else "no"
            elif lower.startswith('"yes') or lower.startswith('"no'):
                exact = "yes" if "yes" in lower[:5] else "no"
            else:
                # Try to extract a short exact answer (first sentence or phrase)
                first_sent = answer.split(".")[0].strip().strip('"')
                if len(first_sent) < 80:
                    exact = first_sent
                else:
                    exact = answer[:80]

            preds[qid] = {"exact_answer": exact, "ideal_answer": answer}
        print(f"  Loaded {len(preds)} predictions (flat dict format)")
        return preds

    print("ERROR: Unknown prediction format")
    sys.exit(1)

def load_gold(path):
    """Load gold standard BioASQ file."""
    with open(path) as f:
        data = json.load(f)
    golds = {}
    for q in data.get("questions", []):
        golds[q["id"]] = q
    print(f"  Loaded {len(golds)} gold questions")
    return golds

def main():
    ap = argparse.ArgumentParser(description="Universal BioASQ Evaluator")
    ap.add_argument("--pred", "-p", required=True, help="Predictions (any format)")
    ap.add_argument("--gold", "-g", required=True, help="Gold standard (BioASQ format)")
    args = ap.parse_args()

    print("=" * 60)
    print("BIOASQ UNIVERSAL EVAL")
    print("=" * 60)

    preds = load_predictions(args.pred)
    golds = load_gold(args.gold)

    stats = {"factoid": [0,0], "yesno": [0,0], "list": [0,0], "summary": [0,0]}
    errors = []
    matched = 0

    for qid, gold in golds.items():
        if qid not in preds:
            continue
        matched += 1
        pred = preds[qid]
        qtype = gold.get("type", "summary")
        body = gold.get("body", "")[:60]

        if qtype == "yesno":
            stats["yesno"][1] += 1
            g = gold.get("exact_answer", "")
            if isinstance(g, list): g = g[0] if g else ""
            g = normalize(g)

            pr = pred.get("exact_answer", "")
            if isinstance(pr, list): pr = pr[0] if pr else ""
            pr = normalize(pr)
            # Extract yes/no from longer answers
            if pr not in ("yes", "no"):
                if pr.startswith("yes"): pr = "yes"
                elif pr.startswith("no"): pr = "no"
                elif "yes" in pr[:20]: pr = "yes"
                elif "no" in pr[:20]: pr = "no"

            if g == pr:
                stats["yesno"][0] += 1
            else:
                errors.append(f"  YESNO WRONG: {body}... gold={g} pred={pr}")

        elif qtype == "factoid":
            stats["factoid"][1] += 1
            gold_ans = gold.get("exact_answer", [])
            gold_set = set()
            if isinstance(gold_ans, list):
                for item in gold_ans:
                    if isinstance(item, list):
                        for s in item: gold_set.add(normalize(s))
                    else: gold_set.add(normalize(str(item)))
            else: gold_set.add(normalize(str(gold_ans)))

            pred_ans = pred.get("exact_answer", "")
            if isinstance(pred_ans, list):
                flat = []
                for item in pred_ans:
                    if isinstance(item, list): flat.extend(item)
                    else: flat.append(str(item))
                pred_first = normalize(flat[0]) if flat else ""
            else:
                pred_first = normalize(str(pred_ans))

            match = any(pred_first in g or g in pred_first for g in gold_set if g)
            if match:
                stats["factoid"][0] += 1
            else:
                gold_show = list(gold_set)[:3]
                errors.append(f"  FACTOID WRONG: {body}... gold={gold_show} pred={pred_first}")

        elif qtype == "list":
            stats["list"][1] += 1
            gold_items = set()
            for item in gold.get("exact_answer", []):
                if isinstance(item, list):
                    for s in item: gold_items.add(normalize(s))
                else: gold_items.add(normalize(str(item)))

            pred_ans = pred.get("exact_answer", "")
            pred_items = set()
            if isinstance(pred_ans, list):
                for item in pred_ans:
                    if isinstance(item, list):
                        for s in item: pred_items.add(normalize(s))
                    else: pred_items.add(normalize(str(item)))
            elif isinstance(pred_ans, str):
                # Try splitting by commas or newlines
                for item in re.split(r'[,;\n]', pred_ans):
                    item = item.strip().strip('-•* ')
                    if item and len(item) > 1:
                        pred_items.add(normalize(item))

            overlap = sum(1 for pi in pred_items
                         if any(pi in g or g in pi for g in gold_items))
            if gold_items and overlap / len(gold_items) >= 0.3:
                stats["list"][0] += 1
            else:
                errors.append(f"  LIST LOW: {body}... gold={len(gold_items)} items, matched={overlap}/{len(pred_items)} pred")

        elif qtype == "summary":
            stats["summary"][1] += 1
            ideal = pred.get("ideal_answer", "")
            if len(ideal) > 20:
                stats["summary"][0] += 1
            else:
                errors.append(f"  SUMMARY EMPTY: {body}...")

    print(f"\n  Matched {matched}/{len(golds)} questions\n")
    print("-" * 40)
    total_right = 0
    total_all = 0
    for qtype, (right, total) in stats.items():
        if total > 0:
            pct = right / total * 100
            print(f"  {qtype:10s}: {right:3d}/{total:3d} correct ({pct:.0f}%)")
            total_right += right
            total_all += total
    if total_all:
        print(f"  {'OVERALL':10s}: {total_right:3d}/{total_all:3d} ({total_right/total_all*100:.0f}%)")

    print(f"\nERRORS ({len(errors)} total):")
    for e in errors:
        print(e)
    print("=" * 60)

if __name__ == "__main__":
    main()
