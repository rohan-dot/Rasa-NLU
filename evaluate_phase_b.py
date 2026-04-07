#!/usr/bin/env python3
"""
BioASQ Phase B — Offline Evaluation Against Training Gold Answers
=================================================================
Run this BEFORE submitting to measure your system's quality on
training13b data. Mimics the official BioASQ evaluation metrics.

Usage:
    python evaluate_phase_b.py \
        --predictions  my_predictions.json \
        --gold         training13b.json \
        --num-questions 100

This will report:
  - Factoid: Strict Accuracy, Lenient Accuracy, MRR
  - List:    Precision, Recall, F1
  - Yes/No:  Accuracy, F1(yes), F1(no), Macro F1
  - Summary: ROUGE-1, ROUGE-2, ROUGE-L (if rouge_score installed)
"""

import argparse
import json
import re
import sys
from collections import defaultdict

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def evaluate_factoid(predictions: list[dict], gold_map: dict):
    """Evaluate factoid questions — strict acc, lenient acc, MRR."""
    strict_correct = 0
    lenient_correct = 0
    reciprocal_rank_sum = 0.0
    total = 0

    for pred in predictions:
        qid = pred["id"]
        if qid not in gold_map:
            continue
        gold = gold_map[qid]
        if gold.get("type") != "factoid":
            continue

        total += 1

        # Gold exact answers can be a list of lists (synonyms)
        gold_exact = gold.get("exact_answer", [])
        gold_normalized = set()
        if isinstance(gold_exact, list):
            for item in gold_exact:
                if isinstance(item, list):
                    for syn in item:
                        gold_normalized.add(normalize(syn))
                else:
                    gold_normalized.add(normalize(str(item)))
        else:
            gold_normalized.add(normalize(str(gold_exact)))

        # Predicted exact answers
        pred_exact = pred.get("exact_answer", [])
        if isinstance(pred_exact, str):
            pred_exact = [pred_exact]
        if not isinstance(pred_exact, list):
            pred_exact = [str(pred_exact)]

        # Strict: first answer matches
        if pred_exact and normalize(pred_exact[0]) in gold_normalized:
            strict_correct += 1

        # Lenient: any answer matches
        found = False
        for i, pa in enumerate(pred_exact[:5]):
            if normalize(pa) in gold_normalized:
                if not found:
                    reciprocal_rank_sum += 1.0 / (i + 1)
                    found = True
                lenient_correct += 1
                break

        if not found:
            reciprocal_rank_sum += 0.0

    if total == 0:
        return {"total": 0}

    return {
        "total": total,
        "strict_accuracy": strict_correct / total,
        "lenient_accuracy": lenient_correct / total,
        "MRR": reciprocal_rank_sum / total,
    }


def evaluate_yesno(predictions: list[dict], gold_map: dict):
    """Evaluate yes/no questions — accuracy and macro F1."""
    correct = 0
    total = 0
    tp_yes = fp_yes = fn_yes = 0
    tp_no = fp_no = fn_no = 0

    for pred in predictions:
        qid = pred["id"]
        if qid not in gold_map:
            continue
        gold = gold_map[qid]
        if gold.get("type") != "yesno":
            continue

        total += 1

        gold_ans = gold.get("exact_answer", "")
        if isinstance(gold_ans, list):
            gold_ans = gold_ans[0] if gold_ans else ""
        gold_ans = str(gold_ans).lower().strip()

        pred_ans = pred.get("exact_answer", "")
        if isinstance(pred_ans, list):
            pred_ans = pred_ans[0] if pred_ans else ""
        pred_ans = str(pred_ans).lower().strip()

        if gold_ans == pred_ans:
            correct += 1

        if gold_ans == "yes":
            if pred_ans == "yes":
                tp_yes += 1
            else:
                fn_yes += 1
                fp_no += 1
        else:
            if pred_ans == "no":
                tp_no += 1
            else:
                fn_no += 1
                fp_yes += 1

    if total == 0:
        return {"total": 0}

    def f1(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "total": total,
        "accuracy": correct / total,
        "f1_yes": f1(tp_yes, fp_yes, fn_yes),
        "f1_no": f1(tp_no, fp_no, fn_no),
        "macro_f1": (f1(tp_yes, fp_yes, fn_yes) +
                     f1(tp_no, fp_no, fn_no)) / 2,
    }


def evaluate_list(predictions: list[dict], gold_map: dict):
    """Evaluate list questions — mean precision, recall, F1."""
    precisions = []
    recalls = []
    f1s = []
    total = 0

    for pred in predictions:
        qid = pred["id"]
        if qid not in gold_map:
            continue
        gold = gold_map[qid]
        if gold.get("type") != "list":
            continue

        total += 1

        # Gold: list of lists (synonyms)
        gold_exact = gold.get("exact_answer", [])
        gold_items = set()
        for item in gold_exact:
            if isinstance(item, list):
                for syn in item:
                    gold_items.add(normalize(syn))
            else:
                gold_items.add(normalize(str(item)))

        # Predicted
        pred_exact = pred.get("exact_answer", [])
        pred_items = set()
        if isinstance(pred_exact, list):
            for item in pred_exact:
                if isinstance(item, list):
                    for syn in item:
                        pred_items.add(normalize(syn))
                else:
                    pred_items.add(normalize(str(item)))
        else:
            pred_items.add(normalize(str(pred_exact)))

        # Calculate overlap
        if not gold_items:
            continue

        matched = 0
        for pi in pred_items:
            if pi in gold_items:
                matched += 1

        prec = matched / len(pred_items) if pred_items else 0
        rec = matched / len(gold_items) if gold_items else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    if total == 0:
        return {"total": 0}

    return {
        "total": total,
        "mean_precision": sum(precisions) / len(precisions),
        "mean_recall": sum(recalls) / len(recalls),
        "mean_f1": sum(f1s) / len(f1s),
    }


def evaluate_summary(predictions: list[dict], gold_map: dict):
    """Evaluate summary / ideal answers with ROUGE."""
    if not HAS_ROUGE:
        return {"error": "Install rouge_score: pip install rouge-score"}

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    r1s, r2s, rLs = [], [], []
    total = 0

    for pred in predictions:
        qid = pred["id"]
        if qid not in gold_map:
            continue
        gold = gold_map[qid]

        gold_ideal = gold.get("ideal_answer", "")
        if isinstance(gold_ideal, list):
            gold_ideal = gold_ideal[0] if gold_ideal else ""

        pred_ideal = pred.get("ideal_answer", "")
        if not gold_ideal or not pred_ideal:
            continue

        total += 1
        scores = scorer.score(gold_ideal, pred_ideal)
        r1s.append(scores['rouge1'].fmeasure)
        r2s.append(scores['rouge2'].fmeasure)
        rLs.append(scores['rougeL'].fmeasure)

    if total == 0:
        return {"total": 0}

    return {
        "total": total,
        "rouge1_f1": sum(r1s) / len(r1s),
        "rouge2_f1": sum(r2s) / len(r2s),
        "rougeL_f1": sum(rLs) / len(rLs),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BioASQ Phase B predictions against gold"
    )
    parser.add_argument("--predictions", "-p", required=True)
    parser.add_argument("--gold", "-g", required=True,
                        help="Training JSON with gold answers")
    parser.add_argument("--num-questions", "-n", type=int, default=None,
                        help="Limit eval to first N gold questions")
    args = parser.parse_args()

    with open(args.predictions) as f:
        preds = json.load(f).get("questions", [])
    with open(args.gold) as f:
        gold_data = json.load(f).get("questions", [])

    if args.num_questions:
        gold_data = gold_data[:args.num_questions]

    gold_map = {q["id"]: q for q in gold_data}

    print("=" * 60)
    print("BioASQ Phase B Evaluation")
    print("=" * 60)

    # Factoid
    factoid_results = evaluate_factoid(preds, gold_map)
    if factoid_results["total"] > 0:
        print(f"\nFACTOID ({factoid_results['total']} questions):")
        print(f"  Strict Accuracy: {factoid_results['strict_accuracy']:.3f}")
        print(f"  Lenient Accuracy: {factoid_results['lenient_accuracy']:.3f}")
        print(f"  MRR:             {factoid_results['MRR']:.3f}")

    # Yes/No
    yesno_results = evaluate_yesno(preds, gold_map)
    if yesno_results["total"] > 0:
        print(f"\nYES/NO ({yesno_results['total']} questions):")
        print(f"  Accuracy: {yesno_results['accuracy']:.3f}")
        print(f"  Macro F1: {yesno_results['macro_f1']:.3f}")

    # List
    list_results = evaluate_list(preds, gold_map)
    if list_results["total"] > 0:
        print(f"\nLIST ({list_results['total']} questions):")
        print(f"  Mean Precision: {list_results['mean_precision']:.3f}")
        print(f"  Mean Recall:    {list_results['mean_recall']:.3f}")
        print(f"  Mean F1:        {list_results['mean_f1']:.3f}")

    # Summary (ROUGE on ideal answers — all types)
    summary_results = evaluate_summary(preds, gold_map)
    if isinstance(summary_results.get("total"), int) and summary_results["total"] > 0:
        print(f"\nIDEAL ANSWERS — ROUGE ({summary_results['total']} questions):")
        print(f"  ROUGE-1 F1: {summary_results['rouge1_f1']:.3f}")
        print(f"  ROUGE-2 F1: {summary_results['rouge2_f1']:.3f}")
        print(f"  ROUGE-L F1: {summary_results['rougeL_f1']:.3f}")
    elif "error" in summary_results:
        print(f"\nROUGE: {summary_results['error']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
