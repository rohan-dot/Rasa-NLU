"""
BioASQ 14b - Main Entry Point
==============================
Hybrid retrieval: BioASQ PubMed service + FAISS training snippets.

Usage:
    # Phase A+ with hybrid retrieval (PubMed + FAISS)
    python main.py --phase A+ --test BioASQ-task14bPhaseA-testset1.json --train trainining14b.json

    # Phase A+ offline only (FAISS training snippets, no PubMed)
    python main.py --phase A+ --test testset.json --train trainining14b.json --no-pubmed

    # Phase B (gold snippets, no retrieval needed)
    python main.py --phase B --test BioASQ-task14bPhaseB-testset1.json --train trainining14b.json

    # Quick debug
    python main.py --phase A+ --test testset.json --train trainining14b.json --limit 3 --no-faiss
"""

import os
import json
import time
import logging
import argparse
from collections import defaultdict

from config import TRAINING_DATA_PATH, TEST_DATA_PATH, OUTPUT_DIR, EMBEDDING_MODEL, LOG_LEVEL
from data_loader import load_questions, TrainingIndex, format_submission
from llm_client import LLMClient
from agent import BioASQAgent

logger = logging.getLogger(__name__)


def setup_logging(level=LOG_LEVEL):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def evaluate(questions):
    results = defaultdict(list)
    for q in questions:
        if q.qtype == "yesno" and q.exact_answer and q.generated_exact:
            gold = q.exact_answer if isinstance(q.exact_answer, str) else str(q.exact_answer)
            pred = q.generated_exact if isinstance(q.generated_exact, str) else str(q.generated_exact)
            results["yesno_acc"].append(1.0 if gold.lower() == pred.lower() else 0.0)

        elif q.qtype == "factoid" and q.exact_answer and q.generated_exact:
            gold_set = set()
            for item in (q.exact_answer if isinstance(q.exact_answer, list) else []):
                if isinstance(item, list):
                    gold_set.update(i.lower() for i in item)
                else:
                    gold_set.add(str(item).lower())
            pred_set = set()
            for item in (q.generated_exact if isinstance(q.generated_exact, list) else []):
                if isinstance(item, list):
                    pred_set.update(i.lower() for i in item)
                else:
                    pred_set.add(str(item).lower())
            results["factoid_lenient"].append(1.0 if gold_set & pred_set else 0.0)

        elif q.qtype == "list" and q.exact_answer and q.generated_exact:
            gold_set = set()
            for item in (q.exact_answer if isinstance(q.exact_answer, list) else []):
                if isinstance(item, list):
                    gold_set.update(i.lower() for i in item)
                else:
                    gold_set.add(str(item).lower())
            pred_set = set()
            for item in (q.generated_exact if isinstance(q.generated_exact, list) else []):
                if isinstance(item, list):
                    pred_set.update(i.lower() for i in item)
                else:
                    pred_set.add(str(item).lower())
            if gold_set:
                p = len(gold_set & pred_set) / max(len(pred_set), 1)
                r = len(gold_set & pred_set) / len(gold_set)
                f1 = 2 * p * r / max(p + r, 1e-8)
                results["list_f1"].append(f1)

        if q.ideal_answer and q.generated_ideal:
            gold_tok = set(q.ideal_answer.lower().split())
            pred_tok = set(q.generated_ideal.lower().split())
            if gold_tok and pred_tok:
                p = len(gold_tok & pred_tok) / len(pred_tok)
                r = len(gold_tok & pred_tok) / len(gold_tok)
                f1 = 2 * p * r / max(p + r, 1e-8)
                results["ideal_word_f1"].append(f1)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for metric, vals in sorted(results.items()):
        if vals:
            print(f"  {metric:30s}: {sum(vals)/len(vals):.4f}  (n={len(vals)})")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="BioASQ 14b Agentic QA")
    parser.add_argument("--phase", choices=["A+", "B"], default="A+")
    parser.add_argument("--test", type=str, default=TEST_DATA_PATH)
    parser.add_argument("--train", type=str, default=TRAINING_DATA_PATH)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--no-faiss", action="store_true",
                        help="Skip FAISS index (random few-shot)")
    parser.add_argument("--no-pubmed", action="store_true",
                        help="Skip BioASQ PubMed retrieval (FAISS only)")
    parser.add_argument("--vllm-url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    setup_logging()

    if args.vllm_url:
        import config
        config.VLLM_BASE_URL = args.vllm_url
    if args.model:
        import config
        config.VLLM_MODEL = args.model

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load Test Data ──────────────────────────────────────────────
    is_phase_b = (args.phase == "B")
    test_questions = load_questions(args.test, is_training=is_phase_b)
    if args.limit:
        test_questions = test_questions[:args.limit]

    # ── Load Training Data & Build FAISS ────────────────────────────
    training_index = None
    if os.path.exists(args.train):
        train_questions = load_questions(args.train, is_training=True)
        training_index = TrainingIndex(train_questions, EMBEDDING_MODEL)
        if not args.no_faiss:
            training_index.build()
        else:
            logger.info("FAISS skipped — random few-shot fallback")
    else:
        logger.warning(f"Training data not found at {args.train}")

    # ── Initialize BioASQ PubMed Retriever ──────────────────────────
    pubmed_retriever = None
    if args.phase == "A+" and not args.no_pubmed:
        from bioasq_retriever import BioASQPubMedRetriever
        pubmed_retriever = BioASQPubMedRetriever()
        logger.info("BioASQ PubMed retriever enabled (no API key needed)")
    elif args.phase == "A+" and args.no_pubmed:
        logger.info("PubMed retrieval disabled — using FAISS training snippets only")

    # ── Initialize & Run ────────────────────────────────────────────
    from config import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY
    llm = LLMClient(base_url=VLLM_BASE_URL, model=VLLM_MODEL, api_key=VLLM_API_KEY)

    agent = BioASQAgent(
        llm=llm,
        training_index=training_index,
        pubmed_retriever=pubmed_retriever,
    )

    start = time.time()
    logger.info(f"\nPhase {args.phase} | {len(test_questions)} questions")
    if pubmed_retriever:
        logger.info("Retrieval: BioASQ PubMed + FAISS training snippets (hybrid)")
    elif args.phase == "B":
        logger.info("Retrieval: Gold snippets from test set")
    else:
        logger.info("Retrieval: FAISS training snippets only")

    answered = agent.answer_batch(test_questions, phase=args.phase)

    elapsed = time.time() - start
    logger.info(f"\nDone in {elapsed:.1f}s ({elapsed/max(len(answered),1):.1f}s/q)")

    # ── Save Submission ─────────────────────────────────────────────
    output_path = args.output or os.path.join(OUTPUT_DIR, f"bioasq_{args.phase.replace('+','plus')}_submission.json")
    submission = format_submission(answered)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved: {output_path}")

    types = {}
    for q in answered:
        types[q.qtype] = types.get(q.qtype, 0) + 1
    has_exact = sum(1 for q in answered if q.generated_exact is not None)
    has_ideal = sum(1 for q in answered if q.generated_ideal)
    logger.info(f"Types: {types}")
    logger.info(f"Generated: {has_exact} exact, {has_ideal} ideal")

    if args.eval:
        evaluate(answered)


if __name__ == "__main__":
    main()
