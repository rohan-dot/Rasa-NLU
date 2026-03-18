"""
BioASQ 14b - Main Entry Point
==============================
Run the agentic QA pipeline for BioASQ Task 14b.

Usage:
    # Phase A+ (questions only → retrieve + answer, default)
    python main.py --phase A+ --test BioASQ-task14bPhaseA-testset1.json

    # Phase B (questions + gold docs/snippets → answer)
    python main.py --phase B --test BioASQ-task14b-testset1-phaseB.json

    # Evaluate on training data (dry-run with gold answers for scoring)
    python main.py --phase B --test trainining14b.json --eval --limit 20

    # Skip FAISS index building (faster startup, random few-shot selection)
    python main.py --phase A+ --test testset.json --no-faiss

Environment variables:
    VLLM_BASE_URL     vLLM server URL (default: http://g52lambda02.example.com:8000/v1)
    VLLM_MODEL        Model name (default: google/gemma-3-27b-it)
    TRAINING_DATA_PATH  Path to training JSON
    PUBMED_EMAIL      Email for NCBI E-utilities
    PUBMED_API_KEY    NCBI API key (optional, raises rate limit)
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

from config import (
    TRAINING_DATA_PATH,
    TEST_DATA_PATH,
    OUTPUT_DIR,
    PHASE_APLUS_OUTPUT,
    PHASE_A_OUTPUT,
    PHASE_B_OUTPUT,
    EMBEDDING_MODEL,
    LOG_LEVEL,
)
from data_loader import (
    load_questions,
    TrainingIndex,
    format_submission_phaseA,
    format_submission_phaseB,
)
from llm_client import LLMClient
from retriever import PubMedRetriever
from agent import BioASQAgent

logger = logging.getLogger(__name__)


def setup_logging(level: str = LOG_LEVEL):
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noise from HTTP libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def evaluate_answers(questions, limit=None):
    """Quick evaluation against gold answers (for dev/debug).

    Computes simple metrics:
    - Exact match accuracy for yesno
    - Token overlap F1 for factoid/list
    - ROUGE-like overlap for ideal answers
    """
    from collections import defaultdict
    results = defaultdict(list)

    for q in questions[:limit]:
        qtype = q.qtype

        # Exact answer evaluation
        if qtype == "yesno" and q.exact_answer and q.generated_exact:
            gold = q.exact_answer.lower() if isinstance(q.exact_answer, str) else str(q.exact_answer).lower()
            pred = q.generated_exact.lower() if isinstance(q.generated_exact, str) else str(q.generated_exact).lower()
            results["yesno_acc"].append(1.0 if gold == pred else 0.0)

        elif qtype == "factoid" and q.exact_answer and q.generated_exact:
            # Lenient: check if any gold answer appears in top-5 predictions
            gold_set = set()
            if isinstance(q.exact_answer, list):
                for item in q.exact_answer:
                    if isinstance(item, list):
                        gold_set.update(i.lower() for i in item)
                    else:
                        gold_set.add(str(item).lower())

            pred_set = set()
            if isinstance(q.generated_exact, list):
                for item in q.generated_exact[:5]:
                    if isinstance(item, list):
                        pred_set.update(i.lower() for i in item)
                    else:
                        pred_set.add(str(item).lower())

            # Lenient accuracy: any overlap?
            hit = 1.0 if gold_set & pred_set else 0.0
            results["factoid_lenient_acc"].append(hit)

        elif qtype == "list" and q.exact_answer and q.generated_exact:
            gold_set = set()
            if isinstance(q.exact_answer, list):
                for item in q.exact_answer:
                    if isinstance(item, list):
                        gold_set.update(i.lower() for i in item)
                    else:
                        gold_set.add(str(item).lower())

            pred_set = set()
            if isinstance(q.generated_exact, list):
                for item in q.generated_exact:
                    if isinstance(item, list):
                        pred_set.update(i.lower() for i in item)
                    else:
                        pred_set.add(str(item).lower())

            if gold_set:
                precision = len(gold_set & pred_set) / max(len(pred_set), 1)
                recall = len(gold_set & pred_set) / len(gold_set)
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                results["list_f1"].append(f1)

        # Ideal answer evaluation (simple word overlap F1)
        if q.ideal_answer and q.generated_ideal:
            gold_tokens = set(q.ideal_answer.lower().split())
            pred_tokens = set(q.generated_ideal.lower().split())
            if gold_tokens and pred_tokens:
                precision = len(gold_tokens & pred_tokens) / len(pred_tokens)
                recall = len(gold_tokens & pred_tokens) / len(gold_tokens)
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                results["ideal_token_f1"].append(f1)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for metric, values in sorted(results.items()):
        if values:
            mean_val = sum(values) / len(values)
            print(f"  {metric:30s}: {mean_val:.4f}  (n={len(values)})")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="BioASQ 14b Agentic QA Pipeline")
    parser.add_argument(
        "--phase", choices=["A", "A+", "B"], default="A+",
        help="Which phase to run: A (retrieval only), A+ (retrieve+answer), B (answer with gold docs)"
    )
    parser.add_argument(
        "--test", type=str, default=TEST_DATA_PATH,
        help="Path to the test set JSON file"
    )
    parser.add_argument(
        "--train", type=str, default=TRAINING_DATA_PATH,
        help="Path to the training set JSON file"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: auto-generated based on phase)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of questions to process (for debugging)"
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run evaluation against gold answers (requires training data as test input)"
    )
    parser.add_argument(
        "--no-faiss", action="store_true",
        help="Skip FAISS index building (use random few-shot selection)"
    )
    parser.add_argument(
        "--vllm-url", type=str, default=None,
        help="Override vLLM base URL"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model name"
    )

    args = parser.parse_args()
    setup_logging()

    # Override config from CLI if provided
    if args.vllm_url:
        os.environ["VLLM_BASE_URL"] = args.vllm_url
        import config
        config.VLLM_BASE_URL = args.vllm_url
    if args.model:
        os.environ["VLLM_MODEL"] = args.model
        import config
        config.VLLM_MODEL = args.model

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load Data ───────────────────────────────────────────────────
    logger.info("Loading test data...")
    is_phase_b = (args.phase == "B")
    test_questions = load_questions(args.test, is_training=is_phase_b)

    if args.limit:
        test_questions = test_questions[:args.limit]
        logger.info(f"Limited to {args.limit} questions")

    # ── Load Training Data & Build Index ────────────────────────────
    training_index = None
    if os.path.exists(args.train):
        logger.info("Loading training data for few-shot retrieval...")
        train_questions = load_questions(args.train, is_training=True)

        if not args.no_faiss:
            logger.info("Building FAISS index over training questions...")
            training_index = TrainingIndex(train_questions, EMBEDDING_MODEL)
            training_index.build()
        else:
            training_index = TrainingIndex(train_questions, EMBEDDING_MODEL)
            logger.info("FAISS index skipped (--no-faiss). Using random few-shot selection.")
    else:
        logger.warning(f"Training data not found at {args.train}. Running without few-shot examples.")

    # ── Initialize Components ───────────────────────────────────────
    logger.info("Initializing LLM client...")
    from config import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY
    llm = LLMClient(base_url=VLLM_BASE_URL, model=VLLM_MODEL, api_key=VLLM_API_KEY)

    retriever = PubMedRetriever() if args.phase in ("A", "A+") else None

    agent = BioASQAgent(
        llm=llm,
        retriever=retriever,
        training_index=training_index,
    )

    # ── Run Pipeline ────────────────────────────────────────────────
    start_time = time.time()
    logger.info(f"\nStarting Phase {args.phase} pipeline for {len(test_questions)} questions...\n")

    answered = agent.answer_batch(test_questions, phase=args.phase)

    elapsed = time.time() - start_time
    logger.info(f"\nPipeline completed in {elapsed:.1f}s ({elapsed/len(answered):.1f}s/question)")

    # ── Save Output ─────────────────────────────────────────────────
    if args.phase == "A":
        output_path = args.output or PHASE_A_OUTPUT
        submission = format_submission_phaseA(answered)
    else:
        output_path = args.output or (PHASE_B_OUTPUT if args.phase == "B" else PHASE_APLUS_OUTPUT)
        submission = format_submission_phaseB(answered)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    logger.info(f"Submission saved to: {output_path}")

    # Print summary
    type_counts = {}
    for q in answered:
        type_counts[q.qtype] = type_counts.get(q.qtype, 0) + 1
    logger.info(f"Question types: {type_counts}")

    has_exact = sum(1 for q in answered if q.generated_exact is not None)
    has_ideal = sum(1 for q in answered if q.generated_ideal)
    logger.info(f"Answers generated: {has_exact} exact, {has_ideal} ideal")

    # ── Optional Evaluation ─────────────────────────────────────────
    if args.eval:
        evaluate_answers(answered)


if __name__ == "__main__":
    main()
