#!/usr/bin/env python3
"""
BioASQ Task 14b — Full Pipeline Runner
=======================================
Ties everything together:
  1. (Optional) Evaluate on training data first to tune prompts
  2. Run the agentic Phase B solver on the real test set
  3. Validate submission format

Usage:
    # Step 1: Dry-run eval on training data (tests your setup)
    python run_pipeline.py eval \
        --training training13b.json \
        --model google/gemma-2-9b-it \
        --num-questions 20

    # Step 2: Real submission run
    python run_pipeline.py submit \
        --test-input BioASQ-task14bPhaseB-testset1.json \
        --training training13b.json \
        --model google/gemma-2-9b-it \
        --output submission_phaseB.json

    # Step 3: Validate the submission file
    python run_pipeline.py validate \
        --submission submission_phaseB.json
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path


def cmd_eval(args):
    """Run the solver on a slice of training data, then evaluate."""
    training_path = Path(args.training)
    if not training_path.exists():
        print(f"ERROR: Training file not found: {training_path}")
        sys.exit(1)

    # Load training data and create a mini test set from it
    with open(training_path) as f:
        data = json.load(f)

    questions = data.get("questions", [])
    n = args.num_questions or 20
    subset = questions[:n]

    # Write a temp test file (same format as Phase B input)
    tmp_test = Path("_eval_test_input.json")
    with open(tmp_test, "w") as f:
        json.dump({"questions": subset}, f)

    tmp_output = Path("_eval_predictions.json")

    print(f"\n{'='*60}")
    print(f"DRY-RUN EVALUATION on {len(subset)} training questions")
    print(f"Model: {args.model}")
    print(f"{'='*60}\n")

    # Run the solver
    cmd = [
        sys.executable, "bioasq_phase_b.py",
        "--test-input", str(tmp_test),
        "--training", args.training,
        "--output", str(tmp_output),
        "--model", args.model,
        "--passes", str(args.passes),
        "--max-model-len", str(args.max_model_len),
        "--dtype", args.dtype,
    ]
    if args.tensor_parallel > 1:
        cmd += ["--tensor-parallel", str(args.tensor_parallel)]
    if args.pubmed_db:
        cmd += ["--pubmed-db", args.pubmed_db,
                "--baseline-articles", str(args.baseline_articles)]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("ERROR: Solver failed!")
        sys.exit(1)

    # Evaluate
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}\n")

    cmd_eval = [
        sys.executable, "evaluate_phase_b.py",
        "--predictions", str(tmp_output),
        "--gold", args.training,
    ]
    subprocess.run(cmd_eval)

    # Cleanup
    if not args.keep_temp:
        tmp_test.unlink(missing_ok=True)

    print(f"\nPredictions saved to: {tmp_output}")
    print("Review them to see where the model struggles.\n")


def cmd_submit(args):
    """Run the solver on the real test set."""
    print(f"\n{'='*60}")
    print(f"GENERATING SUBMISSION")
    print(f"Test set: {args.test_input}")
    print(f"Model:    {args.model}")
    print(f"Output:   {args.output}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "bioasq_phase_b.py",
        "--test-input", args.test_input,
        "--output", args.output,
        "--model", args.model,
        "--passes", str(args.passes),
        "--max-model-len", str(args.max_model_len),
        "--dtype", args.dtype,
    ]
    if args.training:
        cmd += ["--training", args.training]
    if args.tensor_parallel > 1:
        cmd += ["--tensor-parallel", str(args.tensor_parallel)]
    if args.pubmed_db:
        cmd += ["--pubmed-db", args.pubmed_db,
                "--baseline-articles", str(args.baseline_articles)]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("ERROR: Solver failed!")
        sys.exit(1)

    # Auto-validate
    validate_submission(args.output, args.test_input)


def validate_submission(submission_path: str, test_input_path: str = None):
    """Check that the submission JSON is valid for BioASQ."""
    print(f"\n{'='*60}")
    print("VALIDATING SUBMISSION")
    print(f"{'='*60}\n")

    with open(submission_path) as f:
        sub = json.load(f)

    questions = sub.get("questions", [])
    if not questions:
        print("ERROR: No questions in submission!")
        return False

    # Load test input to cross-reference
    test_ids = set()
    test_types = {}
    if test_input_path and Path(test_input_path).exists():
        with open(test_input_path) as f:
            test_data = json.load(f)
        for q in test_data.get("questions", []):
            test_ids.add(q["id"])
            test_types[q["id"]] = q.get("type", "unknown")

    errors = 0
    warnings = 0

    for q in questions:
        qid = q.get("id")
        if not qid:
            print(f"  ERROR: Question missing 'id' field")
            errors += 1
            continue

        if test_ids and qid not in test_ids:
            print(f"  WARNING: Question {qid} not in test set")
            warnings += 1

        qtype = test_types.get(qid, "unknown")

        # Check ideal_answer
        ideal = q.get("ideal_answer")
        if not ideal:
            print(f"  WARNING: {qid} missing ideal_answer")
            warnings += 1
        elif isinstance(ideal, str) and len(ideal) < 10:
            print(f"  WARNING: {qid} ideal_answer very short ({len(ideal)} chars)")
            warnings += 1

        # Check exact_answer for non-summary types
        if qtype != "summary":
            exact = q.get("exact_answer")
            if exact is None:
                print(f"  WARNING: {qid} [{qtype}] missing exact_answer")
                warnings += 1
            elif qtype == "yesno":
                ea = exact if isinstance(exact, str) else (
                    exact[0] if isinstance(exact, list) and exact else "")
                if ea.lower() not in ("yes", "no"):
                    print(f"  ERROR: {qid} yesno answer is '{ea}', "
                          f"must be 'yes' or 'no'")
                    errors += 1
            elif qtype == "list":
                if not isinstance(exact, list):
                    print(f"  ERROR: {qid} list answer must be a list, "
                          f"got {type(exact).__name__}")
                    errors += 1
                elif not exact:
                    print(f"  WARNING: {qid} list answer is empty")
                    warnings += 1

    # Missing questions
    if test_ids:
        sub_ids = {q["id"] for q in questions}
        missing = test_ids - sub_ids
        if missing:
            print(f"  WARNING: {len(missing)} test questions not in submission")
            warnings += 1

    print(f"\nTotal questions: {len(questions)}")
    print(f"Errors:   {errors}")
    print(f"Warnings: {warnings}")

    if errors == 0:
        print("\n✓ Submission format is VALID")
    else:
        print("\n✗ Submission has ERRORS — fix before submitting!")

    return errors == 0


def cmd_validate(args):
    """Validate a submission file."""
    validate_submission(args.submission, args.test_input)


def main():
    parser = argparse.ArgumentParser(
        description="BioASQ Task 14b Full Pipeline Runner"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- eval subcommand ----
    p_eval = subparsers.add_parser("eval",
        help="Dry-run evaluation on training data")
    p_eval.add_argument("--training", "-tr", required=True)
    p_eval.add_argument("--model", "-m", default="google/gemma-2-9b-it")
    p_eval.add_argument("--num-questions", "-n", type=int, default=20)
    p_eval.add_argument("--passes", type=int, default=3)
    p_eval.add_argument("--max-model-len", type=int, default=8192)
    p_eval.add_argument("--tensor-parallel", type=int, default=1)
    p_eval.add_argument("--dtype", default="auto")
    p_eval.add_argument("--keep-temp", action="store_true")
    p_eval.add_argument("--pubmed-db", default=None,
                        help="PubMed baseline SQLite index")
    p_eval.add_argument("--baseline-articles", type=int, default=10)
    p_eval.set_defaults(func=cmd_eval)

    # ---- submit subcommand ----
    p_sub = subparsers.add_parser("submit",
        help="Generate submission for the real test set")
    p_sub.add_argument("--test-input", "-t", required=True)
    p_sub.add_argument("--training", "-tr", default=None)
    p_sub.add_argument("--output", "-o", default="submission_phaseB.json")
    p_sub.add_argument("--model", "-m", default="google/gemma-2-9b-it")
    p_sub.add_argument("--passes", type=int, default=3)
    p_sub.add_argument("--max-model-len", type=int, default=8192)
    p_sub.add_argument("--tensor-parallel", type=int, default=1)
    p_sub.add_argument("--dtype", default="auto")
    p_sub.add_argument("--pubmed-db", default=None,
                       help="PubMed baseline SQLite index")
    p_sub.add_argument("--baseline-articles", type=int, default=10)
    p_sub.set_defaults(func=cmd_submit)

    # ---- validate subcommand ----
    p_val = subparsers.add_parser("validate",
        help="Validate a submission JSON file")
    p_val.add_argument("--submission", "-s", required=True)
    p_val.add_argument("--test-input", "-t", default=None)
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
