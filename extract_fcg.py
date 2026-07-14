#!/usr/bin/env python3
"""BUILD STEP 1 — Extract structured constraints from FCG files (.json or .html).

With your vLLM server (defaults already match it):
    python scripts/extract_fcg.py --fcg-dir /path/FCG/full --out data/constraints_db.json

Override endpoint/model if needed:
    python scripts/extract_fcg.py --fcg-dir ... --base-url http://127.0.0.1:8000 \
        --model gemma-4-31B-it

No model available (offline heuristic, plumbing tests only):
    python scripts/extract_fcg.py --fcg-dir ... --out ... --mock
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flight_planner.constraints.extract import build_db, VLLMClient, MockLLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fcg-dir", required=True, help="dir of country .json/.html files")
    ap.add_argument("--out", default="data/constraints_db.json")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="gemma-4-31B-it")
    ap.add_argument("--mock", action="store_true", help="offline heuristic, no LLM")
    args = ap.parse_args()

    if args.mock:
        llm = MockLLM()
        print("[extract] using MockLLM (no model)")
    else:
        llm = VLLMClient(args.base_url, args.model)
        print(f"[extract] vLLM {args.base_url} model={args.model}")

    n = build_db(llm, args.fcg_dir, args.out)
    print(f"[extract] wrote {n} country records -> {args.out}")


if __name__ == "__main__":
    main()
