"""Pipeline orchestrator.

Runs stages in order with prerequisite checks and HUMAN REVIEW GATES.
By design it STOPS at each gate unless --yes is passed; the agent driving
this (or Jo) is expected to inspect the gate artifact before continuing.

Usage:
  python -m pipeline.run_pipeline preprocess   # checklist + transcripts
  python -m pipeline.run_pipeline stage1
  python -m pipeline.run_pipeline stage2
  python -m pipeline.run_pipeline stage3
  python -m pipeline.run_pipeline all [--yes]  # --yes skips gate stops
  python -m pipeline.run_pipeline status
"""

import argparse
import asyncio
import sys

from . import config


def _exists(path, label: str) -> bool:
    ok = path.exists()
    mark = "OK " if ok else "-- "
    print(f"  {mark} {label}: {path}")
    return ok


def status() -> None:
    print("Inputs:")
    _exists(config.CHECKLIST_DOCX, "checklist docx")
    n_tr = len(list(config.TRANSCRIPTS_DIR.glob("*"))) if config.TRANSCRIPTS_DIR.exists() else 0
    print(f"  {'OK ' if n_tr else '-- '} transcripts dir ({n_tr} files): {config.TRANSCRIPTS_DIR}")
    n_c = len(list(config.FCG_COUNTRY_DIR.glob('*.json'))) if config.FCG_COUNTRY_DIR.exists() else 0
    print(f"  {'OK ' if n_c else '-- '} country JSONs ({n_c} files): {config.FCG_COUNTRY_DIR}")
    print("Artifacts:")
    _exists(config.CHECKLIST_ITEMS_JSONL, "checklist items [GATE]")
    _exists(config.STAGE1_FLAGGED_JSON, "stage1 flagged")
    _exists(config.STAGE2_SCHEMA_JSON, "stage2 schema [GATE]")
    _exists(config.STAGE3_CHECKPOINT_JSONL, "stage3 checkpoint")
    _exists(config.FINAL_CSV, "final CSV")


def preprocess() -> None:
    from . import checklist, transcripts
    config.ensure_dirs()
    if not config.CHECKLIST_DOCX.exists():
        sys.exit(f"Missing checklist: {config.CHECKLIST_DOCX}")
    if not config.TRANSCRIPTS_DIR.exists():
        sys.exit(f"Missing transcripts dir: {config.TRANSCRIPTS_DIR}")
    written = transcripts.normalize_all(config.TRANSCRIPTS_DIR,
                                        config.TRANSCRIPTS_CLEAN_DIR)
    print(f"Normalized {len(written)} transcripts -> {config.TRANSCRIPTS_CLEAN_DIR}")
    checklist.main()


def stage1() -> None:
    from . import stage1_flag_items
    if not config.CHECKLIST_ITEMS_JSONL.exists():
        sys.exit("Run preprocess first (checklist_items.jsonl missing).")
    asyncio.run(stage1_flag_items.run())


def stage2() -> None:
    from . import stage2_verify_schema
    if not config.STAGE1_FLAGGED_JSON.exists():
        sys.exit("Run stage1 first (stage1_flagged.json missing).")
    asyncio.run(stage2_verify_schema.run())


def stage3() -> None:
    from . import stage3_extract
    if not config.STAGE2_SCHEMA_JSON.exists():
        sys.exit("Run stage2 first (stage2_schema.json missing).")
    asyncio.run(stage3_extract.run())


def schema_from_items(items_path: str, llm_refine: bool) -> None:
    """Boss-provided items JSON -> stage2_schema.json (replaces stages 1-2)."""
    import sys as _sys
    from . import items_to_schema
    argv = [items_path] + (["--llm-refine"] if llm_refine else [])
    _sys.argv = ["items_to_schema"] + argv
    items_to_schema.main()


GATE_MSG = """
==================== HUMAN REVIEW GATE ====================
Stopping so {artifact} can be reviewed/edited.
When satisfied, continue with: python -m pipeline.run_pipeline {next_cmd}
(or rerun 'all --yes' to skip gates -- not recommended first time)
===========================================================
"""


def run_all(auto_yes: bool) -> None:
    preprocess()
    if not auto_yes:
        print(GATE_MSG.format(artifact=config.CHECKLIST_ITEMS_JSONL,
                              next_cmd="stage1 (then stage2, stage3)"))
        return
    stage1()
    stage2()
    if not auto_yes:
        print(GATE_MSG.format(artifact=config.STAGE2_SCHEMA_JSON,
                              next_cmd="stage3"))
        return
    stage3()


def main() -> None:
    ap = argparse.ArgumentParser(description="FCG checklist pipeline")
    ap.add_argument("command", choices=["preprocess", "stage1", "stage2",
                                        "schema", "stage3", "all", "status"])
    ap.add_argument("--yes", action="store_true",
                    help="run through review gates without stopping")
    ap.add_argument("--items", help="items JSON for the 'schema' command "
                                    "(boss-provided checklist items)")
    ap.add_argument("--llm-refine", action="store_true",
                    help="with 'schema': LLM pass to split compound items")
    args = ap.parse_args()

    if args.command == "schema":
        if not args.items:
            sys.exit("'schema' needs --items path/to/items.json")
        schema_from_items(args.items, args.llm_refine)
        return

    {"preprocess": preprocess, "stage1": stage1, "stage2": stage2,
     "stage3": stage3, "status": status,
     "all": lambda: run_all(args.yes)}[args.command]()


if __name__ == "__main__":
    main()
