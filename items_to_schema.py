"""Items -> extraction schema, deterministically (replaces stages 1-2 when
the checklist items are provided directly, e.g. hand-picked by the boss).

Input: a JSON file of items. Accepted shapes:
  ["Overflight permissions", "HAZMAT rules", ...]
  [{"title": "Overflight permissions", "notes": "focus on lead time"}, ...]
  {"items": [...either shape...]}

Output: stage2_schema.json in the same format stage 3 already consumes.
Field naming convention: <slug>_raw with a templated verbatim-excerpt
question, honoring the raw-over-summaries principle -- the planner's
dossier builder picks up every *_raw column automatically (planner v14+).

A CORE field set (the columns the route planner needs to function) is
always included first unless --no-core is given. Boss items that slugify
to a core field name simply merge into it (no duplicates).

Deterministic by default. --llm-refine additionally asks the LLM to split
any compound item ("customs AND immigration lead times") into single-fact
fields; the result still goes through the same review gate.

Usage:
  python -m pipeline.items_to_schema path/to/items.json
  python -m pipeline.items_to_schema path/to/items.json --no-core
  python -m pipeline.items_to_schema path/to/items.json --llm-refine
"""

import argparse
import asyncio
import json
import re
import unicodedata
from pathlib import Path

from . import config

# Columns the route planner uses. Order matters: it becomes CSV column order.
CORE_FIELDS = [
    ("overflight_raw",
     "overflight permissions, diplomatic overflight clearance requirements, "
     "and overflight lead times"),
    ("diplomatic_lead_time_raw",
     "diplomatic clearance lead times for landing rights"),
    ("entry_exit_airports_raw",
     "designated entry/exit airports, airports of entry (AOE), and which "
     "airports foreign aircraft may use"),
    ("entry_exit_airports_summary",
     "a comma-separated list of every ICAO/IATA airport code mentioned as a "
     "designated entry/exit airport (codes only, no prose)"),
    ("customs_immigration_raw",
     "customs, immigration, and CIQ requirements"),
    ("airfield_restrictions_raw",
     "airfield restrictions, PPR requirements, and prohibited airfields"),
    ("operating_hours_raw",
     "airfield or ATC operating hours"),
    ("hazmat_raw",
     "HAZMAT, dangerous goods, and munitions rules"),
    ("aircard_cash_raw",
     "AIR Card acceptance, fuel payment, and cash payment rules"),
    ("country_specific_raw",
     "country-specific notes, special requirements, or restrictions not "
     "covered by other categories"),
]

QUESTION_TMPL = ("Quote verbatim all dossier text about {topic}. Preserve "
                 "the original wording. If the dossier says nothing about "
                 "this, answer exactly NA.")


def _slug(title: str) -> str:
    s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    s = re.sub(r"_+", "_", s)[:34] or "item"
    if not s.endswith("_raw") and not s.endswith("_summary"):
        s += "_raw"
    return s


def _load_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("items", [])
    items = []
    for entry in data:
        if isinstance(entry, str):
            items.append({"title": entry.strip(), "notes": ""})
        elif isinstance(entry, dict) and entry.get("title"):
            items.append({"title": str(entry["title"]).strip(),
                          "notes": str(entry.get("notes", "")).strip()})
    if not items:
        raise SystemExit(f"No usable items found in {path}")
    return items


def build_schema(items: list[dict], include_core: bool) -> list[dict]:
    schema, seen = [], set()

    def add(field: str, topic: str, item_id: str):
        if field in seen:
            return
        seen.add(field)
        schema.append({"field": field, "item_id": item_id,
                       "priority": "must_check",
                       "question": QUESTION_TMPL.format(topic=topic),
                       "answer_type": "text", "options": None})

    if include_core:
        for field, topic in CORE_FIELDS:
            add(field, topic, "CORE")

    for i, item in enumerate(items, 1):
        if isinstance(item, str):
            item = {"title": item, "notes": ""}
        topic = item["title"]
        if item.get("notes"):
            topic += f" (focus: {item['notes']})"
        add(_slug(item["title"]), topic, f"BOSS-{i:02d}")
    return schema


async def llm_refine(schema: list[dict]) -> list[dict]:
    """Optionally split compound boss items into single-fact fields."""
    from .llm_client import chat_json
    system = ("You review extraction fields. If a field's topic combines "
              "MULTIPLE distinct facts, split it into one field per fact; "
              "otherwise return it unchanged. Keep the *_raw naming and the "
              "verbatim-quote question style. Respond ONLY with JSON: "
              '{"fields": [{"field": "...", "question": "..."}]}')
    out = []
    for entry in schema:
        if entry["item_id"] == "CORE":
            out.append(entry)
            continue
        resp = await chat_json(system, json.dumps(
            {"field": entry["field"], "question": entry["question"]}))
        fields = (resp or {}).get("fields") or []
        if not fields:
            out.append(entry)
            continue
        for j, f in enumerate(fields):
            name = re.sub(r"[^a-z0-9_]", "", str(f.get("field", "")))[:40]
            q = str(f.get("question", "")).strip()
            if not name or not q:
                continue
            out.append({**entry, "field": name, "question": q,
                        "item_id": entry["item_id"] + (f".{j+1}" if len(fields) > 1 else "")})
    # dedupe, keep first
    seen, deduped = set(), []
    for e in out:
        if e["field"] not in seen:
            seen.add(e["field"])
            deduped.append(e)
    return deduped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items_json", type=Path)
    ap.add_argument("--no-core", action="store_true",
                    help="do not include the planner core field set")
    ap.add_argument("--llm-refine", action="store_true",
                    help="LLM pass to split compound items")
    args = ap.parse_args()

    config.ensure_dirs()
    items = _load_items(args.items_json)
    schema = build_schema(items, include_core=not args.no_core)
    if args.llm_refine:
        schema = asyncio.run(llm_refine(schema))

    config.STAGE2_SCHEMA_JSON.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    n_core = sum(1 for s in schema if s["item_id"] == "CORE")
    print(f"Wrote {len(schema)} fields ({n_core} core, "
          f"{len(schema) - n_core} from items) -> {config.STAGE2_SCHEMA_JSON}")
    for s in schema:
        print(f"  {s['item_id']:<9} {s['field']}")
    print("REVIEW GATE: edit stage2_schema.json, then run stage3.")


if __name__ == "__main__":
    main()
