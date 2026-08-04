#!/usr/bin/env python3
"""fcg_extract_simple.py — checklist items -> FCG extraction CSV, one shot.

The whole pipeline in one file, no package, no gates:

  python fcg_extract_simple.py items.json  --countries fcg_countries/
  python fcg_extract_simple.py items.docx  --countries fcg_countries/

Items file:
  .json : ["NOTAMs", {"title": "Parking MOG", "notes": "hazmat too"}, ...]
          (or {"items": [...]})
  .docx : every non-empty paragraph / table row = one item

Output (default fcg_extract.csv) is exactly what country_route_planner.py
(v14) consumes: 'country' code column + *_raw text columns. The CORE
planner columns (overflight, lead time, entry/exit airports + summary,
customs, airfield restrictions, hours, hazmat, AIR Card, country-specific)
are always included; your items add columns on top.

LLM endpoint from env (same vars as everything else):
  LITELLM_BASE_URL / LITELLM_API_KEY / AGENT_MODEL / AGENT_TLS_VERIFY

Principles kept from the full pipeline: closed-world (answers only from
dossier text, else NA), raw verbatim quotes over summaries, one fact per
question, async with semaphore, per-country resume via checkpoint file,
lenient JSON parsing of model output.
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

# ----------------------------------------------------------------------------
# Config (env)
# ----------------------------------------------------------------------------

def _env(*names, default=""):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


BASE_URL = _env("FCG_LLM_BASE_URL", "LITELLM_BASE_URL",
                default="http://localhost:4000/v1")
API_KEY = _env("FCG_LLM_API_KEY", "LITELLM_API_KEY", default="sk-local")
MODEL = _env("FCG_LLM_MODEL", "AGENT_MODEL", default="claude-opus-4-8")
TLS_VERIFY = _env("FCG_TLS_VERIFY", "AGENT_TLS_VERIFY",
                  default="true").strip().lower() not in ("false", "0", "no")
CONCURRENCY = int(_env("FCG_CONCURRENCY", default="4"))
MAX_TOKENS = int(_env("FCG_MAX_TOKENS", default="3000"))
DOC_CHAR_CAP = int(_env("FCG_COUNTRY_DOC_CHAR_CAP", default="12000"))
RETRIES = 3

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
    ("operating_hours_raw", "airfield or ATC operating hours"),
    ("hazmat_raw", "HAZMAT, dangerous goods, and munitions rules"),
    ("aircard_cash_raw",
     "AIR Card acceptance, fuel payment, and cash payment rules"),
    ("country_specific_raw",
     "country-specific notes, special requirements, or restrictions not "
     "covered by other categories"),
]

QUESTION = ("Quote verbatim all dossier text about {topic}. Preserve the "
            "original wording. If the dossier says nothing about this, "
            "answer exactly NA.")

# ----------------------------------------------------------------------------
# Items loading (.json or .docx)
# ----------------------------------------------------------------------------

def slug(title: str) -> str:
    s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    s = re.sub(r"_+", "_", s)[:34] or "item"
    return s if s.endswith(("_raw", "_summary")) else s + "_raw"


def load_items(path: Path) -> list[dict]:
    if path.suffix.lower() == ".docx":
        from docx import Document
        doc = Document(str(path))
        titles = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    titles.append(" ".join(cells))
        return [{"title": t, "notes": ""} for t in titles]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("items", [])
    out = []
    for e in data:
        if isinstance(e, str) and e.strip():
            out.append({"title": e.strip(), "notes": ""})
        elif isinstance(e, dict) and e.get("title"):
            out.append({"title": str(e["title"]).strip(),
                        "notes": str(e.get("notes", "")).strip()})
    return out


def build_fields(items: list[dict]) -> list[tuple[str, str]]:
    """Returns ordered (field_name, question) pairs: core + items, deduped."""
    fields, seen = [], set()
    for name, topic in CORE_FIELDS:
        fields.append((name, QUESTION.format(topic=topic)))
        seen.add(name)
    for item in items:
        name = slug(item["title"])
        if name in seen:
            continue
        seen.add(name)
        topic = item["title"] + (f" (focus: {item['notes']})"
                                 if item["notes"] else "")
        fields.append((name, QUESTION.format(topic=topic)))
    return fields

# ----------------------------------------------------------------------------
# Lenient JSON parsing (three-tier, truncation-safe)
# ----------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _balance(t: str) -> str:
    stack, in_str, esc = [], False, False
    for ch in t:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = in_str
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    if in_str:
        t += '"'
    for op in reversed(stack):
        t += "}" if op == "{" else "]"
    return t


def parse_lenient(text):
    if not text:
        return None
    m = _FENCE.search(text)
    cand = m.group(1) if m else text
    for attempt in (cand,
                    _balance(re.sub(r",\s*([}\]])", r"\1", cand.strip())),
                    None):
        if attempt is None:
            cut = max(cand.rfind("}"), cand.rfind(","))
            if cut <= 0:
                return None
            attempt = _balance(cand[:cut + 1].rstrip().rstrip(","))
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    return None

# ----------------------------------------------------------------------------
# LLM + extraction
# ----------------------------------------------------------------------------

def flatten(obj, prefix="") -> list[str]:
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lines.extend(flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            lines.extend(flatten(v, f"{prefix}{i}."))
    else:
        t = str(obj).strip()
        if t:
            lines.append(f"{prefix[:-1]}: {t}")
    return lines


_SUFFIX = re.compile(r"_FCG[\d.]*$", re.IGNORECASE)


async def extract_one(client, sem, path: Path, system: str,
                      field_names: list[str]) -> dict:
    country = _SUFFIX.sub("", path.stem).upper()
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        text = "\n".join(flatten(data))
    except json.JSONDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    user = f"FCG DOSSIER FOR {country}:\n\n{text[:DOC_CHAR_CAP]}"

    row = {"country": country, "source_file": path.name, "_status": "ok"}
    async with sem:
        resp = None
        for attempt in range(1, RETRIES + 1):
            try:
                r = await client.chat.completions.create(
                    model=MODEL, temperature=0, max_tokens=MAX_TOKENS,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                resp = parse_lenient(r.choices[0].message.content or "")
                if resp:
                    break
            except Exception as e:  # noqa: BLE001
                print(f"  [{country}] attempt {attempt} failed: {e}")
                await asyncio.sleep(3 * attempt)
    if not isinstance(resp, dict):
        row["_status"] = "llm_failure"
        resp = {}
    for f in field_names:
        v = resp.get(f, "NA")
        row[f] = "NA" if v is None or not str(v).strip() else str(v)
    return row


async def run(items_path: Path, countries_dir: Path, out_csv: Path) -> None:
    import httpx
    from openai import AsyncOpenAI

    items = load_items(items_path)
    fields = build_fields(items)
    field_names = [n for n, _ in fields]
    print(f"[fields] {len(fields)} total ({len(CORE_FIELDS)} core, "
          f"{len(fields) - len(CORE_FIELDS)} from your items):")
    for n, _ in fields:
        print(f"  {n}")

    system = ("You extract facts from a raw Foreign Clearance Guide (FCG) "
              "country dossier.\nSTRICT RULES:\n"
              "- Answer ONLY from the dossier text. If a fact is not stated, "
              "the answer is exactly \"NA\". Never use outside knowledge.\n"
              "- Each field asks one question; answer each independently "
              "with verbatim quotes from the dossier.\n"
              "Respond ONLY with a JSON object with EXACTLY these keys:\n"
              + json.dumps({n: "..." for n in field_names}) + "\n"
              "Field questions:\n"
              + "\n".join(f"- {n}: {q}" for n, q in fields))

    country_files = sorted(countries_dir.glob("*.json"))
    if not country_files:
        sys.exit(f"No country JSON files in {countries_dir}")

    # resume from checkpoint
    ckpt_path = out_csv.with_suffix(".checkpoint.jsonl")
    done = {}
    if ckpt_path.exists():
        with open(ckpt_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    # invalidate checkpoint rows from a different field set
                    if set(field_names) <= set(r.keys()):
                        done[r["country"]] = r
    todo = [p for p in country_files
            if _SUFFIX.sub("", p.stem).upper() not in done]
    print(f"[run] {len(country_files)} countries, {len(done)} cached, "
          f"{len(todo)} to extract")

    http_client = None if TLS_VERIFY else httpx.AsyncClient(verify=False)
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY,
                         http_client=http_client)
    sem = asyncio.Semaphore(CONCURRENCY)

    with open(ckpt_path, "a", encoding="utf-8") as ckpt:
        tasks = [asyncio.create_task(
            extract_one(client, sem, p, system, field_names)) for p in todo]
        for coro in asyncio.as_completed(tasks):
            row = await coro
            done[row["country"]] = row
            ckpt.write(json.dumps(row, ensure_ascii=False) + "\n")
            ckpt.flush()
            print(f"[{len(done)}/{len(country_files)}] {row['country']} "
                  f"({row['_status']})")

    cols = ["country", "source_file", "_status"] + field_names
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in sorted(done):
            w.writerow(done[c])
    fails = sum(1 for r in done.values() if r["_status"] != "ok")
    print(f"\nWrote {len(done)} rows -> {out_csv}"
          + (f"  ({fails} failures: delete their lines from "
             f"{ckpt_path.name} and rerun)" if fails else ""))
    print(f"Next: FCG_CSV={out_csv} python country_route_planner.py "
          f'"A to B to C"')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items", type=Path, help="items file (.json or .docx)")
    ap.add_argument("--countries", type=Path, default=Path("fcg_countries"),
                    help="directory of per-country FCG JSON files")
    ap.add_argument("--out", type=Path, default=Path("fcg_extract.csv"))
    args = ap.parse_args()
    if not args.items.exists():
        sys.exit(f"Items file not found: {args.items}")
    asyncio.run(run(args.items, args.countries, args.out))


if __name__ == "__main__":
    main()
