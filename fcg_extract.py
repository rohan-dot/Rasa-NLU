"""
FCG JSON -> LLM extraction -> CSV pipeline (JSON-native rewrite).

Source files are now structured JSON (one per airport/country, keys vary per file).
Instead of feeding whole documents, each checklist item is ROUTED to its relevant
subtree(s) by key-name search, the embedded HTML fragments are stripped, and the
model answers from only that focused text. This:
  - fits any file regardless of size (no full-doc context needed)
  - is resilient to files having more/fewer keys (missing subtree -> clean NA)
  - structurally prevents cross-section contradictions (maritime never rides along)

Output: one CSV row per file with <item>_raw, <item>_summary, misc-free by default,
plus source_keys (which subtree answered) and flags columns.

Requires:  pip install openai beautifulsoup4
Run (script):    python fcg_extract.py
Run (notebook):  await run()
"""

import os
import sys
import csv
import glob
import json
import re
import html as htmllib
import asyncio
from pathlib import Path

from bs4 import BeautifulSoup
from openai import AsyncOpenAI

# ----------------------------- CONFIG -----------------------------
JSON_DIR      = "./airports"                 # folder of per-airport .json files
OUTPUT_CSV    = "./fcg_extract.csv"
MODEL         = "gemma-4-31B-it"             # must match vLLM --served-model-name
VLLM_BASE_URL = "http://localhost:8000/v1"
API_KEY       = "EMPTY"

CONCURRENCY   = 16          # files in flight at once
MAX_RETRIES   = 3
MAX_TOKENS    = 4000        # per-item calls are small; auto-doubles on retry
VERIFY        = True        # strict second pass per item (cannot blank found values)
DEBUG         = False       # print routed subtree text per item

# If a routed subtree's text exceeds this, it is chunked and answers merged.
SUBTREE_CHAR_LIMIT = 60_000
CHUNK_SIZE         = 25_000
CHUNK_OVERLAP      = 1_500
# ------------------------------------------------------------------

# --------------------------- YOUR CHECKLIST ------------------------
# route: list of key-name candidates searched (case-insensitive substring match)
#        anywhere in the tree. First candidates are preferred; all matches used.
# list=True -> the answer is an enumerated list wanted IN FULL.
QUESTIONS = [
    {"id": "diplomatic_lead_time",
     "question": "Aircraft clearance lead time(s) and validity - every value with its category/qualifier (blanket/annual, DV, HAZMAT, military, unmanned, other)",
     "route": ["leadTimeValidity", "leadTime", "validity"]},

    {"id": "hazmat",
     "question": "HAZMAT / dangerous-goods clearance requirements",
     "route": ["hazmat", "leadTimeValidity", "clearance"]},

    {"id": "entry_exit_airports",
     "question": "Airports/airfields authorized for entry and exit, with their ICAO codes (civilian and military)",
     "route": ["enterDepart", "airportDetails", "airport"],
     "list": True},

    {"id": "airfield_restrictions",
     "question": "Aircraft entry/exit restrictions at specific airfields",
     "route": ["additionalAirportInfo", "airportDetails", "authTechStopAndAltAirport"]},

    {"id": "customs_immigration",
     "question": "Customs, agriculture, and immigration requirements",
     "route": ["customsImmInspect", "customs"]},

    {"id": "operating_hours",
     "question": "Operating hours and holiday closures",
     "route": ["additionalAirportInfo", "airportDetails", "navAndOtherOpsInfo"]},

    {"id": "country_specific",
     "question": "Country-specific forms or information required for clearance requests",
     "route": ["apacsClrRequest", "apacsReqInst", "clrReqInstr", "countryInfo"]},

    {"id": "aircard_cash",
     "question": "Whether an Air Card or any cash payment is required",
     "route": ["additionalAirportInfo", "airportDetails", "generalCountryInfo"]},
]
# ------------------------------------------------------------------

client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=API_KEY)

# Keys that are pure plumbing -> dropped before rendering subtree text.
_SKIP_KEY = re.compile(r"(Path$|^version$|^lastModified|^titlePath$)", re.I)


# ------------------------- JSON -> text ---------------------------
def strip_html(s: str) -> str:
    """Convert an HTML fragment value to readable text (lists kept as lines)."""
    if not isinstance(s, str):
        return str(s)
    if "<" not in s:
        return htmllib.unescape(s).strip()
    soup = BeautifulSoup(s, "html.parser")
    for li in soup.find_all("li"):
        li.insert_before("\n- ")
    for br in soup.find_all(["br", "p", "ol", "ul"]):
        br.insert_before("\n")
    return re.sub(r"\n{2,}", "\n", htmllib.unescape(soup.get_text()).strip())


def render(node, depth=0, max_depth=8) -> str:
    """Render a JSON subtree to compact labeled text the model can read."""
    if depth > max_depth:
        return ""
    pad = "  " * depth
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if _SKIP_KEY.search(k) or v in (None, "", [], {}):
                continue
            if isinstance(v, (dict, list)):
                body = render(v, depth + 1, max_depth)
                if body.strip():
                    out.append(f"{pad}{k}:")
                    out.append(body)
            else:
                txt = strip_html(v)
                if txt:
                    out.append(f"{pad}{k}: {txt}")
    elif isinstance(node, list):
        for i, item in enumerate(node):
            body = render(item, depth + 1, max_depth)
            if body.strip():
                out.append(f"{pad}- [{i}]")
                out.append(body)
    else:
        txt = strip_html(node)
        if txt:
            out.append(pad + txt)
    return "\n".join(out)


def find_subtrees(obj, candidates, path="$"):
    """Yield (path, subtree) for every dict/list value whose KEY matches a candidate
    (case-insensitive substring). Does not descend into an already-matched subtree."""
    cands = [c.lower() for c in candidates]
    hits = []
    def walk(node, p):
        if isinstance(node, dict):
            for k, v in node.items():
                kp = f"{p}.{k}"
                if any(c in k.lower() for c in cands) and isinstance(v, (dict, list, str)):
                    hits.append((kp, v))
                else:
                    walk(v, kp)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{p}[{i}]")
    walk(obj, path)
    return hits


def route_text(doc: dict, q: dict) -> tuple:
    """Return (text, matched_paths) for one checklist item."""
    hits = find_subtrees(doc, q["route"])
    if not hits:
        return "", []
    parts, paths = [], []
    for p, sub in hits:
        body = render(sub) if isinstance(sub, (dict, list)) else strip_html(sub)
        if body.strip():
            parts.append(f"### {p}\n{body}")
            paths.append(p)
    return "\n\n".join(parts), paths


# --------------------------- prompts -------------------------------
def item_schema(q: dict) -> dict:
    raw_d = (f"COMPLETE verbatim list of ALL entries for: {q['question']}. Every entry, no truncation."
             if q.get("list") else
             f"Minimal exact text copied verbatim from the provided data for: {q['question']}. Keep it short.")
    return {"type": "object",
            "properties": {
                "raw":     {"type": "string", "description": raw_d + " 'NA' if absent."},
                "summary": {"type": "string", "description":
                            f"Concise summary using ONLY facts present in raw. 'NA' if absent."}},
            "required": ["raw", "summary"], "additionalProperties": False}


def extract_prompt(q: dict, text: str) -> str:
    mode = ("Copy the COMPLETE list - EVERY matching entry. Do not shorten, sample, or stop early."
            if q.get("list") else
            "Copy ONLY the minimal exact span(s) that answer it. If several values exist with "
            "different qualifiers (e.g. several lead times), copy EACH with its qualifier - never merge.")
    return (
        "You are a precise data-extraction agent. Below is structured data for ONE country, "
        "already narrowed to the relevant fields. Use ONLY this data.\n"
        f"ITEM: {q['question']}\n"
        f"raw     -> {mode}\n"
        "summary -> summarize ONLY what you put in raw; no new numbers, dates, or facts.\n"
        'If the data does not contain the answer, put exactly "NA" in BOTH fields. Never guess.\n\n'
        f"DATA:\n{text}"
    )


def verify_prompt(q: dict, text: str, ans: dict) -> str:
    keep = ("The corrected raw must remain the COMPLETE list - do not drop entries.\n"
            if q.get("list") else "")
    return (
        "You are a strict fact-checker. Below is the source data and an extracted answer.\n"
        "Return a corrected answer in the same schema:\n"
        "- raw must appear in the data (verbatim). Fix paraphrase; supply a missed value if present.\n"
        "- summary may only use facts in the corrected raw.\n"
        "- Use 'NA' ONLY if the value is genuinely not in the data - never because you are unsure.\n"
        f"{keep}\n"
        f"ITEM: {q['question']}\n\nSOURCE DATA:\n{text}\n\n"
        f"EXTRACTED:\n{json.dumps(ans, ensure_ascii=False)}"
    )


# ------------------------- model plumbing --------------------------
def _repair(s: str) -> str:
    start = s.find("{")
    if start == -1:
        return s
    s = s[start:]
    in_str = esc = False
    depth = 0
    for ch in s:
        if esc: esc = False
        elif ch == "\\" and in_str: esc = True
        elif ch == '"': in_str = not in_str
        elif not in_str:
            if ch == "{": depth += 1
            elif ch == "}": depth -= 1
    if in_str: s += '"'
    if depth > 0: s += "}" * depth
    return s


def parse_json(raw: str) -> dict:
    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        txt = txt[4:] if txt[:4].lower() == "json" else txt
        txt = txt.strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return json.loads(_repair(txt))


async def complete(system: str, user: str, schema: dict) -> dict:
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "item", "schema": schema}},
                temperature=0,
                max_tokens=MAX_TOKENS if attempt == 0 else MAX_TOKENS * 2,
            )
            return parse_json(resp.choices[0].message.content or "")
        except Exception as e:
            last = e
            await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"model call failed: {last}")


def chunks(text: str):
    i = 0
    while i < len(text):
        yield text[i:i + CHUNK_SIZE]
        i += CHUNK_SIZE - CHUNK_OVERLAP


def _nz(v) -> str:
    return v.strip() if isinstance(v, str) and v.strip() else "NA"


async def answer_item(q: dict, text: str) -> tuple:
    """Extract (+verify) one checklist item from its routed text. Returns (raw, summary, note)."""
    schema = item_schema(q)
    sys_x = "Extract only what is present. Use 'NA' when absent. Match the JSON schema exactly."

    async def one(txt: str) -> dict:
        return await complete(sys_x, extract_prompt(q, txt), schema)

    if len(text) <= SUBTREE_CHAR_LIMIT:
        ans = await one(text)
        note = ""
    else:                                        # oversized subtree -> chunk & merge
        parts = await asyncio.gather(*[one(c) for c in chunks(text)])
        raws = [p.get("raw", "NA") for p in parts if _nz(p.get("raw")) != "NA"]
        sums = [p.get("summary", "NA") for p in parts if _nz(p.get("summary")) != "NA"]
        ans = {"raw": "\n".join(dict.fromkeys(raws)) if raws else "NA",
               "summary": " ".join(dict.fromkeys(sums)) if sums else "NA"}
        note = "chunked"

    if VERIFY and _nz(ans.get("raw")) != "NA" and len(text) <= SUBTREE_CHAR_LIMIT:
        sys_v = "Fact-check strictly. Match the JSON schema exactly."
        try:
            corr = await complete(sys_v, verify_prompt(q, text, ans), schema)
            new_raw = _nz(corr.get("raw"))
            if new_raw != "NA":                  # verifier may correct, never blank
                if new_raw != _nz(ans.get("raw")):
                    note = (note + ";" if note else "") + "corrected"
                ans = {"raw": new_raw, "summary": _nz(corr.get("summary"))}
        except Exception:
            note = (note + ";" if note else "") + "verify_failed"
    return _nz(ans.get("raw")), _nz(ans.get("summary")), note


# ------------------------------ per file ---------------------------
def fieldnames() -> list:
    cols = ["airport"]
    for q in QUESTIONS:
        cols += [f"{q['id']}_raw", f"{q['id']}_summary"]
    cols += ["source_keys", "flags"]
    return cols


async def extract_file(path: str, sem: asyncio.Semaphore) -> dict:
    cols = fieldnames()
    row = {k: "NA" for k in cols}
    row["airport"] = Path(path).stem
    async with sem:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8", errors="ignore"))
            notes, sources = [], []
            for q in QUESTIONS:
                text, paths = route_text(doc, q)
                if DEBUG:
                    print(f"[DEBUG] {row['airport']}/{q['id']}: {len(text)} chars from {paths}")
                if not text.strip():             # subtree absent in this file -> clean NA
                    notes.append(f"{q['id']}:no_route")
                    continue
                raw, summ, note = await answer_item(q, text)
                row[f"{q['id']}_raw"] = raw
                row[f"{q['id']}_summary"] = summ
                if note:
                    notes.append(f"{q['id']}:{note}")
                sources.append(f"{q['id']}={'|'.join(paths[:3])}")
            row["source_keys"] = "; ".join(sources) if sources else "NA"
            row["flags"] = "; ".join(notes) if notes else "ok"
        except Exception as e:
            row["flags"] = f"ERROR:{e}"
    return row


# ------------------------------- main ------------------------------
async def run():
    cols = fieldnames()
    files = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))
    if not files:
        print(f"No .json files found in {JSON_DIR}")
        return

    good, done = [], set()
    if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not str(r.get("flags", "")).startswith("ERROR"):
                    good.append(r); done.add(r["airport"])

    todo = [p for p in files if Path(p).stem not in done]
    print(f"{len(files)} files | {len(done)} done | {len(todo)} to process | concurrency={CONCURRENCY}")
    if not todo:
        print("Nothing to do."); return

    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in good:
            w.writerow({k: r.get(k, "NA") for k in cols})
        f.flush()
        tasks = [asyncio.create_task(extract_file(p, sem)) for p in todo]
        n = 0
        for coro in asyncio.as_completed(tasks):
            row = await coro
            async with lock:
                w.writerow(row); f.flush()
            n += 1
            print(f"[{n}/{len(todo)}] {row['airport']}  flags={row['flags']}")
    print(f"\nDone -> {OUTPUT_CSV}")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except RuntimeError as e:
        if "running event loop" not in str(e):
            raise
        try:
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.run(run())
        except ImportError:
            raise SystemExit("Inside Jupyter: run `await run()` in a cell, or pip install nest_asyncio.")
