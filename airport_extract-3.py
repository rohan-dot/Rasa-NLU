"""
Airport HTML/CFM -> LLM extraction -> CSV pipeline (consolidated, robust).

Loops a folder of airport pages. For each file a vLLM-served Gemma model answers a
fixed checklist using constrained JSON output, and writes ONE CSV row per file:
    <id>_raw      -> minimal supporting text copied verbatim from the page
    <id>_summary  -> an LLM summary of the main points
plus `airport` (from filename), optional `misc`, and `flags` (QA notes).

Handles, out of the box:
  - served-model-name vs path (set MODEL below)
  - Jupyter event loop (await run()) AND plain script (python airport_extract.py)
  - runaway generation         -> MAX_TOKENS cap
  - truncated / malformed JSON  -> auto token escalation + salvage of partial JSON
  - concurrency                 -> saturates vLLM batching
  - resume that RETRIES failed rows (ERROR rows are re-processed, not skipped)
  - oversized pages             -> map-reduce fallback

Requires:  pip install openai beautifulsoup4
Run (script):    python airport_extract.py
Run (notebook):  await run()
"""

import os
import sys
import csv
import glob
import json
import re
import asyncio
from pathlib import Path

from bs4 import BeautifulSoup
from openai import AsyncOpenAI

# ----------------------------- CONFIG -----------------------------
HTML_DIR      = "./airports"                 # folder containing the page files
OUTPUT_CSV    = "./airport_extract.csv"
MODEL         = "gemma-4-31B-it"             # MUST match vLLM --served-model-name (not the path)
VLLM_BASE_URL = "http://localhost:8000/v1"
API_KEY       = "EMPTY"

FILE_GLOBS    = ("*.html", "*.htm", "*.cfm") # extensions to treat as pages
CONCURRENCY   = 16                           # in-flight requests
MAX_RETRIES   = 3
MAX_TOKENS    = 6000                         # output cap; auto-doubles once on truncation
INCLUDE_MISC  = True
DEBUG         = False                        # True -> print raw model output on parse failure

# Restrict extraction to specific sections of each page. The text is sliced to
# [SECTION_START .. SECTION_END) BEFORE the model sees it, so it cannot pull from
# anywhere else. Set SECTION_START = None to use the whole page.
# These are regexes -> set them to match your actual section headers.
SECTION_START = r"SECTION\s+1\b"             # where the relevant region begins
SECTION_END   = r"SECTION\s+3\b"             # where it ends (start of the first section to drop)

# Map-reduce only triggers for unusually large pages (~4 chars/token).
CHUNK_CHAR_THRESHOLD = 80_000
CHUNK_SIZE           = 30_000
CHUNK_OVERLAP        = 2_000
# ------------------------------------------------------------------

# --------------------------- YOUR CHECKLIST ------------------------
# id       -> short, no-space key used for the column names
# question -> what to extract (a question OR a plain label both work)
QUESTIONS = [
    {"id": "runway",    "question": "Runway designations, lengths, and surface types"},
    {"id": "frequency", "question": "Tower, ATC, or CTAF frequencies"},
    {"id": "elevation", "question": "Field elevation"},
    {"id": "fuel",      "question": "Fuel types and services available"},
    # ... add the rest of your items here ...
]
# ------------------------------------------------------------------

client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=API_KEY)


# ------------------------------ helpers ----------------------------
def clean_html(path: str) -> str:
    """Conservative clean: drop noise tags, keep all visible body text (incl. tables)."""
    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head", "noscript", "svg", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    lines = [ln for ln in (l.strip() for l in text.splitlines()) if ln]
    return "\n".join(lines)


def slice_sections(text: str) -> str:
    """Keep only [SECTION_START .. SECTION_END). Falls back to full text if not found."""
    if not SECTION_START:
        return text
    m_start = re.search(SECTION_START, text, re.IGNORECASE)
    if not m_start:
        return text                                # marker missing -> don't silently blank it
    start = m_start.start()
    end = len(text)
    if SECTION_END:
        m_end = re.search(SECTION_END, text[start:], re.IGNORECASE)
        if m_end:
            end = start + m_end.start()
    return text[start:end]


def fieldnames() -> list:
    cols = ["airport"]
    for q in QUESTIONS:
        cols += [f"{q['id']}_raw", f"{q['id']}_summary"]
    if INCLUDE_MISC:
        cols.append("misc")
    cols.append("flags")
    return cols


def build_schema() -> dict:
    props, required = {}, []
    for q in QUESTIONS:
        raw_key, sum_key = f"{q['id']}_raw", f"{q['id']}_summary"
        props[raw_key] = {"type": "string", "description": (
            f"Minimal exact text copied verbatim from the page for: {q['question']}. "
            "Keep it short. 'NA' if absent.")}
        props[sum_key] = {"type": "string", "description": (
            f"Concise summary of: {q['question']}. 'NA' if absent.")}
        required += [raw_key, sum_key]
    if INCLUDE_MISC:
        props["misc"] = {"type": "string", "description":
                         "Other clearly relevant info not covered above. 'NA' if none."}
        required.append("misc")
    return {"type": "object", "properties": props, "required": required,
            "additionalProperties": False}


def build_prompt(text: str) -> str:
    qlist = "\n".join(f"- {q['id']}: {q['question']}" for q in QUESTIONS)
    one_shot = (
        "EXAMPLE. Items: 'length: runway length', 'cafe: is there a cafe'.\n"
        "Page: '...Runway 09/27 is 8,200 ft, asphalt...'\n"
        'Output: {"length_raw": "Runway 09/27 is 8,200 ft, asphalt", '
        '"length_summary": "Runway 09/27, 8,200 ft, asphalt", '
        '"cafe_raw": "NA", "cafe_summary": "NA"}\n'
    )
    return (
        "You are a precise data-extraction agent working on ONE airport's page.\n"
        "The text below has been restricted to the relevant sections; use ONLY this text.\n"
        "For each item produce two values:\n"
        "  *_raw     -> copy ONLY the minimal exact span that answers it (a value, phrase, or a\n"
        "              sentence or two) VERBATIM. Keep it SHORT - at most a few lines. If the answer\n"
        "              is a long table/list, copy only the few most relevant rows.\n"
        "  *_summary -> summarize ONLY what you put in *_raw. Do NOT introduce any number, date, or\n"
        "              fact that is not present in *_raw.\n"
        "If the text gives conflicting values for one item (e.g. two different lead times), copy the\n"
        "relevant value(s) verbatim into *_raw WITH their qualifiers and do NOT merge or average them.\n"
        'If the answer is not in the text, put exactly "NA" in BOTH fields. Never guess or invent.\n\n'
        f"{one_shot}\n"
        f"Items:\n{qlist}\n\n"
        f"PAGE TEXT:\n{text}"
    )


_ws = re.compile(r"\s+")
def _norm(s: str) -> str:
    return _ws.sub(" ", s).strip().lower()


def verify_verbatim(data: dict, source: str) -> list:
    """Return *_raw fields whose value isn't found in the source (likely paraphrased)."""
    src = _norm(source)
    bad = []
    for q in QUESTIONS:
        key = f"{q['id']}_raw"
        val = data.get(key, "NA")
        if isinstance(val, str) and val.strip() and val.strip() != "NA":
            if _norm(val) not in src:
                bad.append(key)
    return bad


def chunk_text(text: str) -> list:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ----------------------- JSON parsing / repair --------------------
_PAIR = re.compile(r'"([^"]+)"\s*:\s*"((?:[^"\\]|\\.)*)"')     # keys may contain spaces

def _repair(s: str) -> str:
    """Close any dangling string / unbalanced braces in malformed-or-truncated JSON."""
    start = s.find("{")
    if start == -1:
        return s
    s = s[start:]
    in_str = esc = False
    depth = 0
    for ch in s:
        if esc:
            esc = False
        elif ch == "\\" and in_str:
            esc = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
    if in_str:                                    # the model left a string open -> close it
        s += '"'
    if depth > 0:                                 # ...and close the object
        s += "}" * depth
    return s


def _salvage(raw: str) -> dict:
    """Last resort: pull every intact "key":"value" pair out of the text."""
    out = {}
    for m in _PAIR.finditer(raw):
        key, val = m.group(1), m.group(2)
        try:
            val = json.loads(f'"{val}"')          # unescape
        except json.JSONDecodeError:
            pass
        out[key] = val
    return out


def parse_payload(raw: str, finish_reason: str) -> tuple:
    """Return (data, note). Tries strict parse, then auto-repair, then regex salvage."""
    txt = raw.strip()
    if txt.startswith("```"):                     # tolerate stray ```json fences
        txt = txt.strip("`")
        txt = txt[4:] if txt[:4].lower() == "json" else txt
        txt = txt.strip()
    try:
        return json.loads(txt), ""
    except json.JSONDecodeError:
        pass
    try:                                          # close dangling quote/braces, retry
        note = "truncated" if finish_reason == "length" else "repaired"
        return json.loads(_repair(txt)), note
    except json.JSONDecodeError:
        pass
    salvaged = _salvage(txt)                      # keep whatever pairs are intact
    if salvaged:
        return salvaged, "salvaged"
    raise json.JSONDecodeError("unparseable model output", txt or " ", 0)


# --------------------------- model calls ---------------------------
async def call_model(text: str) -> tuple:
    schema = build_schema()
    prompt = build_prompt(text)
    last_err = None
    for attempt in range(MAX_RETRIES):
        budget = MAX_TOKENS if attempt == 0 else MAX_TOKENS * 2   # escalate on retry
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content":
                     "Extract only what is present. Use 'NA' when absent. "
                     "Output must match the JSON schema exactly."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "airport_extract", "schema": schema}},
                temperature=0,
                max_tokens=budget,
            )
            choice = resp.choices[0]
            raw = choice.message.content or ""
            try:
                return parse_payload(raw, choice.finish_reason)
            except json.JSONDecodeError as je:
                if DEBUG:
                    print(f"[DEBUG] finish_reason={choice.finish_reason} len={len(raw)} :: {je}")
                    print("[DEBUG] raw >>>"); print(raw[:1200]); print("[DEBUG] <<<")
                last_err = je
        except Exception as e:
            last_err = e
        await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"model call failed after {MAX_RETRIES} tries: {last_err}")


def _merge(parts: list) -> dict:
    """Merge per-chunk extractions: union of non-NA values per field."""
    merged = {}
    keys = [f"{q['id']}{s}" for q in QUESTIONS for s in ("_raw", "_summary")]
    if INCLUDE_MISC:
        keys.append("misc")
    for k in keys:
        vals = [p.get(k, "NA") for p in parts
                if isinstance(p.get(k), str) and p.get(k).strip() and p.get(k).strip() != "NA"]
        merged[k] = "\n".join(dict.fromkeys(vals)) if vals else "NA"
    return merged


async def extract_payload(text: str) -> tuple:
    """Return (data, notes_list, chunked_flag)."""
    if len(text) <= CHUNK_CHAR_THRESHOLD:
        data, note = await call_model(text)
        return data, ([note] if note else []), False
    results = await asyncio.gather(*[call_model(c) for c in chunk_text(text)])
    notes = sorted({n for _, n in results if n})
    return _merge([d for d, _ in results]), notes, True


async def extract_file(path: str, sem: asyncio.Semaphore) -> dict:
    cols = fieldnames()
    row = {k: "NA" for k in cols}
    row["airport"] = Path(path).stem
    async with sem:
        try:
            text = slice_sections(clean_html(path))
            if DEBUG:
                print(f"[DEBUG] {row['airport']}: extracted {len(text)} chars of page text")
                print("[DEBUG] page preview >>>")
                print(text[:800])
                print("[DEBUG] <<<")
            data, notes, chunked = await extract_payload(text)
            for k in cols:
                if k in ("airport", "flags"):
                    continue
                v = data.get(k, "NA")
                row[k] = v if (isinstance(v, str) and v.strip()) else "NA"
            flags = verify_verbatim(data, text) + notes
            if chunked:
                flags.append("chunked")
            row["flags"] = ";".join(flags) if flags else "NA"
        except Exception as e:
            row["flags"] = f"ERROR:{e}"
    return row


# ------------------------------- main ------------------------------
async def run():
    cols = fieldnames()
    files = []
    for pat in FILE_GLOBS:
        files += glob.glob(os.path.join(HTML_DIR, pat))
    files = sorted(set(files))
    if not files:
        print(f"No page files found in {HTML_DIR} (looked for {FILE_GLOBS})")
        return

    # Resume: keep good rows, DROP error rows so they get retried (no duplicates).
    good_rows, done = [], set()
    if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not str(r.get("flags", "")).startswith("ERROR"):
                    good_rows.append(r)
                    done.add(r["airport"])

    todo = [p for p in files if Path(p).stem not in done]
    print(f"{len(files)} files | {len(done)} done | {len(todo)} to process | concurrency={CONCURRENCY}")
    if not todo:
        print("Nothing to do.")
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    write_lock = asyncio.Lock()

    # Rewrite file: header + retained good rows, then append fresh results.
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in good_rows:
            writer.writerow({k: r.get(k, "NA") for k in cols})
        f.flush()

        tasks = [asyncio.create_task(extract_file(p, sem)) for p in todo]
        completed = 0
        for coro in asyncio.as_completed(tasks):
            row = await coro
            async with write_lock:
                writer.writerow(row)
                f.flush()
            completed += 1
            print(f"[{completed}/{len(todo)}] {row['airport']}  flags={row['flags']}")

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
            raise SystemExit("Inside a running loop (Jupyter): run `await run()` in a cell, "
                             "or `pip install nest_asyncio`.")
