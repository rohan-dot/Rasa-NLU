"""
Airport HTML -> LLM extraction -> CSV pipeline (v2, robust + concurrent).

For every .html file in a folder, a vLLM-served Gemma model answers a fixed
checklist of questions using constrained JSON output, and writes ONE CSV row per
file. Each question gets two columns:
    <id>_raw      -> supporting text copied from the page, verbatim
    <id>_summary  -> an LLM summary of the main points for that question
Plus `airport` (from filename), an optional `misc` column, and a `flags` column
that lists any *_raw fields whose text could NOT be verified as a substring of the
page (i.e. the model likely paraphrased -> worth a spot check).

Anything absent is written as "NA"; the model is instructed never to invent.

Upgrades over v1:
  - Async concurrency (saturates vLLM's continuous batching)        -> CONCURRENCY
  - Resume: skips airports already present in the output CSV
  - Retry with backoff on API / JSON failures
  - Verbatim verification of every *_raw field against the source
  - Stronger one-shot prompt (NA case + verbatim case, quote-then-summarize)
  - Map-reduce fallback for oversized pages (instead of truncating)

Requires:  pip install openai beautifulsoup4
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
HTML_DIR      = "./airports"                 # main folder -> all the .html files
OUTPUT_CSV    = "./airport_extract.csv"
MODEL         = "gemma-4-31B-it"             # must match vLLM's --served-model-name (NOT the path)
VLLM_BASE_URL = "http://localhost:8000/v1"
API_KEY       = "EMPTY"

CONCURRENCY   = 16          # in-flight requests; raise to push vLLM harder
MAX_RETRIES   = 3
MAX_TOKENS    = 6000        # cap output, but high enough that the JSON can always CLOSE.
                            # Too low truncates the JSON mid-string -> parse error. Raise if you
                            # add many questions or expect long tabular answers.
INCLUDE_MISC  = True

# Map-reduce only kicks in for unusually large pages. ~4 chars/token, so 80k chars
# ~= 20k tokens. Your ~8-page files sit far below this and take the fast single path.
CHUNK_CHAR_THRESHOLD = 80_000
CHUNK_SIZE           = 30_000
CHUNK_OVERLAP        = 2_000
# ------------------------------------------------------------------

# --------------------------- YOUR CHECKLIST ------------------------
QUESTIONS = [
    {"id": "runway",    "question": "What are the runway designations, lengths, and surface types?"},
    {"id": "frequency", "question": "What are the tower, ATC, or CTAF frequencies?"},
    {"id": "elevation", "question": "What is the field elevation?"},
    {"id": "fuel",      "question": "What fuel types and services are available?"},
    # ... add the rest of your questions here ...
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
            f"Exact text copied verbatim from the page that answers: {q['question']} "
            "Do not paraphrase. 'NA' if absent.")}
        props[sum_key] = {"type": "string", "description": (
            f"Concise summary of the main points answering: {q['question']} 'NA' if absent.")}
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
        "EXAMPLE (illustrative only). Questions: 'length: runway length?', 'cafe: is there a cafe?'\n"
        "Page text: '...Runway 09/27 is 8,200 ft, asphalt...'\n"
        "Correct output:\n"
        '{"length_raw": "Runway 09/27 is 8,200 ft, asphalt", '
        '"length_summary": "Single runway 09/27, 8,200 ft, asphalt surface", '
        '"cafe_raw": "NA", "cafe_summary": "NA"}\n'
        "Note: 'length' was answered with text copied verbatim; 'cafe' was absent from the "
        "page, so BOTH its fields are 'NA' rather than guessed.\n"
    )
    return (
        "You are a precise data-extraction agent working on ONE airport's page.\n"
        "For each question, first locate the exact supporting text, then summarize it:\n"
        "  *_raw     -> copy ONLY the minimal exact span that answers it (a value, phrase, or a\n"
        "              sentence or two) VERBATIM from the page. Keep it SHORT -- at most a few lines.\n"
        "              If the answer is a long table or list, copy only the few most relevant rows,\n"
        "              never the entire table or the whole page.\n"
        "  *_summary -> a short, clean summary of the main points\n"
        'If the page does not contain the answer, put exactly "NA" in BOTH fields for that '
        "question. Never guess, infer beyond the text, or invent information.\n\n"
        f"{one_shot}\n"
        f"Questions:\n{qlist}\n\n"
        f"PAGE TEXT:\n{text}"
    )


_ws = re.compile(r"\s+")
def _norm(s: str) -> str:
    return _ws.sub(" ", s).strip().lower()


def verify_verbatim(data: dict, source: str) -> list:
    """Return the list of *_raw fields whose value isn't found in the source text."""
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


# --------------------------- model calls ---------------------------
async def call_model(text: str) -> dict:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content":
                     "Extract only what is present in the page. Use 'NA' when absent. "
                     "Output must match the JSON schema exactly."},
                    {"role": "user", "content": build_prompt(text)},
                ],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "airport_extract",
                                                 "schema": build_schema()}},
                temperature=0,
                max_tokens=MAX_TOKENS,
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            if choice.finish_reason == "length":
                raise ValueError("output hit max_tokens before JSON closed -> raise MAX_TOKENS")
            txt = content.strip()
            if txt.startswith("```"):                # tolerate stray ```json fences
                txt = txt.strip("`")
                txt = txt[4:] if txt.lower().startswith("json") else txt
                txt = txt.strip()
            return json.loads(txt)
        except Exception as e:                      # transient API / parse errors
            last_err = e
            await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"model call failed after {MAX_RETRIES} tries: {last_err}")


async def extract_payload(text: str) -> tuple:
    """Returns (data, chunked_flag). Single call if small; map-reduce if oversized."""
    if len(text) <= CHUNK_CHAR_THRESHOLD:
        return await call_model(text), False

    # Map-reduce fallback: extract each chunk, then merge per field.
    parts = await asyncio.gather(*[call_model(c) for c in chunk_text(text)])
    merged = {}
    for q in QUESTIONS:
        for suffix in ("_raw", "_summary"):
            key = f"{q['id']}{suffix}"
            vals = [p.get(key, "NA") for p in parts
                    if isinstance(p.get(key), str) and p.get(key).strip()
                    and p.get(key).strip() != "NA"]
            merged[key] = "\n".join(dict.fromkeys(vals)) if vals else "NA"
    if INCLUDE_MISC:
        vals = [p.get("misc", "NA") for p in parts
                if isinstance(p.get("misc"), str) and p.get("misc").strip()
                and p.get("misc").strip() != "NA"]
        merged["misc"] = "\n".join(dict.fromkeys(vals)) if vals else "NA"
    return merged, True


async def extract_file(path: str, sem: asyncio.Semaphore) -> dict:
    cols = fieldnames()
    row = {k: "NA" for k in cols}
    row["airport"] = Path(path).stem
    async with sem:
        try:
            text = clean_html(path)
            data, chunked = await extract_payload(text)
            for k in cols:
                if k in ("airport", "flags"):
                    continue
                v = data.get(k, "NA")
                row[k] = v if (isinstance(v, str) and v.strip()) else "NA"
            flags = verify_verbatim(data, text)
            if chunked:
                flags.append("chunked")
            row["flags"] = ";".join(flags) if flags else "NA"
        except Exception as e:
            row["flags"] = f"ERROR:{e}"
    return row


# ------------------------------- main ------------------------------
async def run():
    files = sorted(glob.glob(os.path.join(HTML_DIR, "*.html"))
                   + glob.glob(os.path.join(HTML_DIR, "*.htm")))
    if not files:
        sys.exit(f"No HTML files found in {HTML_DIR}")

    cols = fieldnames()

    # Resume: collect airports already written so a re-run only fills the gaps.
    done = set()
    if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            done = {r["airport"] for r in csv.DictReader(f)}

    todo = [p for p in files if Path(p).stem not in done]
    print(f"{len(files)} files | {len(done)} already done | {len(todo)} to process "
          f"| concurrency={CONCURRENCY}")
    if not todo:
        print("Nothing to do.")
        return

    new_file = not done
    sem = asyncio.Semaphore(CONCURRENCY)
    write_lock = asyncio.Lock()

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if new_file:
            writer.writeheader()
            f.flush()

        tasks = [asyncio.create_task(extract_file(p, sem)) for p in todo]
        completed = 0
        for coro in asyncio.as_completed(tasks):
            row = await coro
            async with write_lock:
                writer.writerow(row)
                f.flush()                          # crash-safe + keeps resume accurate
            completed += 1
            print(f"[{completed}/{len(todo)}] {row['airport']}  flags={row['flags']}")

    print(f"\nDone -> {OUTPUT_CSV}")


if __name__ == "__main__":
    try:
        asyncio.run(run())                       # plain `python airport_html_extract_v2.py`
    except RuntimeError as e:
        # Jupyter / IPython already has a running loop -> asyncio.run() is illegal there.
        if "running event loop" not in str(e):
            raise
        try:
            import nest_asyncio                   # lets asyncio.run nest inside the notebook loop
            nest_asyncio.apply()
            asyncio.run(run())
        except ImportError:
            raise SystemExit(
                "Detected a running event loop (you're likely in Jupyter).\n"
                "  -> In a notebook cell, run:   await run()\n"
                "  -> Or install the shim:       pip install nest_asyncio")
