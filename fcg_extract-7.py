#!/usr/bin/env python3
"""
fcg_extract.py — Two-phase FCG checklist extraction pipeline.

Phase 1: For each country FCG JSON, answer all checklist items using ONLY the
         FCG text (closed-world: verbatim quotes or literal "NA").
Phase 2: For every field still NA, build an ICAO-joined dossier from ancillary
         sources (SVC_RMK txt + MEIS airfield JSONs) and re-ask ONLY those
         fields. Fill-NA-only: Phase 2 never overwrites a Phase 1 answer.

Output: one CSV row per country; for each checklist item a value column and a
        provenance column "<item>__src" (FCG | ANCILLARY | NA).

Model backend: LiteLLM OpenAI-compatible gateway.
  Env: LITELLM_BASE_URL   e.g. https://gateway:4000/v1
       LITELLM_API_KEY
       AGENT_MODEL        e.g. claude-opus-4-8
       AGENT_TLS_VERIFY   "false" to skip TLS verification (self-signed cert)
       FCG_MAX_TOKENS     per-call response cap (default 6000)
       FCG_CONCURRENCY    parallel batch calls (default 2)
       FCG_BUDGET_FCG     char budget for FCG text      (default 9000)
       FCG_BUDGET_MEIS    char budget for MEIS section  (default 20000)
       FCG_BUDGET_SVC     char budget for SVC section   (default 10000)

Usage (single line!):
  python fcg_extract.py --checklist checklist.json --fcg-dir data/fcg \
      --svc data/svc_rmk.txt --meis data/meis1.json data/meis2.json data/meis3.json \
      --airports data/airports.csv --codes data/countries_codes_and_coordinates.csv \
      --out fcg_extract.csv

--airports/--codes are optional; without them, ICAO resolution falls back to
matching MEIS entries by CountryName against the FCG filename.
"""

import argparse
import concurrent.futures as cf
import csv
import hashlib
import html as html_mod
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import urllib3

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BASE_URL = os.environ.get("LITELLM_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("LITELLM_API_KEY", "")
MODEL = os.environ.get("AGENT_MODEL", "claude-opus-4-8")
TLS_VERIFY = os.environ.get("AGENT_TLS_VERIFY", "true").lower() not in ("false", "0", "no")
MAX_TOKENS = int(os.environ.get("FCG_MAX_TOKENS", "6000"))
CONCURRENCY = int(os.environ.get("FCG_CONCURRENCY", "2"))
BUDGET_FCG = int(os.environ.get("FCG_BUDGET_FCG", "60000"))
BUDGET_MEIS = int(os.environ.get("FCG_BUDGET_MEIS", "20000"))
BUDGET_SVC = int(os.environ.get("FCG_BUDGET_SVC", "10000"))
BATCH_SIZE = 20
RETRIES = 4

if not TLS_VERIFY:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

WORK_DIR = Path("data/work")
DOSSIER_DIR = WORK_DIR / "dossiers"
CHECKPOINT_PATH = WORK_DIR / "checkpoint.json"

SYSTEM_PROMPT = (
    "You are a strict closed-world extraction engine for aviation mission "
    "planning. You are given a SOURCE DOSSIER and a list of FIELDS. For each "
    "field, find the passage(s) of the dossier relevant to that field and "
    "answer ONLY with text quoted verbatim from the dossier. Join multiple "
    "verbatim snippets with ' | '. The dossier may use different terminology "
    "than the field name — quote the passage that addresses the topic even if "
    "the wording differs. For compound fields, quote whatever parts the "
    "dossier does cover. Respond with the literal string \"NA\" ONLY when "
    "nothing in the dossier is relevant to the field. Never infer, summarize, "
    "paraphrase, or use outside knowledge. Respond ONLY with a single JSON "
    "object mapping each field name exactly as given to its answer string. "
    "Escape newlines as \\n and double quotes as \\\" inside JSON string "
    "values. No markdown fences, no commentary."
)


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()[:16]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def die(msg: str) -> None:
    log(f"FATAL: {msg}")
    sys.exit(1)


def sanitize_col(name: str) -> str:
    c = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_")
    return c[:80] if c else "item"


# ----------------------------------------------------------------------------
# Checklist loading — accepts list of strings, or list of dicts with a
# name/question-ish key, or a dict of name->question.
# ----------------------------------------------------------------------------
def load_checklist(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    items = []  # list of (column_name, question_text)
    if isinstance(data, dict):
        data = [{"name": k, "question": v} for k, v in data.items()]
    if not isinstance(data, list):
        die("Checklist JSON must be a list or object.")
    for i, it in enumerate(data):
        if isinstance(it, str):
            items.append((sanitize_col(it), it))
        elif isinstance(it, dict):
            name = None
            for k in ("name", "item", "field", "id", "title", "checklist_item"):
                if it.get(k):
                    name = str(it[k])
                    break
            q = None
            for k in ("question", "prompt", "description", "text", "query"):
                if it.get(k):
                    q = str(it[k])
                    break
            if name is None and q is None:
                die(f"Checklist item {i} has no recognizable name/question keys: {list(it.keys())}")
            name = name or q
            q = q or name
            items.append((sanitize_col(name), q))
        else:
            die(f"Checklist item {i} has unsupported type {type(it)}")
    # de-dupe column names
    seen = {}
    out = []
    for col, q in items:
        n = seen.get(col, 0)
        seen[col] = n + 1
        out.append((f"{col}_{n+1}" if n else col, q))
    return out


# ----------------------------------------------------------------------------
# Country / ICAO resolution
# ----------------------------------------------------------------------------
def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def load_codes_csv(path: Path):
    """countries_codes_and_coordinates.csv: name -> (alpha2, alpha3)."""
    mapping = {}
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3 or norm_name(row[0]) in ("country", ""):
                continue
            name, a2, a3 = row[0].strip(), row[1].strip().strip('" '), row[2].strip().strip('" ')
            if len(a2) == 2 and len(a3) == 3:
                mapping[norm_name(name)] = (a2.upper(), a3.upper())
    return mapping


def load_airports_csv(path: Path):
    """OurAirports airports.csv: iso_country(alpha2) -> [icao idents]."""
    by_a2 = {}
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            ident = (r.get("ident") or "").strip().upper()
            a2 = (r.get("iso_country") or "").strip().upper()
            typ = (r.get("type") or "").strip()
            if len(ident) == 4 and ident.isalpha() and a2 and typ != "closed":
                by_a2.setdefault(a2, []).append(ident)
    return by_a2


def resolve_country(fcg_file: Path, codes_map):
    """Return (country_key, alpha2, alpha3, display_name). Falls back gracefully."""
    stem = fcg_file.stem
    disp = re.sub(r"[_\-]+", " ", stem).strip()
    a2 = a3 = None
    if len(stem) == 3 and stem.isalpha():
        a3 = stem.upper()
        for _, (c2, c3) in (codes_map or {}).items():
            if c3 == a3:
                a2 = c2
                break
    elif len(stem) == 2 and stem.isalpha():
        a2 = stem.upper()
        for _, (c2, c3) in (codes_map or {}).items():
            if c2 == a2:
                a3 = c3
                break
    elif codes_map:
        hit = codes_map.get(norm_name(disp))
        if hit:
            a2, a3 = hit
    key = a3 or (a2 or sanitize_col(stem).upper()[:8])
    return key, a2, a3, disp


# ----------------------------------------------------------------------------
# Ancillary sources
# ----------------------------------------------------------------------------
def find_key_recursive(obj, key_pat: re.Pattern, max_depth=6):
    """Find first string value whose key matches pattern (case-insensitive)."""
    if max_depth < 0:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and key_pat.search(k):
                s = v.strip().upper()
                if s:
                    return s
        for v in obj.values():
            r = find_key_recursive(v, key_pat, max_depth - 1)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key_recursive(v, key_pat, max_depth - 1)
            if r:
                return r
    return None


ICAO_PAT = re.compile(r"icao", re.I)
CTRY_PAT = re.compile(r"country", re.I)


def load_meis(paths):
    """Load and concat MEIS JSON arrays; index by ICAO and by CountryName."""
    by_icao, by_country, seen_ids = {}, {}, set()
    total = 0
    for p in paths:
        log(f"Loading MEIS {p} ...")
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8", errors="replace"))
        except Exception as e:
            die(f"Failed to parse MEIS file {p}: {e}")
        if isinstance(data, dict):
            data = [data]
        for entry in data:
            if not isinstance(entry, dict):
                continue
            eid = entry.get("id") or ""
            icao = find_key_recursive(entry.get("AirfieldBasicInformation", entry), ICAO_PAT) \
                or find_key_recursive(entry, ICAO_PAT)
            dedupe_key = eid or icao or json.dumps(entry, sort_keys=True)[:200]
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)
            total += 1
            if icao and len(icao) == 4:
                by_icao.setdefault(icao, []).append(entry)
            cname = find_key_recursive(entry.get("AirfieldBasicInformation", entry), CTRY_PAT)
            if cname:
                by_country.setdefault(norm_name(cname), []).append((icao, entry))
    log(f"MEIS loaded: {total} entries, {len(by_icao)} ICAO-keyed, {len(by_country)} countries")
    return by_icao, by_country


def strip_nulls(obj):
    if isinstance(obj, dict):
        return {k: strip_nulls(v) for k, v in obj.items() if v is not None and v != "" and v != []}
    if isinstance(obj, list):
        return [strip_nulls(v) for v in obj if v is not None]
    return obj


def load_svc(path: Path):
    """SVC_RMK tab-separated (cp1252). Columns include ICAO + REMARKS + CYCLE_DATE.
    Returns icao -> [line, ...] (raw lines preserved for verbatim quoting)."""
    by_icao = {}
    header_cols = None
    icao_idx = None
    with path.open(encoding="cp1252", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if header_cols is None:
                header_cols = [c.strip().upper() for c in parts]
                if "ICAO" in header_cols:
                    icao_idx = header_cols.index("ICAO")
                    continue
                # no header row — fall through and detect ICAO per-line
                header_cols = []
            icao = None
            if icao_idx is not None and len(parts) > icao_idx:
                cand = parts[icao_idx].strip().upper()
                if len(cand) == 4 and cand.isalpha():
                    icao = cand
            if icao is None:
                for tok in parts:
                    t = tok.strip().upper()
                    if len(t) == 4 and t.isalpha():
                        icao = t
                        break
            if icao:
                by_icao.setdefault(icao, []).append(line)
    log(f"SVC_RMK loaded: {sum(len(v) for v in by_icao.values())} rows across {len(by_icao)} ICAOs")
    return by_icao


# ----------------------------------------------------------------------------
# Dossier building
# ----------------------------------------------------------------------------
def cap(text: str, budget: int, label: str) -> str:
    if len(text) <= budget:
        return text
    return text[:budget] + f"\n[...{label} truncated at {budget} chars...]"


TAG_A = re.compile(r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.I | re.S)
TAG_BREAK = re.compile(r"</?(?:li|ol|ul|p|br|tr|div|h[1-6])\b[^>]*>", re.I)
TAG_ANY = re.compile(r"<[^>]+>")
META_KEY = re.compile(r"(?:^|\.)(?:[A-Za-z0-9_]*[Pp]ath)$")


def html_to_text(s: str) -> str:
    """Strip HTML markup, keep visible text and link URLs, collapse whitespace."""
    if "<" not in s:
        return re.sub(r"\s+", " ", s).strip()
    s = TAG_A.sub(lambda m: f"{m.group(2).strip()} ({m.group(1)})", s)
    s = TAG_BREAK.sub("\n", s)
    s = TAG_ANY.sub(" ", s)
    s = html_mod.unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()


def fcg_text_for(fcg_file: Path) -> str:
    """Flatten a country FCG JSON into readable key: value text.
    Drops *Path metadata keys and strips HTML from values."""
    try:
        data = json.loads(fcg_file.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        die(f"Failed to parse FCG file {fcg_file}: {e}")

    lines = []

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if META_KEY.search(k):
                    continue  # titlePath/htmlPath/additionalInfoPath etc.
                walk(v, f"{prefix}{k}." if prefix else f"{k}.")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{prefix}{i}.")
        else:
            if obj is not None and str(obj).strip():
                txt = html_to_text(str(obj))
                if txt:
                    lines.append(f"{prefix.rstrip('.')}: {txt}")

    walk(data)
    return "\n".join(lines)


def build_ancillary_dossier(icaos, svc_by_icao, meis_by_icao, meis_by_country, country_disp):
    meis_parts, svc_parts = [], []
    hit_icaos = set()
    for icao in icaos:
        for entry in meis_by_icao.get(icao, []):
            meis_parts.append(f"=== MEIS AIRFIELD {icao} ===\n"
                              + json.dumps(strip_nulls(entry), ensure_ascii=False, indent=1))
            hit_icaos.add(icao)
        if icao in svc_by_icao:
            svc_parts.append(f"=== SVC_RMK {icao} ===\n" + "\n".join(svc_by_icao[icao]))
            hit_icaos.add(icao)
    # CountryName fallback for MEIS entries with no ICAO join (e.g. military fields)
    for icao2, entry in meis_by_country.get(norm_name(country_disp), []):
        if icao2 not in hit_icaos:
            meis_parts.append(f"=== MEIS AIRFIELD {icao2 or 'UNKNOWN-ICAO'} (country match) ===\n"
                              + json.dumps(strip_nulls(entry), ensure_ascii=False, indent=1))
    meis_txt = cap("\n\n".join(meis_parts), BUDGET_MEIS, "MEIS")
    svc_txt = cap("\n\n".join(svc_parts), BUDGET_SVC, "SVC_RMK")
    parts = []
    if meis_txt.strip():
        parts.append("### SOURCE: MEIS AIRFIELD DETAILS ###\n" + meis_txt)
    if svc_txt.strip():
        parts.append("### SOURCE: ENROUTE SUPPORT REMARKS (SVC_RMK) ###\n" + svc_txt)
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------
# LLM call
# ----------------------------------------------------------------------------
class BudgetExceeded(RuntimeError):
    pass


def llm_extract(dossier: str, fields):
    """fields: list of (col, question). Returns dict col->answer. Raises on hard failure."""
    field_lines = "\n".join(f"- {c}: {q}" for c, q in fields)
    user = (f"SOURCE DOSSIER:\n{dossier}\n\n"
            f"FIELDS (answer every one; JSON keys must match the part before the colon exactly):\n"
            f"{field_lines}")
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, verify=TLS_VERIFY, timeout=300)
            if r.status_code != 200:
                body = r.text[:300]
                if "budget" in body.lower():
                    raise BudgetExceeded(f"LiteLLM budget cap hit: {body}")
                raise RuntimeError(f"HTTP {r.status_code}: {body}")
            data = r.json()
            choice = data["choices"][0]
            msg = choice.get("message") or {}
            content = msg.get("content")
            if content is None or not str(content).strip():
                # reasoning models may put text in reasoning_content instead
                content = msg.get("reasoning_content")
            if content is None or not str(content).strip():
                fr = choice.get("finish_reason")
                raise RuntimeError(
                    f"Model returned empty content (finish_reason={fr}). "
                    f"If finish_reason=length, the model spent all tokens; "
                    f"raise FCG_MAX_TOKENS. Raw msg: {json.dumps(msg)[:200]}")
            return parse_json_answer(str(content), [c for c, _ in fields])
        except BudgetExceeded:
            raise  # non-retryable: raising budget requires admin action
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            log(f"  LLM attempt {attempt}/{RETRIES} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {RETRIES} attempts: {last_err}")


def repair_json(txt: str) -> str:
    """Best-effort repair for model JSON: escape raw newlines/tabs and interior
    unescaped quotes inside string literals; close truncated tails."""
    out, in_str, esc = [], False, False
    n = len(txt)
    for i, ch in enumerate(txt):
        if in_str:
            if esc:
                out.append(ch)
                esc = False
            elif ch == "\\":
                out.append(ch)
                esc = True
            elif ch == '"':
                # A real string terminator is followed by , : } ] or end.
                j = i + 1
                while j < n and txt[j] in " \t\r\n":
                    j += 1
                if j >= n or txt[j] in ",:}]":
                    out.append(ch)
                    in_str = False
                else:
                    out.append('\\"')  # interior quote -> escape it
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\t":
                out.append("\\t")
            elif ch == "\r":
                continue
            else:
                out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_str = True
    s = "".join(out)
    if in_str:
        s += '"'
    # balance braces (truncated tail): drop a dangling partial pair after last comma
    if s.count("{") > s.count("}"):
        if not re.search(r'"\s*[,}]?\s*$', s):
            s = s[: s.rfind(",")] if "," in s else s
        s += "}" * (s.count("{") - s.count("}"))
    return s


def parse_json_answer(content: str, expected_cols):
    txt = content.strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.S).strip()
    m = re.search(r"\{.*", txt, flags=re.S)
    if m:
        txt = m.group(0)
    obj = None
    for candidate in (txt, repair_json(txt)):
        try:
            obj = json.loads(candidate)
            break
        except Exception:
            continue
    if obj is None:
        raise RuntimeError(f"Unparseable JSON from model even after repair: {content[:200]}")
    if not isinstance(obj, dict):
        raise RuntimeError("Model returned non-object JSON")
    out = {}
    for c in expected_cols:
        v = obj.get(c, "NA")
        if v is None or (isinstance(v, str) and not v.strip()):
            v = "NA"
        out[c] = str(v).strip()
    return out


# ----------------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------------
def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        try:
            return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        except Exception:
            log("WARNING: checkpoint unreadable, starting fresh")
    return {}


def save_checkpoint(ck):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(ck), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)


def batch_key(country, phase, batch_idx, fields, dossier):
    fh = sha("|".join(c for c, _ in fields))
    dh = sha(dossier)
    return f"{country}|{phase}|{batch_idx}|{fh}|{dh}"


# ----------------------------------------------------------------------------
# Per-country processing
# ----------------------------------------------------------------------------
def run_batches(country, phase, dossier, fields, ck):
    """Run field batches (with cache) against a dossier. Returns col->answer."""
    batches = [fields[i:i + BATCH_SIZE] for i in range(0, len(fields), BATCH_SIZE)]
    results = {}
    todo = []
    for bi, batch in enumerate(batches):
        k = batch_key(country, phase, bi, batch, dossier)
        if k in ck:
            results.update(ck[k])
        else:
            todo.append((bi, batch, k))
    if todo:
        log(f"  {country} {phase}: {len(todo)}/{len(batches)} batches to run "
            f"({len(batches) - len(todo)} cached)")
        with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futs = {ex.submit(llm_extract, dossier, b): (bi, b, k) for bi, b, k in todo}
            for fut in cf.as_completed(futs):
                bi, b, k = futs[fut]
                ans = fut.result()  # raise -> abort country, checkpoint keeps prior work
                ck[k] = ans
                results.update(ans)
                save_checkpoint(ck)
    return results


def process_country(fcg_file, items, codes_map, airports_by_a2,
                    svc_by_icao, meis_by_icao, meis_by_country, ck):
    country, a2, a3, disp = resolve_country(fcg_file, codes_map)
    log(f"== {country} ({disp}) ==")

    # ---- Phase 1: FCG only
    fcg_txt = cap(fcg_text_for(fcg_file), BUDGET_FCG, "FCG")
    dossier1 = f"### SOURCE: FOREIGN CLEARANCE GUIDE ({disp}) ###\n{fcg_txt}"
    DOSSIER_DIR.mkdir(parents=True, exist_ok=True)
    (DOSSIER_DIR / f"{country}_phase1.txt").write_text(dossier1, encoding="utf-8")

    p1 = run_batches(country, "P1", dossier1, items, ck)

    row = {"country": country, "country_name": disp, "alpha2": a2 or "", "alpha3": a3 or ""}
    src = {}
    for col, _ in items:
        v = p1.get(col, "NA")
        row[col] = v
        src[col] = "FCG" if v != "NA" else "NA"

    # ---- Phase 2: ancillary fill of NA fields only
    na_fields = [(c, q) for c, q in items if row[c] == "NA"]
    if na_fields:
        icaos = airports_by_a2.get(a2, []) if a2 else []
        dossier2_body = build_ancillary_dossier(icaos, svc_by_icao, meis_by_icao,
                                                meis_by_country, disp)
        if dossier2_body.strip():
            dossier2 = (f"### COUNTRY: {disp} ({country}) — ANCILLARY SOURCES ###\n\n"
                        + dossier2_body)
            (DOSSIER_DIR / f"{country}_phase2.txt").write_text(dossier2, encoding="utf-8")
            p2 = run_batches(country, "P2", dossier2, na_fields, ck)
            for col, _ in na_fields:
                v = p2.get(col, "NA")
                if v != "NA":
                    row[col] = v
                    src[col] = "ANCILLARY"
            log(f"  {country}: Phase 2 filled "
                f"{sum(1 for c, _ in na_fields if row[c] != 'NA')}/{len(na_fields)} NA fields")
        else:
            log(f"  {country}: no ancillary data found (ICAOs tried: {len(icaos)})")
    else:
        log(f"  {country}: Phase 1 answered everything; skipping Phase 2")

    for col, _ in items:
        row[f"{col}__src"] = src[col]
    return row


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Two-phase FCG checklist extraction")
    ap.add_argument("--checklist", required=True, help="Checklist JSON (94 items)")
    ap.add_argument("--fcg-dir", required=True, help="Folder of per-country FCG JSON files")
    ap.add_argument("--svc", help="SVC_RMK tab-separated txt (cp1252)")
    ap.add_argument("--meis", nargs="*", default=[], help="MEIS airfield JSON files (up to 3+)")
    ap.add_argument("--airports", help="OurAirports airports.csv (ICAO resolution)")
    ap.add_argument("--codes", help="countries_codes_and_coordinates.csv (name->alpha2/3)")
    ap.add_argument("--countries", nargs="*", help="Limit to these FCG filename stems")
    ap.add_argument("--out", default="fcg_extract.csv")
    args = ap.parse_args()

    if not BASE_URL or not API_KEY:
        die("Set LITELLM_BASE_URL and LITELLM_API_KEY")

    items = load_checklist(Path(args.checklist))
    log(f"Checklist: {len(items)} items")

    codes_map = load_codes_csv(Path(args.codes)) if args.codes else {}
    airports_by_a2 = load_airports_csv(Path(args.airports)) if args.airports else {}
    svc_by_icao = load_svc(Path(args.svc)) if args.svc else {}
    meis_by_icao, meis_by_country = load_meis(args.meis) if args.meis else ({}, {})

    fcg_files = sorted(Path(args.fcg_dir).glob("*.json"))
    if args.countries:
        want = {c.lower() for c in args.countries}
        fcg_files = [f for f in fcg_files if f.stem.lower() in want]
    if not fcg_files:
        die("No FCG files matched")
    log(f"Countries to process: {len(fcg_files)}")

    ck = load_checkpoint()
    rows, failures = [], []
    for f in fcg_files:
        try:
            rows.append(process_country(f, items, codes_map, airports_by_a2,
                                        svc_by_icao, meis_by_icao, meis_by_country, ck))
        except BudgetExceeded as e:
            log(f"BUDGET EXHAUSTED at {f.stem}: {e}")
            log("Aborting run — raise/reset the key budget in LiteLLM, then re-run; "
                "checkpoint will resume completed batches.")
            break
        except Exception as e:
            log(f"ERROR processing {f.stem}: {e} — continuing with next country")
            failures.append(f.stem)

    if rows:
        cols = (["country", "country_name", "alpha2", "alpha3"]
                + [c for c, _ in items] + [f"{c}__src" for c, _ in items])
        outp = Path(args.out)
        with outp.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        log(f"Wrote {len(rows)} rows -> {outp}")
    if failures:
        log(f"FAILED countries ({len(failures)}): {', '.join(failures)}")
        sys.exit(2)


if __name__ == "__main__":
    main()
