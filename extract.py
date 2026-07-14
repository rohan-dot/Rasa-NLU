"""Offline batch: FCG country files -> structured Constraints DB.

Supports BOTH source formats:
  - legacy *.cfm.html  (whole-page prose)
  - new *.json         (nested dicts: content fields like notes/restrictions/
                        attention/prohibition paired with '<field>Path' keys,
                        values are HTML-fragment prose or null)

Either way the content is PROSE — the agent reads it and DERIVES each checklist
answer, returning {value, evidence} per field so a human can confirm without
re-reading the source. Missing -> null -> REVIEW. LLM runs only here, never in
the routing loop.
"""
from __future__ import annotations

import json
import os
import re
from glob import glob

from ..schema import Constraints

# ---------------------------------------------------------------- schema
def _vw(value_type):
    return {"type": "object",
            "properties": {"value": {"type": value_type},
                           "evidence": {"type": ["string", "null"]}},
            "required": ["value", "evidence"]}

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "overflight_allowed": _vw(["boolean", "null"]),
        "dip_clearance_lead_days": _vw(["integer", "null"]),          # 29.7
        "hazmat_allowed": _vw(["boolean", "null"]),                    # 29.7
        "aircraft_entry_exit_restrictions": _vw(["string", "null"]),   # 29.8
        "ops_hours_notes": _vw(["string", "null"]),                    # 29.9
        "customs_immigration_notes": _vw(["string", "null"]),          # 29.10
        "country_specific_forms": _vw(["array"]),                      # 25.16
        "air_card_or_cash_required": _vw(["boolean", "null"]),         # 29.11
    },
    "required": ["overflight_allowed"],
}

SYSTEM_PROMPT = (
    "You read a Foreign Clearance Guide country entry. The text is organized as "
    "lines of the form [json.path] content. DERIVE answers for: overflight "
    "permitted?; 29.7 diplomatic clearance lead time (days) and HazMat allowed?; "
    "29.8 aircraft entry/exit airfield restrictions; 29.9 ops hours / holiday "
    "closures; 29.10 customs/ag/immigration; 25.16 country-specific forms; 29.11 "
    "AIR Card / cash payment. For EACH field return {value, evidence} where "
    "evidence is the [json.path] label plus the exact sentence supporting the "
    "value. If not stated, value=null and evidence=null. NEVER guess. Return "
    "ONLY the JSON object."
)

# ------------------------------------------------------------- text prep
def _strip_html(s: str) -> str:
    s = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = re.sub(r"&nbsp;|&amp;|&#\d+;|&quot;|&lt;|&gt;", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_SKIP_TOP_KEYS = {"lastModifiedDate", "lastModifiedBy", "timezone",
                  "status", "code", "name"}


def flatten_fcg_json(node, path: str = "") -> list[tuple[str, str]]:
    """Recursively collect (json_path, prose) pairs from the nested FCG JSON.
    Skips nulls, '<field>Path' pointer keys, 'version' counters, and top-level
    file metadata."""
    out: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if v is None or k.endswith("Path") or k == "version" \
               or (path == "" and k in _SKIP_TOP_KEYS):
                continue
            p = f"{path}.{k}" if path else k
            out.extend(flatten_fcg_json(v, p))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.extend(flatten_fcg_json(v, f"{path}[{i}]"))
    elif isinstance(node, str):
        text = _strip_html(node)
        if text:
            out.append((path, text))
    # bare numbers/bools: not prose, ignore
    return out


def _json_to_labeled_text(raw: dict) -> str:
    return "\n".join(f"[{p}] {t}" for p, t in flatten_fcg_json(raw))


def _iso3_from_filename(path: str) -> str:
    return os.path.basename(path).split(".")[0].upper()


# ------------------------------------------------------------- extraction
def extract_one(llm, doc_text: str, iso3: str, source: str) -> Constraints:
    user = f"Country entry (ISO3 {iso3}):\n\n{doc_text[:16000]}"
    raw = llm.extract_json(SYSTEM_PROMPT, user, EXTRACTION_SCHEMA)
    fields, evidence = {}, {}
    for key, vw in raw.items():
        if isinstance(vw, dict) and "value" in vw:
            fields[key] = vw["value"]
            if vw.get("evidence"):
                evidence[key] = vw["evidence"]
        else:
            fields[key] = vw
    fields["region_id"] = iso3
    fields["source"] = source
    fields["raw_excerpts"] = evidence
    return Constraints.from_dict(fields)


def load_country_file(fp: str) -> tuple[str, str]:
    """Return (iso3, prose_text) for either a .json or .html FCG file."""
    if fp.lower().endswith(".json"):
        with open(fp, encoding="utf-8", errors="ignore") as f:
            raw = json.load(f)
        iso3 = _iso3_from_filename(fp)
        if not re.fullmatch(r"[A-Z]{3}", iso3):
            iso3 = str(raw.get("code", iso3)).upper()   # fall back to embedded code
        return iso3, _json_to_labeled_text(raw)
    with open(fp, encoding="utf-8", errors="ignore") as f:
        return _iso3_from_filename(fp), _strip_html(f.read())


def build_db(llm, fcg_dir: str, out_path: str) -> int:
    files = sorted(glob(os.path.join(fcg_dir, "*.json"))
                   + glob(os.path.join(fcg_dir, "*.html")))
    db: dict[str, dict] = {}
    for fp in files:
        iso3, text = load_country_file(fp)
        db[iso3] = extract_one(llm, text, iso3,
                               source=f"FCG:{os.path.basename(fp)}").to_dict()
    with open(out_path, "w") as f:
        json.dump(db, f, indent=2)
    return len(db)
