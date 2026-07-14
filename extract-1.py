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

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


# ------------------------------------------------------------- LLM clients
class VLLMClient:
    """OpenAI-compatible chat client for a self-hosted vLLM server.
    Defaults match: vllm serve ... --served-model-name gemma-4-31B-it
                    --host 127.0.0.1 --port 8000"""

    def __init__(self, base_url: str = "http://127.0.0.1:8000",
                 model: str = "gemma-4-31B-it",
                 api_key: str = "EMPTY", timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def extract_json(self, system: str, user: str, schema: dict) -> dict:
        if requests is None:
            raise RuntimeError("requests not available")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.0,
            "max_tokens": 2048,
            # vLLM structured output: constrain decoding to the JSON schema
            "guided_json": schema,
        }
        r = requests.post(f"{self.base_url}/v1/chat/completions",
                          headers={"Authorization": f"Bearer {self.api_key}",
                                   "Content-Type": "application/json"},
                          data=json.dumps(payload), timeout=self.timeout)
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])


class MockLLM:
    """Offline stand-in (regex heuristics) so the pipeline runs with no model.
    Returns null for facts it can't find, like the real prompt demands."""

    def extract_json(self, system: str, user: str, schema: dict) -> dict:
        text = user.lower()

        def vw(value, evidence=None):
            return {"value": value, "evidence": evidence}

        out: dict = {}
        out["overflight_allowed"] = vw(
            False if ("overflight prohibited" in text or "no overflight" in text)
            else (True if "overflight" in text else None))
        m = re.search(r"(\d+)\s*(?:days|day)\s*prior", text)
        out["dip_clearance_lead_days"] = vw(int(m.group(1)) if m else None,
                                            m.group(0) if m else None)
        out["hazmat_allowed"] = vw(
            False if ("hazmat prohibited" in text or "no hazmat" in text)
            else (True if "hazmat" in text else None))
        out["air_card_or_cash_required"] = vw(
            True if ("air card" in text or "cash payment" in text) else None)
        return out

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
# Max characters of flattened prose per LLM call. Tune to your Gemma context
# window (chars ~= 4x tokens; 60k chars ~ 15k tokens, safe for 32k+ ctx).
CHUNK_CHARS = 60000


def _extract_chunk(llm, chunk: str, iso3: str) -> tuple[dict, dict]:
    user = f"Country entry (ISO3 {iso3}):\n\n{chunk}"
    raw = llm.extract_json(SYSTEM_PROMPT, user, EXTRACTION_SCHEMA)
    fields, evidence = {}, {}
    for key, vw in raw.items():
        if isinstance(vw, dict) and "value" in vw:
            fields[key] = vw["value"]
            if vw.get("evidence"):
                evidence[key] = vw["evidence"]
        else:
            fields[key] = vw
    return fields, evidence


def _split_chunks(doc_text: str) -> list[str]:
    """Split the flattened doc on line boundaries so no [path] entry is cut."""
    if len(doc_text) <= CHUNK_CHARS:
        return [doc_text]
    chunks, cur, size = [], [], 0
    for line in doc_text.split("\n"):
        if size + len(line) > CHUNK_CHARS and cur:
            chunks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _merge(base: dict, extra: dict, base_ev: dict, extra_ev: dict) -> None:
    """Combine chunk results: first non-null wins; list fields are unioned.
    A False overrides True for *_allowed fields (a restriction anywhere in the
    doc beats a permission elsewhere — conservative on safety-relevant fields)."""
    for k, v in extra.items():
        if v is None:
            continue
        cur = base.get(k)
        if isinstance(v, list):
            base[k] = list(dict.fromkeys((cur or []) + v))
        elif cur is None:
            base[k] = v
            if k in extra_ev:
                base_ev[k] = extra_ev[k]
        elif k.endswith("_allowed") and cur is True and v is False:
            base[k] = False
            if k in extra_ev:
                base_ev[k] = extra_ev[k]
        elif k.endswith("_required") and cur is False and v is True:
            base[k] = True
            if k in extra_ev:
                base_ev[k] = extra_ev[k]


def extract_one(llm, doc_text: str, iso3: str, source: str) -> Constraints:
    chunks = _split_chunks(doc_text)
    fields, evidence = _extract_chunk(llm, chunks[0], iso3)
    for chunk in chunks[1:]:
        f2, e2 = _extract_chunk(llm, chunk, iso3)
        _merge(fields, f2, evidence, e2)
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
