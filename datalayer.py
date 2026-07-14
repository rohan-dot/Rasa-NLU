"""
Source-agnostic airport + FCG data layer.

You do NOT trust any particular dataset; you trust your OWN ingest gate.
Any aeronautical CSV (DAFIF export, country AIP extract, FAA NASR, OurAirports,
a spreadsheet someone hands you) is mapped into ONE canonical table via a
SCHEMA_MAP entry, and every row must pass validation to get in:
  - ICAO: 4 chars [A-Z0-9], starts with a letter
  - lat in [-90, 90], lon in [-180, 180]
  - provenance recorded per row (source name + file)

Add a new source = add a dict. No code changes.
"""

import csv
import re
from pathlib import Path

# --------------------- canonical fields ---------------------------
# icao, name, lat, lon, country (ISO2 or name), runway_ft (0 if unknown), type

SCHEMA_MAPS = {
    # OurAirports layout (kept as an example; swap in whatever you trust)
    "ourairports": {
        "icao": ["gps_code", "ident"],      # first non-empty wins
        "name": ["name"],
        "lat": ["latitude_deg"],
        "lon": ["longitude_deg"],
        "country": ["iso_country"],
        "type": ["type"],
    },
    # DAFIF-style flat export (typical column names; adjust to your cut)
    "dafif": {
        "icao": ["ICAO", "ARPT_IDENT"],
        "name": ["NAME", "ARPT_NAME"],
        "lat": ["WGS_DLAT", "LAT_DECIMAL"],
        "lon": ["WGS_DLONG", "LON_DECIMAL"],
        "country": ["CTRY", "COUNTRY_CODE"],
        "type": ["TYPE"],
    },
    # Minimal generic file you make yourself: icao,name,lat,lon,country,runway_ft
    "generic": {
        "icao": ["icao"], "name": ["name"], "lat": ["lat"], "lon": ["lon"],
        "country": ["country"], "runway_ft": ["runway_ft"], "type": ["type"],
    },
}

ICAO_RE = re.compile(r"^[A-Z][A-Z0-9]{3}$")


def _pick(row, keys):
    for k in keys:
        v = (row.get(k) or "").strip()
        if v:
            return v
    return ""


def ingest_airports(path, schema="generic", source_name=None):
    """CSV -> {icao: record}. Invalid rows are counted, not silently kept."""
    m = SCHEMA_MAPS[schema]
    src = source_name or Path(path).name
    out, rejected = {}, 0
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            icao = _pick(row, m["icao"]).upper()
            try:
                lat, lon = float(_pick(row, m["lat"])), float(_pick(row, m["lon"]))
            except ValueError:
                rejected += 1
                continue
            if not ICAO_RE.match(icao) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
                rejected += 1
                continue
            rwy = 0
            if "runway_ft" in m:
                try:
                    rwy = int(float(_pick(row, m["runway_ft"]) or 0))
                except ValueError:
                    rwy = 0
            out[icao] = {
                "icao": icao,
                "name": _pick(row, m["name"]) or icao,
                "lat": lat, "lon": lon,
                "country": _pick(row, m.get("country", [])),
                "type": _pick(row, m.get("type", [])) or "unknown",
                "runway_ft": rwy,
                "source": src,
            }
    return out, rejected


def merge_runways(airports, path, ident_col="airport_ident", len_col="length_ft"):
    """Optional second file with runway lengths (keeps the longest per airport)."""
    if not path or not Path(path).exists():
        return airports
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        for r in csv.DictReader(f):
            icao = (r.get(ident_col) or "").strip().upper()
            try:
                ft = int(float(r.get(len_col) or 0))
            except ValueError:
                continue
            if icao in airports and ft > airports[icao]["runway_ft"]:
                airports[icao]["runway_ft"] = ft
    return airports


def load_country_names(path, code_col="code", name_col="name"):
    out = {}
    if path and Path(path).exists():
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[r[code_col].strip().upper()] = r[name_col].strip()
    return out


# ------------------------- FCG extract ----------------------------
FCG_ICAO_RE = re.compile(r"\b([A-Z][A-Z0-9]{3})\b")
LEAD_RE = re.compile(r"(\d+)\s*(business\s+|working\s+|calendar\s+)?(day|week|hour)s?", re.I)


def parse_lead_days(text):
    """Max (most conservative) lead time in days found in free text; None if none parse."""
    if not text or text.strip().upper() == "NA":
        return None
    worst = None
    for num, qual, unit in LEAD_RE.findall(text):
        n = int(num)
        days = n / 24 if unit.lower().startswith("hour") else n * 7 if unit.lower().startswith("week") else n
        if qual and qual.strip().lower() in ("business", "working"):
            days *= 7 / 5
        worst = max(worst or 0, days)
    return None if worst is None else int(worst + 0.999)


def load_fcg(path):
    """fcg_extract.csv -> list of country records with authorized-ICAO sets."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            entry = r.get("entry_exit_airports_raw", "") or ""
            rows.append({
                "key": r.get("airport", "?"),
                "icaos": set(FCG_ICAO_RE.findall(entry.upper())) if entry.strip().upper() != "NA" else set(),
                "lead_days": parse_lead_days(r.get("diplomatic_lead_time_raw", "")),
                "lead_raw": (r.get("diplomatic_lead_time_summary") or r.get("diplomatic_lead_time_raw") or "NA"),
                "hazmat": (r.get("hazmat_summary") or r.get("hazmat_raw") or "NA"),
                "restrictions": r.get("airfield_restrictions_summary", "NA"),
                "customs": r.get("customs_immigration_summary", "NA"),
                "hours": r.get("operating_hours_summary", "NA"),
                "forms": r.get("country_specific_summary", "NA"),
                "cash": r.get("aircard_cash_summary", "NA"),
                # overflight columns exist once fcg_extract.py includes the
                # "overflight" question; loader degrades gracefully if absent
                "overflight": (r.get("overflight_summary") or r.get("overflight_raw") or "NA"),
                "overflight_lead_days": parse_lead_days(r.get("overflight_raw", "")),
                # question-ids whose subtree was absent from the source document
                # (extractor flag "<id>:no_route") -> NA means "not published",
                # any other NA -> "not found in section (possible extraction miss)"
                "no_route_ids": {p.split(":")[0] for p in (r.get("flags", "") or "").split(";")
                                 if ":no_route" in p},
            })
    return rows
