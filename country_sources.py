#!/usr/bin/env python3
"""country_sources.py — per-country slices of the three ancillary sources.

Built to the probed format spec (scratch/source_format_spec.txt):
  MEIS  : JSON array of airfield objects. ICAO at
          AirfieldBasicInformation.AirfieldNames.PrimaryICAO.ICAO, with
          AlternateICAO[] fallback; CountryName (e.g. "IRAQ") present but
          CountryCode is FIPS 2-letter — never used here.
  SVC_RMK: tab-separated; header ARPT_IDENT TYPE RMK_SEQ ICAO REMARKS
          CYCLE_DATE; split on literal tabs only (REMARKS has commas but
          no tabs); cp1252 bytes possible; CYCLE_DATE = YYYYMM.
  udl   : blank-line-separated NOTAM blocks; `A)` = ICAO location (may be
          FIR); two-tier match = exact ident OR country-derived 2-letter
          prefix (tagged scope=FIR).

Join chain per ICAO: exact OurAirports ident -> alpha2 -> alpha3.
MEIS fallback when the ICAO is unknown to OurAirports (some military
fields): CountryName matched against the coords CSV country name.

Usage (standalone):
  python country_sources.py --countries data/fcg_countries \
      --sources data/sources --out data/work/sources_index
Produces one <A3>.json slice per target country plus a printed summary
table. fcg_extract_simple.py imports build_or_load_slices() directly.
"""

import argparse
import json
import re
import sys
from pathlib import Path

MEIS_KEEP_KEYS = [
    "AirfieldOperatingHours", "AirfieldRestriction", "AirfieldRemarks",
    "PlanningRemarks", "FuelList", "MaximumOnGround", "AMCSuitabilityCodes",
    "SuitableAircraftList", "UnsuitableAircraftList",
    "AircraftRescueFireFighting", "Waivers", "ExceptionalFactors",
]
MEIS_PER_KEY_CAP = 1500          # chars of flattened text per kept key
_SUFFIX = re.compile(r"_FCG[\d.]*$", re.IGNORECASE)


# ----------------------------------------------------------------------------
# Reference joins (same files the planner uses, tolerant parsing)
# ----------------------------------------------------------------------------

def _clean(s):
    return s.strip().strip('"').strip() if isinstance(s, str) else s


def load_coords(path="countries_codes_and_coordinates.csv"):
    """Returns (a2_to_a3, a3_to_name)."""
    import csv as _csv
    a2_to_a3, a3_to_name = {}, {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = _csv.reader(f)
        header = [_clean(c).lower() for c in next(reader)]
        idx = {}
        for i, c in enumerate(header):
            if "alpha-3" in c:
                idx["a3"] = i
            elif "alpha-2" in c:
                idx["a2"] = i
            elif c == "country":
                idx["name"] = i
        for row in reader:
            try:
                a3 = _clean(row[idx["a3"]]).upper()
                a2 = _clean(row[idx["a2"]]).upper()
                name = _clean(row[idx["name"]])
            except (IndexError, KeyError):
                continue
            if a2 and a3 and a2 not in a2_to_a3:
                a2_to_a3[a2] = a3
            if a3 and a3 not in a3_to_name:
                a3_to_name[a3] = name
    return a2_to_a3, a3_to_name


def load_ident_to_a3(airports_csv="airports.csv",
                     coords_csv="countries_codes_and_coordinates.csv"):
    """OurAirports ident -> alpha3, ALL airport types (broad set, as decided)."""
    import csv as _csv
    a2_to_a3, _ = load_coords(coords_csv)
    ident_to_a3 = {}
    with open(airports_csv, encoding="utf-8", newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            ident = (row.get("ident") or "").strip().upper()
            a2 = (row.get("iso_country") or "").strip().upper()
            a3 = a2_to_a3.get(a2)
            if ident and a3 and ident not in ident_to_a3:
                ident_to_a3[ident] = a3
    return ident_to_a3


def build_target_index(target_a3s, ident_to_a3, a3_to_name):
    """Per target country: ident set, derived 2-letter prefixes, name."""
    index = {}
    for a3 in target_a3s:
        idents = {i for i, c in ident_to_a3.items() if c == a3}
        prefixes = {i[:2] for i in idents if len(i) == 4 and i[:2].isalpha()}
        index[a3] = {"idents": idents, "prefixes": prefixes,
                     "name": (a3_to_name.get(a3) or "").upper()}
    return index


# ----------------------------------------------------------------------------
# MEIS (streaming: files up to ~227 MB, never json.load the whole array)
# ----------------------------------------------------------------------------

def stream_json_records(path):
    """Yield each top-level object of a JSON array as a raw string."""
    depth, in_str, esc, buf, capturing = 0, False, False, [], False
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            for ch in chunk:
                if capturing:
                    buf.append(ch)
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
                if ch == "{":
                    if depth == 0:
                        capturing = True
                        buf = ["{"]
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and capturing:
                        yield "".join(buf)
                        buf, capturing = [], False


def _flatten(obj, prefix=""):
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lines.extend(_flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            lines.extend(_flatten(v, f"{prefix}{i}."))
    else:
        t = str(obj).strip()
        if t and t.lower() not in ("none", "null"):
            lines.append(f"{prefix[:-1]}: {t}")
    return lines


def _condense_runways(rec):
    out = []
    for rw in rec.get("RunwayDetailsList") or []:
        if not isinstance(rw, dict):
            continue
        keep = {k: v for k, v in rw.items()
                if any(t in k.lower() for t in
                       ("name", "ident", "length", "width", "surface",
                        "pcn", "weight"))
                and not isinstance(v, (dict, list))}
        if keep:
            out.append(", ".join(f"{k}={v}" for k, v in keep.items()))
    return out


def meis_record_summary(rec):
    """Compact but verbatim-valued text for one airfield."""
    abi = rec.get("AirfieldBasicInformation") or {}
    names = abi.get("AirfieldNames") or {}
    icao = ((names.get("PrimaryICAO") or {}).get("ICAO") or "").upper()
    loc = (names.get("PrimaryICAO") or {}).get("LocationName") or ""
    op = abi.get("AirfieldOperationType") or ""
    parts = [f"AIRFIELD {icao} ({loc}) type={op}"]
    for key in MEIS_KEEP_KEYS:
        if key in rec and rec[key] not in (None, "", [], {}):
            text = "\n".join(_flatten(rec[key], f"{key}."))[:MEIS_PER_KEY_CAP]
            if text:
                parts.append(text)
    runways = _condense_runways(rec)
    if runways:
        parts.append("Runways: " + " | ".join(runways)[:MEIS_PER_KEY_CAP])
    return icao, "\n".join(parts)


def meis_icaos(rec):
    names = (rec.get("AirfieldBasicInformation") or {}).get("AirfieldNames") or {}
    primary = ((names.get("PrimaryICAO") or {}).get("ICAO") or "").upper()
    alts = [str(a).upper() for a in (names.get("AlternateICAO") or []) if a]
    return [c for c in [primary] + alts if c]


def meis_country_name(rec):
    info = ((rec.get("AirfieldBasicInformation") or {})
            .get("AirfieldAreaInformation") or {})
    return str((info.get("CountryInformation") or {})
               .get("CountryName") or "").upper()


def slice_meis(paths, index, ident_to_a3):
    slices = {a3: [] for a3 in index}
    name_to_a3 = {v["name"]: a3 for a3, v in index.items() if v["name"]}
    for path in paths:
        for raw in stream_json_records(path):
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            target = None
            for icao in meis_icaos(rec):
                a3 = ident_to_a3.get(icao)
                if a3 in slices:
                    target = a3
                    break
            if target is None:
                target = name_to_a3.get(meis_country_name(rec))
            if target is None:
                continue
            icao, text = meis_record_summary(rec)
            slices[target].append({"icao": icao, "text": text,
                                   "file": Path(path).name})
    return slices


# ----------------------------------------------------------------------------
# SVC_RMK
# ----------------------------------------------------------------------------

def slice_svc(path, index, ident_to_a3):
    slices = {a3: {"rows": [], "cycle_dates": []} for a3 in index}
    with open(path, encoding="cp1252", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            icao_i = header.index("ICAO")
            date_i = header.index("CYCLE_DATE")
        except ValueError:
            sys.exit(f"[svc] unexpected header: {header}")
        n_cols = len(header)
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            cols = line.split("\t", n_cols - 1)
            if len(cols) <= icao_i:
                continue
            a3 = ident_to_a3.get(cols[icao_i].strip().upper())
            if a3 in slices:
                slices[a3]["rows"].append(line)
                if len(cols) > date_i and cols[date_i].strip().isdigit():
                    slices[a3]["cycle_dates"].append(cols[date_i].strip())
    for a3, s in slices.items():
        dates = sorted(s.pop("cycle_dates"))
        s["date_range"] = [dates[0], dates[-1]] if dates else None
        s["header"] = "\t".join(header)
    return slices


# ----------------------------------------------------------------------------
# udl NOTAMs (two-tier: exact ident OR derived prefix -> scope=FIR)
# ----------------------------------------------------------------------------

_A_FIELD = re.compile(r"\bA\)\s*([A-Z]{3,4})\b")
_DT10 = re.compile(r"\b(\d{10})\b")


def slice_notam(path, index):
    slices = {a3: {"blocks": [], "years": []} for a3 in index}
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        idents = set(_A_FIELD.findall(block))
        if not idents:
            continue
        for a3, meta in index.items():
            scope = None
            if idents & meta["idents"]:
                scope = "AIRFIELD"
            elif any(i[:2] in meta["prefixes"] for i in idents if len(i) == 4):
                scope = "FIR"
            if scope:
                slices[a3]["blocks"].append({"scope": scope, "text": block})
                for m in _DT10.findall(block):
                    slices[a3]["years"].append("20" + m[:2])
    for a3, s in slices.items():
        years = sorted(set(s.pop("years")))
        s["date_range"] = [years[0], years[-1]] if years else None
    return slices


# ----------------------------------------------------------------------------
# Orchestration + persistence
# ----------------------------------------------------------------------------

def target_a3s_from_dir(countries_dir):
    return sorted(_SUFFIX.sub("", p.stem).upper()
                  for p in Path(countries_dir).glob("*.json"))


def build_or_load_slices(countries_dir, sources_dir, out_dir,
                         airports_csv="airports.csv",
                         coords_csv="countries_codes_and_coordinates.csv",
                         rebuild=False):
    """Returns {A3: slice_dict}; caches one JSON per country in out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = target_a3s_from_dir(countries_dir)
    cached = {a3: out_dir / f"{a3}.json" for a3 in targets}
    if not rebuild and all(p.exists() for p in cached.values()):
        return {a3: json.loads(p.read_text(encoding="utf-8"))
                for a3, p in cached.items()}

    sources_dir = Path(sources_dir)
    meis_paths = sorted(p for p in sources_dir.iterdir()
                        if p.suffix.lower() == ".json" and "meis" in p.name.lower())
    svc_path = next((p for p in sources_dir.iterdir()
                     if "svc_rmk" in p.name.lower()), None)
    udl_path = next((p for p in sources_dir.iterdir()
                     if "udl" in p.name.lower()), None)

    ident_to_a3 = load_ident_to_a3(airports_csv, coords_csv)
    _, a3_to_name = load_coords(coords_csv)
    index = build_target_index(targets, ident_to_a3, a3_to_name)

    print(f"[sources] targets: {targets}")
    meis = slice_meis(meis_paths, index, ident_to_a3) if meis_paths \
        else {a3: [] for a3 in targets}
    svc = slice_svc(svc_path, index, ident_to_a3) if svc_path \
        else {a3: {"rows": [], "date_range": None, "header": ""}
              for a3 in targets}
    notam = slice_notam(udl_path, index) if udl_path \
        else {a3: {"blocks": [], "date_range": None} for a3 in targets}

    result = {}
    for a3 in targets:
        result[a3] = {"meis": meis[a3], "svc": svc[a3], "notam": notam[a3]}
        cached[a3].write_text(json.dumps(result[a3], ensure_ascii=False,
                                         indent=1), encoding="utf-8")
    return result


def summary_table(slices):
    lines = [f"{'A3':<5} {'MEIS':>5} {'SVC':>5} {'NOTAM(A/F)':>11} "
             f"{'SVC dates':>15} {'NOTAM dates':>12}"]
    for a3, s in sorted(slices.items()):
        nb = s["notam"]["blocks"]
        n_a = sum(1 for b in nb if b["scope"] == "AIRFIELD")
        n_f = len(nb) - n_a
        sd = "-".join(s["svc"]["date_range"] or ["--"])
        nd = "-".join(s["notam"]["date_range"] or ["--"])
        lines.append(f"{a3:<5} {len(s['meis']):>5} {len(s['svc']['rows']):>5} "
                     f"{f'{n_a}/{n_f}':>11} {sd:>15} {nd:>12}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", default="data/fcg_countries")
    ap.add_argument("--sources", default="data/sources")
    ap.add_argument("--out", default="data/work/sources_index")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    slices = build_or_load_slices(args.countries, args.sources, args.out,
                                  rebuild=args.rebuild)
    print(summary_table(slices))


if __name__ == "__main__":
    main()
