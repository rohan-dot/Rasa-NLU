"""
Broom Route Planner: FCG extract CSV + OurAirports -> compliant itinerary + report.

Fully offline. Inputs:
  1. fcg_extract.csv        (output of fcg_extract.py - one row per country/airport file)
  2. airports.csv           (OurAirports - https://davidmegginson.github.io/ourairports-data/airports.csv)
  3. runways.csv            (OurAirports - .../runways.csv)   [optional, runway-length filter]
  4. countries.csv          (OurAirports - .../countries.csv) [maps ISO code -> country name]

Download the three OurAirports files ONCE on a connected machine, then carry them in.

What it does:
  - Great-circle route origin -> destination.
  - Finds every airport within CORRIDOR_KM of the route line.
  - Keeps only airports that pass the FCG criteria from your CSV:
      * appears in that country's authorized entry/exit list (entry_exit_airports_raw)
      * diplomatic lead time is achievable before DEPARTURE_DATE
      * HAZMAT permitted if CARRYING_HAZMAT
      * airfield-restriction and hours fields surfaced as warnings (free text -> report)
  - Greedily selects the fewest stops with legs <= MAX_LEG_KM.
  - Emits itinerary.md: legs, per-stop requirements, and a clearance-filing
    deadline timeline sorted soonest-first.

Run:  python broom_planner.py HOGW DRMS 2026-08-10
      (origin ICAO, destination ICAO, departure date)
"""

import csv
import math
import re
import sys
from datetime import date, timedelta
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
FCG_CSV       = "./fcg_extract.csv"
AIRPORTS_CSV  = "./data/airports.csv"
RUNWAYS_CSV   = "./data/runways.csv"      # optional; set None to skip
COUNTRIES_CSV = "./data/countries.csv"

MAX_LEG_KM      = 1500     # max distance between consecutive stops (config, not aircraft range)
CORRIDOR_KM     = 400      # how far off the great-circle line a stop may sit
MIN_RUNWAY_FT   = 6000     # 0 to disable
ALLOWED_TYPES   = {"large_airport", "medium_airport"}   # OurAirports 'type' column
CARRYING_HAZMAT = False
TODAY           = date.today()          # clearance clock starts now
REPORT_PATH     = "./itinerary.md"

# When a country's lead time can't be parsed to a number of days, treat it as
# this many days and flag it in the report rather than silently passing it.
UNPARSED_LEAD_DAYS = 30
# ------------------------------------------------------------------

EARTH_R = 6371.0
ICAO_RE = re.compile(r"\b([A-Z][A-Z0-9]{3})\b")
LEAD_RE = re.compile(
    r"(\d+)\s*(business\s+|working\s+|calendar\s+)?(day|week|hour)s?",
    re.I,
)


# --------------------------- geometry ------------------------------
def rad(deg):
    return math.radians(float(deg))


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = rad(lat1), rad(lat2)
    dphi, dlmb = rad(lat2 - lat1), rad(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def cross_track_km(lat, lon, lat1, lon1, lat2, lon2):
    """Perpendicular distance of point from great-circle path 1->2, plus the
    along-track distance from point 1 (used for ordering stops along the route)."""
    d13 = haversine_km(lat1, lon1, lat, lon) / EARTH_R
    if d13 == 0:
        return 0.0, 0.0
    brng = lambda a1, o1, a2, o2: math.atan2(
        math.sin(rad(o2 - o1)) * math.cos(rad(a2)),
        math.cos(rad(a1)) * math.sin(rad(a2))
        - math.sin(rad(a1)) * math.cos(rad(a2)) * math.cos(rad(o2 - o1)),
    )
    t13 = brng(lat1, lon1, lat, lon)
    t12 = brng(lat1, lon1, lat2, lon2)
    dxt = math.asin(math.sin(d13) * math.sin(t13 - t12))
    dat = math.acos(max(-1.0, min(1.0, math.cos(d13) / max(math.cos(dxt), 1e-12))))
    return abs(dxt) * EARTH_R, dat * EARTH_R


# --------------------------- data layer ----------------------------
def load_airports():
    """{icao: {name, lat, lon, iso_country, type}} from OurAirports."""
    out = {}
    with open(AIRPORTS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            icao = (r.get("gps_code") or r.get("ident") or "").strip().upper()
            if len(icao) != 4 or r.get("type") not in ALLOWED_TYPES | {"small_airport"}:
                continue
            try:
                out[icao] = {
                    "icao": icao,
                    "name": r["name"],
                    "lat": float(r["latitude_deg"]),
                    "lon": float(r["longitude_deg"]),
                    "iso": r["iso_country"],
                    "type": r["type"],
                }
            except (ValueError, KeyError):
                continue
    return out


def load_runway_max():
    """{icao: longest runway ft}. Optional filter input."""
    if not RUNWAYS_CSV or not Path(RUNWAYS_CSV).exists():
        return {}
    best = {}
    with open(RUNWAYS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            icao = (r.get("airport_ident") or "").strip().upper()
            try:
                ft = int(float(r.get("length_ft") or 0))
            except ValueError:
                continue
            if ft > best.get(icao, 0):
                best[icao] = ft
    return best


def load_countries():
    """{iso2: country name lower} for matching FCG rows to OurAirports iso codes."""
    out = {}
    if Path(COUNTRIES_CSV).exists():
        with open(COUNTRIES_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[r["code"].strip().upper()] = r["name"].strip().lower()
    return out


def parse_lead_days(text):
    """Best-effort: max lead time in days found in free text. None if nothing parses."""
    if not text or text.strip().upper() == "NA":
        return None
    worst = None
    for num, qual, unit in LEAD_RE.findall(text):
        n = int(num)
        days = n / 24 if unit.lower().startswith("hour") else n * 7 if unit.lower().startswith("week") else n
        if qual and qual.strip().lower() in ("business", "working"):
            days = days * 7 / 5  # rough calendar conversion
        worst = max(worst or 0, math.ceil(days))
    return worst


def load_fcg():
    """One record per FCG row: authorized ICAO set, lead days, hazmat/notes text."""
    rows = []
    with open(FCG_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            entry = r.get("entry_exit_airports_raw", "") or ""
            icaos = set(ICAO_RE.findall(entry.upper())) if entry.strip().upper() != "NA" else set()
            rows.append({
                "key": r.get("airport", "?"),                 # file stem = country/airport name
                "icaos": icaos,
                "lead_days": parse_lead_days(r.get("diplomatic_lead_time_raw", "")),
                "lead_raw": r.get("diplomatic_lead_time_summary") or r.get("diplomatic_lead_time_raw", "NA"),
                "hazmat": r.get("hazmat_summary") or r.get("hazmat_raw", "NA"),
                "restrictions": r.get("airfield_restrictions_summary", "NA"),
                "customs": r.get("customs_immigration_summary", "NA"),
                "hours": r.get("operating_hours_summary", "NA"),
                "forms": r.get("country_specific_summary", "NA"),
                "cash": r.get("aircard_cash_summary", "NA"),
            })
    return rows


def fcg_for_icao(fcg_rows, icao, country_name):
    """Match an airport to its FCG record: prefer explicit ICAO listing, else
    fall back to country-name match on the file stem."""
    for r in fcg_rows:
        if icao in r["icaos"]:
            return r, True
    cn = (country_name or "").lower()
    for r in fcg_rows:
        if cn and cn in r["key"].replace("_", " ").lower():
            return r, False
    return None, False


# ------------------------- constraint check ------------------------
def check_stop(fcg, authorized, dep_date, hazmat):
    """Return (ok, blockers, warnings) for one candidate stop."""
    blockers, warns = [], []
    if fcg is None:
        return False, ["no FCG data for this country"], []
    if not authorized:
        blockers.append("not on authorized entry/exit airport list")
    lead = fcg["lead_days"]
    if lead is None and fcg["lead_raw"].strip().upper() != "NA":
        lead = UNPARSED_LEAD_DAYS
        warns.append(f"lead time unparsed; assumed {UNPARSED_LEAD_DAYS} days: '{fcg['lead_raw'][:120]}'")
    if lead is not None:
        deadline = dep_date - timedelta(days=lead)
        if deadline < TODAY:
            blockers.append(f"lead time {lead}d cannot be met (needed filing by {deadline})")
    if hazmat and fcg["hazmat"].strip().upper() != "NA":
        low = fcg["hazmat"].lower()
        if any(w in low for w in ("prohibit", "not permitted", "not allowed", "forbidden")):
            blockers.append("HAZMAT prohibited")
        else:
            warns.append(f"HAZMAT conditions: {fcg['hazmat'][:160]}")
    for field, label in (("restrictions", "airfield restriction"), ("hours", "hours")):
        v = fcg[field]
        if v and v.strip().upper() != "NA":
            warns.append(f"{label}: {v[:160]}")
    return not blockers, blockers, warns


# --------------------------- route planning ------------------------
def plan(origin, dest, dep_date):
    airports = load_airports()
    rmax = load_runway_max()
    countries = load_countries()
    fcg_rows = load_fcg()

    if origin not in airports or dest not in airports:
        missing = [c for c in (origin, dest) if c not in airports]
        raise SystemExit(f"ICAO not found in airports.csv: {missing}")
    A, B = airports[origin], airports[dest]
    total = haversine_km(A["lat"], A["lon"], B["lat"], B["lon"])

    # 1. corridor candidates, compliance-checked
    cands = []
    for ap in airports.values():
        if ap["icao"] in (origin, dest) or ap["type"] not in ALLOWED_TYPES:
            continue
        if MIN_RUNWAY_FT and rmax.get(ap["icao"], 0) < MIN_RUNWAY_FT:
            continue
        dxt, dat = cross_track_km(ap["lat"], ap["lon"], A["lat"], A["lon"], B["lat"], B["lon"])
        if dxt > CORRIDOR_KM or dat <= 0 or dat >= total:
            continue
        fcg, listed = fcg_for_icao(fcg_rows, ap["icao"], countries.get(ap["iso"]))
        ok, blockers, warns = check_stop(fcg, listed, dep_date, CARRYING_HAZMAT)
        cands.append({**ap, "along": dat, "off": dxt, "fcg": fcg,
                      "ok": ok, "blockers": blockers, "warns": warns})
    cands.sort(key=lambda c: c["along"])

    # 2. greedy min-stop selection over compliant candidates
    stops, pos, pos_ap = [], 0.0, A
    compliant = [c for c in cands if c["ok"]]
    while haversine_km(pos_ap["lat"], pos_ap["lon"], B["lat"], B["lon"]) > MAX_LEG_KM:
        reach = [c for c in compliant
                 if c["along"] > pos
                 and haversine_km(pos_ap["lat"], pos_ap["lon"], c["lat"], c["lon"]) <= MAX_LEG_KM]
        if not reach:
            return None, cands, total, "no compliant stop reachable — raise MAX_LEG_KM/CORRIDOR_KM or review blockers"
        nxt = max(reach, key=lambda c: c["along"])   # farthest reachable = fewest stops
        stops.append(nxt)
        pos, pos_ap = nxt["along"], nxt
    return stops, cands, total, None


# ------------------------------ report -----------------------------
def fmt_stop(s):
    lines = [f"### {s['icao']} — {s['name']} ({s['iso']})",
             f"- Along route: {s['along']:.0f} km, {s['off']:.0f} km off-track"]
    f = s["fcg"]
    if f:
        lines.append(f"- FCG source: `{f['key']}`")
        lines.append(f"- Lead time: {f['lead_raw'][:200]}")
        for label, key in (("Customs/immigration", "customs"), ("Hours", "hours"),
                           ("Forms", "forms"), ("Air Card / cash", "cash")):
            v = f[key]
            if v and v.strip().upper() != "NA":
                lines.append(f"- {label}: {v[:200]}")
    for w in s["warns"]:
        lines.append(f"- ⚠️ {w}")
    for b in s["blockers"]:
        lines.append(f"- ⛔ {b}")
    return "\n".join(lines)


def write_report(origin, dest, dep_date, stops, cands, total, err):
    out = [f"# Route plan: {origin} → {dest}",
           f"Departure {dep_date} | filed as of {TODAY} | route {total:.0f} km | "
           f"max leg {MAX_LEG_KM} km | corridor {CORRIDOR_KM} km | HAZMAT={CARRYING_HAZMAT}", ""]
    if err:
        out += ["## ❌ No compliant route", err, "",
                "### Nearest blocked candidates (why they failed)"]
        for c in [c for c in cands if not c["ok"]][:15]:
            out.append(f"- {c['icao']} {c['name']}: {'; '.join(c['blockers'])}")
    else:
        out.append(f"## Itinerary ({len(stops)} stop{'s' if len(stops) != 1 else ''})")
        out.append(f"{origin} → " + " → ".join(s["icao"] for s in stops) + f" → {dest}\n")
        for s in stops:
            out += [fmt_stop(s), ""]
        # filing-deadline timeline
        out.append("## Clearance filing timeline (soonest first)")
        seen, rows = set(), []
        for s in stops:
            f = s["fcg"]
            if not f or f["key"] in seen:
                continue
            seen.add(f["key"])
            lead = f["lead_days"] or (UNPARSED_LEAD_DAYS if f["lead_raw"].strip().upper() != "NA" else None)
            if lead:
                rows.append((dep_date - timedelta(days=lead), f["key"], lead))
        for dl, key, lead in sorted(rows):
            flag = " ⚠️ OVERDUE" if dl < TODAY else ""
            out.append(f"- **{dl}** — file {key} clearance ({lead}d lead){flag}")
    Path(REPORT_PATH).write_text("\n".join(out), encoding="utf-8")
    print(f"Report -> {REPORT_PATH}")


# ------------------------------- main -------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit("Usage: python broom_planner.py ORIGIN_ICAO DEST_ICAO YYYY-MM-DD")
    o, d = sys.argv[1].upper(), sys.argv[2].upper()
    dep = date.fromisoformat(sys.argv[3])
    stops, cands, total, err = plan(o, d, dep)
    write_report(o, d, dep, stops or [], cands, total, err)
