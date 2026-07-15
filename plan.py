"""
plan.py - agentic route planner using ONLY fcg_extract.csv + your vLLM model.
No external airport tables. No runways.csv. Nothing to download.

  python plan.py IAD Delhi 2026-09-15
  python plan.py KIAD VIDP 2026-09-15 --hazmat

How it works:
  1. Origin/destination resolved to ICAO by the model (IAD -> KIAD, Delhi -> VIDP).
  2. Candidate stops = ONLY the ICAOs on your FCG extract's authorized
     entry/exit lists. The legal stop set comes from your own data.
  3. Coordinates for those ICAOs come from the model ONCE, are range-validated,
     and cached in coords_cache.json - open it, review it, correct it by hand;
     it is never re-asked once cached. All distance math is then deterministic.
  4. Route = fewest authorized stops along the corridor with every FCG check
     applied (entry authorization, lead-time feasibility, HAZMAT).
  5. Overflight: the model estimates which countries each leg crosses; every
     estimate is then CHECKED against your FCG overflight data (prohibited ->
     blocker). Crossing lists are flagged as model-estimated in the report.

Trust model: clearance facts are 100% from your CSV, deterministically checked.
Geography (coords, crossings) is model-provided, cached, and always labeled
approximate - it steers the search, it never overrides an FCG rule.
"""

import json
import math
import re
import sys
from datetime import date, timedelta
from pathlib import Path

from openai import OpenAI
from datalayer import load_fcg, parse_lead_days

# ----------------------------- CONFIG -----------------------------
MODEL         = "gemma-4-31B-it"
VLLM_BASE_URL = "http://localhost:8000/v1"
API_KEY       = "EMPTY"

FCG_CSV       = "./fcg_extract.csv"
COORDS_CACHE  = "./coords_cache.json"
REPORT_PATH   = "./plan_report.md"

MAX_LEG_KM    = 5000       # max distance between consecutive stops
CORRIDOR_KM   = 2500       # how far off the direct path a stop may sit
TODAY         = date.today()
PROHIBIT_WORDS = ("prohibit", "suspend", "not being granted", "not permitted",
                  "not allowed", "forbidden", "closed", "denied")
# ------------------------------------------------------------------

EARTH_R = 6371.0
ICAO_RE = re.compile(r"^[A-Z][A-Z0-9]{3}$")


# --------------------------- verified math -------------------------
def _rad(x): return math.radians(float(x))


def dist_km(a, b):
    p1, p2 = _rad(a[0]), _rad(b[0])
    dphi, dlmb = _rad(b[0] - a[0]), _rad(b[1] - a[1])
    h = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * EARTH_R * math.asin(math.sqrt(h))


def cross_track_km(pt, a, b):
    d13 = dist_km(a, pt) / EARTH_R
    if d13 == 0: return 0.0, 0.0
    def brng(p, q):
        return math.atan2(math.sin(_rad(q[1]-p[1]))*math.cos(_rad(q[0])),
                          math.cos(_rad(p[0]))*math.sin(_rad(q[0]))
                          - math.sin(_rad(p[0]))*math.cos(_rad(q[0]))*math.cos(_rad(q[1]-p[1])))
    dxt = math.asin(math.sin(d13) * math.sin(brng(a, pt) - brng(a, b)))
    dat = math.acos(max(-1, min(1, math.cos(d13)/max(math.cos(dxt), 1e-12))))
    return abs(dxt)*EARTH_R, dat*EARTH_R


# ----------------------- model-provided geography ------------------
def ask_json(client, prompt, schema, max_tokens=1200):
    r = client.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "x", "schema": schema}})
    txt = (r.choices[0].message.content or "").strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        txt = txt[4:] if txt[:4].lower() == "json" else txt
    return json.loads(txt[txt.find("{"):])


def resolve_icao(client, term):
    if ICAO_RE.match(term.upper()):
        return term.upper()
    out = ask_json(client,
        f"What is the 4-letter ICAO code of the airport '{term}'? If ambiguous, "
        f"pick the main international airport. Respond only with JSON.",
        {"type": "object", "properties": {"icao": {"type": "string"}},
         "required": ["icao"]}, 100)
    icao = out.get("icao", "").upper()
    if not ICAO_RE.match(icao):
        raise SystemExit(f"Could not resolve '{term}' to an ICAO code (model said '{icao}')")
    print(f"  resolved '{term}' -> {icao} (model-provided)")
    return icao


def get_coords(client, icaos):
    """{icao: (lat, lon, name)} - from cache first, model for the rest, validated."""
    cache = json.loads(Path(COORDS_CACHE).read_text()) if Path(COORDS_CACHE).exists() else {}
    missing = [i for i in icaos if i not in cache]
    for chunk_start in range(0, len(missing), 40):
        chunk = missing[chunk_start:chunk_start + 40]
        out = ask_json(client,
            "Give decimal lat/lon coordinates and name for each of these airport "
            f"ICAO codes: {', '.join(chunk)}. Skip any code that is not a real "
            "airport. Respond only with JSON.",
            {"type": "object", "properties": {"airports": {"type": "array",
                "items": {"type": "object", "properties": {
                    "icao": {"type": "string"}, "lat": {"type": "number"},
                    "lon": {"type": "number"}, "name": {"type": "string"}},
                    "required": ["icao", "lat", "lon"]}}},
             "required": ["airports"]}, 3000)
        for a in out.get("airports", []):
            icao = str(a.get("icao", "")).upper()
            try:
                lat, lon = float(a["lat"]), float(a["lon"])
            except (KeyError, TypeError, ValueError):
                continue
            if ICAO_RE.match(icao) and -90 <= lat <= 90 and -180 <= lon <= 180:
                cache[icao] = [lat, lon, a.get("name", icao)]
    Path(COORDS_CACHE).write_text(json.dumps(cache, indent=1))
    found = {i: tuple(cache[i]) for i in icaos if i in cache}
    dropped = [i for i in icaos if i not in cache]
    if dropped:
        print(f"  no coordinates for {dropped} (skipped - not real airports or model unsure)")
    return found


def countries_crossed(client, a_icao, a, b_icao, b):
    out = ask_json(client,
        f"A great-circle flight goes from {a_icao} ({a[0]:.2f},{a[1]:.2f}) to "
        f"{b_icao} ({b[0]:.2f},{b[1]:.2f}). List the countries whose airspace "
        f"it crosses, in order, excluding departure and arrival countries. "
        f"Respond only with JSON.",
        {"type": "object", "properties": {"countries": {"type": "array",
            "items": {"type": "string"}}}, "required": ["countries"]}, 500)
    return out.get("countries", [])


# --------------------------- FCG checks ----------------------------
def find_by_icao(fcg, icao):
    for r in fcg:
        if icao in r["icaos"]:
            return r
    return None


def find_by_country(fcg, name):
    n = name.lower().strip()
    for r in fcg:
        if n in r["key"].replace("_", " ").lower():
            return r
    return None


def stop_check(rec, icao, dep, hazmat):
    blockers, warns, file_by = [], [], None
    lead = rec["lead_days"]
    if lead is None and rec["lead_raw"].strip().upper() != "NA":
        warns.append(f"{icao} ({rec['key']}): lead time not machine-parseable")
    elif lead is not None:
        file_by = dep - timedelta(days=lead)
        if file_by < TODAY:
            blockers.append(f"{icao} ({rec['key']}): lead {lead}d unmeetable (file_by {file_by})")
    if hazmat:
        hz = (rec["hazmat"] or "").lower()
        if any(w in hz for w in PROHIBIT_WORDS):
            blockers.append(f"{icao} ({rec['key']}): HAZMAT prohibited")
        elif hz.strip() in ("", "na"):
            warns.append(f"{icao} ({rec['key']}): HAZMAT rules unknown")
    return blockers, warns, file_by


# ------------------------------ planner ----------------------------
def plan(origin_term, dest_term, dep_str, hazmat):
    dep = date.fromisoformat(dep_str)
    fcg = load_fcg(FCG_CSV)
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=API_KEY)

    origin = resolve_icao(client, origin_term)
    dest = resolve_icao(client, dest_term)

    authorized = sorted({i for r in fcg for i in r["icaos"]})
    print(f"  {len(authorized)} authorized entry/exit ICAOs in FCG extract")
    coords = get_coords(client, sorted(set(authorized + [origin, dest])))
    if origin not in coords or dest not in coords:
        raise SystemExit("No coordinates for origin/destination - add them to coords_cache.json by hand")
    A, B = coords[origin], coords[dest]
    total = dist_km(A, B)

    # candidates: authorized + coords + in corridor + FCG-feasible
    cands, rejected = [], []
    for icao in authorized:
        if icao in (origin, dest) or icao not in coords:
            continue
        off, along = cross_track_km(coords[icao], A, B)
        if off > CORRIDOR_KM or not (0 < along < total):
            continue
        rec = find_by_icao(fcg, icao)
        b, w, fb = stop_check(rec, icao, dep, hazmat)
        (cands if not b else rejected).append(
            {"icao": icao, "rec": rec, "along": along, "off": off,
             "warns": w, "blockers": b, "file_by": fb})
    cands.sort(key=lambda c: c["along"])

    # greedy fewest stops
    stops, pos_pt, pos_along = [], A, 0.0
    while dist_km(pos_pt, B) > MAX_LEG_KM:
        reach = [c for c in cands if c["along"] > pos_along
                 and dist_km(pos_pt, coords[c["icao"]]) <= MAX_LEG_KM]
        if not reach:
            return None, rejected, total, coords, fcg, client, dep, origin, dest
        nxt = max(reach, key=lambda c: c["along"])
        stops.append(nxt)
        pos_pt, pos_along = coords[nxt["icao"]], nxt["along"]
    return stops, rejected, total, coords, fcg, client, dep, origin, dest


def report(stops, rejected, total, coords, fcg, client, dep, origin, dest, hazmat):
    out = [f"# Route plan: {origin} -> {dest}",
           f"Departure {dep} | planned {TODAY} | direct {total:.0f} km | "
           f"max leg {MAX_LEG_KM} km | HAZMAT={hazmat}",
           "*Coordinates and airspace crossings are model-provided approximations "
           "(cached in coords_cache.json). All clearance facts are from your FCG extract.*", ""]
    if stops is None:
        out += ["## ❌ No compliant route found",
                "Closest rejected candidates and why:"]
        for c in rejected[:12]:
            out.append(f"- {c['icao']}: {'; '.join(c['blockers'])}")
        Path(REPORT_PATH).write_text("\n".join(out)); print("\n".join(out)); return

    chain = [origin] + [s["icao"] for s in stops] + [dest]
    out.append("## Itinerary")
    out.append(" -> ".join(chain))
    for a, b in zip(chain, chain[1:]):
        out.append(f"- {a} -> {b}: {dist_km(coords[a], coords[b]):.0f} km")
    out.append("")

    deadlines, warns, blockers = {}, [], []
    for s in stops:
        r = s["rec"]
        out += [f"### {s['icao']} - {coords[s['icao']][2]}",
                f"- Country record: {r['key']}",
                f"- Lead time: {r['lead_raw'][:200]}"
                + (f" (file by **{s['file_by']}**)" if s["file_by"] else "")]
        for lbl, key in (("Customs", "customs"), ("Hours", "hours"),
                         ("Forms", "forms"), ("Payment", "cash")):
            v = r[key]
            if v and v.strip().upper() != "NA":
                out.append(f"- {lbl}: {v[:180]}")
        warns += s["warns"]
        if s["file_by"] and r["lead_days"]:
            deadlines[r["key"]] = (s["file_by"], r["lead_days"])
        out.append("")

    # destination country lead time counts too
    dest_rec = find_by_icao(fcg, dest)
    if dest_rec and dest_rec["lead_days"]:
        fb = dep - timedelta(days=dest_rec["lead_days"])
        deadlines[dest_rec["key"]] = (fb, dest_rec["lead_days"])
        if fb < TODAY:
            blockers.append(f"{dest} ({dest_rec['key']}): lead unmeetable (file_by {fb})")
    elif dest_rec is None:
        warns.append(f"{dest}: destination not on any FCG entry/exit list - verify separately")

    out.append("## Overflight (model-estimated crossings, FCG-checked)")
    for a, b in zip(chain, chain[1:]):
        crossed = countries_crossed(client, a, coords[a], b, coords[b])
        parts = []
        for name in crossed:
            rec = find_by_country(fcg, name)
            if rec is None:
                parts.append(f"{name}[no FCG data]")
                warns.append(f"overflight {a}->{b}: {name} - no FCG row")
                continue
            of = rec["overflight"]
            if any(w in (of or "").lower() for w in PROHIBIT_WORDS):
                parts.append(f"{rec['key']}[⛔ PROHIBITED]")
                blockers.append(f"overflight {a}->{b}: {rec['key']} PROHIBITED")
            elif of.strip().upper() == "NA":
                parts.append(f"{rec['key']}[❓]")
                warns.append(f"overflight {a}->{b}: {rec['key']} requirements unknown")
            else:
                lead = rec.get("overflight_lead_days") or parse_lead_days(of)
                if lead:
                    fb = dep - timedelta(days=lead)
                    deadlines[f"{rec['key']} (overflight)"] = (fb, lead)
                    if fb < TODAY:
                        blockers.append(f"overflight {a}->{b}: {rec['key']} lead unmeetable")
                parts.append(f"{rec['key']}[ok]")
        out.append(f"- {a} -> {b}: " + (", ".join(parts) if parts else "(none reported)"))

    out.append("\n## Verdict: " + ("✅ NO BLOCKERS" if not blockers else "⛔ BLOCKED - reroute needed"))
    out += [f"- ⛔ {b}" for b in blockers]
    if warns:
        out.append("\n## Warnings / unknowns")
        out += [f"- ⚠️ {w}" for w in dict.fromkeys(warns)]
    if deadlines:
        out.append("\n## Filing timeline (soonest first)")
        for key, (fb, lead) in sorted(deadlines.items(), key=lambda kv: kv[1][0]):
            out.append(f"- **{fb}** - file {key} ({lead}d lead)"
                       + (" ⚠️ OVERDUE" if fb < TODAY else ""))
    Path(REPORT_PATH).write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nSaved -> {REPORT_PATH}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit("Usage: python plan.py ORIGIN DEST YYYY-MM-DD [--hazmat]")
    hz = "--hazmat" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--hazmat"]
    stops, rejected, total, coords, fcg, client, dep, o, d = plan(args[0], args[1], args[2], hz)
    report(stops, rejected, total, coords, fcg, client, dep, o, d, hz)
