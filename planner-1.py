"""
planner.py - route planner over YOUR fcg_extract.csv + YOUR mapping file.
Needs: fcg_extract.csv, mapping.csv (code,name), datalayer.py, your vLLM.
Nothing external. Nothing downloaded.

  python planner.py hog dur 2026-09-15
  python planner.py HOGW DRMS 2026-09-15 --hazmat
  python planner.py "Hogwarts" "Durmstrang Institute" 2026-09-15

Input can be: a country code from your mapping file (hog), a country name
(Hogwarts), an authorized ICAO (HOGW), or any airport/city name (the model
resolves it). If you give a country, the planner automatically picks that
country's authorized airport that best fits the route.

Division of labor:
  YOUR CSVs   -> all clearance facts: authorized entry/exit ICAOs, lead times,
                 HAZMAT, overflight rules, customs/hours/forms. Deterministic.
  MATH TOOLS  -> haversine, cross-track corridor, greedy fewest-stop selection.
                 Written here, deterministic, no model involvement.
  YOUR MODEL  -> only geography: coordinates (asked once, cached to
                 coords_cache.json - open and hand-correct it any time) and
                 which countries each leg overflies (every estimate is then
                 CHECKED against your CSV; prohibited -> blocker).
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
MAPPING_CSV   = "./mapping.csv"        # columns: code,name
COORDS_CACHE  = "./coords_cache.json"
REPORT_PATH   = "./plan_report.md"

MAX_LEG_KM    = 5000
CORRIDOR_KM   = 2500
TODAY         = date.today()
PROHIBIT_WORDS = ("prohibit", "suspend", "not being granted", "not permitted",
                  "not allowed", "forbidden", "closed", "denied")
# ------------------------------------------------------------------

EARTH_R = 6371.0
ICAO_RE = re.compile(r"^[A-Z][A-Z0-9]{3}$")


# ------------------------- verified math ---------------------------
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


# -------------------- mapping + FCG country match ------------------
def load_mapping(path):
    """mapping.csv (code,name) -> ({CODE: Name}, {name_lower: CODE})"""
    import csv as _csv
    code2name, name2code = {}, {}
    if not Path(path).exists():
        print(f"  warning: {path} not found - country names disabled")
        return code2name, name2code
    with open(path, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            code = (r.get("code") or "").strip().upper()
            name = (r.get("name") or "").strip()
            if code and name:
                code2name[code] = name
                name2code[name.lower()] = code
    return code2name, name2code


def key_stem(key):
    return key.split("_")[0].upper()       # "HOG_FCG2" -> "HOG"


def display_name(key, code2name):
    return code2name.get(key_stem(key), key)


def find_row_by_country(fcg, country, code2name, name2code):
    """Match a country given as code, name, or model-produced name to its FCG row."""
    c = country.strip().lower()
    code = name2code.get(c) or (c.upper() if c.upper() in code2name else None)
    for r in fcg:
        stem = key_stem(r["key"])
        if code and stem == code:
            return r
        nm = code2name.get(stem, "").lower()
        if nm and (nm == c or nm in c or c in nm):
            return r
    return None


# ---------------------- model-provided geography -------------------
def ask_json(client, prompt, schema, max_tokens=1500):
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


def get_coords(client, icaos):
    cache = json.loads(Path(COORDS_CACHE).read_text()) if Path(COORDS_CACHE).exists() else {}
    missing = [i for i in icaos if i not in cache]
    for s in range(0, len(missing), 40):
        chunk = missing[s:s+40]
        out = ask_json(client,
            f"Give decimal lat/lon and name for each airport ICAO code: "
            f"{', '.join(chunk)}. Skip codes that are not real airports. JSON only.",
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
    dropped = [i for i in icaos if i not in cache]
    if dropped:
        print(f"  no coordinates for {dropped} (skipped)")
    return {i: tuple(cache[i]) for i in icaos if i in cache}


def crossings(client, a_icao, a, b_icao, b):
    out = ask_json(client,
        f"A great-circle flight goes from {a_icao} ({a[0]:.2f},{a[1]:.2f}) to "
        f"{b_icao} ({b[0]:.2f},{b[1]:.2f}). List countries whose airspace it "
        f"crosses, in order, excluding the departure and arrival countries. JSON only.",
        {"type": "object", "properties": {"countries": {"type": "array",
            "items": {"type": "string"}}}, "required": ["countries"]}, 600)
    return out.get("countries", [])


# --------------------- endpoint resolution -------------------------
def resolve_endpoint(term, fcg, authorized, code2name, name2code, client):
    """Return ("icao", ICAO) or ("country", row). Accepts ICAO, mapping code,
    country name, or any airport/city name (model fallback)."""
    t = term.strip()
    if ICAO_RE.match(t.upper()) and t.upper() in authorized:
        return "icao", t.upper()
    row = find_row_by_country(fcg, t, code2name, name2code)
    if row is not None:
        return "country", row
    if ICAO_RE.match(t.upper()):
        return "icao", t.upper()           # ICAO outside FCG list (e.g. home base)
    out = ask_json(client,
        f"What is the 4-letter ICAO code of the airport '{t}'? Pick the main "
        f"international airport if ambiguous. JSON only.",
        {"type": "object", "properties": {"icao": {"type": "string"}},
         "required": ["icao"]}, 100)
    icao = out.get("icao", "").upper()
    if not ICAO_RE.match(icao):
        raise SystemExit(f"Could not resolve '{term}' (model said '{icao}')")
    print(f"  resolved '{term}' -> {icao} (model-provided)")
    return "icao", icao


def pick_country_airport(row, other_pt, coords):
    """Country endpoint -> its authorized airport closest to the other endpoint."""
    best, bd = None, float("inf")
    for icao in sorted(row["icaos"]):
        if icao in coords:
            d = dist_km(coords[icao], other_pt)
            if d < bd:
                best, bd = icao, d
    return best


# ------------------------- stop-level checks -----------------------
def stop_check(rec, icao, dep, hazmat):
    blockers, warns, file_by = [], [], None
    lead = rec["lead_days"]
    if lead is None and rec["lead_raw"].strip().upper() != "NA":
        warns.append(f"{icao}: lead time not machine-parseable")
    elif lead is not None:
        file_by = dep - timedelta(days=lead)
        if file_by < TODAY:
            blockers.append(f"{icao}: lead {lead}d unmeetable (file_by {file_by})")
    if hazmat:
        hz = (rec["hazmat"] or "").lower()
        if any(w in hz for w in PROHIBIT_WORDS):
            blockers.append(f"{icao}: HAZMAT prohibited")
        elif hz.strip() in ("", "na"):
            warns.append(f"{icao}: HAZMAT rules unknown")
    return blockers, warns, file_by


# ------------------------------ planner ----------------------------
def main(origin_term, dest_term, dep_str, hazmat):
    dep = date.fromisoformat(dep_str)
    fcg = load_fcg(FCG_CSV)
    code2name, name2code = load_mapping(MAPPING_CSV)
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=API_KEY)

    icao_to_row = {}
    for r in fcg:
        for i in r["icaos"]:
            icao_to_row.setdefault(i, r)
    authorized = sorted(icao_to_row)
    print(f"  {len(fcg)} FCG rows, {len(authorized)} authorized ICAOs, "
          f"{len(code2name)} mapping entries")

    o_kind, o_val = resolve_endpoint(origin_term, fcg, authorized, code2name, name2code, client)
    d_kind, d_val = resolve_endpoint(dest_term, fcg, authorized, code2name, name2code, client)

    need = set(authorized)
    for kind, val in ((o_kind, o_val), (d_kind, d_val)):
        if kind == "icao":
            need.add(val)
    coords = get_coords(client, sorted(need))

    # country endpoints -> best authorized airport for the route
    if o_kind == "country" and d_kind == "country":
        best = None
        for oi in sorted(o_val["icaos"]):
            for di in sorted(d_val["icaos"]):
                if oi in coords and di in coords:
                    d = dist_km(coords[oi], coords[di])
                    if best is None or d < best[0]:
                        best = (d, oi, di)
        if not best:
            raise SystemExit("No coordinates for any airport pair between those countries")
        origin, dest = best[1], best[2]
    else:
        missing = [v for k, v in ((o_kind, o_val), (d_kind, d_val))
                   if k == "icao" and v not in coords]
        if missing:
            raise SystemExit(f"No coordinates for {missing} - add to {COORDS_CACHE} by hand")
        if o_kind == "country":
            origin, dest = pick_country_airport(o_val, coords[d_val], coords), d_val
        elif d_kind == "country":
            origin, dest = o_val, pick_country_airport(d_val, coords[o_val], coords)
        else:
            origin, dest = o_val, d_val
    if not origin or not dest:
        raise SystemExit("Could not choose airports for the given endpoints")
    print(f"  route endpoints: {origin} -> {dest}")

    A, B = coords[origin], coords[dest]
    total = dist_km(A, B)

    # candidates: authorized, in corridor, FCG-feasible
    cands, rejected = [], []
    for icao in authorized:
        if icao in (origin, dest) or icao not in coords:
            continue
        off, along = cross_track_km(coords[icao], A, B)
        if off > CORRIDOR_KM or not (0 < along < total):
            continue
        rec = icao_to_row[icao]
        b, w, fb = stop_check(rec, icao, dep, hazmat)
        entry = {"icao": icao, "rec": rec, "along": along, "off": off,
                 "warns": w, "blockers": b, "file_by": fb}
        (cands if not b else rejected).append(entry)
    cands.sort(key=lambda c: c["along"])

    # greedy fewest stops
    stops, pos_pt, pos_along, fail = [], A, 0.0, None
    while dist_km(pos_pt, B) > MAX_LEG_KM:
        reach = [c for c in cands if c["along"] > pos_along
                 and dist_km(pos_pt, coords[c["icao"]]) <= MAX_LEG_KM]
        if not reach:
            fail = "no compliant authorized stop reachable - raise MAX_LEG_KM/CORRIDOR_KM or review rejections"
            break
        nxt = max(reach, key=lambda c: c["along"])
        stops.append(nxt)
        pos_pt, pos_along = coords[nxt["icao"]], nxt["along"]

    # ------------------------------ report -------------------------
    dn = lambda key: display_name(key, code2name)
    out = [f"# Route plan: {origin} -> {dest}",
           f"Departure {dep} | planned {TODAY} | direct {total:.0f} km | "
           f"max leg {MAX_LEG_KM} km | HAZMAT={hazmat}",
           "*Coordinates & crossings are model-provided approximations "
           f"(cache: {COORDS_CACHE}). All clearance facts from your FCG extract.*", ""]
    if fail:
        out += ["## X No compliant route", fail, "", "Rejected candidates:"]
        out += [f"- {c['icao']} ({dn(c['rec']['key'])}): {'; '.join(c['blockers'])}"
                for c in rejected[:12]]
        Path(REPORT_PATH).write_text("\n".join(out)); print("\n".join(out)); return

    chain = [origin] + [s["icao"] for s in stops] + [dest]
    out.append("## Itinerary\n" + " -> ".join(chain))
    for a, b in zip(chain, chain[1:]):
        out.append(f"- {a} -> {b}: {dist_km(coords[a], coords[b]):.0f} km")
    out.append("")

    deadlines, warns, blockers = {}, [], []
    for s in stops:
        r = s["rec"]
        out += [f"### {s['icao']} - {coords[s['icao']][2]} ({dn(r['key'])})",
                f"- Lead time: {r['lead_raw'][:200]}"
                + (f" (file by **{s['file_by']}**)" if s["file_by"] else "")]
        for lbl, key in (("Customs", "customs"), ("Hours", "hours"),
                         ("Forms", "forms"), ("Payment", "cash")):
            v = r[key]
            if v and v.strip().upper() != "NA":
                out.append(f"- {lbl}: {v[:180]}")
        warns += s["warns"]
        if s["file_by"] and r["lead_days"]:
            deadlines[dn(r["key"])] = (s["file_by"], r["lead_days"])
        out.append("")

    for endpoint, label in ((dest, "destination"), (origin, "origin")):
        rec = icao_to_row.get(endpoint)
        if rec and rec["lead_days"]:
            fb = dep - timedelta(days=rec["lead_days"])
            deadlines[dn(rec["key"])] = (fb, rec["lead_days"])
            if fb < TODAY:
                blockers.append(f"{endpoint} ({dn(rec['key'])}): lead unmeetable (file_by {fb})")
        elif rec is None:
            warns.append(f"{endpoint}: {label} not on any FCG entry/exit list - verify separately")

    out.append("## Overflight (model-estimated crossings, checked against your CSV)")
    for a, b in zip(chain, chain[1:]):
        parts = []
        for name in crossings(client, a, coords[a], b, coords[b]):
            rec = find_row_by_country(fcg, name, code2name, name2code)
            if rec is None:
                parts.append(f"{name}[no FCG data]")
                warns.append(f"overflight {a}->{b}: {name} - no FCG row matched")
                continue
            of = rec["overflight"]
            if any(w in (of or "").lower() for w in PROHIBIT_WORDS):
                parts.append(f"{dn(rec['key'])}[PROHIBITED]")
                blockers.append(f"overflight {a}->{b}: {dn(rec['key'])} PROHIBITED")
            elif of.strip().upper() == "NA":
                parts.append(f"{dn(rec['key'])}[?]")
                warns.append(f"overflight {a}->{b}: {dn(rec['key'])} requirements unknown (NA)")
            else:
                lead = rec.get("overflight_lead_days") or parse_lead_days(of)
                if lead:
                    fb = dep - timedelta(days=lead)
                    deadlines[f"{dn(rec['key'])} (overflight)"] = (fb, lead)
                    if fb < TODAY:
                        blockers.append(f"overflight {a}->{b}: {dn(rec['key'])} lead unmeetable")
                parts.append(f"{dn(rec['key'])}[ok]")
        out.append(f"- {a} -> {b}: " + (", ".join(parts) if parts else "(none reported)"))

    out.append("\n## Verdict: " + ("NO BLOCKERS" if not blockers else "BLOCKED - reroute needed"))
    out += [f"- BLOCKER: {b}" for b in blockers]
    if warns:
        out.append("\n## Warnings / unknowns")
        out += [f"- WARN: {w}" for w in dict.fromkeys(warns)]
    if deadlines:
        out.append("\n## Filing timeline (soonest first)")
        for key, (fb, lead) in sorted(deadlines.items(), key=lambda kv: kv[1][0]):
            out.append(f"- **{fb}** - file {key} ({lead}d lead)"
                       + (" OVERDUE" if fb < TODAY else ""))
    Path(REPORT_PATH).write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nSaved -> {REPORT_PATH}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit("Usage: python planner.py ORIGIN DEST YYYY-MM-DD [--hazmat]")
    hz = "--hazmat" in sys.argv
    a = [x for x in sys.argv[1:] if x != "--hazmat"]
    main(a[0], a[1], a[2], hz)
