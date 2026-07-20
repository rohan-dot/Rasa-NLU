#!/usr/bin/env python3
"""
Agentic flight route planner — v3 (airport-precise stops).

What v3 adds over v2:
  * OurAirports airports.csv join: every airport code extracted from your FCG
    `entry_exit_airports_summary` gets real lat/lon (ICAO `ident` join).
  * STAGE-GRAPH OPTIMIZATION: for a query like "hog to dur to fleur", each
    stop expands into ALL designated candidate airports for that country
    (from YOUR FCG data). A dynamic program then picks one airport per stop
    minimizing total great-circle distance. Alternates are shown with their
    distance penalty so users see what they gave up.
  * If the user types a specific airport (e.g. KCHS), that stop is pinned to
    it (candidate set of one).
  * Country-level overflight feasibility (LLM, plan-check-replan) unchanged.
    LLM also vets each CHOSEN stop airport against the FCG text; a denied
    airport is removed from the candidate set and the DP re-runs.
  * --list mode: show designated airports for one country, with coordinates
    and LLM-structured usable/restricted verdicts.

Data files required (all in the working directory, all offline-friendly):
  fcg_extract.csv                        your FCG constraints (country-level)
  mapping.csv                            your code -> country name
  countries_codes_and_coordinates.csv    tadast gist (alpha-2, alpha-3, lat/lon)
  airports.csv                           OurAirports dump (ident, lat/lon)

Usage:
  python country_route_planner.py "hog to dur to fleur to hog"
  python country_route_planner.py "kchs to sazb to scdl to segu to kvps to kchs"
  python country_route_planner.py "usa to chl" --no-llm
  python country_route_planner.py "dur" --list
"""

import argparse
import heapq
import json
import math
import re
import sys
from collections import OrderedDict

import pandas as pd
import requests

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
FCG_CSV      = "fcg_extract.csv"
MAPPING_CSV  = "mapping.csv"
COORDS_CSV   = "countries_codes_and_coordinates.csv"
AIRPORTS_CSV = "airports.csv"            # OurAirports dump

VLLM_URL     = "http://localhost:8000/v1/chat/completions"
VLLM_MODEL   = "gemma-4-31B-it"
LLM_TIMEOUT  = 120
K_NEIGHBOURS = 8
MAX_LEG_KM   = 4000.0                    # country-graph edge cap; None = off
MAX_ALTERNATES_SHOWN = 4                 # per stop, in the report

FCG_CODE_COL = "airport"
AIRPORT_COL  = "entry_exit_airports_summary"
FEASIBILITY_COLS = [
    "overflight_summary",
    "entry_exit_airports_summary",
    "customs_immigration_summary",
    "diplomatic_lead_time_summary",
    "airfield_restrictions_summary",
    "operating_hours_summary",
    "hazmat_summary",
]
# OurAirports rows to accept as landable candidates
AIRPORT_TYPES_OK = {"large_airport", "medium_airport", "small_airport"}

COORD_OVERRIDES = {
    # "XKS": ("Kosovo", 42.6, 20.9),
    # "DGA": ("Diego Garcia", -7.3, 72.4),
}

# ----------------------------------------------------------------------------
# 1. LOADING (fail-fast)
# ----------------------------------------------------------------------------

def _clean(s):
    return s.strip().strip('"').strip() if isinstance(s, str) else s


def load_coords(path: str) -> pd.DataFrame:
    """tadast gist: returns iso3, iso2, country, lat, lon (one row per iso3)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "alpha-3" in lc:
            col_map[c] = "iso3"
        elif "alpha-2" in lc:
            col_map[c] = "iso2"
        elif "latitude" in lc:
            col_map[c] = "lat"
        elif "longitude" in lc:
            col_map[c] = "lon"
        elif lc == "country":
            col_map[c] = "country"
    df = df.rename(columns=col_map)
    missing = {"iso3", "iso2", "lat", "lon", "country"} - set(df.columns)
    if missing:
        sys.exit(f"[FATAL] coords CSV missing {missing}; found {list(df.columns)}")
    for c in ("iso3", "iso2", "country"):
        df[c] = df[c].map(_clean)
    for c in ("lat", "lon"):
        df[c] = pd.to_numeric(df[c].map(_clean), errors="coerce")
    df["iso3"] = df["iso3"].str.upper()
    df["iso2"] = df["iso2"].str.upper()
    df = (df.dropna(subset=["lat", "lon"])
            .drop_duplicates(subset="iso3", keep="first"))
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    if COORD_OVERRIDES:
        extra = pd.DataFrame(
            [{"iso3": k, "iso2": "", "country": v[0],
              "lat": float(v[1]), "lon": float(v[2])}
             for k, v in COORD_OVERRIDES.items() if k not in set(df["iso3"])])
        if len(extra):
            df = pd.concat([df, extra], ignore_index=True)
    if df.empty:
        sys.exit("[FATAL] coords CSV parsed to 0 usable rows.")
    print(f"[load] coords: {len(df)} countries with lat/lon")
    return df[["iso3", "iso2", "country", "lat", "lon"]]


def load_mapping(path: str) -> dict:
    mapping = {}
    with open(path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or "," not in line:
                continue
            code, name = line.split(",", 1)
            code = code.strip().strip('"').upper()
            name = name.strip().strip('"')
            if i == 0 and (code.lower() in ("code", "airport", "abbr")
                           or name.lower() in ("name", "country")):
                continue
            mapping[code] = name
    if not mapping:
        sys.exit("[FATAL] mapping.csv parsed to 0 entries.")
    print(f"[load] mapping: {len(mapping)} codes")
    return mapping


def load_fcg(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if FCG_CODE_COL not in df.columns:
        sys.exit(f"[FATAL] FCG CSV lacks '{FCG_CODE_COL}'; "
                 f"found {list(df.columns)}")
    df[FCG_CODE_COL] = (df[FCG_CODE_COL].astype(str)
                        .str.strip().str.upper()
                        .str.replace(r"_FCG.*$", "", regex=True))
    print(f"[load] FCG: {len(df)} country rows")
    return df


def load_ourairports(path: str, coords: pd.DataFrame) -> pd.DataFrame:
    """OurAirports airports.csv -> index of ICAO ident -> name/lat/lon/iso3."""
    usecols = ["ident", "type", "name", "latitude_deg", "longitude_deg",
               "iso_country"]
    df = pd.read_csv(path, usecols=usecols)
    df = df[df["type"].isin(AIRPORT_TYPES_OK)].copy()
    df["ident"] = df["ident"].astype(str).str.strip().str.upper()
    df["iso_country"] = df["iso_country"].astype(str).str.strip().str.upper()
    df["latitude_deg"] = pd.to_numeric(df["latitude_deg"], errors="coerce")
    df["longitude_deg"] = pd.to_numeric(df["longitude_deg"], errors="coerce")
    df = df.dropna(subset=["latitude_deg", "longitude_deg"])
    # alpha-2 -> alpha-3 bridge via the countries file
    a2_to_a3 = dict(zip(coords["iso2"], coords["iso3"]))
    df["iso3"] = df["iso_country"].map(a2_to_a3)
    df = df.dropna(subset=["iso3"]).drop_duplicates(subset="ident",
                                                    keep="first")
    df = df.set_index("ident")
    print(f"[load] OurAirports: {len(df)} usable airports "
          f"({', '.join(sorted(AIRPORT_TYPES_OK))})")
    return df


AIRPORT_CODE_RE = re.compile(r"\b[A-Z0-9]{3,4}\b")
AIRPORT_STOPWORDS = {
    "THE", "AND", "FOR", "NOT", "ALL", "ANY", "PER", "VIA", "ICAO", "IATA",
    "AOE", "N/A", "TBD", "UTC", "GMT", "VIP", "CIQ", "PPR", "NOTAM", "HRS",
}


def build_airport_index(fcg: pd.DataFrame, ourairports: pd.DataFrame):
    """Two products:
       airport_to_country : code -> FCG country code (for resolving input)
       candidates         : country code -> [airport codes WITH coordinates]
    """
    airport_to_country, collisions = {}, set()
    if AIRPORT_COL not in fcg.columns:
        print(f"[airports] column '{AIRPORT_COL}' missing — airport features off")
        return {}, {}
    country_codes = set(fcg[FCG_CODE_COL])
    for _, row in fcg.iterrows():
        text = str(row.get(AIRPORT_COL, "") or "")
        for tok in AIRPORT_CODE_RE.findall(text.upper()):
            if tok in AIRPORT_STOPWORDS or tok in country_codes:
                continue
            if tok.isdigit():
                continue
            if tok in airport_to_country and \
                    airport_to_country[tok] != row[FCG_CODE_COL]:
                collisions.add(tok)
                continue
            airport_to_country[tok] = row[FCG_CODE_COL]
    for tok in collisions:
        airport_to_country.pop(tok, None)

    candidates, no_coords = {}, []
    for ap, ctry in airport_to_country.items():
        if ap in ourairports.index:
            candidates.setdefault(ctry, []).append(ap)
        else:
            no_coords.append(ap)
    print(f"[airports] {len(airport_to_country)} codes extracted from FCG; "
          f"{sum(len(v) for v in candidates.values())} matched to OurAirports "
          f"coords across {len(candidates)} countries")
    if no_coords:
        print(f"[airports] no coords (unusable for distance optimization, "
              f"still listed): {sorted(no_coords)[:30]}"
              f"{' ...' if len(no_coords) > 30 else ''}")
    return airport_to_country, candidates


def build_dataset():
    coords = load_coords(COORDS_CSV)
    mapping = load_mapping(MAPPING_CSV)
    fcg = load_fcg(FCG_CSV)
    ourairports = load_ourairports(AIRPORTS_CSV, coords)
    fcg_codes = set(fcg[FCG_CODE_COL])
    joined = fcg_codes & set(coords["iso3"])
    unmatched = sorted(fcg_codes - set(coords["iso3"]))
    print(f"[join] {len(joined)}/{len(fcg_codes)} FCG codes matched to coords")
    if unmatched:
        print(f"[join] unmatched: {unmatched}")
    if len(joined) < 2:
        sys.exit("[FATAL] fewer than 2 FCG countries have coordinates.")
    nodes = coords[coords["iso3"].isin(joined)].set_index("iso3")
    fcg_idx = fcg.drop_duplicates(FCG_CODE_COL).set_index(FCG_CODE_COL)
    airport_to_country, candidates = build_airport_index(fcg, ourairports)
    airport_to_country = {a: c for a, c in airport_to_country.items()
                          if c in nodes.index}
    candidates = {c: aps for c, aps in candidates.items() if c in nodes.index}
    return nodes, fcg_idx, mapping, airport_to_country, candidates, ourairports

# ----------------------------------------------------------------------------
# 2. RESOLUTION (closed world; countries and airports)
# ----------------------------------------------------------------------------

def resolve_waypoint(user_token, mapping, nodes, airport_to_country, use_llm,
                     ourairports=None):
    tok = user_token.strip().upper()
    if tok in nodes.index:
        return tok, None
    if tok in airport_to_country:
        return airport_to_country[tok], tok
    # Fallback: any airport in OurAirports is a valid user-typed waypoint,
    # even if not listed in the FCG entry/exit text (e.g. home base KCHS).
    if ourairports is not None and tok in ourairports.index:
        ctry = str(ourairports.loc[tok, "iso3"])
        if ctry in nodes.index:
            print(f"[resolve] '{tok}' found in OurAirports -> {ctry} "
                  f"({ourairports.loc[tok, 'name']}) "
                  f"[not in FCG entry/exit list — flagged for review]")
            return ctry, tok
    name_to_code = {v.upper(): k for k, v in mapping.items()}
    if tok in name_to_code and name_to_code[tok] in nodes.index:
        return name_to_code[tok], None
    hit = nodes[nodes["country"].str.upper() == tok]
    if len(hit) == 1:
        return hit.index[0], None
    hit = nodes[nodes["country"].str.upper().str.contains(re.escape(tok))]
    if len(hit) == 1:
        return hit.index[0], None
    if use_llm:
        valid_countries = sorted(nodes.index)
        sample_airports = sorted(airport_to_country)[:400]
        ans = llm_json(
            f"The user wrote '{user_token}'. Match to EITHER one ISO alpha-3 "
            f"COUNTRY code from:\n{valid_countries}\nOR one AIRPORT code "
            f"from:\n{sample_airports}\nCountry codes mean countries "
            f"(FRA=France, never Frankfurt).\n"
            f'Respond ONLY JSON: {{"kind": "country"|"airport"|null, '
            f'"code": "XXX"|null}}') or {}
        code = (ans.get("code") or "").upper()
        if ans.get("kind") == "country" and code in nodes.index:
            print(f"[resolve] LLM: '{user_token}' -> country {code} [validated]")
            return code, None
        if ans.get("kind") == "airport" and code in airport_to_country:
            print(f"[resolve] LLM: '{user_token}' -> airport {code} [validated]")
            return airport_to_country[code], code
    sys.exit(f"[FATAL] cannot resolve '{user_token}' to any country or "
             f"airport in your dataset.")

# ----------------------------------------------------------------------------
# 3. GEOMETRY
# ----------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def airport_pt(ap, ourairports):
    row = ourairports.loc[ap]
    return float(row["latitude_deg"]), float(row["longitude_deg"])


def build_graph(nodes, k=K_NEIGHBOURS, max_leg_km=MAX_LEG_KM):
    codes = list(nodes.index)
    pts = {c: (float(nodes.loc[c, "lat"]), float(nodes.loc[c, "lon"]))
           for c in codes}
    graph = {c: {} for c in codes}
    for c in codes:
        dists = []
        for o in codes:
            if o == c:
                continue
            d = haversine_km(*pts[c], *pts[o])
            if max_leg_km is None or d <= max_leg_km:
                dists.append((d, o))
        dists.sort()
        for d, o in dists[:k]:
            graph[c][o] = d
            graph[o][c] = d
    print(f"[graph] {len(codes)} country nodes, "
          f"{sum(len(v) for v in graph.values()) // 2} edges (k={k})")
    return graph


def dijkstra(graph, src, dst, blocked):
    dist, prev, pq, seen = {src: 0.0}, {}, [(0.0, src)], set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == dst:
            break
        for v, w in graph[u].items():
            if v in blocked and v != dst:
                continue
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    if dst not in dist:
        return None, None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return list(reversed(path)), dist[dst]

# ----------------------------------------------------------------------------
# 4. STAGE-GRAPH AIRPORT SELECTION (dynamic program)
# ----------------------------------------------------------------------------

def choose_airports(stop_countries, pinned, candidates, ourairports, nodes,
                    banned):
    """Pick one airport per stop minimizing total consecutive-leg distance.

    stop_countries : ordered country codes for the stops
    pinned         : dict stage_index -> airport code the user typed
    candidates     : country -> [airport codes with coords]
    banned         : set of airport codes vetoed by the feasibility agent
    Countries with no usable candidate fall back to a synthetic
    'CENTROID:<code>' point at the country centroid.

    Returns (chosen list, total_km, per-stage candidate cost table).
    """
    def stage_options(i, ctry):
        if i in pinned:
            ap = pinned[i]
            if ap in banned:
                sys.exit(f"[FATAL] user-pinned airport {ap} was judged "
                         f"infeasible; pick another.")
            if ap in ourairports.index:
                return [ap]
            print(f"[stage] pinned {ap} has no coords; using centroid of {ctry}")
            return [f"CENTROID:{ctry}"]
        opts = [a for a in candidates.get(ctry, []) if a not in banned]
        return opts or [f"CENTROID:{ctry}"]

    def pt(opt):
        if opt.startswith("CENTROID:"):
            c = opt.split(":", 1)[1]
            return float(nodes.loc[c, "lat"]), float(nodes.loc[c, "lon"])
        return airport_pt(opt, ourairports)

    stages = [stage_options(i, c) for i, c in enumerate(stop_countries)]

    # DP over stages
    INF = float("inf")
    cost = [{opt: (0.0 if i == 0 else INF) for opt in stages[i]}
            for i in range(len(stages))]
    back = [dict() for _ in stages]
    for i in range(1, len(stages)):
        for cur in stages[i]:
            pcur = pt(cur)
            for prv in stages[i - 1]:
                c_new = cost[i - 1][prv] + haversine_km(*pt(prv), *pcur)
                if c_new < cost[i][cur]:
                    cost[i][cur] = c_new
                    back[i][cur] = prv
    last = min(cost[-1], key=cost[-1].get)
    total = cost[-1][last]
    chosen = [last]
    for i in range(len(stages) - 1, 0, -1):
        chosen.append(back[i][chosen[-1]])
    chosen.reverse()

    # Alternates report: best achievable total if stage i were forced to alt
    alternates = []
    for i, ctry in enumerate(stop_countries):
        alts = []
        for opt in stages[i]:
            if opt == chosen[i]:
                continue
            # cheap estimate: swap the single stop, keep neighbours fixed
            delta = 0.0
            if i > 0:
                delta += (haversine_km(*pt(chosen[i - 1]), *pt(opt))
                          - haversine_km(*pt(chosen[i - 1]), *pt(chosen[i])))
            if i < len(stages) - 1:
                delta += (haversine_km(*pt(opt), *pt(chosen[i + 1]))
                          - haversine_km(*pt(chosen[i]), *pt(chosen[i + 1])))
            alts.append((delta, opt))
        alts.sort()
        alternates.append(alts[:MAX_ALTERNATES_SHOWN])
    return chosen, total, alternates

# ----------------------------------------------------------------------------
# LLM plumbing
# ----------------------------------------------------------------------------

def llm_chat(prompt, system="You are a precise flight-clearance analyst. "
             "Follow output format instructions exactly."):
    payload = {"model": VLLM_MODEL,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": prompt}],
               "temperature": 0.0, "max_tokens": 800}
    r = requests.post(VLLM_URL, json=payload, timeout=LLM_TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def llm_json(prompt):
    try:
        txt = llm_chat(prompt)
    except Exception as e:
        print(f"[llm] call failed: {e}")
        return None
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        print(f"[llm] no JSON in reply: {txt[:200]!r}")
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        print(f"[llm] bad JSON: {m.group(0)[:200]!r}")
        return None

# ----------------------------------------------------------------------------
# 5. FEASIBILITY AGENT
# ----------------------------------------------------------------------------

def country_dossier(code, fcg):
    if code not in fcg.index:
        return "(no FCG data for this country)"
    row = fcg.loc[code]
    parts = [f"{c}: {str(row[c]).strip()[:600]}"
             for c in FEASIBILITY_COLS
             if c in row.index and pd.notna(row[c]) and str(row[c]).strip()]
    return "\n".join(parts) if parts else "(FCG row present but summaries empty)"


def judge_country(code, name, role, travel_date, fcg, airport=None,
                  airport_in_fcg=True):
    if airport and not str(airport).startswith("CENTROID:"):
        if airport_in_fcg:
            via = (f" The plan lands at airport '{airport}' here; confirm it "
                   f"is listed/permitted in the entry-exit data.")
        else:
            via = (f" The plan lands at airport '{airport}' here. This "
                   f"airport was chosen by the user and is NOT in the "
                   f"entry-exit list; do not deny for that reason alone — "
                   f"judge only the country-level rules and flag caution if "
                   f"the data requires using designated airports of entry.")
    else:
        via = ""
    ans = llm_json(
        f"Planned mission date: {travel_date}.\n"
        f"Country: {name} ({code}). Role: {role} "
        f"({'overflight only' if role == 'overflight' else 'landing/departure'})."
        f"{via}\nOfficial clearance data:\n---\n{country_dossier(code, fcg)}"
        f"\n---\nBased ONLY on the data above, can we use this country in "
        f"this role? Consider overflight permission, lead times vs mission "
        f"date, entry/exit and customs rules if landing. If data is missing "
        f"or ambiguous, allow but flag caution.\n"
        f'Respond ONLY JSON: {{"allowed": true/false, "caution": true/false, '
        f'"airport_ok": true/false, "reason": "<one sentence>"}}')
    if not ans or "allowed" not in ans:
        return {"allowed": True, "caution": True, "airport_ok": True,
                "reason": "LLM verdict unavailable; unverified."}
    ans.setdefault("airport_ok", True)
    return ans

# ----------------------------------------------------------------------------
# MULTI-LEG PLANNING (country corridors + airport stage optimization)
# ----------------------------------------------------------------------------

def plan(query, travel_date, use_llm):
    (nodes, fcg, mapping, airport_to_country,
     candidates, ourairports) = build_dataset()

    tokens = [t for t in re.split(r"\s+to\s+", query.strip(),
                                  flags=re.IGNORECASE) if t.strip()]
    if len(tokens) < 2:
        sys.exit('[FATAL] query must be "A to B" or "A to B to C [to A]"')

    stop_countries, pinned = [], {}
    for i, t in enumerate(tokens):
        ctry, ap = resolve_waypoint(t, mapping, nodes,
                                    airport_to_country, use_llm, ourairports)
        stop_countries.append(ctry)
        if ap:
            pinned[i] = ap
    round_trip = (stop_countries[0] == stop_countries[-1]
                  and len(stop_countries) > 2)
    print("[route] stops: " + " -> ".join(
        f"{c}({pinned[i]})" if i in pinned else c
        for i, c in enumerate(stop_countries))
        + ("  [round trip]" if round_trip else ""))

    graph = build_graph(nodes)
    verdicts, blocked, banned_airports = OrderedDict(), set(), set()

    # Outer loop: choose airports -> validate stops+corridors -> ban/block ->
    # re-choose. Each iteration permanently removes something, so it converges.
    for outer in range(1, 11):
        chosen, total_km, alternates = choose_airports(
            stop_countries, pinned, candidates, ourairports, nodes,
            banned_airports)
        print(f"[opt {outer}] chosen airports: "
              + " -> ".join(chosen) + f"  ({total_km:,.0f} km direct)")

        # country corridors per leg (for overflight clearances)
        corridors, corridor_fail = [], False
        for a, b in zip(stop_countries, stop_countries[1:]):
            if a == b:
                corridors.append([a])
                continue
            path, _ = dijkstra(graph, a, b, blocked)
            if path is None:
                sys.exit(f"[FATAL] no overflight corridor {a}->{b} with "
                         f"{sorted(blocked)} blocked.")
            corridors.append(path)

        if not use_llm:
            return (chosen, total_km, alternates, stop_countries,
                    corridors, verdicts, nodes, ourairports)

        changed = False
        # judge stop countries (+ their chosen airports)
        for i, (ctry, ap) in enumerate(zip(stop_countries, chosen)):
            key = (ctry, "stop", ap)
            if key not in verdicts:
                v = judge_country(ctry, nodes.loc[ctry, "country"], "stop",
                                  travel_date, fcg, ap,
                                  airport_in_fcg=(ap in airport_to_country))
                verdicts[key] = v
                ok = v["allowed"] and v.get("airport_ok", True)
                print(f"    [{'OK ' if ok else 'DENY'}] stop {ctry}/{ap}: "
                      f"{v['reason']}")
                if not v["allowed"]:
                    sys.exit(f"[FATAL] stop country {ctry} infeasible: "
                             f"{v['reason']}")
                if not v.get("airport_ok", True) \
                        and not ap.startswith("CENTROID:"):
                    banned_airports.add(ap)
                    changed = True
        # judge overflight countries in corridors
        for path in corridors:
            for code in path:
                if code in stop_countries:
                    continue
                key = (code, "overflight", None)
                if key in verdicts:
                    v = verdicts[key]
                else:
                    v = judge_country(code, nodes.loc[code, "country"],
                                      "overflight", travel_date, fcg)
                    verdicts[key] = v
                    print(f"    [{'OK ' if v['allowed'] else 'DENY'}] "
                          f"overflight {code}: {v['reason']}")
                if not v["allowed"] and code not in blocked:
                    blocked.add(code)
                    changed = True
        if not changed:
            return (chosen, total_km, alternates, stop_countries,
                    corridors, verdicts, nodes, ourairports)
        print(f"    replanning (banned airports: {sorted(banned_airports)}; "
              f"blocked countries: {sorted(blocked)})")
    sys.exit("[FATAL] plan did not stabilise within 10 iterations.")


def narrate(chosen, total_km, alternates, stop_countries, corridors,
            verdicts, nodes, ourairports, travel_date, use_llm):
    def ap_name(ap):
        if ap.startswith("CENTROID:"):
            c = ap.split(":", 1)[1]
            return f"{nodes.loc[c, 'country']} centroid (no designated " \
                   f"airport with coords)"
        return f"{ap} ({ourairports.loc[ap, 'name']})"

    print("\n=== FINAL PLAN ===")
    for i, (ctry, ap) in enumerate(zip(stop_countries, chosen)):
        print(f"Stop {i + 1}: {nodes.loc[ctry, 'country']} ({ctry}) — "
              f"{ap_name(ap)}")
        alts = alternates[i]
        if alts:
            for delta, alt in alts:
                print(f"        alt: {ap_name(alt)}  ({delta:+,.0f} km)")
    print("\nLegs:")
    for i, ((a, b), corridor) in enumerate(
            zip(zip(chosen, chosen[1:]), corridors)):
        def pt(opt):
            if opt.startswith("CENTROID:"):
                c = opt.split(":", 1)[1]
                return float(nodes.loc[c, "lat"]), float(nodes.loc[c, "lon"])
            return airport_pt(opt, ourairports)
        d = haversine_km(*pt(a), *pt(b))
        print(f"  {a} -> {b}: {d:,.0f} km | overflight corridor: "
              f"{' -> '.join(corridor)}")
    print(f"Total direct distance: {total_km:,.0f} km")
    cautions = {k: v for k, v in verdicts.items() if v.get("caution")}
    if cautions:
        print("Cautions:")
        for (code, role, ap), v in cautions.items():
            tag = f"{code}/{ap}" if ap else code
            print(f"  - {tag} ({role}): {v['reason']}")
    if use_llm:
        summary = llm_chat(
            f"Mission date {travel_date}.\nStops and chosen airports:\n"
            + "\n".join(f"{c}: {a}" for c, a in zip(stop_countries, chosen))
            + "\nOverflight corridors per leg:\n"
            + "\n".join(" -> ".join(p) for p in corridors)
            + "\nVerdicts:\n"
            + json.dumps({f"{c}/{r}/{a}": v
                          for (c, r, a), v in verdicts.items()}, indent=1)
            + "\nWrite a short operational briefing (<=200 words): per-leg "
              "rationale, why each airport was chosen, required lead-time "
              "actions, cautions.")
        print("\n=== BRIEFING ===\n" + summary)

# ----------------------------------------------------------------------------
# --list MODE
# ----------------------------------------------------------------------------

def list_airports(country_query, use_llm):
    (nodes, fcg, mapping, airport_to_country,
     candidates, ourairports) = build_dataset()
    code, _ = resolve_waypoint(country_query, mapping, nodes,
                               airport_to_country, use_llm, ourairports)
    name = nodes.loc[code, "country"]
    print(f"\n=== ENTRY/EXIT AIRPORTS: {name} ({code}) ===")
    extracted = sorted(a for a, c in airport_to_country.items() if c == code)
    for ap in extracted:
        if ap in ourairports.index:
            r = ourairports.loc[ap]
            print(f"  {ap}: {r['name']}  "
                  f"({r['latitude_deg']:.3f}, {r['longitude_deg']:.3f})")
        else:
            print(f"  {ap}: (no coordinates in OurAirports)")
    if not extracted:
        print("  (none extracted from FCG text)")
    raw = ""
    if code in fcg.index and AIRPORT_COL in fcg.columns:
        raw = str(fcg.loc[code].get(AIRPORT_COL, "") or "").strip()
    print(f"\nRaw FCG entry/exit text:\n---\n{raw or '(empty)'}\n---")
    restr = ""
    if code in fcg.index and "airfield_restrictions_summary" in fcg.columns:
        restr = str(fcg.loc[code].get("airfield_restrictions_summary", "")
                    or "").strip()
    if restr:
        print(f"\nAirfield restrictions:\n---\n{restr}\n---")
    if use_llm and (raw or restr):
        ans = llm_json(
            f"Country: {name} ({code}). Official entry/exit airport data:\n"
            f"---\n{raw}\n---\nAirfield restrictions:\n---\n"
            f"{restr or '(none)'}\n---\nBased ONLY on this text, list the "
            f"designated airports and their status. Use ONLY airports "
            f"mentioned in the text — never invent.\n"
            f'Respond ONLY JSON: {{"usable": [{{"code": "...", '
            f'"conditions": "<short or empty>"}}], '
            f'"restricted_or_prohibited": [{{"code": "...", '
            f'"reason": "<short>"}}], "notes": "<one sentence>"}}')
        if ans:
            print("\n=== STRUCTURED VERDICT (LLM, grounded in text above) ===")
            for a in ans.get("usable", []):
                cond = f" — {a['conditions']}" if a.get("conditions") else ""
                print(f"  USABLE     {a.get('code', '?')}{cond}")
            for a in ans.get("restricted_or_prohibited", []):
                print(f"  RESTRICTED {a.get('code', '?')} — "
                      f"{a.get('reason', '')}")
            if ans.get("notes"):
                print(f"  Notes: {ans['notes']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query",
                    help='route "A to B to C" — or one country with --list')
    ap.add_argument("--date", default="2026-09-15")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--list", action="store_true",
                    help="list designated entry/exit airports for a country")
    args = ap.parse_args()
    use_llm = not args.no_llm
    if args.list:
        list_airports(args.query, use_llm)
        sys.exit(0)
    result = plan(args.query, args.date, use_llm)
    narrate(*result, args.date, use_llm)
