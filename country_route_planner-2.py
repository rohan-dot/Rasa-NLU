#!/usr/bin/env python3
"""
Country-level agentic flight route planner — v2.

New in v2:
  * Multi-stop & round-trip queries: "fra to afg", "fra to tur to afg",
    "fra to tur to afg to fra" (round trip = end where you started).
  * Airport-level input: users can type an airport code that appears in a
    country's `entry_exit_airports_summary`. The airport resolves to its
    country (routing stays country-level); the chosen entry/exit airport is
    carried through to the final briefing.
  * All fixes validated on the real dataset:
      - mapping.csv parsed manually (first-comma split) so names containing
        commas ("Korea, Republic of") don't break loading
      - FCG code suffixes like "NOR_FCG2.0" stripped automatically
      - coordinates CSV deduplicated per ISO3 + explicit float casts
        (prevents math.radians() receiving a pandas Series)
      - fail-fast validation with clear messages at every load/join step

Architecture (unchanged): deterministic geometry (haversine + kNN graph +
Dijkstra) plans candidate paths; the LLM only (a) fuzzy-resolves user input
against the closed world of your dataset, (b) judges per-country feasibility
from YOUR FCG text in strict JSON, (c) narrates. Denied countries are blocked
and the route replans until stable.

Usage:
  python country_route_planner.py "fra to afg"
  python country_route_planner.py "hog to dur to fleur to hog" --date 2026-09-15
  python country_route_planner.py "CDG to KBL" --no-llm
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

VLLM_URL     = "http://localhost:8000/v1/chat/completions"
VLLM_MODEL   = "gemma-4-31B-it"
LLM_TIMEOUT  = 120
K_NEIGHBOURS = 8
MAX_LEG_KM   = 4000.0     # None to disable per-edge cap

FCG_CODE_COL = "airport"  # column holding the country code in your FCG CSV
AIRPORT_COL  = "entry_exit_airports_summary"  # column listing usable airports
FEASIBILITY_COLS = [
    "overflight_summary",
    "entry_exit_airports_summary",
    "customs_immigration_summary",
    "diplomatic_lead_time_summary",
    "airfield_restrictions_summary",
    "operating_hours_summary",
    "hazmat_summary",
]

# Manual coordinates for codes missing from the public coords file
# (non-ISO / territory codes seen in your data). Extend as needed.
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
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "alpha-3" in lc:
            col_map[c] = "iso3"
        elif "latitude" in lc:
            col_map[c] = "lat"
        elif "longitude" in lc:
            col_map[c] = "lon"
        elif lc == "country":
            col_map[c] = "country"
    df = df.rename(columns=col_map)
    missing = {"iso3", "lat", "lon", "country"} - set(df.columns)
    if missing:
        sys.exit(f"[FATAL] coords CSV missing columns {missing}; "
                 f"found {list(df.columns)}")
    for c in ("iso3", "country"):
        df[c] = df[c].map(_clean)
    for c in ("lat", "lon"):
        df[c] = pd.to_numeric(df[c].map(_clean), errors="coerce")
    df["iso3"] = df["iso3"].str.upper()
    df = df.dropna(subset=["lat", "lon"])
    df = df.drop_duplicates(subset="iso3", keep="first")   # one row per code
    df["lat"] = df["lat"].astype(float)                    # scalar floats,
    df["lon"] = df["lon"].astype(float)                    # never a Series
    if COORD_OVERRIDES:
        extra = pd.DataFrame(
            [{"iso3": k, "country": v[0], "lat": float(v[1]), "lon": float(v[2])}
             for k, v in COORD_OVERRIDES.items()
             if k not in set(df["iso3"])])
        if len(extra):
            df = pd.concat([df, extra], ignore_index=True)
    if df.empty:
        sys.exit("[FATAL] coords CSV parsed to 0 usable rows.")
    print(f"[load] coords: {len(df)} countries with lat/lon")
    return df[["iso3", "country", "lat", "lon"]]


def load_mapping(path: str) -> dict:
    """code,name — manual first-comma split; comma-in-name safe."""
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
    print(f"[load] mapping: {len(mapping)} codes "
          f"(e.g. {list(mapping.items())[:3]})")
    return mapping


def load_fcg(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if FCG_CODE_COL not in df.columns:
        sys.exit(f"[FATAL] FCG CSV lacks '{FCG_CODE_COL}'; "
                 f"found {list(df.columns)}")
    df[FCG_CODE_COL] = (df[FCG_CODE_COL].astype(str)
                        .str.strip().str.upper()
                        .str.replace(r"_FCG.*$", "", regex=True))  # NOR_FCG2.0 -> NOR
    print(f"[load] FCG: {len(df)} country rows")
    return df


AIRPORT_CODE_RE = re.compile(r"\b[A-Z]{3,4}\b")
# Words that look like codes but aren't airports — extend if noise appears.
AIRPORT_STOPWORDS = {
    "THE", "AND", "FOR", "NOT", "ALL", "ANY", "PER", "VIA", "ICAO", "IATA",
    "AOE", "N/A", "TBD", "UTC", "GMT", "VIP", "CIQ", "PPR", "NOTAM", "HRS",
}


def build_airport_index(fcg: pd.DataFrame) -> dict:
    """airport code -> country code, extracted from AIRPORT_COL text.

    Heuristic: any standalone 3-4 letter uppercase token in the entry/exit
    airports text is treated as an airport code, minus stopwords and minus
    codes that collide with country codes in the dataset (country wins).
    """
    index = {}
    collisions = set()
    if AIRPORT_COL not in fcg.columns:
        print(f"[airports] column '{AIRPORT_COL}' not found — "
              f"airport-level input disabled")
        return index
    country_codes = set(fcg[FCG_CODE_COL])
    for _, row in fcg.iterrows():
        text = str(row.get(AIRPORT_COL, "") or "")
        for tok in AIRPORT_CODE_RE.findall(text.upper()):
            if tok in AIRPORT_STOPWORDS or tok in country_codes:
                continue
            if tok in index and index[tok] != row[FCG_CODE_COL]:
                collisions.add(tok)      # same code claimed by 2 countries
                continue
            index[tok] = row[FCG_CODE_COL]
    for tok in collisions:
        index.pop(tok, None)             # ambiguous -> refuse to guess
    print(f"[airports] indexed {len(index)} airport codes from FCG text"
          + (f"; dropped ambiguous: {sorted(collisions)}" if collisions else ""))
    return index


def build_dataset():
    coords = load_coords(COORDS_CSV)
    mapping = load_mapping(MAPPING_CSV)
    fcg = load_fcg(FCG_CSV)
    fcg_codes = set(fcg[FCG_CODE_COL])
    joined = fcg_codes & set(coords["iso3"])
    unmatched = sorted(fcg_codes - set(coords["iso3"]))
    print(f"[join] {len(joined)}/{len(fcg_codes)} FCG codes matched to coordinates")
    if unmatched:
        print(f"[join] unmatched (excluded, or add to COORD_OVERRIDES): {unmatched}")
    if len(joined) < 2:
        sys.exit("[FATAL] fewer than 2 FCG countries have coordinates.")
    nodes = coords[coords["iso3"].isin(joined)].set_index("iso3")
    fcg_idx = fcg.drop_duplicates(FCG_CODE_COL).set_index(FCG_CODE_COL)
    airport_index = build_airport_index(fcg)
    # airports pointing at countries that lack coordinates are unusable
    airport_index = {a: c for a, c in airport_index.items() if c in nodes.index}
    return nodes, fcg_idx, mapping, airport_index

# ----------------------------------------------------------------------------
# 2. CLOSED-WORLD RESOLUTION (countries AND airports)
# ----------------------------------------------------------------------------

def resolve_waypoint(user_token: str, mapping: dict, nodes: pd.DataFrame,
                     airport_index: dict, use_llm: bool):
    """Returns (country_code, airport_code_or_None)."""
    tok = user_token.strip().upper()

    # a) country code in dataset (your mapping codes ARE the country codes)
    if tok in nodes.index:
        return tok, None
    # b) airport code from the FCG entry/exit text
    if tok in airport_index:
        return airport_index[tok], tok
    # c) country name, exact then substring
    name_to_code = {v.upper(): k for k, v in mapping.items()}
    if tok in name_to_code and name_to_code[tok] in nodes.index:
        return name_to_code[tok], None
    hit = nodes[nodes["country"].str.upper() == tok]
    if len(hit) == 1:
        return hit.index[0], None
    hit = nodes[nodes["country"].str.upper().str.contains(re.escape(tok))]
    if len(hit) == 1:
        return hit.index[0], None

    # d) LLM fuzzy resolve, validated against the closed world
    if use_llm:
        valid_countries = sorted(nodes.index)
        sample_airports = sorted(airport_index)[:400]
        prompt = (
            f"The user wrote the location '{user_token}'.\n"
            f"Match it to EITHER one ISO alpha-3 COUNTRY code from:\n"
            f"{valid_countries}\n"
            f"OR one AIRPORT code from:\n{sample_airports}\n"
            f"These country codes mean countries (FRA=France, never Frankfurt).\n"
            f'Respond ONLY JSON: {{"kind": "country"|"airport"|null, '
            f'"code": "XXX"|null}}'
        )
        ans = llm_json(prompt) or {}
        code = (ans.get("code") or "").upper()
        if ans.get("kind") == "country" and code in nodes.index:
            print(f"[resolve] LLM: '{user_token}' -> country {code} [validated]")
            return code, None
        if ans.get("kind") == "airport" and code in airport_index:
            print(f"[resolve] LLM: '{user_token}' -> airport {code} "
                  f"({airport_index[code]}) [validated]")
            return airport_index[code], code

    sys.exit(f"[FATAL] cannot resolve '{user_token}' to any country or "
             f"airport in your dataset.")

# ----------------------------------------------------------------------------
# 3. GEOMETRY + GRAPH
# ----------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def build_graph(nodes: pd.DataFrame, k=K_NEIGHBOURS, max_leg_km=MAX_LEG_KM):
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
    n_edges = sum(len(v) for v in graph.values()) // 2
    print(f"[graph] {len(codes)} nodes, {n_edges} edges (k={k})")
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
# LLM plumbing
# ----------------------------------------------------------------------------

def llm_chat(prompt, system="You are a precise flight-clearance analyst. "
             "Follow output format instructions exactly."):
    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 800,
    }
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
# 4. FEASIBILITY AGENT
# ----------------------------------------------------------------------------

def country_dossier(code, fcg):
    if code not in fcg.index:
        return "(no FCG data for this country)"
    row = fcg.loc[code]
    parts = [f"{c}: {str(row[c]).strip()[:600]}"
             for c in FEASIBILITY_COLS
             if c in row.index and pd.notna(row[c]) and str(row[c]).strip()]
    return "\n".join(parts) if parts else "(FCG row present but summaries empty)"


def judge_country(code, name, role, travel_date, fcg, airport=None):
    dossier = country_dossier(code, fcg)
    via = (f" The user intends to use airport '{airport}' for entry/exit here; "
           f"confirm it is listed/permitted in the entry-exit data."
           if airport else "")
    prompt = (
        f"Planned mission date: {travel_date}.\n"
        f"Country: {name} ({code}). Role: {role} "
        f"({'overflight only' if role == 'overflight' else 'landing/departure'})."
        f"{via}\n"
        f"Official clearance data:\n---\n{dossier}\n---\n"
        f"Based ONLY on the data above, can we use this country in this role? "
        f"Consider overflight permission, lead times vs mission date, "
        f"entry/exit and customs rules if landing. If data is missing or "
        f"ambiguous, allow but flag caution.\n"
        f'Respond ONLY JSON: {{"allowed": true/false, "caution": true/false, '
        f'"reason": "<one sentence>"}}'
    )
    ans = llm_json(prompt)
    if not ans or "allowed" not in ans:
        return {"allowed": True, "caution": True,
                "reason": "LLM verdict unavailable; unverified."}
    return ans

# ----------------------------------------------------------------------------
# MULTI-LEG PLANNING
# ----------------------------------------------------------------------------

def plan_leg(graph, nodes, fcg, src, dst, travel_date, use_llm,
             verdicts, blocked, airports):
    """Plan one leg with plan-check-replan. Shares verdict cache + blocklist
    across legs so a country judged/denied once stays judged/denied."""
    for iteration in range(1, 11):
        path, km = dijkstra(graph, src, dst, blocked)
        if path is None:
            sys.exit(f"[FATAL] no feasible route {src}->{dst} with "
                     f"{sorted(blocked)} blocked. Raise K_NEIGHBOURS/MAX_LEG_KM "
                     f"or review denials.")
        print(f"  [leg {src}->{dst} | iter {iteration}] "
              f"{' -> '.join(path)} ({km:,.0f} km)")
        if not use_llm:
            return path, km
        new_blocks = []
        for code in path:
            role = "stop" if code in (src, dst) else "overflight"
            cache_key = (code, role)
            if cache_key in verdicts:
                v = verdicts[cache_key]
            else:
                v = judge_country(code, nodes.loc[code, "country"], role,
                                  travel_date, fcg, airports.get(code))
                verdicts[cache_key] = v
                print(f"    [{'OK ' if v['allowed'] else 'DENY'}] "
                      f"{code}/{role}: {v['reason']}")
            if not v["allowed"]:
                if code in (src, dst):
                    sys.exit(f"[FATAL] {role} country {code} infeasible: "
                             f"{v['reason']}")
                new_blocks.append(code)
        if not new_blocks:
            return path, km
        blocked.update(new_blocks)
        print(f"    re-routing around: {new_blocks}")
    sys.exit("[FATAL] leg did not stabilise within 10 iterations.")


def plan(query, travel_date, use_llm):
    nodes, fcg, mapping, airport_index = build_dataset()

    tokens = [t for t in re.split(r"\s+to\s+", query.strip(),
                                  flags=re.IGNORECASE) if t.strip()]
    if len(tokens) < 2:
        sys.exit('[FATAL] query must be "A to B" or "A to B to C [to A]"')

    waypoints, airports = [], {}   # airports: country -> chosen airport
    for t in tokens:
        country, airport = resolve_waypoint(t, mapping, nodes,
                                            airport_index, use_llm)
        waypoints.append(country)
        if airport:
            airports[country] = airport
    pretty = " -> ".join(
        f"{c} ({nodes.loc[c,'country']}"
        + (f", via {airports[c]}" if c in airports else "") + ")"
        for c in waypoints)
    round_trip = waypoints[0] == waypoints[-1] and len(waypoints) > 2
    print(f"[route] {pretty}{'  [round trip]' if round_trip else ''}")

    # collapse accidental repeats like "fra to fra"
    dedup = [waypoints[0]]
    for w in waypoints[1:]:
        if w != dedup[-1]:
            dedup.append(w)
    if len(dedup) < 2:
        sys.exit("[FATAL] all waypoints are the same country.")

    graph = build_graph(nodes)
    verdicts, blocked = OrderedDict(), set()
    full_path, total_km, leg_summaries = [], 0.0, []
    for src, dst in zip(dedup, dedup[1:]):
        path, km = plan_leg(graph, nodes, fcg, src, dst, travel_date,
                            use_llm, verdicts, blocked, airports)
        leg_summaries.append((src, dst, path, km))
        total_km += km
        full_path.extend(path if not full_path else path[1:])
    return full_path, total_km, leg_summaries, verdicts, nodes, airports


def narrate(full_path, total_km, legs, verdicts, nodes, airports,
            travel_date, use_llm):
    print("\n=== FINAL ROUTE ===")
    for src, dst, path, km in legs:
        print(f"Leg {src} -> {dst} ({km:,.0f} km): {' -> '.join(path)}")
    print(f"Overall: {' -> '.join(full_path)}")
    print(f"Total great-circle distance: {total_km:,.0f} km")
    if airports:
        print("Entry/exit airports: "
              + ", ".join(f"{c}: {a}" for c, a in airports.items()))
    cautions = {k: v for k, v in verdicts.items() if v.get("caution")}
    if cautions:
        print("Cautions:")
        for (code, role), v in cautions.items():
            print(f"  - {code} ({role}): {v['reason']}")
    if use_llm:
        summary = llm_chat(
            f"Mission date {travel_date}. Multi-leg route:\n"
            + "\n".join(f"{s}->{d}: {' -> '.join(p)} ({k:,.0f} km)"
                        for s, d, p, k in legs)
            + (f"\nDesignated entry/exit airports: {airports}" if airports else "")
            + "\nPer-country verdicts:\n"
            + json.dumps({f"{c}/{r}": v for (c, r), v in verdicts.items()},
                         indent=1)
            + "\nWrite a short operational briefing (<=180 words): route "
              "rationale per leg, required lead-time actions, cautions.")
        print("\n=== BRIEFING ===\n" + summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query",
                    help='"fra to afg" | "hog to dur to fleur to hog" | "CDG to KBL"')
    ap.add_argument("--date", default="2026-09-15")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    use_llm = not args.no_llm
    full_path, km, legs, verdicts, nodes, airports = plan(
        args.query, args.date, use_llm)
    narrate(full_path, km, legs, verdicts, nodes, airports,
            args.date, use_llm)
