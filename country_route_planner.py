#!/usr/bin/env python3
"""
Country-level agentic flight route planner.

Architecture (deterministic math + LLM judgment, cleanly separated):

  1. LOAD      : your FCG constraints CSV (country-level rows, ISO alpha-3 in
                 'airport' column), your mapping.csv (code -> country name),
                 and the public countries_codes_and_coordinates.csv (tadast gist).
                 Fail-fast validation: crash loudly if joins produce nothing.
  2. RESOLVE   : user input ("fra to afg") -> ISO alpha-3, CLOSED-WORLD.
                 Only codes present in YOUR mapping/FCG data are legal.
                 The LLM is only consulted for fuzzy input ("France to
                 Afganistan" with a typo), and its answer is validated against
                 the dataset. It can never invent EDDF/Frankfurt again.
  3. GRAPH     : nodes = countries with coordinates; edges = k-nearest
                 neighbours by haversine (approximates "you can fly from a
                 country to a nearby country"). Weights = great-circle km.
  4. FEASIBILITY (LLM): for each candidate country on/near the route, the LLM
                 reads your overflight/customs/lead-time summary text and
                 returns strict JSON {allowed, reason}. Denied countries are
                 removed from the graph.
  5. ROUTE     : Dijkstra over the filtered graph = shortest feasible path.
  6. NARRATE   (LLM): turns the final path + feasibility notes into a
                 human-readable plan.

Usage:
  CUDA_VISIBLE_DEVICES=1 python country_route_planner.py "fra to afg"
  python country_route_planner.py "fra to afg" --date 2026-09-15
  python country_route_planner.py "fra to afg" --no-llm     # pure math, skip agent

Requires: pandas, requests (and your vLLM server running with an
OpenAI-compatible endpoint, e.g. `vllm serve <model>`).
"""

import argparse
import heapq
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
import requests

# ----------------------------------------------------------------------------
# CONFIG — edit these paths / endpoint for your environment
# ----------------------------------------------------------------------------
FCG_CSV      = "fcg_extract.csv"                      # your constraints data
MAPPING_CSV  = "mapping.csv"                          # your code->name file
COORDS_CSV   = "countries_codes_and_coordinates.csv"  # tadast gist download

VLLM_URL     = "http://localhost:8000/v1/chat/completions"
VLLM_MODEL   = "gemma-4-31B-it"   # must match the model name vLLM is serving
LLM_TIMEOUT  = 120                # seconds per LLM call
K_NEIGHBOURS = 8                  # graph connectivity (higher = denser graph)
MAX_LEG_KM   = 4000.0             # optional hard cap on a single leg; None to disable

# Column in your FCG CSV that holds the ISO alpha-3 code
FCG_CODE_COL = "airport"
# FCG summary columns the feasibility agent reads (edit to taste)
FEASIBILITY_COLS = [
    "overflight_summary",
    "entry_exit_airports_summary",
    "customs_immigration_summary",
    "diplomatic_lead_time_summary",
    "airfield_restrictions_summary",
    "operating_hours_summary",
    "hazmat_summary",
]

# ----------------------------------------------------------------------------
# 1. LOADING (with fail-fast validation)
# ----------------------------------------------------------------------------

def _clean(s):
    """The tadast CSV wraps values like ' \"AFG\"' — strip quotes/spaces."""
    if isinstance(s, str):
        return s.strip().strip('"').strip()
    return s


def load_coords(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    # Normalise expected columns from the gist
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
        sys.exit(f"[FATAL] coords CSV missing columns {missing}. "
                 f"Found: {list(df.columns)}")
    for c in ("iso3", "country"):
        df[c] = df[c].map(_clean)
    for c in ("lat", "lon"):
        df[c] = pd.to_numeric(df[c].map(_clean), errors="coerce")
    df["iso3"] = df["iso3"].str.upper()
    df = df.dropna(subset=["lat", "lon"])
    if df.empty:
        sys.exit("[FATAL] coords CSV parsed to 0 usable rows — check quoting.")
    print(f"[load] coords: {len(df)} countries with lat/lon")
    return df[["iso3", "country", "lat", "lon"]]


def load_mapping(path: str) -> dict:
    """mapping.csv: your short code -> country name (e.g. fra,France)."""
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        sys.exit(f"[FATAL] mapping.csv needs >=2 columns, got {list(df.columns)}")
    code_col, name_col = df.columns[0], df.columns[1]
    mapping = {
        str(r[code_col]).strip().upper(): str(r[name_col]).strip()
        for _, r in df.iterrows()
    }
    print(f"[load] mapping: {len(mapping)} codes (e.g. "
          f"{list(mapping.items())[:3]})")
    return mapping


def load_fcg(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if FCG_CODE_COL not in df.columns:
        sys.exit(f"[FATAL] FCG CSV has no '{FCG_CODE_COL}' column. "
                 f"Found: {list(df.columns)}")
    df[FCG_CODE_COL] = df[FCG_CODE_COL].astype(str).str.strip().str.upper()
    print(f"[load] FCG: {len(df)} country rows")
    return df


def build_dataset():
    coords = load_coords(COORDS_CSV)
    mapping = load_mapping(MAPPING_CSV)
    fcg = load_fcg(FCG_CSV)

    # Join FCG codes -> coordinates. FAIL FAST if the join is bad.
    fcg_codes = set(fcg[FCG_CODE_COL])
    coord_codes = set(coords["iso3"])
    joined = fcg_codes & coord_codes
    unmatched = sorted(fcg_codes - coord_codes)
    print(f"[join] {len(joined)}/{len(fcg_codes)} FCG codes matched to coordinates")
    if unmatched:
        print(f"[join] unmatched (no coords, excluded from graph): {unmatched}")
    if len(joined) < 2:
        sys.exit("[FATAL] fewer than 2 FCG countries have coordinates — "
                 "the join failed. Verify your codes are ISO alpha-3.")

    nodes = coords[coords["iso3"].isin(joined)].set_index("iso3")
    fcg_idx = fcg.drop_duplicates(FCG_CODE_COL).set_index(FCG_CODE_COL)
    return nodes, fcg_idx, mapping

# ----------------------------------------------------------------------------
# 2. CLOSED-WORLD RESOLUTION
# ----------------------------------------------------------------------------

def resolve_code(user_token: str, mapping: dict, nodes: pd.DataFrame,
                 use_llm: bool) -> str:
    """Resolve user input to an ISO3 code that EXISTS in the dataset."""
    tok = user_token.strip().upper()

    # a) exact code in your mapping / dataset
    if tok in mapping and tok in nodes.index:
        return tok
    if tok in nodes.index:
        return tok

    # b) exact country-name match (case-insensitive) via mapping or coords
    name_to_code = {v.upper(): k for k, v in mapping.items()}
    if tok in name_to_code and name_to_code[tok] in nodes.index:
        return name_to_code[tok]
    hit = nodes[nodes["country"].str.upper() == tok]
    if len(hit) == 1:
        return hit.index[0]

    # c) substring match
    hit = nodes[nodes["country"].str.upper().str.contains(re.escape(tok))]
    if len(hit) == 1:
        return hit.index[0]

    # d) LLM fuzzy resolve — but validated against dataset (closed world)
    if use_llm:
        valid = sorted(nodes.index)
        prompt = (
            f"The user wrote the location '{user_token}'. Map it to exactly one "
            f"ISO alpha-3 COUNTRY code from this list (these are countries, NOT "
            f"airports; e.g. FRA means France, never Frankfurt):\n{valid}\n"
            f'Respond with ONLY JSON: {{"code": "XXX"}} or {{"code": null}} '
            f"if no reasonable match."
        )
        ans = llm_json(prompt)
        code = (ans or {}).get("code")
        if code and code.upper() in nodes.index:
            print(f"[resolve] LLM: '{user_token}' -> {code.upper()} "
                  f"({nodes.loc[code.upper(), 'country']})  [validated]")
            return code.upper()

    sys.exit(f"[FATAL] cannot resolve '{user_token}' to any country in your "
             f"dataset. Valid codes include: {sorted(nodes.index)[:20]} ...")

# ----------------------------------------------------------------------------
# 3. GEOMETRY + GRAPH
# ----------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def build_graph(nodes: pd.DataFrame, k: int = K_NEIGHBOURS,
                max_leg_km=MAX_LEG_KM) -> dict:
    """k-nearest-neighbour graph over country centroids."""
    codes = list(nodes.index)
    pts = {c: (nodes.loc[c, "lat"], nodes.loc[c, "lon"]) for c in codes}
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
            graph[o][c] = d  # symmetric
    n_edges = sum(len(v) for v in graph.values()) // 2
    print(f"[graph] {len(codes)} nodes, {n_edges} edges (k={k})")
    return graph


def dijkstra(graph: dict, src: str, dst: str, blocked: set):
    """Shortest path avoiding blocked countries (src/dst never blocked)."""
    dist = {src: 0.0}
    prev = {}
    pq = [(0.0, src)]
    seen = set()
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
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dst not in dist:
        return None, None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return list(reversed(path)), dist[dst]

# ----------------------------------------------------------------------------
# LLM plumbing (vLLM OpenAI-compatible endpoint)
# ----------------------------------------------------------------------------

def llm_chat(prompt: str, system: str = "You are a precise flight-clearance "
             "analyst. Follow output format instructions exactly.") -> str:
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 800,
    }
    r = requests.post(VLLM_URL, json=payload, timeout=LLM_TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def llm_json(prompt: str):
    """Call LLM and parse a JSON object out of the reply (fence-tolerant)."""
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
# 4. FEASIBILITY AGENT (reads YOUR FCG text, returns strict verdicts)
# ----------------------------------------------------------------------------

def country_dossier(code: str, fcg: pd.DataFrame) -> str:
    if code not in fcg.index:
        return "(no FCG data for this country)"
    row = fcg.loc[code]
    parts = []
    for col in FEASIBILITY_COLS:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            parts.append(f"{col}: {str(row[col]).strip()[:600]}")
    return "\n".join(parts) if parts else "(FCG row present but summaries empty)"


def judge_country(code: str, name: str, role: str, travel_date: str,
                  fcg: pd.DataFrame):
    """role: 'overflight' for intermediate hops, 'stop' for origin/destination."""
    dossier = country_dossier(code, fcg)
    prompt = (
        f"Planned mission date: {travel_date}.\n"
        f"Country: {name} ({code}). Role in route: {role} "
        f"({'we only fly over it' if role == 'overflight' else 'we land / depart here'}).\n"
        f"Official clearance data for this country:\n---\n{dossier}\n---\n"
        f"Based ONLY on the data above (do not use outside knowledge), can we "
        f"use this country in the given role? Consider overflight permission, "
        f"lead times vs the mission date, entry/exit and customs rules if "
        f"landing. If data is missing or ambiguous, allow but flag it.\n"
        f'Respond ONLY with JSON: {{"allowed": true/false, '
        f'"caution": true/false, "reason": "<one sentence>"}}'
    )
    ans = llm_json(prompt)
    if not ans or "allowed" not in ans:
        # Fail open with a caution flag — a broken LLM reply shouldn't
        # silently forbid the whole map.
        return {"allowed": True, "caution": True,
                "reason": "LLM verdict unavailable; unverified."}
    return ans

# ----------------------------------------------------------------------------
# MAIN PLANNING LOOP
# ----------------------------------------------------------------------------

def plan(query: str, travel_date: str, use_llm: bool):
    nodes, fcg, mapping = build_dataset()

    # Parse "X to Y"
    m = re.split(r"\s+to\s+", query.strip(), flags=re.IGNORECASE)
    if len(m) != 2:
        sys.exit('[FATAL] query must look like "fra to afg"')
    src = resolve_code(m[0], mapping, nodes, use_llm)
    dst = resolve_code(m[1], mapping, nodes, use_llm)
    print(f"[route] {src} ({nodes.loc[src,'country']}) -> "
          f"{dst} ({nodes.loc[dst,'country']})")

    graph = build_graph(nodes)

    # Iterative plan-check-replan: find shortest path, judge each country,
    # block denials, re-route. Converges quickly because each iteration
    # permanently removes at least one country.
    blocked, verdicts = set(), {}
    for iteration in range(1, 11):
        path, total_km = dijkstra(graph, src, dst, blocked)
        if path is None:
            sys.exit(f"[FATAL] no feasible route: {sorted(blocked)} blocked "
                     f"disconnects {src} from {dst}. Consider raising "
                     f"K_NEIGHBOURS or MAX_LEG_KM.")
        print(f"[iter {iteration}] candidate path "
              f"{' -> '.join(path)} ({total_km:,.0f} km)")

        if not use_llm:
            return path, total_km, verdicts, nodes

        new_blocks = []
        for code in path:
            if code in verdicts:
                continue
            role = "stop" if code in (src, dst) else "overflight"
            v = judge_country(code, nodes.loc[code, "country"], role,
                              travel_date, fcg)
            verdicts[code] = v
            flag = "OK " if v["allowed"] else "DENY"
            print(f"    [{flag}] {code}: {v['reason']}")
            if not v["allowed"] and code not in (src, dst):
                new_blocks.append(code)
            if not v["allowed"] and code in (src, dst):
                sys.exit(f"[FATAL] {role} country {code} is infeasible per "
                         f"FCG data: {v['reason']}")
        if not new_blocks:
            return path, total_km, verdicts, nodes
        blocked.update(new_blocks)
        print(f"    re-routing around: {new_blocks}")

    sys.exit("[FATAL] no stable route within 10 replanning iterations.")


def narrate(path, total_km, verdicts, nodes, travel_date, use_llm):
    legs = []
    for a, b in zip(path, path[1:]):
        d = haversine_km(nodes.loc[a, "lat"], nodes.loc[a, "lon"],
                         nodes.loc[b, "lat"], nodes.loc[b, "lon"])
        legs.append(f"{nodes.loc[a,'country']} ({a}) -> "
                    f"{nodes.loc[b,'country']} ({b}): {d:,.0f} km")
    print("\n=== FINAL ROUTE ===")
    print("\n".join(legs))
    print(f"Total great-circle distance: {total_km:,.0f} km")
    cautions = {c: v for c, v in verdicts.items() if v.get("caution")}
    if cautions:
        print("Cautions:")
        for c, v in cautions.items():
            print(f"  - {c}: {v['reason']}")
    if use_llm:
        summary = llm_chat(
            f"Mission date {travel_date}. Route legs:\n" + "\n".join(legs) +
            "\nPer-country clearance verdicts:\n" +
            json.dumps(verdicts, indent=1) +
            "\nWrite a short operational briefing (<=150 words) for the crew: "
            "route rationale, required lead-time actions, cautions.")
        print("\n=== BRIEFING ===\n" + summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help='e.g. "fra to afg"')
    ap.add_argument("--date", default="2026-09-15", help="mission date")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip agent; pure geometric shortest path")
    args = ap.parse_args()
    use_llm = not args.no_llm
    path, km, verdicts, nodes = plan(args.query, args.date, use_llm)
    narrate(path, km, verdicts, nodes, args.date, use_llm)
