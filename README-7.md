# Agentic Flight-Path Planner (FCG-constrained)

Generate and verify military airlift routes that **minimize fuel** subject to
**hard diplomatic-overflight feasibility** derived from Foreign Clearance Guide
(FCG) data. Deterministic geometry/routing/optimization; an LLM (Gemma on vLLM)
does only document extraction and checklist justification.

```
Origin/Dest ─▶ great-circle ─▶ countries crossed ─▶ constraints (FCG/DB)
        │                                                   │
        └────────── route AROUND forbidden airspace ◀───────┘  (Stage 1, networkx)
                            │
                     fuel optimize (Stage 2, openap-top) ─▶ checklist verify ─▶ report
```

## Why it's built this way
The LLM never does geometry or search. Three **pluggable seams** keep it open to
new angles without touching the core:

| Seam | v1 implementation | Future drop-in |
|------|-------------------|----------------|
| `AirspaceProvider` (`airspace/base.py`) | `CountryProvider` — political borders, ISO3 | `FIRProvider` — FIR/airspace polygons |
| `ConstraintProvider` (`constraints/base.py`) | `FCGConstraintProvider` — structured DB | theater/NOTAM sources via `MergedConstraintProvider` |
| `Objective` (`objectives/base.py`) | `FeasibilityConstraint` (hard) + `FuelCost` (soft) | `TimeCost`, `ThreatAvoidance`, `ContrailCost` |

Region ids are opaque downstream, so swapping countries→FIRs is one new class.

## Layout
```
flight_planner/
  schema.py            typed contracts (Constraints, Route, FlightQuery, PlanResult)
  geo.py               pure-python geodesy + polygon predicates (swap for shapely/pyproj)
  airspace/base.py     SEAM 1
  constraints/base.py  SEAM 2  (+ extract.py = FCG HTML -> DB, llm/client.py = vLLM)
  objectives/base.py   SEAM 3
  routing/router.py    Stage 1: visibility-graph avoidance (networkx)
  routing/optimize.py  Stage 2: fuel (openap-top, with fallback)
  checklist/verify.py  rule-based checklist (maps items 29.7/29.8/... to fields)
  agent/engine.py      unified generate + verify state machine
scripts/   extract_fcg.py | plan_route.py | verify_route.py
data/      countries.geojson | constraints_db.json | airports.csv   (ALL SYNTHETIC DEMO)
tests/     test_smoke.py
```

## Run the demo (works today, stdlib + networkx + numpy)
```bash
python3 tests/test_smoke.py                      # 3 tests

# GENERATE: Travis AFB -> Ramstein, C-17, 14-day lead
python3 scripts/plan_route.py --from KSUU --to ETAR --aircraft C17 --lead-days 14

# VERIFY a proposed track that cuts through forbidden airspace
python3 scripts/verify_route.py --aircraft C17 --lead-days 14 \
    --waypoints "38.26,-121.93;57,-30;49.44,7.60"
```
The demo ships a **fictional forbidden zone "XAA"** in the mid-Atlantic; the
generator routes around it (via Iceland) and verify-mode flags + replaces a route
that crosses it.

## Build order → production (air-gapped)
**1. Extract FCG → constraints DB** (the only LLM-in-the-loop step, run once)
```bash
# point vLLM/Gemma at your real FCG/full/*.cfm.html
python3 scripts/extract_fcg.py --fcg-dir /path/FCG/full --out data/constraints_db.json \
    --base-url http://localhost:8000 --model google/gemma-3-27b-it
# (--mock runs an offline heuristic for plumbing tests, no model needed)
```
Uses vLLM **`guided_json`** to force schema-valid output — the big reliability win.

**2. Real borders.** Convert Natural Earth `ne_110m_admin_0_countries` to GeoJSON
(field `ADM0_A3` is read automatically), simplify ~0.3°, point `--geojson` at it.

**3. Real fuel.** `pip install openap openap-top casadi cfgrib` then pass `--openap`
(and `--wind-grib ERA5.grib`). `routing/optimize.py` already calls
`top.CompleteFlight(...).trajectory(objective="fuel")` per leg.

**4. FIRs / extra objectives.** Add a `FIRProvider` (Seam 1) and/or new `Objective`
subclasses (Seam 3). No core changes.

## Swapping geometry for production accuracy
`geo.py` treats lat/lon as planar for visibility tests — fine for simplified
polygons, but watch the antimeridian and high latitudes. For production replace
`point_in_ring`/`visible` with shapely on a projected CRS, and edge weights with
`pyproj.Geod` geodesic lengths. The function names are the seam.

## Porting the engine to LangGraph
`agent/engine.py` is an explicit state machine that maps 1:1 onto LangGraph nodes:
`great_circle → countries_crossed → [LLM]extract → check_feasible → (reroute loop)
→ optimize_fuel → [LLM]verify_checklist → report`. Use LangGraph when you want
durable checkpoints and the LLM to own the extract/justify nodes.

## Important caveat
All files in `data/` are **synthetic** and the geometry is simplified. Nothing here
is a real overflight authorization. Validate against authoritative FCG/APACS data
and your own airspace sources before any operational use.
