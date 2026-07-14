"""
Agentic route planner: your vLLM model orchestrates VERIFIED tools.

Design contract:
  - The model never states a fact; it requests tools and reasons over results.
  - One JSON action per turn:  {"thought": "...", "action": "<tool>", "args": {...}}
    or                         {"thought": "...", "action": "final",
                                "route": ["ICAO", ...], "uncertainties": ["..."]}
  - Before a final answer is accepted, THIS CODE (not the model) runs
    Toolbox.validate_route(). Blockers -> the result is fed back and the model
    must revise. Warnings -> surfaced as flagged uncertainties in the report.
  - Hard cap on iterations; full tool-call audit trail saved with the report.

Run:  python agent.py HOGW DRMS 2026-08-10
Env:  same vLLM endpoint as fcg_extract.py (edit CONFIG below).
"""

import json
import sys
from datetime import date
from pathlib import Path

try:
    from openai import OpenAI          # same dependency as fcg_extract.py
except ImportError:                    # allows offline testing with a fake client
    OpenAI = None

from tools import Toolbox, TOOL_SPECS

# ----------------------------- CONFIG -----------------------------
MODEL         = "gemma-4-31B-it"
VLLM_BASE_URL = "http://localhost:8000/v1"
API_KEY       = "EMPTY"

AIRPORTS_CSV  = "./data/airports.csv"
SCHEMA        = "ourairports"        # "dafif" / "generic" - see datalayer.SCHEMA_MAPS
RUNWAYS_CSV   = "./data/runways.csv"
COUNTRIES_CSV = "./data/countries.csv"
FCG_CSV       = "./fcg_extract.csv"

MAX_LEG_KM    = 1500
MIN_RUNWAY_FT = 6000
HAZMAT        = False
MAX_TURNS     = 20
REPORT_PATH   = "./agent_itinerary.md"
AUDIT_PATH    = "./agent_audit.json"
# ------------------------------------------------------------------

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {"type": "string"},
        "args": {"type": "object"},
        "route": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["thought", "action"],
}

SYSTEM = """You are a flight-route planning agent operating on VERIFIED tools only.

Hard rules:
1. You know NO airports, distances, or clearance rules yourself. Every fact must
   come from a tool result in this conversation. Never invent an ICAO code.
2. One action per turn, as JSON: {"thought", "action", "args"}.
3. Workflow: airports_near_route -> investigate candidates with fcg_requirements /
   check_lead_time / distance_km -> run check_overflight on each planned leg
   (a leg over a PROHIBITED country will fail validation - reroute around it)
   -> propose {"action":"final","route":[stops],"uncertainties":[...]}
   (intermediate stops only, exclude origin/destination).
4. Your final route will be independently validated. If blockers come back,
   revise using different candidates. Do not resubmit an unchanged route.
5. List EVERY uncertainty: unknown runway lengths, unparseable lead times,
   missing FCG data, hours/restriction caveats. Flag, never guess.

Available tools and their args:
""" + json.dumps({k: v["args"] for k, v in TOOL_SPECS.items()}, indent=1)


def parse_action(raw):
    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        txt = txt[4:] if txt[:4].lower() == "json" else txt
    start = txt.find("{")
    return json.loads(txt[start:]) if start >= 0 else json.loads(txt)


def run(origin, dest, departure):
    tb = Toolbox(AIRPORTS_CSV, FCG_CSV, schema=SCHEMA,
                 runways_csv=RUNWAYS_CSV, countries_csv=COUNTRIES_CSV)
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=API_KEY)
    print(f"Ingest: {tb.ingest_report}")

    task = (f"Plan a route {origin} -> {dest}, departure {departure} (today {tb.today}). "
            f"Constraints: max leg {MAX_LEG_KM} km, min runway {MIN_RUNWAY_FT} ft, "
            f"HAZMAT={HAZMAT}. Use as few stops as possible. Unknown/NA data is "
            f"acceptable for this first route but must be listed as uncertainties.")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    audit = []
    option_a = option_b = None          # (stops, verdict, uncertainties)
    phase = "A"                          # A: best route, unknowns ok; B: strict

    for turn in range(MAX_TURNS):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0, max_tokens=1500,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "action", "schema": ACTION_SCHEMA}})
        try:
            act = parse_action(resp.choices[0].message.content or "")
        except Exception as e:
            messages.append({"role": "user", "content": f"Unparseable JSON ({e}). Reply with one valid action."})
            continue
        messages.append({"role": "assistant", "content": json.dumps(act)})
        audit.append({"turn": turn, "phase": phase, "action": act})
        print(f"[{turn}/{phase}] {act['action']}  - {act.get('thought','')[:90]}")

        if act["action"] == "final":
            stops = [s.upper() for s in act.get("route", [])]
            strict = (phase == "B")
            verdict = tb.validate_route(stops, origin, dest, departure,
                                        MAX_LEG_KM, HAZMAT, MIN_RUNWAY_FT, strict=strict)
            audit.append({"turn": turn, "phase": phase, "validation": verdict})
            if verdict["ok"]:
                if phase == "A":
                    option_a = (stops, verdict, act.get("uncertainties", []))
                    phase = "B"
                    messages.append({"role": "user", "content":
                        "Route A accepted. Now propose an ALTERNATIVE route with ZERO "
                        "critical unknowns: every stop needs a parseable lead time, known "
                        "overflight status for all non-oceanic crossings, and (if HAZMAT) "
                        "known HAZMAT rules. It may use more stops. It will be validated "
                        "in strict mode. If you conclude no such route exists, reply "
                        '{"action":"final","route":[],"uncertainties":["<why>"]}.'})
                    continue
                option_b = (stops, verdict, act.get("uncertainties", []))
                break
            if phase == "B" and not stops:      # model says no strict route exists
                option_b = (None, verdict, act.get("uncertainties", []))
                break
            messages.append({"role": "user", "content":
                f"Independent validation ({'STRICT' if strict else 'standard'}) REJECTED:\n"
                + json.dumps(verdict["blockers"]) + "\nRevise with different stops."})
            continue

        if act["action"] not in TOOL_SPECS:
            messages.append({"role": "user", "content":
                f"Unknown tool '{act['action']}'. Allowed: {list(TOOL_SPECS)}"})
            continue
        args = {k: v for k, v in (act.get("args") or {}).items()
                if k in TOOL_SPECS[act["action"]]["args"]}
        try:
            result = getattr(tb, act["action"])(**args)
        except TypeError as e:
            result = {"error": f"bad args: {e}"}
        audit.append({"turn": turn, "phase": phase, "result": result})
        messages.append({"role": "user", "content": "TOOL RESULT:\n"
                         + json.dumps(result, default=str)[:8000]})

    Path(AUDIT_PATH).write_text(json.dumps(audit, indent=1, default=str))
    if option_a:
        write_report(origin, dest, departure, option_a, option_b, tb)
        print(f"REPORT -> {REPORT_PATH}")
    else:
        print("No validated route within turn budget - see audit trail.")


def _stop_section(tb, s, departure):
    req = tb.fcg_requirements(s)
    lt = tb.check_lead_time(s, departure)
    out = [f"### {s} - {tb.airport_info(s)['name']}",
           f"- FCG source: {req.get('fcg_source','NONE')}",
           f"- Lead time: {req.get('lead_time','NA')}"
           + (f" (file by **{lt['file_by']}**)" if lt.get("file_by") else "")]
    for k, lbl in (("customs", "Customs"), ("forms", "Forms"),
                   ("payment", "Payment"), ("hours", "Hours")):
        v = req.get(k, "NA")
        if v and v.strip().upper() != "NA":
            out.append(f"- {lbl}: {v[:180]}")
    for u in req.get("unknowns", []):
        out.append(f"- ❓ {u['field']}: NA ({u['kind']})")
    return out


def _option_section(tb, title, opt, origin, dest, departure):
    stops, verdict, uncertainties = opt
    out = [f"## {title}"]
    if stops is None:
        out += ["No fully verified route found.",
                *(f"- {u}" for u in uncertainties), ""]
        return out
    out.append(f"{origin} -> " + " -> ".join(stops) + f" -> {dest}")
    out += [f"- {l['from']} -> {l['to']}: {l['km']} km" for l in verdict["legs"]]
    out.append("")
    for s in stops:
        out += _stop_section(tb, s, departure) + [""]
    out.append("### Overflight (approximate, nearest-airport proxy)")
    chain = [origin] + stops + [dest]
    for a, b in zip(chain, chain[1:]):
        of = tb.check_overflight(a, b, departure)
        if "error" not in of:
            out.append(f"- {of['leg']}: " + ", ".join(
                f"{r['country']}[{r.get('status','?')}]" for r in of["results"]))
    flagged = list(dict.fromkeys(verdict["warnings"] + uncertainties))
    if flagged:
        out.append("### Flagged uncertainties")
        out += [f"- ⚠️ {w}" for w in flagged]
    out.append("")
    return out


def write_report(origin, dest, departure, option_a, option_b, tb):
    out = [f"# Agentic route plan: {origin} -> {dest}",
           f"Departure {departure} | planned {tb.today} | validated independently",
           f"Data ingest: {tb.ingest_report}", ""]
    out += _option_section(tb, "Option A - best route (unknowns flagged, NOT fully verified)",
                           option_a, origin, dest, departure)
    if option_b:
        out += _option_section(tb, "Option B - fully verified route (strict: zero critical unknowns)",
                               option_b, origin, dest, departure)
    else:
        out += ["## Option B - fully verified route", "Strict search did not complete."]
    Path(REPORT_PATH).write_text("\n".join(out), encoding="utf-8")




if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit("Usage: python agent.py ORIGIN DEST YYYY-MM-DD")
    run(sys.argv[1].upper(), sys.argv[2].upper(), sys.argv[3])
