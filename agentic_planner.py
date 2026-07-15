"""
agentic_planner.py - agentic layer on top of planner.py. Drag-and-drop next to
planner.py, datalayer.py, fcg_extract.csv, mapping.csv. Same config as planner.py.

  python agentic_planner.py "hog to dur on 2026-09-15, avoid Azkaban airspace, max 2 stops"
  python agentic_planner.py "plan HOGW to DRMS departing 2026-09-15 with hazmat"

What's agentic vs planner.py:
  - You type a natural-language request; the model turns it into constraints.
  - The model PROPOSES routes using verified tools (corridor candidates,
    distances, FCG requirements, overflight checks).
  - Every proposal is judged by deterministic validate_route(). Blockers are
    fed back and the model REROUTES (e.g. detour around prohibited airspace)
    until a route passes or the turn budget ends.
  - Facts only enter through tools. The model never asserts a lead time,
    an authorization, or a distance.

Speed fixes included (planner.py untouched):
  - negative coordinate caching: junk ICAOs are remembered as null and never
    re-asked (this was making every planner.py run slow)
  - crossings cached per leg pair in crossings_cache.json
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import planner as P               # reuses your planner.py config + functions
from openai import OpenAI

MAX_TURNS       = 24
CROSSINGS_CACHE = "./crossings_cache.json"
REPORT_PATH     = "./agentic_plan_report.md"

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"}, "action": {"type": "string"},
        "args": {"type": "object"},
        "route": {"type": "array", "items": {"type": "string"}},
        "origin": {"type": "string"}, "dest": {"type": "string"},
        "departure": {"type": "string"}, "hazmat": {"type": "boolean"},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["thought", "action"],
}


# ------------------------ cached geography -------------------------
def get_coords_cached(client, icaos):
    """planner.get_coords + negative caching: unknown codes stored as null."""
    cache = json.loads(Path(P.COORDS_CACHE).read_text()) if Path(P.COORDS_CACHE).exists() else {}
    missing = [i for i in icaos if i not in cache]
    if missing:
        found = P.get_coords(client, missing)     # writes positives to cache file
        cache = json.loads(Path(P.COORDS_CACHE).read_text())
        for i in missing:
            if i not in found:
                cache[i] = None                   # negative-cache junk codes
        Path(P.COORDS_CACHE).write_text(json.dumps(cache, indent=1))
    return {i: tuple(cache[i]) for i in icaos if cache.get(i)}


def crossings_cached(client, a_icao, a, b_icao, b):
    cache = json.loads(Path(CROSSINGS_CACHE).read_text()) if Path(CROSSINGS_CACHE).exists() else {}
    key = f"{a_icao}->{b_icao}"
    if key not in cache:
        cache[key] = P.crossings(client, a_icao, a, b_icao, b)
        Path(CROSSINGS_CACHE).write_text(json.dumps(cache, indent=1))
    return cache[key]


# ----------------------------- toolbox -----------------------------
class Toolbox:
    def __init__(self, client):
        self.client = client
        self.fcg = P.load_fcg(P.FCG_CSV)
        self.code2name, self.name2code = P.load_mapping(P.MAPPING_CSV)
        self.icao_to_row = {}
        for r in self.fcg:
            for i in r["icaos"]:
                self.icao_to_row.setdefault(i, r)
        self.coords = get_coords_cached(self.client, sorted(self.icao_to_row))

    def dn(self, key):
        return P.display_name(key, self.code2name)

    # ---- tools the model may call ----
    def resolve(self, term):
        kind, val = P.resolve_endpoint(term, self.fcg, sorted(self.icao_to_row),
                                       self.code2name, self.name2code, self.client)
        if kind == "country":
            return {"kind": "country", "country": self.dn(val["key"]),
                    "authorized_icaos": sorted(val["icaos"])}
        if val not in self.coords:
            self.coords.update(get_coords_cached(self.client, [val]))
        return {"kind": "icao", "icao": val,
                "has_coords": val in self.coords}

    def distance(self, icao_a, icao_b):
        a, b = self.coords.get(icao_a.upper()), self.coords.get(icao_b.upper())
        if not a or not b:
            return {"error": f"no coordinates for {icao_a if not a else icao_b}"}
        return {"km": round(P.dist_km(a, b), 1)}

    def candidates(self, origin, dest, corridor_km=None, max_leg_km=None):
        A, B = self.coords.get(origin.upper()), self.coords.get(dest.upper())
        if not A or not B:
            return {"error": "no coordinates for origin/dest"}
        corridor_km = corridor_km or P.CORRIDOR_KM
        total = P.dist_km(A, B)
        out = []
        for icao, rec in self.icao_to_row.items():
            if icao in (origin.upper(), dest.upper()) or icao not in self.coords:
                continue
            off, along = P.cross_track_km(self.coords[icao], A, B)
            if off <= corridor_km and 0 < along < total:
                out.append({"icao": icao, "country": self.dn(rec["key"]),
                            "along_km": round(along), "off_track_km": round(off)})
        out.sort(key=lambda c: c["along_km"])
        return {"direct_km": round(total), "candidates": out[:40]}

    def requirements(self, icao):
        rec = self.icao_to_row.get(icao.upper())
        if not rec:
            return {"icao": icao.upper(),
                    "warning": "not on any authorized entry/exit list"}
        return {"icao": icao.upper(), "country": self.dn(rec["key"]),
                "lead_time": rec["lead_raw"][:200], "lead_days": rec["lead_days"],
                "hazmat": rec["hazmat"][:150], "overflight": rec["overflight"][:150],
                "restrictions": rec["restrictions"][:150], "customs": rec["customs"][:150],
                "hours": rec["hours"][:150], "forms": rec["forms"][:150],
                "payment": rec["cash"][:150]}

    def check_overflight_leg(self, icao_a, icao_b, departure):
        a, b = self.coords.get(icao_a.upper()), self.coords.get(icao_b.upper())
        if not a or not b:
            return {"error": "no coordinates"}
        dep = date.fromisoformat(departure)
        results = []
        for name in crossings_cached(self.client, icao_a.upper(), a, icao_b.upper(), b):
            rec = P.find_row_by_country(self.fcg, name, self.code2name, self.name2code)
            if rec is None:
                results.append({"country": name, "status": "no FCG data"})
                continue
            of = rec["overflight"]
            if any(w in (of or "").lower() for w in P.PROHIBIT_WORDS):
                results.append({"country": self.dn(rec["key"]), "status": "PROHIBITED"})
            elif of.strip().upper() == "NA":
                results.append({"country": self.dn(rec["key"]), "status": "unknown (NA)"})
            else:
                lead = rec.get("overflight_lead_days") or P.parse_lead_days(of)
                st = "ok"
                if lead and dep - timedelta(days=lead) < P.TODAY:
                    st = f"lead {lead}d unmeetable"
                results.append({"country": self.dn(rec["key"]), "status": st,
                                "detail": of[:100]})
        return {"leg": f"{icao_a.upper()}->{icao_b.upper()}",
                "results": results, "note": "crossings are model-estimated"}

    # ---- deterministic judge (not callable-by-name, run on 'final') ----
    def validate(self, stops, origin, dest, departure, hazmat, max_leg_km):
        dep = date.fromisoformat(departure)
        chain = [origin] + stops + [dest]
        blockers, warnings, legs = [], [], []
        for a, b in zip(chain, chain[1:]):
            ca, cb = self.coords.get(a), self.coords.get(b)
            if not ca or not cb:
                blockers.append(f"no coordinates for leg {a}->{b}")
                continue
            km = P.dist_km(ca, cb)
            legs.append({"leg": f"{a}->{b}", "km": round(km)})
            if km > max_leg_km:
                blockers.append(f"leg {a}->{b} is {km:.0f} km > max {max_leg_km}")
        for s in stops:
            rec = self.icao_to_row.get(s)
            if rec is None:
                blockers.append(f"{s}: not on any authorized entry/exit list")
                continue
            bl, wa, _ = P.stop_check(rec, s, dep, hazmat)
            blockers += bl
            warnings += wa
        for a, b in zip(chain, chain[1:]):
            of = self.check_overflight_leg(a, b, departure)
            for r in of.get("results", []):
                if r["status"] == "PROHIBITED":
                    blockers.append(f"overflight {a}->{b}: {r['country']} PROHIBITED")
                elif "unmeetable" in r["status"]:
                    blockers.append(f"overflight {a}->{b}: {r['country']} {r['status']}")
                elif r["status"] != "ok":
                    warnings.append(f"overflight {a}->{b}: {r['country']} - {r['status']}")
        return {"ok": not blockers, "legs": legs,
                "blockers": blockers, "warnings": warnings}


TOOLS = ["resolve", "distance", "candidates", "requirements", "check_overflight_leg"]

SYSTEM = """You are a route-planning agent over VERIFIED tools. Rules:
1. You know NO airports, distances, or clearance rules. Every fact must come
   from a tool result in this conversation. Never invent an ICAO code.
2. One JSON action per turn: {"thought","action","args"}. Tools: """ + ", ".join(TOOLS) + """.
   Args: resolve(term), distance(icao_a,icao_b), candidates(origin,dest,corridor_km),
   requirements(icao), check_overflight_leg(icao_a,icao_b,departure).
3. First parse the user's request (origin, dest, departure date, hazmat,
   special constraints like airspace to avoid or max stops). Resolve endpoints.
4. Explore with candidates/requirements/check_overflight_leg, honoring the
   user's constraints, then finalize:
   {"action":"final","origin":ICAO,"dest":ICAO,"departure":"YYYY-MM-DD",
    "hazmat":bool,"route":[intermediate stop ICAOs],"notes":["uncertainties"]}
5. Your final route is independently validated. If blockers return, REROUTE
   with different stops (detour around prohibited airspace by picking stops
   that change the legs). Never resubmit an unchanged route."""


def parse_action(raw):
    t = raw.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t[:4].lower() == "json" else t
    return json.loads(t[t.find("{"):])


def run(request):
    client = OpenAI(base_url=P.VLLM_BASE_URL, api_key=P.API_KEY)
    tb = Toolbox(client)
    print(f"  {len(tb.fcg)} FCG rows, {len(tb.coords)} ICAOs with coordinates")
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Today is {P.TODAY}. Request: {request}. "
             f"Default max leg {P.MAX_LEG_KM} km unless the user says otherwise."}]
    audit = []
    for turn in range(MAX_TURNS):
        r = client.chat.completions.create(
            model=P.MODEL, messages=msgs, temperature=0, max_tokens=1500,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "action", "schema": ACTION_SCHEMA}})
        try:
            act = parse_action(r.choices[0].message.content or "")
        except Exception as e:
            msgs.append({"role": "user", "content": f"Unparseable JSON ({e}). One valid action."})
            continue
        msgs.append({"role": "assistant", "content": json.dumps(act)})
        audit.append(act)
        print(f"[{turn}] {act['action']} - {act.get('thought','')[:80]}")

        if act["action"] == "final":
            stops = [s.upper() for s in act.get("route", [])]
            origin, dest = act.get("origin", "").upper(), act.get("dest", "").upper()
            dep, hz = act.get("departure", ""), bool(act.get("hazmat"))
            try:
                verdict = tb.validate(stops, origin, dest, dep, hz, P.MAX_LEG_KM)
            except Exception as e:
                msgs.append({"role": "user", "content": f"Validation error: {e}. Fix and refinalize."})
                continue
            if verdict["ok"]:
                write_report(request, origin, dest, dep, hz, stops, verdict,
                             act.get("notes", []), tb)
                Path("./agentic_audit.json").write_text(json.dumps(audit, indent=1, default=str))
                print(f"VALIDATED -> {REPORT_PATH}")
                return
            msgs.append({"role": "user", "content":
                "Independent validation REJECTED your route:\n"
                + json.dumps(verdict["blockers"])
                + "\nReroute with different stops (do not resubmit unchanged)."})
            continue

        if act["action"] not in TOOLS:
            msgs.append({"role": "user", "content": f"Unknown tool. Allowed: {TOOLS} or 'final'."})
            continue
        try:
            result = getattr(tb, act["action"])(**(act.get("args") or {}))
        except TypeError as e:
            result = {"error": f"bad args: {e}"}
        msgs.append({"role": "user", "content": "TOOL RESULT:\n"
                     + json.dumps(result, default=str)[:8000]})
    Path("./agentic_audit.json").write_text(json.dumps(audit, indent=1, default=str))
    print("No validated route within turn budget - see agentic_audit.json")


def write_report(request, origin, dest, dep, hz, stops, verdict, notes, tb):
    chain = [origin] + stops + [dest]
    out = [f"# Agentic route plan", f"Request: {request}",
           f"Validated {P.TODAY} | HAZMAT={hz} | departure {dep}",
           "*Coordinates & crossings model-provided (cached); clearance facts from your FCG extract.*",
           "", "## Route", " -> ".join(chain)]
    out += [f"- {l['leg']}: {l['km']} km" for l in verdict["legs"]]
    out.append("")
    deadlines = {}
    for s in stops + [dest, origin]:
        rec = tb.icao_to_row.get(s)
        if not rec:
            continue
        req = tb.requirements(s)
        if s in stops:
            out += [f"### {s} - {req['country']}", f"- Lead time: {req['lead_time']}"]
            for k in ("customs", "hours", "forms", "payment"):
                if req[k].strip().upper() != "NA":
                    out.append(f"- {k.capitalize()}: {req[k]}")
            out.append("")
        if rec["lead_days"]:
            deadlines[req["country"]] = (date.fromisoformat(dep) - timedelta(days=rec["lead_days"]),
                                         rec["lead_days"])
    flagged = list(dict.fromkeys(verdict["warnings"] + notes))
    if flagged:
        out.append("## Warnings / uncertainties")
        out += [f"- WARN: {w}" for w in flagged]
    if deadlines:
        out.append("\n## Filing timeline (soonest first)")
        for k, (fb, lead) in sorted(deadlines.items(), key=lambda kv: kv[1][0]):
            out.append(f"- **{fb}** - file {k} ({lead}d lead)"
                       + (" OVERDUE" if fb < P.TODAY else ""))
    Path(REPORT_PATH).write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python agentic_planner.py "hog to dur on 2026-09-15, avoid Azkaban"')
    run(" ".join(sys.argv[1:]))
