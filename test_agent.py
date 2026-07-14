"""Offline tests: verified tools + agent loop plumbing with a scripted model.
The fake model first proposes AZKA (blocked: unlisted... actually blocked by
lead time + hazmat-free but 45 business days) to prove validation rejects and
the loop recovers."""

import json
from types import SimpleNamespace

import agent
from tools import Toolbox, haversine_km

tb = Toolbox("./data/airports.csv", "./fcg_extract.csv", schema="ourairports",
             runways_csv="./data/runways.csv", countries_csv="./data/countries.csv")

# ---------- tool unit tests ----------
# known-distance sanity: HOGW(57.2,-3.8) -> DRMS(69,33) ~2215 km (matches v1 planner)
d = tb.distance_km("HOGW", "DRMS")["km"]
assert 2150 < d < 2280, d
assert "error" in tb.airport_info("XXXX")
near = tb.airports_near_route("HOGW", "DRMS", corridor_km=400, min_runway_ft=6000)
icaos = {c["icao"] for c in near["candidates"]}
assert "GRIN" in icaos and "SMAL" not in icaos, icaos       # short strip filtered
req = tb.fcg_requirements("GRIN")
assert req["listed_entry_exit"] is True
lt = tb.check_lead_time("AZKA", "2026-08-10")
assert lt["feasible"] is False                               # 45 business days -> unmeetable
bad = tb.validate_route(["AZKA"], "HOGW", "DRMS", "2026-08-10", 1500, False, 6000)
assert not bad["ok"] and any("lead time" in b for b in bad["blockers"]), bad
good = tb.validate_route(["GRIN"], "HOGW", "DRMS", "2026-08-10", 1500, False, 6000)
assert good["ok"], good

# strict mode: an unparseable lead time is a WARNING leniently, a BLOCKER strictly
norway = next(r for r in tb.fcg if r["key"] == "Norway")
saved = norway["lead_days"]; norway["lead_days"] = None
len_v = tb.validate_route(["GRIN"], "HOGW", "DRMS", "2026-08-10", 1500, False, 6000, strict=False)
str_v = tb.validate_route(["GRIN"], "HOGW", "DRMS", "2026-08-10", 1500, False, 6000, strict=True)
assert len_v["ok"] and not str_v["ok"], (len_v, str_v)
norway["lead_days"] = saved

# NA classification from extractor flags
norway["no_route_ids"] = {"customs_immigration"}
norway["customs"] = "NA"
req = tb.fcg_requirements("GRIN")
kinds = {u["field"]: u["kind"] for u in req["unknowns"]}
assert kinds.get("customs_immigration") == "not published in source document", kinds
print("TOOL TESTS PASSED (incl. strict mode + NA classification)")

# ---------- scripted fake model to exercise the loop ----------
SCRIPT = [
    {"thought": "find candidates", "action": "airports_near_route",
     "args": {"origin": "HOGW", "dest": "DRMS", "corridor_km": 400,
              "min_runway_ft": 6000, "limit": 25}},
    {"thought": "check AZKA clearance", "action": "check_lead_time",
     "args": {"icao": "AZKA", "departure_date": "2026-08-10"}},
    {"thought": "try AZKA anyway (should be rejected)", "action": "final",
     "route": ["AZKA"], "uncertainties": []},
    {"thought": "revise: check GRIN", "action": "fcg_requirements",
     "args": {"icao": "GRIN"}},
    {"thought": "GRIN authorized and feasible", "action": "final",
     "route": ["GRIN"], "uncertainties": ["BEAU closed Sundays per Norway hours note"]},
    {"thought": "GRIN also has no critical unknowns - submit as strict route", "action": "final",
     "route": ["GRIN"], "uncertainties": []},
]


class FakeClient:
    def __init__(self):
        self.i = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        act = SCRIPT[self.i]; self.i += 1
        msg = SimpleNamespace(content=json.dumps(act))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


agent.OpenAI = lambda **kw: FakeClient()
agent.run("HOGW", "DRMS", "2026-08-10")

report = open(agent.REPORT_PATH).read()
assert "GRIN" in report and "Option A" in report and "Option B" in report
audit = json.load(open(agent.AUDIT_PATH))
rejections = [a for a in audit if "validation" in a and not a["validation"]["ok"]]
assert rejections, "expected the bad route to be rejected by the validator"
print("AGENT LOOP TESTS PASSED (bad route rejected, revision validated)")
