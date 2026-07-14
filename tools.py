"""
Verified deterministic tools. This is the ONLY place facts come from.

Every function here is pure Python + your ingested data: no model, no network,
unit-testable. The agent (agent.py) may call these and nothing else, and its
final route is re-checked by validate_route() OUTSIDE the model. If the model
says something these tools didn't return, it doesn't survive validation.
"""

import math
from datetime import date, timedelta

from datalayer import ingest_airports, merge_runways, load_country_names, load_fcg

EARTH_R = 6371.0


# ------------------------- verified math --------------------------
def _rad(x):
    return math.radians(float(x))


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = _rad(lat1), _rad(lat2)
    dphi, dlmb = _rad(lat2 - lat1), _rad(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def cross_track_km(lat, lon, lat1, lon1, lat2, lon2):
    """(off-track km, along-track km) of a point relative to great circle 1->2."""
    d13 = haversine_km(lat1, lon1, lat, lon) / EARTH_R
    if d13 == 0:
        return 0.0, 0.0
    def brng(a1, o1, a2, o2):
        return math.atan2(
            math.sin(_rad(o2 - o1)) * math.cos(_rad(a2)),
            math.cos(_rad(a1)) * math.sin(_rad(a2))
            - math.sin(_rad(a1)) * math.cos(_rad(a2)) * math.cos(_rad(o2 - o1)))
    dxt = math.asin(math.sin(d13) * math.sin(brng(lat1, lon1, lat, lon) - brng(lat1, lon1, lat2, lon2)))
    dat = math.acos(max(-1.0, min(1.0, math.cos(d13) / max(math.cos(dxt), 1e-12))))
    return abs(dxt) * EARTH_R, dat * EARTH_R


# ------------------------ toolbox object --------------------------
class Toolbox:
    def __init__(self, airports_csv, fcg_csv, schema="generic",
                 runways_csv=None, countries_csv=None, today=None):
        self.airports, rejected = ingest_airports(airports_csv, schema=schema)
        self.airports = merge_runways(self.airports, runways_csv)
        self.country_names = load_country_names(countries_csv)
        self.fcg = load_fcg(fcg_csv)
        self.today = today or date.today()
        self.ingest_report = {"accepted": len(self.airports), "rejected": rejected,
                              "source": airports_csv, "schema": schema}

    # ---- lookups ----
    def airport_info(self, icao):
        ap = self.airports.get(icao.upper())
        if not ap:
            return {"error": f"{icao} not in verified airport table"}
        return ap

    def distance_km(self, icao_a, icao_b):
        a, b = self.airport_info(icao_a), self.airport_info(icao_b)
        if "error" in a or "error" in b:
            return {"error": a.get("error") or b.get("error")}
        return {"from": a["icao"], "to": b["icao"],
                "km": round(haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]), 1)}

    def _fcg_for(self, icao, country):
        """(record, listed_as_authorized). ICAO listing preferred, country-name fallback."""
        icao = icao.upper()
        for r in self.fcg:
            if icao in r["icaos"]:
                return r, True
        cn = (self.country_names.get(country, country) or "").lower()
        for r in self.fcg:
            if cn and cn in r["key"].replace("_", " ").lower():
                return r, False
        return None, False

    def fcg_requirements(self, icao):
        """Everything the FCG extract knows about this airport's country."""
        ap = self.airport_info(icao)
        if "error" in ap:
            return ap
        rec, listed = self._fcg_for(icao, ap["country"])
        if rec is None:
            return {"icao": icao.upper(), "fcg": None,
                    "warning": "no FCG data for this country - UNVERIFIED territory"}
        # classify every NA field: absent from source doc vs possible extraction miss
        field_qid = {"lead_raw": "diplomatic_lead_time", "hazmat": "hazmat",
                     "restrictions": "airfield_restrictions", "customs": "customs_immigration",
                     "hours": "operating_hours", "forms": "country_specific",
                     "cash": "aircard_cash", "overflight": "overflight"}
        unknowns = []
        for fld, qid in field_qid.items():
            if (rec.get(fld) or "NA").strip().upper() == "NA":
                kind = ("not published in source document" if qid in rec["no_route_ids"]
                        else "not found in section (possible extraction miss)")
                unknowns.append({"field": qid, "kind": kind})
        return {"icao": icao.upper(), "fcg_source": rec["key"],
                "listed_entry_exit": listed,
                "lead_time": rec["lead_raw"], "lead_days_parsed": rec["lead_days"],
                "hazmat": rec["hazmat"], "restrictions": rec["restrictions"],
                "customs": rec["customs"], "hours": rec["hours"],
                "forms": rec["forms"], "payment": rec["cash"],
                "unknowns": unknowns}

    def check_lead_time(self, icao, departure_date):
        """Can this country's clearance still be filed in time?"""
        req = self.fcg_requirements(icao)
        if req.get("fcg") is None and "fcg_source" not in req:
            return {**req, "feasible": None, "uncertainty": "no FCG record to check"}
        dep = date.fromisoformat(departure_date)
        lead = req.get("lead_days_parsed")
        if lead is None:
            return {"icao": req["icao"], "feasible": None,
                    "uncertainty": f"lead time not machine-parseable: '{req['lead_time'][:120]}'"}
        deadline = dep - timedelta(days=lead)
        return {"icao": req["icao"], "lead_days": lead,
                "file_by": deadline.isoformat(),
                "feasible": deadline >= self.today}

    # ---------------------- overflight ----------------------------
    @staticmethod
    def _gc_point(lat1, lon1, lat2, lon2, f):
        """Point at fraction f along the great circle 1->2."""
        d = haversine_km(lat1, lon1, lat2, lon2) / EARTH_R
        if d == 0:
            return lat1, lon1
        a = math.sin((1 - f) * d) / math.sin(d)
        b = math.sin(f * d) / math.sin(d)
        x = a * math.cos(_rad(lat1)) * math.cos(_rad(lon1)) + b * math.cos(_rad(lat2)) * math.cos(_rad(lon2))
        y = a * math.cos(_rad(lat1)) * math.sin(_rad(lon1)) + b * math.cos(_rad(lat2)) * math.sin(_rad(lon2))
        z = a * math.sin(_rad(lat1)) + b * math.sin(_rad(lat2))
        return math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x))

    def countries_crossed(self, icao_a, icao_b, sample_km=200, proximity_km=300):
        """APPROXIMATE countries overflown on leg a->b, inferred by sampling the
        great circle and taking the country of the nearest verified airport within
        proximity_km. Ocean gaps -> 'OCEANIC/UNKNOWN'. This is a proxy until you
        ingest an authoritative boundary dataset - treat as an uncertainty source."""
        A, B = self.airport_info(icao_a), self.airport_info(icao_b)
        if "error" in A or "error" in B:
            return {"error": A.get("error") or B.get("error")}
        total = haversine_km(A["lat"], A["lon"], B["lat"], B["lon"])
        n = max(2, int(total // sample_km))
        seq, gaps = [], 0
        for i in range(n + 1):
            lat, lon = self._gc_point(A["lat"], A["lon"], B["lat"], B["lon"], i / n)
            best, bd = None, proximity_km
            for ap in self.airports.values():
                d = haversine_km(lat, lon, ap["lat"], ap["lon"])
                if d < bd:
                    best, bd = ap["country"], d
            tag = best or "OCEANIC/UNKNOWN"
            if tag == "OCEANIC/UNKNOWN":
                gaps += 1
            if not seq or seq[-1] != tag:
                seq.append(tag)
        return {"leg": f"{A['icao']}->{B['icao']}", "countries": seq,
                "method": "nearest-verified-airport proxy (approximate)",
                "unresolved_samples": gaps}

    def check_overflight(self, icao_a, icao_b, departure_date):
        """FCG overflight verdict for every (approximate) country on the leg."""
        crossed = self.countries_crossed(icao_a, icao_b)
        if "error" in crossed:
            return crossed
        dep = date.fromisoformat(departure_date)
        results = []
        for c in crossed["countries"]:
            if c == "OCEANIC/UNKNOWN":
                results.append({"country": c, "status": "no verified data along segment"})
                continue
            name = self.country_names.get(c, c)
            rec = next((r for r in self.fcg
                        if name.lower() in r["key"].replace("_", " ").lower()), None)
            if rec is None:
                results.append({"country": name, "status": "NO FCG DATA - unverified overflight"})
                continue
            of = (rec.get("overflight") or "NA")
            entry = {"country": name, "overflight": of[:200]}
            low = of.lower()
            if any(w in low for w in ("prohibit", "suspend", "not being granted",
                                      "not permitted", "closed", "denied")):
                entry["status"] = "PROHIBITED"
            elif of.strip().upper() == "NA":
                entry["status"] = "unknown - flag as uncertainty"
            else:
                lead = rec.get("overflight_lead_days")
                if lead is not None:
                    dl = dep - timedelta(days=lead)
                    entry["status"] = "feasible" if dl >= self.today else f"lead unmeetable (file_by {dl})"
                    entry["file_by"] = dl.isoformat()
                else:
                    entry["status"] = "lead time unparsed - flag as uncertainty"
            results.append(entry)
        return {"leg": crossed["leg"], "results": results,
                "note": crossed["method"]}

    def airports_near_route(self, origin, dest, corridor_km=400, min_runway_ft=0, limit=25):
        """Verified candidates within the corridor, ordered along the route."""
        A, B = self.airport_info(origin), self.airport_info(dest)
        if "error" in A or "error" in B:
            return {"error": A.get("error") or B.get("error")}
        total = haversine_km(A["lat"], A["lon"], B["lat"], B["lon"])
        out = []
        for ap in self.airports.values():
            if ap["icao"] in (A["icao"], B["icao"]):
                continue
            if min_runway_ft and ap["runway_ft"] and ap["runway_ft"] < min_runway_ft:
                continue
            off, along = cross_track_km(ap["lat"], ap["lon"], A["lat"], A["lon"], B["lat"], B["lon"])
            if off <= corridor_km and 0 < along < total:
                out.append({"icao": ap["icao"], "name": ap["name"], "country": ap["country"],
                            "runway_ft": ap["runway_ft"], "off_track_km": round(off),
                            "along_km": round(along)})
        out.sort(key=lambda c: c["along_km"])
        return {"route_km": round(total), "candidates": out[:limit],
                "note": "runway_ft=0 means unknown, treat as an uncertainty" }

    # ------------------ independent validator ----------------------
    def validate_route(self, stops, origin, dest, departure_date,
                       max_leg_km, hazmat=False, min_runway_ft=0, strict=False):
        """Deterministic re-check of a full proposed route. Model output is
        NEVER accepted unless this passes.
        strict=False: unknowns (NA lead times, missing overflight data) are
                      WARNINGS - route can pass with flagged uncertainties.
        strict=True:  critical unknowns become BLOCKERS - a passing route is
                      fully verified end to end (lead times parseable, overflight
                      known for every non-oceanic crossing, HAZMAT rules known
                      if carrying). Non-critical NAs (hours, forms) stay warnings."""
        chain = [origin] + list(stops) + [dest]
        blockers, warnings, legs = [], [], []
        for a, b in zip(chain, chain[1:]):
            d = self.distance_km(a, b)
            if "error" in d:
                blockers.append(d["error"]); continue
            legs.append(d)
            if d["km"] > max_leg_km:
                blockers.append(f"leg {a}->{b} is {d['km']} km > max {max_leg_km} km")
        for s in stops:
            ap = self.airport_info(s)
            if "error" in ap:
                blockers.append(ap["error"]); continue
            if min_runway_ft:
                if ap["runway_ft"] == 0:
                    warnings.append(f"{s}: runway length unknown")
                elif ap["runway_ft"] < min_runway_ft:
                    blockers.append(f"{s}: runway {ap['runway_ft']} ft < {min_runway_ft} ft")
            req = self.fcg_requirements(s)
            if "fcg_source" not in req:
                blockers.append(f"{s}: no FCG data - cannot verify entry authorization")
                continue
            if not req["listed_entry_exit"]:
                blockers.append(f"{s}: not on {req['fcg_source']} authorized entry/exit list")
            lt = self.check_lead_time(s, departure_date)
            if lt.get("feasible") is False:
                blockers.append(f"{s}: clearance lead time unmeetable (file_by {lt['file_by']})")
            elif lt.get("feasible") is None:
                (blockers if strict else warnings).append(
                    f"{s}: {lt.get('uncertainty')}" + (" [strict]" if strict else ""))
            if hazmat:
                hz = (req["hazmat"] or "").lower()
                if any(w in hz for w in ("prohibit", "not permitted", "not allowed", "forbidden")):
                    blockers.append(f"{s}: HAZMAT prohibited ({req['fcg_source']})")
                elif hz.strip() in ("", "na"):
                    (blockers if strict else warnings).append(f"{s}: HAZMAT rules unknown")
            for fld in ("restrictions", "hours"):
                v = req.get(fld, "NA")
                if v and v.strip().upper() != "NA":
                    warnings.append(f"{s} {fld}: {v[:140]}")
        # ---- overflight along every leg (skip countries you land in;
        #      their landing clearance record already governs them) ----
        landing_keys, landing_countries = set(), set()
        for c in chain:
            ap = self.airports.get(c.upper())
            if ap:
                landing_countries.add(self.country_names.get(ap["country"], ap["country"]).lower())
                rec, _ = self._fcg_for(c, ap["country"])
                if rec:
                    landing_keys.add(rec["key"])
        for a, b in zip(chain, chain[1:]):
            of = self.check_overflight(a, b, departure_date)
            if "error" in of:
                warnings.append(f"overflight {a}->{b}: {of['error']}")
                continue
            for r in of["results"]:
                key = r.get("country", "?")
                if (key.lower() in landing_countries
                        or key in landing_keys
                        or any(key.lower() in k.replace("_", " ").lower() for k in landing_keys)):
                    continue
                st = r.get("status", "")
                if st == "PROHIBITED":
                    blockers.append(f"overflight {of['leg']}: {key} PROHIBITED")
                elif "unmeetable" in st:
                    blockers.append(f"overflight {of['leg']}: {key} {st}")
                elif st.startswith(("unknown", "NO FCG", "lead time unparsed")):
                    (blockers if strict else warnings).append(
                        f"overflight {of['leg']}: {key} - {st}")
                elif st.startswith("no verified"):
                    warnings.append(f"overflight {of['leg']}: {key} - {st}")  # oceanic: never blocks
        return {"ok": not blockers, "legs": legs,
                "blockers": blockers, "warnings": warnings}


# Names + JSON arg specs the agent is allowed to use (agent.py enforces this).
TOOL_SPECS = {
    "airport_info":        {"args": ["icao"]},
    "distance_km":         {"args": ["icao_a", "icao_b"]},
    "fcg_requirements":    {"args": ["icao"]},
    "check_lead_time":     {"args": ["icao", "departure_date"]},
    "airports_near_route": {"args": ["origin", "dest", "corridor_km", "min_runway_ft", "limit"]},
    "countries_crossed":   {"args": ["icao_a", "icao_b", "sample_km", "proximity_km"]},
    "check_overflight":    {"args": ["icao_a", "icao_b", "departure_date"]},
    "validate_route":      {"args": ["stops", "origin", "dest", "departure_date",
                                     "max_leg_km", "hazmat", "min_runway_ft", "strict"]},
}
