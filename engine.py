"""The engine. ONE pipeline, two entry points:

  plan(query)   -> GENERATE: build the shortest feasible fuel-optimal route, verify.
  verify(query) -> VERIFY:   score a PROPOSED route; if infeasible, fall through to
                             generate and offer the optimized alternative.

This is written as an explicit state machine so the control flow is obvious and
testable. It maps 1:1 onto LangGraph nodes (see README) when you want the LLM to
own the extract/justify nodes and durable checkpoints.
"""
from __future__ import annotations

from ..schema import FlightQuery, Route, PlanResult
from ..geo import path_length_km
from ..airspace.base import AirspaceProvider
from ..constraints.base import FCGConstraintProvider
from ..objectives.base import Objective, FeasibilityConstraint, FuelCost
from ..routing.router import plan_lateral
from ..routing.optimize import estimate_fuel_kg
from ..checklist.verify import verify as run_checklist


class Engine:
    def __init__(self, airspace: AirspaceProvider, constraints: FCGConstraintProvider,
                 objectives: list[Objective] | None = None,
                 use_openap: bool = False, wind_grib: str | None = None):
        self.airspace = airspace
        self.constraints = constraints
        self.objectives = objectives or [FeasibilityConstraint(), FuelCost(weight=1.0)]
        self.use_openap = use_openap
        self.wind_grib = wind_grib

    # ---- shared steps -------------------------------------------------------
    def _resolve_constraints(self, ids: list[str]) -> dict:
        return {rid: c for rid in ids
                if (c := self.constraints.constraints_for(rid)) is not None}

    def _score(self, route: Route, ids: list[str], cons: dict) -> tuple[bool, float]:
        feasible = True
        for obj in self.objectives:
            v = obj.cost(route, cons, ids)
            if obj.hard and v == float("inf"):
                feasible = False
        fuel = estimate_fuel_kg(route, use_openap=self.use_openap, wind_grib=self.wind_grib)
        return feasible, fuel

    def _finalize(self, query: FlightQuery, route: Route, forbidden_hit: list[str],
                  messages: list[str]) -> PlanResult:
        ids = self.airspace.region_ids_crossed(route.waypoints)
        cons = self._resolve_constraints(ids)
        feasible, fuel = self._score(route, ids, cons)
        checklist = run_checklist(query, ids, cons)
        return PlanResult(
            feasible=feasible, route=route, countries_overflown=ids,
            forbidden_encountered=forbidden_hit,
            distance_km=path_length_km(route.waypoints), fuel_kg=fuel,
            checklist=checklist, messages=messages,
        )

    # ---- GENERATE -----------------------------------------------------------
    def plan(self, query: FlightQuery) -> PlanResult:
        messages: list[str] = []
        # forbidden set is known up-front from the pre-extracted DB (no LLM in loop)
        forbidden_ids = self.constraints.forbidden_overflight_ids()
        forbidden_rings = self.airspace.rings_for(forbidden_ids)
        wps = plan_lateral(query.origin, query.destination, forbidden_rings)
        if wps is None:
            messages.append("No feasible route: origin/destination in forbidden "
                            "airspace or fully enclosed.")
            route = Route([query.origin, query.destination], query.aircraft)
            res = self._finalize(query, route, sorted(forbidden_ids), messages)
            res.feasible = False
            return res
        route = Route(wps, query.aircraft)
        # which forbidden regions did we actually route around?
        direct_ids = set(self.airspace.region_ids_crossed([query.origin, query.destination]))
        avoided = sorted(forbidden_ids & direct_ids)
        if avoided:
            messages.append(f"Re-routed around forbidden airspace: {', '.join(avoided)}")
        return self._finalize(query, route, avoided, messages)

    # ---- VERIFY -------------------------------------------------------------
    def verify(self, query: FlightQuery) -> PlanResult:
        assert query.proposed_waypoints, "verify() needs query.proposed_waypoints"
        messages: list[str] = []
        route = Route(query.proposed_waypoints, query.aircraft)
        res = self._finalize(query, route, [], messages)
        if not res.feasible:
            res.messages.append("Proposed route is INFEASIBLE — generating optimized "
                                "alternative:")
            alt = self.plan(query)
            res.messages.append(alt.summary())
        return res
