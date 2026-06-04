#!/usr/bin/env python3
"""GENERATE mode — build the shortest feasible, fuel-optimized route + checklist.

Example
-------
python scripts/plan_route.py --from KSUU --to ETAR --aircraft C17 \
    --lead-days 14 --hazmat
"""
import argparse
from _common import load_airports, resolve, build_engine
from flight_planner.schema import FlightQuery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="origin", required=True, help="ICAO or lat,lon")
    ap.add_argument("--to", dest="dest", required=True, help="ICAO or lat,lon")
    ap.add_argument("--aircraft", default="C17")
    ap.add_argument("--lead-days", type=int, default=None,
                    help="days available before departure (for dip-clearance check)")
    ap.add_argument("--hazmat", action="store_true")
    ap.add_argument("--openap", action="store_true", help="use openap-top for fuel")
    ap.add_argument("--wind-grib", default=None)
    ap.add_argument("--geojson", default=None)
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    airports = load_airports()
    q = FlightQuery(
        origin=resolve(args.origin, airports),
        destination=resolve(args.dest, airports),
        aircraft=args.aircraft, lead_days_available=args.lead_days,
        carrying_hazmat=args.hazmat,
    )
    eng = build_engine(args.geojson, args.db, use_openap=args.openap, wind_grib=args.wind_grib)
    print(eng.plan(q).summary())


if __name__ == "__main__":
    main()
