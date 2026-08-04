Go ahead with the probe, with these changes:
1. Don't build an ICAO-prefix table. Map ICAO→country via OurAirports
   airports.csv (ident → iso_country alpha-2) then alpha-2→alpha-3 via
   countries_codes_and_coordinates.csv — same join country_route_planner.py
   already does. Copy those two files into the repo if not present.
2. For udl_notifications: slice per country by the exact ICAO idents
   belonging to that country (from the airports.csv join), not FIR prefix.
   Also note the newest timestamp per country — teammate says this data is
   very old, so each dossier slice should be labeled with its date range.
3. Probe output I want to see: for CHE, list its ICAOs found in each of the
   three sources, with counts.
