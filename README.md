Don't write country_index.py. Instead write and run ONE small probe that
prints a precise format spec I can hand to another developer:
1. One complete MEIS airfield object (all keys, values truncated to 80
   chars) and where its ICAO ident lives.
2. The SVC_RMK header row + 2 full data rows.
3. Two complete NOTAM blocks verbatim.
4. The alpha2->alpha3 join columns you used from the two CSVs.
Keep it under 100 lines of output.
