Good probe. Decisions:
1. Keep the broad airport-type ident set for slicing (don't match the
   planner's filter).
2. NOTAM=0 is a matching gap, not ground truth: NOTAMs are often filed
   against the FIR (LSAS), not an airfield. Match NOTAMs by exact ident
   OR by 2-letter prefix derived from that country's own ident set
   (CHE -> {LS}). Mark prefix-matched blocks as scope=FIR in the slice.
   Exception: for USA don't prefix-match on single-K ICAOs' prefixes
   beyond K/PA/PH etc as derived — derive strictly from the ident set,
   never hand-code.
3. Then build country_index.py as you proposed and show me the 7-country
   table: MEIS records / SVC rows / NOTAM blocks (exact vs FIR) / date
   range per country. Go.
