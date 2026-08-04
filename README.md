Session crashed and restarted — context is gone, so re-derive what you need
from files in scratch/ and data/. Prior decisions, all confirmed: broad
all-airport-types ident set; two-tier NOTAM matching (exact ident OR
prefix derived from the country's own idents, scope=FIR tag); NOTAM=0 for
all 7 countries is verified ground truth (feed is US/Germany only) and
accepted — keep the slicer wired for future batches. Now build
pipeline/country_index.py as you proposed and show me the 7-country table:
MEIS records / SVC rows / NOTAM (expected 0) / date range per country.
