**Slide 4 — "How it works" — add the data source to each bullet:**

Deterministic code:
- Great-circle geometry — *airport coordinates from OurAirports (public database, ~47k airports); country centroids from ISO 3166 reference data*
- Exact overflight detection against country borders — *Natural Earth admin-0 boundaries (public domain)*
- Dijkstra over countries as stepping stones — *same border and centroid data; clearance denials from the FCG judge; geopolitical avoid list (manual + US State Dept travel advisories)*
- Dijkstra over designated airports for fuel stops — *entry/exit airports extracted from the FCG; aircraft range from an aircraft specification table*
- Dynamic program for the best entry airport per stop — *FCG designated-airport lists; OurAirports coordinates*
- Lead-time arithmetic — *minimum lead days extracted from FCG text by the judge; mission date from the user*

Local LLM:
- Resolves user input — *ISO country codes, FCG country list, OurAirports ICAO codes*
- Reads each country's raw FCG text — *the Foreign Clearance Guide extract (country-level rows: overflight, diplomatic clearance lead time, entry/exit airfields, customs, HazMat, operating hours, NOTAMs)*
- Fuel and payment notes — *DLA Energy standard fuel prices (public) + FCG AIR Card / cash fields*

One-line footer: *Live layers (optional): OpenSky Network for current air traffic, aviationweather.gov for METAR/TAF — snapshots, not forecasts.*

---

**Slide 22 — "Is the FCG judge trustworthy?" — replace the body:**

**Short answer: not yet proven. Measurement is set up; labeling is in progress.**

The judge is the one component that reads text and makes a call, so it is the one component whose error rate must be measured against a human. We have generated 120 test cases (a country, a role — overflight or landing — and its FCG text). For each, the model's verdict and extracted lead time were recorded three times. A human now labels the same cases without seeing the model's answer. Results will be reported here as the four numbers below.

Until those numbers exist, the verdicts should be treated as a first draft for a human clearance officer to confirm — which is how the system is designed to be used.

**Footnote (define the metrics):**
- *Accuracy* — the share of cases where the model's allow/deny matches the human's.
- *Cohen's kappa* — agreement corrected for chance; 0.6–0.8 is substantial, above 0.8 near-perfect.
- *Deny precision / recall* — when the model says "deny," how often it is right / how many true denials it catches. Recall matters most: a missed denial is a route through prohibited airspace.
- *Lead-time exact match* — the number of days the model read from the text equals what the human read.
- *Self-consistency* — the model gives the same verdict on three repeated runs (stability).

---

If you rename globally: replace "the planner" / "country route planner" with **Agentic Route Planning** on the title and section headers; in body text "the planner" still reads naturally as the system's short name.
