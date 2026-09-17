**

The 300 NM clearance penalty is a design parameter, not a measured constant. At ~430 kt it equals about 40 minutes of flying — the detour a planner would plausibly accept to avoid one clearance request (APACS submission, lead-time exposure, embassy coordination, risk of refusal). The penalty sweep shows sensitivity to this choice; it can be calibrated by fitting to routes human planners actually chose.

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
