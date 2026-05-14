You're right — that was way too dense for four small boxes on one slide. Let me give you a version that actually fits. Each box gets ~2 lines max, then put the rich detail in a single dense "Why this works" footer band that runs full slide width.

---

## Slide 4 — What is new and why should it work?

---

### Title

**What is new and why should it work?**

---

### Headline (one line, bold, under title)

**We fit the same physics-informed model to real data and to our simulation. The agents must recover the parameters we measured in the wild — or we don't run experiments.**

---

### Box 1 — Measure real discussions

Threaded community (Reddit / X / Telegram); stance classifier → per-user trajectories U[i,t]; reply graph G with weighted edges.

*Output:* longitudinal opinion data on a social graph — the canonical input for opinion-dynamics ODEs.

---

### Box 2 — Fit physics-informed model

PINN (Raissi 2019; SINN, Okawa & Iwata KDD 2022) jointly fits trajectories + Friedkin-Johnsen / Hegselmann-Krause / DCR equations.

*Output:* interpretable coefficients with 95% CIs — α (susceptibility), ε (bounded-confidence radius), w (homophily), D/v/R (spread, drift, reinforcement).

---

### Box 3 — Calibrate synthetic community

~500 LLM agents with personas + network positions. Re-fit PINN to sim trajectories; pass three gates.

*Output:* audit certificate when (1) coefficients land in real CIs, (2) cascade/drop-out/leader-rank match, (3) ChangeMyView Δ ≥ 70%.

---

### Gate annotation (between Boxes 3 and 4, small italic)

***No calibration certificate, no experiment.***

---

### Box 4 — Run red/blue experiments

Hackenburg et al. (*Science* 2025) lever sweep, replicated inside the chamber with networked propagation. Each tactic paired with an inoculation defense.

*Output:* ranked lever-effect matrix + matched defense effectiveness, with cascade and backlash measured.

---

### Footer band — "Why this works" (full slide width, light gray box, smaller font)

α and ε are not invented for this project — they are the susceptibility and bounded-confidence parameters from **Friedkin & Johnsen (1990, 1999)** and **Hegselmann & Krause (2002)**, with decades of empirical calibration (Friedkin et al., Lorenz-Spreen et al.). The homophily asymmetry w follows **McPherson et al. (2001)** and Bail et al. The PINN approach is established (**Raissi, Perdikaris & Karniadakis, *JCP* 2019**) and has been applied to opinion dynamics specifically (**SINN, KDD 2022**; **OPINN, 2026**). Persuasion ground truth uses **Tan et al.'s ChangeMyView corpus (WWW 2016)**; inoculation defenses follow **Roozenbeek & van der Linden (*Sci. Adv.* 2022)**.

**A closed loop:** real data produce parameters with established sociological meaning → those parameters audit the agents → experiments only run after the audit passes.

---

## How this fits on one slide

- **Top 10%:** title + headline
- **Middle 55%:** four boxes in a row, same size as your co-author's current diagram
- **Bottom 25%:** the gray "Why this works" footer with citations
- **Footer line:** page number + Lincoln Lab tag

Each box is now down to ~3 short lines. The citations and grounding moved to the bottom band, where they read as a unified scientific-rigor argument instead of cluttering every box.

---

## One honest note again

Same caveat as before — Friedkin & Johnsen, Hegselmann & Krause, Raissi et al., Okawa & Iwata, Tan et al., Hackenburg et al., Roozenbeek & van der Linden, McPherson et al. are load-bearing and verifiable. The OPINN 2026 citation, the Lorenz-Spreen attribution, and the Bail reference I cited from memory — verify them on Google Scholar before pasting, or drop the year and just cite by author if you're short on time.

If even this still overflows your boxes, the first cut is the parenthetical author lists inside each box's *Output* line — keep them in the footer only.
