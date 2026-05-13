Got it. Here are detailed inset rewrites for all four boxes — each one swaps the current vague text for something grounded in numbers, parameters, or concrete artifacts. Pen-ready.

---

**Box 1 — Measure real discussions**

*Cross out:* posts, stance, replies, narrative traces

*Write in:*

> 6 months of r/Brexit (≈100k posts, ≈4,000 active users)
> → stance trajectories U[i,t] in [−1, +1]
> → directed reply graph with weights w_ij = log(1 + replies)
> → shock events flagged (referendum, leadership changes)

This grounds it in *which* community, *how much* data, and *what specific structures* come out.

---

**Box 2 — Fit behavior model**

*Cross out:* reach | skepticism | switching | hardening

*Write in:*

> Network-coupled PINN extracts the community fingerprint:
> α stubbornness · ε chamber tightness · w_in/w_out group asymmetry
> D diffusion · v external push · R extremism reinforcement
> Each with 95% CI from bootstrap resampling.

Now Box 2 reads as real statistics with named parameters and confidence intervals — the heaviest-lifting box gets the credit it deserves.

---

**Box 3 — Calibrate synthetic community**

*Cross out:* 500 agents with behavioral roles and uncertainty bands

*Write in:*

> 500 LLM-driven agents (persona, network position, reply policy)
> Calibration must pass three independent checks:
> (1) fingerprint parameters land inside real-data 95% CIs
> (2) cascade timing, drop-out, and opinion-leader rankings match
> (3) agents grant ChangeMyView Δ-awards at ≥ 70% human rate

This converts "calibrate" from a wave-hand into a concrete pass/fail gate with three named criteria. Reviewers can see *exactly* what calibration means.

---

**Box 4 — Run red/blue experiments**

*Cross out:* persuasion tactics paired with defenses

*Write in:*

> Example outcome — info-dense persuader at network bridge:
> ΔAttitude +0.18, cascade reach 38%, factual accuracy 81%
> (9× more persuasive than baseline; 4× higher false-claim rate)
> Pre-bunking defense neutralizes 39% of the effect.

This is the single highest-leverage edit in the whole deck. It turns Box 4 from "we'll run experiments" into "here's what one experiment produces." A reviewer who reads only this box now knows exactly what the program outputs.

---

**One more line, if you have room:**

Between Box 3 and Box 4, in the gap, write small:

> ***Gate: no calibration certificate, no experiment.***

This visually emphasizes the closed-loop thesis your co-author already has in the footer.

---

That's all four boxes plus the gate annotation. Five pen edits total. After this, slide 4 stops being the weakest slide in the deck and becomes the most concrete one.
