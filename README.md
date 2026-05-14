Here you go. Read this once tonight, sleep, skim it once more in the morning. Q&A section at the end covers the questions you'll actually get hit with.

---

# CALIBER — Technical Reference for Q&A
*Brief: approach, PINN methodology, calibration logic, common reviewer questions*

---

## 1. The Approach in Technical Terms

CALIBER is a closed measurement loop for online influence research. The loop has three legs:

**Leg 1 — Measurement.** We fit a physics-informed neural network (SINN, Okawa & Iwata, KDD 2022) to a rights-cleared threaded corpus. SINN simultaneously learns a neural surrogate for opinion trajectories `u_θ(t, i)` and the coefficients of a chosen opinion-dynamics ODE. We fit three priors independently — Friedkin-Johnsen, Hegselmann-Krause, and a diffusion-convection-reaction (DCR) variant — and report the coefficient ensemble. Each coefficient comes with a 95% confidence interval from block-bootstrap resampling. This coefficient set is the community fingerprint.

**Leg 2 — Synthetic construction.** We build a population of approximately 500 LLM-driven agents with personas, network positions, and reply policies sampled from the empirical distributions of the real community. The agents simulate the same time horizon as the real corpus (typically 24 weeks). We then re-fit SINN to the simulated trajectories using identical architecture and hyperparameters.

**Leg 3 — Calibration gate.** The recovered coefficients from the simulated community must fall inside the real-community 95% CIs across all six parameters (α, ε, w_in/w_out, D, v, R), plus pass five behavioral gates the loss did not directly optimize for, plus match the human rate of ChangeMyView Δ-awards at ≥70% on held-out cases. Failure at any layer halts experimentation. Thresholds are pre-registered before the synthetic runs begin.

Once the gate passes, persuader agents are inserted into the calibrated community across a factorial sweep — varying post-training method, rhetorical strategy, and network position. The primary measurements are shifts in the same coefficients used for calibration: Δε, Δα, Δw. Effective tactics are paired with prebunking, friction injection, and structural rewiring defenses scored by their reduction in induced parameter shifts.

The thesis is: **measure the real community, build a bounded scale model, prove the model is calibrated against the measurement, and only then use it to compare attacks and defenses.**

---

## 2. How the PINN Fit Actually Works

### 2.1 Inputs

- **Stance matrix `U[i, t]`**, shape (N users × T weeks). Each entry is user i's stance at week t, a scalar in [-1, +1]. We extract this by running a fine-tuned RoBERTa stance classifier on every post and averaging per (user, week).
- **Directed reply graph `G = (V, E, W)`** with edge weights `w_ij = log(1 + reply_count_ij)`. Nodes are users with ≥5 posts in the window; edges go from replier to replied-to.
- **Side information per user** — embedded post histories from a frozen language model, fed as auxiliary features.

### 2.2 Architecture

A neural surrogate `u_θ(t, i)` is parameterized as a feedforward network with 5 hidden layers, hidden dimension 8 (per the SINN reference implementation). The influence matrix is low-rank factorized as `W ≈ M^T Q` with M, Q each of size N × K and K = 16. This is the trick that makes the method tractable at Reddit scale — without factorization the full influence matrix is N² entries (16M for our population), which is infeasible.

### 2.3 Objective

The loss has two simultaneous terms:

```
L = λ_data · L_data + λ_phys · L_physics
```

- **L_data** = mean over observed (i, t) of `(u_θ(t, i) − U[i, t])²` — the neural surrogate matches observed stance.
- **L_physics** = mean over collocation points of `(∂u_θ/∂t − F_φ(u_θ, G))²` — the predicted rate of change must satisfy the chosen ODE `F_φ`.

`F_φ` is one of three priors fit independently:

**Friedkin-Johnsen:**
```
du_i/dt = α_i · (s_i − u_i) + Σ_j w_ij · (u_j − u_i)
```

**Hegselmann-Krause (bounded confidence):**
```
du_i/dt = (1/|N_i|) Σ_{j ∈ N_i} (u_j − u_i)
where N_i = {j : |u_j − u_i| ≤ ε_i}
```

**Diffusion-Convection-Reaction (Gong et al.):**
```
du/dt = −D · L_G · u + v · B · u + R · u · (1 − u²)
```
on the graph Laplacian.

The coefficients α, ε, w, D, v, R are themselves learnable parameters of `F_φ`, optimized jointly with the network weights `θ`. Stochastic interaction mechanisms (the discrete choice of which neighbor to attend to at each step) are handled via Gumbel-Softmax reparameterization, which makes the categorical sampling differentiable end-to-end.

### 2.4 Critical point: no ODE solver in the loop

The neural network outputs `u_θ(t, i)` directly. The time-derivative is computed by automatic differentiation through the network. The ODE is enforced as a pointwise loss term at collocation points — it is not integrated forward in time. This is the standard PINN approach (Raissi, Perdikaris & Karniadakis, JCP 2019). The "physics" comes from the constraint, not from numerical integration. If a reviewer asks about Neural-ODEs, that's a different architecture (Chen et al., NeurIPS 2018) — we are not using it.

### 2.5 Why three priors

Different communities follow different rules. Friedkin-Johnsen captures susceptibility-and-anchoring dynamics. Hegselmann-Krause captures bounded-confidence (echo-chamber) dynamics. DCR captures graph-PDE-style spread-and-reinforcement. Fitting all three and reporting the ensemble protects against committing to one mechanism and getting fooled. The prior that fits a given community best is itself an informative finding.

---

## 3. Interpretation of the Coefficients (the "Echo-Chamber Verbiage")

The coefficient names map to operational meanings as follows.

**α (susceptibility / stubbornness):** Per-user. The weight a user places on their innate prior relative to incoming social influence. Low α means a user drifts with whoever they read most recently. High α means they barely move. Operationally: identifies who is becoming harder to move.

**ε (bounded-confidence radius):** Per-user or per-community. The maximum opinion distance at which a user still incorporates a neighbor's view. Small ε means the user only listens to people who already largely agree with them. **This is the direct quantitative measure of echo-chamber tightness in the bounded-confidence literature** (Hegselmann & Krause, 2002). A community-level mean ε that shrinks over time is the strongest signal of an emerging echo chamber.

**w_ij (influence weight):** Per-edge, low-rank factorized. The actual learned influence user i grants user j — distinct from the raw reply count. Used to compute w_in/w_out, the asymmetry between in-group and out-group influence, which operationalizes homophily (McPherson et al., 2001).

**D (diffusion rate):** Community-level. From the DCR formulation. Captures how fast opinions spread across the graph independent of directional forcing.

**v (convection / external push):** Community-level. Captures whether the community is being systematically pushed in one direction by an external force — a coordinated campaign, a news shock, an institutional intervention. A non-zero v during a known event window is the fingerprint of exogenous influence.

**R (reaction / extremism reinforcement):** Community-level. Captures self-reinforcement of extreme opinions — the tendency of opinions near the bounds (±1) to harden further over time even without new input. High R communities polarize toward the edges.

**Together,** these six parameters quantitatively define the community's information-dynamics regime. They have decades of empirical use in opinion-dynamics research (Friedkin & Johnsen 1990 onward) and were not invented for this project. That matters for defensibility: when we say "we measure echo-chamber tightness," we mean a parameter that has been studied for forty years.

---

## 4. The Calibration Procedure

After the LLM-agent simulation produces synthetic data, calibration proceeds in three layers.

**Layer 1 — Coefficient match.** Re-fit SINN to simulated trajectories with identical architecture. Compare:
- Community-level coefficients (ε, D, v, R): each must fall inside real-data 95% CI.
- Per-user distributions (α_i, ε_i): Wasserstein-1 distance between simulated and real CDFs must be below pre-registered threshold (e.g., W_1 < 0.05).
- Graph summary statistics (degree distribution, clustering coefficient, top-k Laplacian eigenvalues): each within 10% of real value.

**Layer 2 — Behavioral gates.**
- Cascade timing — KS test on time-to-half-reach distributions, p > 0.05.
- Drop-out rates — fraction of users halting posting within two weeks of hostile reply, distributions matched.
- Sentiment trajectories — normalized dynamic time warping distance below threshold.
- Cross-topic transfer — coefficient shifts on a held-out topic must directionally match real shifts.
- Opinion-leader centrality — Spearman ρ ≥ 0.5 between sim and real top-k by attention received.

**Layer 3 — ChangeMyView Δ ground truth.** Initialize 50 simulated CMV-OP agents with the real human OP's stance and reasoning. Replay the actual human comments. Simulated agents must grant Δ-awards at ≥70% agreement with the human OPs' actual awards.

Failure at any layer means re-tuning agent population parameters (persona consistency, memory window, reply policy, network position assignment) and re-running. Three to five iterations is typical before all gates pass.

---

## 5. Probable Reviewer Questions and Answers

### Q1. What if you can't get the simulated agents to match real-world priors?

A null calibration result is itself informative and was anticipated in the design.

If Layer 1 fails on a specific coefficient — say, simulated ⟨α⟩ is systematically lower than real ⟨α⟩ — that points to a specific defect: agents are less stubborn than real users. The fix is mechanical: strengthen the persona system prompt, lengthen the memory window so agents have more persistent identity, weight the innate-prior posts more heavily in the agent context. Re-run.

If after three to five tuning iterations calibration still cannot pass, that is reported as the program outcome for Q2-Q3. We do not move to the persuasion sweep with an uncalibrated community. The conclusion would be: LLM agents at current capability cannot reproduce the dynamics of this specific community type within the tolerance bands required for trustworthy experimentation. That is a publishable negative finding — and it is exactly the kind of finding that has to precede any operational investment. Better to discover this in a research line item than after a sponsor builds infrastructure on top of it.

### Q2. Why use a PINN at all instead of just fitting an ODE directly to the data?

Direct ODE fitting cannot handle the non-stationarity, missing data, and partial observability of real Reddit threads. Users go silent for weeks; stance scores are noisy classifier outputs, not ground truth; the graph evolves. A pure ODE solver requires clean continuous trajectories.

A pure neural network fits the data well but recovers nothing interpretable. You cannot audit a synthetic community against the weights of a black-box network.

The PINN combines both: it fits the messy real data through the neural surrogate while imposing the ODE as a soft constraint via the loss term. After training you have a network that predicts well AND interpretable coefficients with established sociological meaning. SINN demonstrated this works on real Twitter data three years ago.

### Q3. The Reddit data is a surrogate. How do you know findings transfer to operational platforms?

We don't claim they will transfer cleanly. The Reddit corpus is a development surrogate, chosen for legal accessibility and labeling density. Operational transfer is itself a measured outcome in Q3-Q4 — we do an initial transfer check on mission-relevant multimodal data. If transfer is poor, that is a finding. The closed-loop calibration methodology, however, ports independently of the specific corpus: the same PINN-fit, same audit gates, same coefficient-based experiment scoring would apply to any threaded discussion data.

The platform-mismatch risk is real and disclosed up front. Threaded forums and recommendation-algorithm-driven platforms (TikTok, short-video) have different propagation dynamics. We address this by keeping the Q4 transfer test bounded — we are not claiming Reddit findings predict TikTok outcomes.

### Q4. Are you fitting ODEs or using ODE solvers in the training loop?

No solver in the loop. The neural surrogate outputs u_θ(t, i) directly. The time-derivative comes from autodiff. The ODE is enforced as a pointwise loss term at collocation points, not integrated forward. This is the standard PINN approach since Raissi et al. 2019. If you are thinking of Neural-ODEs (Chen et al. 2018), that is a different architecture where du/dt is parameterized directly and integration happens during training — we are not using that.

### Q5. Why three opinion-dynamics priors instead of one?

Different communities follow different mechanisms. Friedkin-Johnsen captures anchoring; Hegselmann-Krause captures bounded confidence; DCR captures graph-PDE spread. Fitting all three protects against model misspecification. Which prior fits a community best is itself an informative finding — for r/Brexit we hypothesize HK will dominate given the polarized topic, but we let the data decide.

### Q6. How does CALIBER differ from BRIES (Volkova, Aptima)?

BRIES is a compound-AI architecture for persuasion attack/defense. It is the closest operational competitor. Its known limitation, which Volkova herself has flagged publicly, is that the audience layer is not validated against real-world diffusion data — the agents behave plausibly but their population-level dynamics are not audited.

CALIBER fills exactly that gap. The transition target is BRIES with CALIBER's calibrated audience layer underneath. The two are complementary, not competing.

### Q7. What about overfitting to the calibration corpus?

Three structural defenses. First, cross-topic transfer is one of the five behavioral gates — agents calibrated on one topic must produce directionally correct coefficient shifts on a held-out topic in the same community. Second, the persuasion experiment uses a held-out time period the calibration did not see. Third, the ChangeMyView ground truth comes from an entirely separate corpus from a different community. An overfit calibration that passes Layer 1 will fail Layer 3.

### Q8. What is the dual-use risk?

Real and disclosed. The persuasion sweep, by design, identifies which LLM-training/rhetorical-style/network-position combinations produce the largest attitude shifts in a calibrated chamber. Two mitigations. First, the published artifact is the *measurement protocol* and the *defense library*, not a deployable persuader — we publish what works and what stops it together. Second, the work stays in a human-gated sandbox with IRB review on the persuasion experiments. The defense library is the primary deliverable from a sponsor-value perspective.

### Q9. How do you know the LLM-driven agents are not just memorizing patterns from their pretraining?

Two checks. First, the personas are sampled from empirical distributions of the specific community, not generic templates — the agents have to act as specific kinds of Reddit users, not as a generic "social media user." Second, the calibration gate measures behavior at the population level, not at the individual-utterance level. An agent that produces fluent text but does not contribute to the right coefficient distributions fails Layer 1. The Layer 3 CMV check specifically measures persuasion responses on cases the underlying LLM may have seen in pretraining — if memorization drives the answers, the agents will agree with each other more than with the real OPs, and the 70% threshold will catch it.

### Q10. Why $775K? What does the money buy?

Twelve months of personnel — PI fraction, two ML engineers, RA support for data labeling and graph construction. IRB review fees. Data acquisition costs (Pushshift archives, annotation labor for the held-out CMV slice). No external API spend — compute uses open-weight LLMs (Llama-3, Qwen, Mistral) on existing Lab GPU resources. The cost is dominated by people-time, which reflects that the technical hard part is calibration, not compute.

### Q11. What is the single most important success metric for Year 1?

Whether the three-layer calibration gate passes on at least one corpus with pre-registered thresholds. If it does, CALIBER has demonstrated a validated synthetic experimentation environment, which no current LLM-agent work can claim. If it does not, we have a published negative result with the parameter-fit data showing exactly where the gap is, which informs the next round of work and is itself a contribution to the field.

---

## 6. The One-Sentence Defense

If asked at any point what makes CALIBER fundamentally different: *the same physics-informed measurement instrument that fits real-world community dynamics is used to audit our synthetic community and to score every experimental intervention inside it — measurement, calibration, and evaluation are a single closed loop, not three separate methodologies.*

---

That should cover you. The questions in Section 5 are the ones most likely to actually come up. If you get something off-script, fall back to the one-sentence defense in Section 6 and work outward from there.

Sleep well. You're prepared.
