Let me pull the paper and read it carefully before tying it to your work.

None of those are the paper. Let me try fetching it directly through a different path.

Got it — this is the SINN paper. I've been referencing it throughout your work but only mentioning it as a citation. Let me get the actual content.

Now I have a thorough understanding. Let me also pull a bit more on the empirical results to ground my explanation.

I have enough now to write you a really thorough explanation. Let me put this together.

---

# SINN Paper (Okawa & Iwata, KDD 2022): Full Explanation + How It Ties to CALIBER

## Quick orientation

**Paper:** *Predicting Opinion Dynamics via Sociologically-Informed Neural Networks*
**Authors:** Maya Okawa, Tomoharu Iwata (NTT Communication Science Laboratories)
**Venue:** KDD 2022 (top-tier ML venue)
**Code:** github.com/mayaokawa/opinion_dynamics

This paper is the **direct intellectual ancestor of what you're proposing in CALIBER.** Your boss may not realize this yet, but every time you say "we fit a PINN to opinion data and recover interpretable coefficients," you're standing on the methodology this paper invented. Understanding it deeply will make your slide 4 bulletproof.

---

## The problem the paper sets up

Researchers studying opinion dynamics on social media had two camps that didn't talk to each other:

**Camp 1 — Theoretical sociologists.** They had clean, interpretable mathematical models of opinion change going back decades: the DeGroot model (1974), Friedkin-Johnsen (1990), Hegselmann-Krause (2002). These models say things like "your next opinion is a weighted average of your friends' opinions plus a pull toward your own innate baseline." Beautiful theory, but the parameters of these models (how stubborn each person is, how much they listen to whom) have to be calibrated by hand to real data, which is painful and small-scale.

**Camp 2 — Deep learning practitioners.** They had massive social media datasets and trained neural networks to predict opinion trajectories. These networks fit the data well but produced no interpretable parameters and respected no sociological theory. You couldn't tell *why* the network predicted what it did, or whether its predictions obeyed any plausible mechanism of human influence.

Okawa and Iwata's insight: **PINNs (physics-informed neural networks) solve exactly this gap in physics, so port the idea to sociology.**

---

## What a PINN does (the physics analogy)

In physics, a PINN is a neural network trained against two simultaneous losses:

1. **Data loss** — the network's predictions should match observations
2. **Physics loss** — the network's predictions should also satisfy a known differential equation (e.g., Navier-Stokes for fluids, Schrödinger for quantum)

The physics loss is computed at "collocation points" — places in space-time where you didn't observe data but where the equation should still hold. The neural network's outputs are differentiated via autodiff and compared against what the equation predicts. If the network violates the equation, gradient descent pushes it back into compliance.

You get the best of both worlds: a flexible learner that fits messy real data **and** respects known physical law everywhere.

---

## What SINN does (the sociology version)

Okawa & Iwata's move: replace "physics laws" with "sociology laws." The opinion dynamics models from Camp 1 above can all be written as ordinary differential equations (ODEs). So:

1. **Data loss** — neural network predicts user *i*'s opinion at time *t*, must match observed opinions
2. **Sociology loss** — the network's predicted opinion trajectory must obey one of the classical ODEs (DeGroot / Friedkin-Johnsen / Hegselmann-Krause)
3. **The ODE's coefficients are themselves learnable** — so the network simultaneously learns the trajectories AND discovers what coefficient values let the chosen sociology theory best explain the data

That third point is the genius of the paper. They don't just *use* the ODE as a constraint — they *learn its parameters jointly with the network*. After training, you have:

- A neural network that predicts opinions well, AND
- A set of fitted sociological coefficients (susceptibility, confidence radius, influence weights) with all the interpretability of the classical theory

---

## The three opinion-dynamics ODEs they ported

This is the bit you actually need to internalize because CALIBER uses these same three.

### Friedkin-Johnsen (FJ)

> Your next opinion = a weighted average of your neighbors' opinions, plus a pull toward your innate baseline (the opinion you started with).

In equation form:

du_i/dt = α_i · (s_i − u_i) + Σ_j w_ij · (u_j − u_i)

- **α_i** — user *i*'s susceptibility to social influence (how strongly they're pulled back to their innate prior)
- **s_i** — user *i*'s innate baseline opinion
- **w_ij** — how much user *i* listens to user *j*

SINN recovers all of these from data.

### Hegselmann-Krause Bounded Confidence Model (HK/BCM)

> You only listen to people whose opinion is close enough to yours. Everyone outside your "confidence radius" gets tuned out.

In equation form, du_i/dt averages only over neighbors *j* such that |u_j − u_i| < ε_i. The key parameter is:

- **ε_i** — user *i*'s confidence radius. Smaller ε → tighter echo chamber → only listens to like-minded folks.

This is the model that gives you **interpretable echo-chamber tightness directly as a fitted parameter.** That's exactly what your boss cares about for early echo-chamber detection.

### DeGroot

Simpler, no innate prior, no bounded confidence — just everyone averages their neighbors over and over. Useful as a baseline.

SINN fits all three and reports which one best explains a given dataset.

---

## The clever extensions in the paper

Okawa & Iwata didn't just plug ODEs into a PINN — they added three engineering contributions that matter for real social media data:

### 1. Matrix factorization for the influence matrix

The full influence matrix w_ij has N² entries, which is impossibly large for N = 100,000 users. They factor it: w_ij ≈ M_i^T · Q_j where M and Q are low-rank latent matrices of dimension K << N. This makes the model tractable.

### 2. Language-model side information

For each user they pull profile descriptions (Twitter bios) and embed them with a language model, then use those embeddings as features feeding into the neural network. So the network knows that "pro-life Christian mother" and "abortion-rights advocate" are different kinds of users *before* it sees their opinion trajectories. Significantly improves prediction quality.

### 3. Gumbel-Softmax for stochastic dynamics

Real opinion models often have stochastic interaction mechanisms (people randomly choose which neighbor to talk to). You can't backprop through a discrete random choice. They use the Gumbel-Softmax reparameterization trick to make the stochastic interaction differentiable so the whole thing trains end-to-end.

---

## How well does it actually work?

They tested SINN against six baselines on:

- **Synthetic datasets** generated from known opinion-dynamics models (consensus, clustering, polarization)
- **A real Twitter dataset** on the Abortion topic

Results:

- SINN beats every baseline on synthetic data — including pure neural network methods that have no sociology constraint
- On the real Twitter Abortion data, SINN better predicts the full distribution of opinions (e.g., correctly captures that there are many highly negative samples — pure NN underestimates this)
- The attention mechanism in SINN focuses on **meaningful words** in user bios ("Pro", "Liberty", "Freedom") while a pure NN focuses on **junk** ("https", "co", "year")

That last finding is the cherry on top: imposing the sociology constraint doesn't just give you interpretable parameters, it makes the network pay attention to the *right features* of the data.

---

## Now: how this ties to CALIBER

Every single piece of CALIBER you've been building rests on this paper. Let me map it explicitly.

### Tie-in 1: Step 4 of your pipeline (Fit PINN) — this IS SINN

Step 4 of CALIBER's 8-step flow is literally re-running SINN on a new corpus (r/Brexit, ChangeMyView) and recovering the same parameters they recovered (α, ε, w, plus DCR coefficients from Gong et al. 2026). When you say "Network-coupled PINN extracts α, ε, w, D/v/R," you're using SINN as the method.

**What this means for your proposal:** you can cite this paper as the methodological foundation. You don't have to invent PINNs for opinion dynamics — they exist, they work, they're published at KDD, the code is open source. You're applying an established method to a new question. That's a *much* easier sell to a Lincoln review committee than "we invented a new method."

### Tie-in 2: SINN proves the parameters are recoverable from real social media

A reasonable reviewer might ask: "Sure, you say you'll extract α, ε, w from Reddit data, but can you actually?" SINN's experiments on the Twitter Abortion dataset show that **yes, on a real, noisy social media corpus, the method recovers interpretable sociological parameters and beats pure-NN baselines.** That's a 3-year-old, peer-reviewed, code-published demonstration of feasibility. Your proposal isn't speculative — it's a known feasible step.

### Tie-in 3: SINN does NOT do what CALIBER does — and that's the novelty gap

This is the most important point. SINN stops at Step 4. It fits the PINN to real data, extracts coefficients, and predicts future opinions. **It does not build LLM-agent simulations. It does not calibrate agents. It does not run persuasion experiments. It does not pair tactics with defenses.**

The whole back half of CALIBER (Steps 5-8) is genuinely new. SINN gives you the measuring stick; CALIBER uses the measuring stick to audit synthetic communities and then run controlled experiments inside them. **That's the novel contribution you can defend.**

If a reviewer asks "what's new beyond SINN?" your answer is: SINN proved you can recover sociological parameters from real social media; CALIBER uses those parameters as the calibration target for an LLM-agent population and as the measurement instrument for persuasion experiments inside it. SINN measures the world; CALIBER measures the world *and* uses those measurements to validate a synthetic version of the world for controlled experimentation.

### Tie-in 4: The matrix factorization trick matters for your scale

Your r/Brexit corpus has ~4,000 active users. The influence matrix has 16M entries. You cannot fit a dense w_ij. You will need SINN's matrix-factorization trick (low-rank decomposition into M and Q matrices). When you write up methods, this is the engineering detail that makes the approach tractable at Reddit scale.

### Tie-in 5: The language-model side information ports naturally to your setting

SINN used Twitter bios as side information. You have Reddit user histories, post patterns, subreddit memberships. All of that can be embedded with a language model (the same kind of LLM you're using to build agents in Step 5) and fed into the PINN as auxiliary features. This is a free improvement that already has a published precedent.

---

## What to do with this knowledge

**Three concrete actions for your proposal and slides:**

1. **Add a single citation line wherever you mention PINN.** Something like: "Following the SINN architecture (Okawa & Iwata, KDD 2022), the network jointly fits opinion trajectories and learns the coefficients of a classical opinion-dynamics ODE (FJ / HK / DeGroot)." This anchors your method in established work.

2. **In your "What's new" section, position SINN explicitly as the prior work you're building on, then state your novel step.** Something like: "SINN demonstrated that sociological coefficients can be recovered from real social media via PINN. CALIBER extends this by using the recovered coefficients as the calibration target for LLM-agent simulations, enabling controlled persuasion experiments to be run inside an empirically validated synthetic community."

3. **When your co-author asks again "is an ODE solver involved?" — refer to the paper.** SINN does not use a solver. It uses the standard PINN approach: autodiff for ∂u/∂t, ODE enforced as pointwise loss. That answer is exactly correct because it's the answer the published method gives.

---

## One critical reality check

I want to be honest: I've been citing this paper from memory throughout our conversations, and I just now read it properly. The good news is that the way I described PINNs to your co-author (no solver, autodiff for derivatives, ODE enforced as loss term, coefficients learned jointly) is exactly what SINN does. The technical claims in your proposal hold up.

The version differences to flag:
- Okawa & Iwata use FJ, HK, and DeGroot. The DCR (Diffusion-Convection-Reaction) variant comes from a separate, later line of work (Gong et al., OPINN). If you cite DCR, cite OPINN; if you cite FJ/HK, cite SINN. Don't blur them.
- The "echo-chamber tightness" interpretation of ε (HK's confidence radius) is yours — SINN doesn't frame it that way explicitly. It's a fair and accurate sociological reading, but it's a framing you bring, not something the paper states.

That's it. The rest of the story holds. SINN is the foundation, CALIBER is the new building.
