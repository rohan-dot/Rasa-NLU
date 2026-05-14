Fair. Here's a mature version — no rhetorical flourishes, no "thank you," no audience-warming. PhD-to-PhD register. Direct, dense, technically grounded.

---

# CALIBER — 5-Minute Talk

*~720 words, technical register*

---

Current practice in online influence analysis is retrospective. A narrative is identified after diffusion, dashboards track reach, analysts assess, and only then is action considered. By the time the trend is observable on standard instruments, the persuasion and hardening have largely happened. Reach gets routinely mistaken for persuasion, simulations look plausible without being calibrated, and there is no principled way to compare a tactic against a defense before deployment.

CALIBER is a 12-month effort to build a calibrated, audited experimentation environment for narrative influence — quantitative, bounded, and gated by held-out empirical tests. Not a digital twin. Not an autonomous persuader. A measurement instrument.

---

The technical core is a closed measurement loop. We fit a physics-informed neural network — specifically the SINN architecture from Okawa and Iwata, KDD 2022 — to a rights-cleared threaded corpus. SINN jointly fits trajectories and the parameters of a chosen opinion-dynamics ODE: Friedkin-Johnsen, Hegselmann-Krause, or the diffusion-convection-reaction variant from Gong et al. The recovered parameters are interpretable in established sociological terms: per-user susceptibility α, per-community bounded-confidence radius ε, factorized influence matrix w, and the DCR terms D, v, R. Each is reported with 95% bootstrap confidence intervals.

That coefficient set is the community's fingerprint. It quantifies how fast opinions propagate, how open users are to dissonant content, how stubbornly individual users anchor to prior conviction, and whether the system is drifting toward greater insularity.

---

In parallel we build the synthetic community: roughly 500 LLM-driven agents with personas, network positions, and reply policies sampled from the real community's empirical distributions. The agents do not earn experimental use by behavioral plausibility. They earn it by passing a three-layer audit.

Layer one: we re-fit the same PINN architecture to the simulated trajectories, with identical hyperparameters. The recovered coefficients must fall inside the real-data 95% confidence intervals across all six parameters.

Layer two: five behavioral gates the calibration loss did not directly optimize for — cascade timing, drop-out under hostile exposure, sentiment trajectory matching via dynamic time warping, cross-topic transfer of coefficient shifts, and opinion-leader rank correlation.

Layer three: persuasion ground truth. Using the ChangeMyView corpus from Tan et al., simulated CMV-OP agents must grant delta-awards at the human rate on held-out cases, pre-registered threshold of seventy percent agreement.

Failure at any layer halts the program at that quarter. Pre-registered thresholds, committed before simulation runs.

---

Once the audit passes, persuader agents are inserted into the calibrated community across a factorial sweep: post-training method, rhetorical strategy, and network position. We measure conventional Hackenburg-style attitude shifts and cascade reach, but the primary measurements are the *parameter shifts* induced by each intervention. Δε, Δα, Δw — the same coefficients that defined the chamber now quantify what the intervention did to it. Effective tactics are paired with prebunking, friction injection, and structural rewiring defenses, scored by their reduction in induced parameter shifts.

The output is a ranked tactic-by-defense matrix grounded in interpretable parameters, with explicit reporting of side effects: factual-accuracy degradation, polarization increase, chamber tightening. A tactic that spreads while hardening the out-group is not equivalent to a tactic that spreads while reducing in-group/out-group asymmetry. CALIBER distinguishes them.

---

The known limitations are real and addressed structurally. Platform mismatch — a threaded-forum-fit model may not transfer to recommendation-heavy platforms — is mitigated by initial transfer testing on multimodal mission-relevant data in Q4. Overfitting to one corpus is mitigated by cross-topic transfer as a calibration gate. Dual-use concerns are mitigated by keeping the work in a human-gated sandbox; the published artifact is the measurement protocol and the defense library, not a deployable persuader.

A null calibration result is also informative. If Reddit data does not produce a tight enough fingerprint to constrain the synthetic community, that finding precedes any sponsor investment in operational transition.

---

The FY27 budget is $775K, twelve months, open-weights local LLMs on existing Lab GPU resources. No external inference spend. Cost is dominated by personnel and IRB review.

For 1st TIAD and 90th Cyberspace Operations, the deliverable is a bounded experimentation capability: a calibrated test range for narrative maneuver, an early-warning layer derived from coefficient drift, and a ranked countermeasure scorecard with confidence intervals and side-effect reporting. The transition target is the BRIES architecture from Volkova at Aptima, which currently lacks the validated audience layer CALIBER produces.

---

The technical thesis is single-sentence: measure the real community, build a bounded scale model, prove the model is calibrated against the measurement, and only then use it to compare attacks and defenses.

---

# What changed from Olga's version

- No "Good morning." No "Thank you." No bumper-sticker close. PhD audience does not need framing devices.
- Specific architectures named: SINN, Friedkin-Johnsen, Hegselmann-Krause, ChangeMyView, BRIES. A technical audience expects to hear the actual methods, not metaphors about wind tunnels.
- Parameter names and what they measure are given directly — α, ε, w, D, v, R — instead of being paraphrased as "how fast views spread, how open people are."
- The three-layer audit is broken out at sentence level with pre-registered thresholds, because that's the part a methods-trained listener will judge most carefully.
- Risk section is one paragraph of structural mitigations, not a list of fears.
- Transition section names the actual gap CALIBER fills in the Volkova/Aptima BRIES architecture, instead of generic "value for sponsors" language.

That's the talk. She can deliver it cold and it will land with a technical reviewer.
