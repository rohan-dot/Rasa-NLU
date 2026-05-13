**Input:** Stance matrix `U` (N users × T weeks, values in [−1,+1] from RoBERTa) + directed reply graph `G` with edge weights `w_ij = log(1 + replies)`.

**Output:** Fitted opinion-dynamics coefficients with 95% CIs — per-user stubbornness α_i, per-edge social influence w_ij, plus community-level confidence radius ε and diffusion-convection-reaction terms D, v, R.

**Objective:** Minimize `L = L_data + λ·L_physics`, where a neural surrogate `u_θ(t,i)` must (a) match observed stance and (b) satisfy a chosen opinion-dynamics ODE (Friedkin-Johnsen, Hegselmann-Krause, or DCR) whose coefficients are themselves learnable.

**Why:** Those coefficients are the community's fingerprint — later, the LLM-agent sim is validated by re-fitting the PINN to simulated trajectories and checking whether the recovered coefficients land inside the real-data CIs.
