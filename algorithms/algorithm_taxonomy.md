# Algorithm Taxonomy

Seven algorithms in `algorithms/` for inferring the latent subgoal (coin position) from
observed gridworld navigation trajectories. All share the same output format:
`score_grid[row_from_top, col]` + `predicted_json_coords (col, row_from_top)`.

> **Note:** the default `run_algorithms.py` suite now runs 5 of these
> (Visit Frequency — split into `visit_frequency.py` cell-level and
> `trajectory_visit_frequency.py` trajectory-level baselines — Surprise v2,
> Inverse Planning, MaxEnt IRL). BNIRL, BIRL, and T-REX (sections 4, 5, 7
> below) have moved to `algorithms/archive/` and are excluded from the
> default suite, but remain directly importable. See `algorithms.md`.

---

## Broad Classification

| Algorithm | File | Family |
|---|---|---|
| Visit Frequency | `visit_frequency.py` / `trajectory_visit_frequency.py` | Non-parametric baseline |
| Surprise v2 | `surprise_v2.py` | Information-theoretic (non-Bayesian) |
| Inverse Planning | `inv_planning.py` | Bayesian IRL / ToM |
| BNIRL | `archive/bnirl.py` | Bayesian non-parametric IRL |
| BIRL | `archive/birl_wrapper.py` | Bayesian IRL (MCMC) |
| MaxEnt IRL | `maxent_irl.py` | Maximum Entropy IRL |
| T-REX | `archive/trex.py` | Preference-based IRL (deep learning) |

---

## Detailed Taxonomy

### 1. Visit Frequency

| Dimension | Detail |
|---|---|
| **Objective** | Identify which intermediate cell the agent visits most frequently |
| **Family** | Non-parametric frequency counting — no probabilistic model |
| **How it works** | Walk every consecutive position pair in each trajectory and increment that cell's count. Sum counts across all trajectories. Mask walls, the start cell(s), and the goal cell. Return the cell with the highest count as the predicted subgoal. |
| **Key Assumptions** | Subgoal = most-visited non-terminal cell. All visits equally informative. No assumption about agent rationality or task structure. |
| **Inputs** | `paths` (list of (col, row) sequences), `layout` (grid dict) |
| **Output** | Raw visit count per cell. Highest-count non-start, non-goal cell is predicted subgoal. |
| **Implications for LLM** | Most forgiving — makes no rationality assumptions. However, LLMs that revisit cells heavily (oscillation, backtracking) inflate counts in cells unrelated to the subgoal, potentially misleading the prediction. |

---

### 2. Surprise v2

| Dimension | Detail |
|---|---|
| **Objective** | Identify where the agent deviates most from *expected* terminal-directed motion |
| **Family** | Information-theoretic — Shannon surprise under a hand-crafted navigation prior |
| **How it works** | At each step, build a prior distribution over the four cardinal actions by multiplying two components: (1) a *movement prior* favouring the current heading (controlled by momentum λ and forward/lateral ratio γ), and (2) a *state prior* favouring actions that reduce Manhattan distance to the terminal goal (controlled by decay α). Compute surprise at that step as `−log₂ P(actual_action \| combined_prior)`. Accumulate surprise per visited cell across all trajectories. The cell with the highest total surprise is the predicted subgoal — it is where the agent most consistently deviated from expected terminal-directed motion. |
| **Key Assumptions** | (1) Agent has forward momentum (continuing is more likely than turning). (2) Agent prefers steps that decrease Manhattan distance to the terminal goal. (3) Deviation from these priors (high surprise) signals a subgoal is nearby. Parameters λ (momentum), γ (forward/lateral ratio), α (distance decay) are tuned to human navigation data. |
| **Inputs** | `paths`, `layout`, optional hyperparameters (λ=0.10, γ=2.14, α=1.84) |
| **Output** | Surprise score per cell (sum of Shannon surprise at each visit). Highest-surprise cell is the predicted subgoal. |
| **Implications for LLM** | Parameters calibrated for human walkers. LLMs may naturally lack directional momentum (they respond to the current grid state, not physical inertia). LLMs also sometimes make suboptimal moves not driven by goal-directedness, which would produce spurious high-surprise cells unrelated to the coin. Also uses Manhattan distance in the prior — ignores walls. |

---

### 3. Inverse Planning (Baker et al. 2011)

| Dimension | Detail |
|---|---|
| **Objective** | Infer the agent's latent subgoal by computing a Bayesian posterior P(subgoal \| trajectory) |
| **Family** | Bayesian inverse planning / Theory of Mind |
| **How it works** | For each candidate cell c: (1) run soft value iteration backward from c to produce Q-values Q_c(s,a); (2) run soft value iteration backward from the terminal goal to produce Q_g(s,a); (3) split each trajectory at its first visit to c — steps before c are scored under the Boltzmann policy derived from Q_c, steps from c onward under Q_g; log-likelihood = −∞ if the trajectory never visits c. Sum log-likelihoods across all trajectories to get a score for c. Apply log-softmax normalisation across all candidates (uniform prior) to get a posterior probability P(c \| data). The argmax cell is the predicted subgoal. |
| **Key Assumptions** | (1) Agent is *approximately rational* — follows a soft-optimal (Boltzmann) policy. (2) Two-phase task: agent first navigates to subgoal, then to terminal. (3) Full observability. (4) Rationality controlled by β: β→0 is uniform random, β→∞ is perfectly optimal. Default β=2 (moderately stochastic). (5) Candidate subgoal must be *visited* by the trajectory, else likelihood = −∞. |
| **Inputs** | `paths`, `layout`, `beta` (softmax temperature, default 2.0) |
| **Output** | Posterior probability over candidate cells. Predicted subgoal = argmax. |
| **Implications for LLM** | The soft-optimal model was designed for human intentional agents. LLMs are inconsistent in a different way — they are *context-sensitive* rather than noisy around an optimal policy. If an LLM never visits the coin cell (e.g. because it fails), likelihood is −∞ for that candidate and the model has no signal. With low reasoning effort, LLMs may be far more stochastic than β=2 implies, underweighting valid subgoal hypotheses. |

---

### 4. BNIRL (Bayesian Non-parametric IRL)

| Dimension | Detail |
|---|---|
| **Objective** | Infer a latent subgoal by treating each trajectory step as generated by one of K=2 "intention clusters" (subgoal vs terminal) |
| **Family** | Bayesian non-parametric IRL — Gibbs sampler |
| **How it works** | Concatenate all trajectory steps into a single (pos, next_pos) sequence. Initialise a subgoal estimate g randomly from the visited cells. Repeat for n_iter iterations: (1) *Assignment* — for each step, compute the softmax likelihood of that step under "heading toward g" vs "heading toward terminal" using Manhattan distance as reward; assign the step stochastically to the subgoal or terminal cluster. (2) *Update* — set g to the cell-wise median of next_pos values for all steps assigned to the subgoal cluster. After burn_in iterations, record g at each remaining step. The cell appearing most frequently across post-burn-in samples is the predicted subgoal. |
| **Key Assumptions** | (1) K=2 fixed (hardcoded): each step belongs to either the subgoal-seeking or terminal-seeking intention. (2) Step likelihood = softmax over Manhattan distance reward to the target. (3) Subgoal is updated as the median of positions whose steps are currently assigned to the subgoal cluster. (4) Agent preference is governed by Manhattan distance only — walls are ignored in the step likelihood. |
| **Inputs** | `paths`, `layout`, `n_iter` (500), `burn_in` (100), `alpha` (softmax temp, 3.0), `seed` |
| **Output** | Post-burn-in sample histogram over cells. Mode = predicted subgoal. |
| **Implications for LLM** | Ignoring walls is a significant limitation when the LLM trajectory is wall-aware. LLMs operating in a walled environment may take paths that look "irrational" under Manhattan distance but are optimal given the actual geometry. K=2 is appropriate for the coin task. Because it pools all trajectory steps into one sequence, LLM inconsistency (different strategies across different runs) may blur the cluster boundary. |

---

### 5. BIRL — Bayesian IRL / PolicyWalk (Ramachandran & Amir 2007)

| Dimension | Detail |
|---|---|
| **Objective** | Infer a full tabular reward function R(s) for every state via MCMC, then identify the state with highest inferred reward |
| **Family** | Bayesian IRL — Metropolis-Hastings MCMC over reward space |
| **How it works** | Initialise a reward function R(s) = 0 for all states. Run Metropolis-Hastings for 1500 iterations: (1) *Propose* — randomly pick one state and perturb its reward by ±0.5, clamped to [r_min, r_max]. (2) *Evaluate* — compute the optimal policy π* under the proposed R via policy iteration; score all observed (state, action) pairs as `exp(α · Σ Q_R(s,a))` with α=20. (3) *Accept/reject* — accept with probability `min(1, posterior_ratio)`. After all iterations, the accepted R is returned. The non-start, non-goal cell with the highest inferred reward is the predicted subgoal. |
| **Key Assumptions** | (1) Agent is *perfectly rational* — follows the greedy optimal policy given R. (2) Reward function is tabular: one scalar per non-wall state. (3) Reward bounded: r ∈ [r_min, r_max], default [−2.1, −0.1] — assumes all rewards are negative (step costs). (4) Stationary reward across all trajectories. (5) Likelihood ∝ exp(α · Q(s, a)) under optimal policy for each (s, a) pair observed. Default α=20 (strong confidence in optimality). |
| **Inputs** | `paths`, `layout`, optional kwargs overriding default config |
| **Output** | Inferred reward per cell. Highest-reward non-start, non-goal cell = predicted subgoal. |
| **Implications for LLM** | Strongest rationality assumption of all algorithms. LLMs are not optimal agents — they may choose suboptimal actions (especially at low reasoning effort), and the α=20 likelihood term will assign near-zero probability to suboptimal actions, causing the MCMC chain to either reject updates frequently or fit an inconsistent reward function. The negative-only reward range ([−2.1, −0.1]) means the model can only express "cells the agent avoids less" rather than true positive rewards. |

---

### 6. MaxEnt IRL (Ziebart 2010)

| Dimension | Detail |
|---|---|
| **Objective** | Infer a tabular reward function R(s) by matching the *expected state visitation frequency* under the model to the empirical distribution in the trajectories |
| **Family** | Maximum Causal Entropy IRL — gradient ascent on a feature-matching objective |
| **How it works** | Initialise reward parameters θ(s) = 1 for all states. Iterate until convergence: (1) *Backward pass* — run soft Bellman equations backward from the terminal goal under current θ to compute local causal action probabilities (the MaxCausalEntropy policy). (2) *Forward pass* — propagate the empirical start-state distribution forward through this policy to compute expected state visitation frequencies (SVF). (3) *Gradient* — `∇θ = empirical_SVF − expected_SVF` (since features = identity, this is simply the difference in per-state visit counts). (4) *Update* — exponentiated gradient ascent: `θ ← θ · exp(lr · ∇θ)` with decaying `lr(k) = 0.2 / (1 + k)`. Convergence when `‖θ_new − θ_old‖ < 1e-4`. The highest-θ non-start, non-goal cell is the predicted subgoal. |
| **Key Assumptions** | (1) Agent follows the *maximum causal entropy* policy — among all policies matching observed feature expectations, prefers the most spread-out (least committed) one. (2) Feature matrix = identity: one feature per state (no hand-crafted features needed). (3) Full observability. (4) Trajectories are exchangeable (pooled). (5) Tabular state space with 0.99 discount. |
| **Inputs** | `paths`, `layout`, `discount` (0.99), convergence thresholds `eps` (1e-4), `eps_svf` (1e-5) |
| **Output** | Inferred reward per cell. Highest-reward non-start, non-goal cell = predicted subgoal. |
| **Implications for LLM** | MaxEnt's "least-committed rationality" assumption is more forgiving than BIRL — it does not require the agent to follow the exact optimal policy, only to match aggregate state visitation patterns. This makes it the most theoretically robust of the classical IRL methods for LLM trajectories. However, it still requires a sufficient number of trajectories to estimate the SVF reliably. With only 1–2 successful trajectories (common for hard grids), the SVF estimate is noisy. |

---

### 7. T-REX (Brown et al. 2019)

| Dimension | Detail |
|---|---|
| **Objective** | Learn a neural reward function R_θ(s) by training on pairwise trajectory *preferences* derived from a quality proxy ranking, then identify the highest-reward cell |
| **Family** | Preference-based IRL — deep learning with Bradley-Terry preference model |
| **How it works** | (1) *Rank* — score each trajectory as `w_eff · path_efficiency + w_ent · (1 − normalised_entropy)` using wall-aware BFS for efficiency; unsuccessful trajectories are always placed below successful ones. (2) *Sample clip pairs* — draw n_pairs pairs (i, j) where j is ranked higher than i; for each pair sample a random start time t_i and t_j ≥ t_i (temporal constraint); extract fixed-length clips and encode each position as `[x_norm, y_norm, coin_flag]`. (3) *Train* — for each pair compute the cumulative return `R(clip) = Σ reward(state)` from a 3→64→64→1 MLP; apply Bradley-Terry preference loss `−log σ(R(clip_j) − R(clip_i))`; optimise with Adam. (4) *Extract* — query the trained network at every valid cell with `coin_flag=0`; return the full reward grid and the highest-reward cell as the predicted subgoal. |
| **Key Assumptions** | (1) Trajectories can be meaningfully ranked by observable proxies: path efficiency (SPL formula, wall-aware BFS) and per-trajectory action entropy. (2) The ranking captures latent reward signal: better-ranked trajectories should have higher cumulative reward under R_θ. (3) Temporal ordering constraint: clips from better trajectories are sampled no earlier than clips from worse ones. (4) Unsuccessful trajectories are always ranked below successful ones (hard group separation). (5) Reward is spatially smooth and representable by a 3→64→64→1 MLP. (6) State = (x_norm, y_norm, coin_flag): coin collection is a binary phase transition. |
| **Inputs** | `paths` + `success_flags` (all trajectories, not just successful), `layout`, `clip_len`, `n_pairs`, `lr`, `hidden_dim` |
| **Output** | Neural network predicted reward per cell (queried with coin_flag=0). Highest-reward cell = predicted subgoal. |
| **Implications for LLM** | Does not require explicit rationality — instead asks "which trajectories are *relatively* better?" and uses this to shape R_θ. This is well-suited to LLMs: even if no trajectory is optimal, relative quality differences still provide a training signal. Key vulnerability: the quality proxy (efficiency + entropy) must be meaningful. LLMs with high reasoning effort produce more efficient, low-entropy trajectories, which should rank correctly. Low reasoning effort LLMs may have noisy efficiency/entropy that disrupts the ranking. The model also uses unsuccessful trajectories, which is important given that many LLM runs fail. Requires ≥2 trajectories with different ranks; degrades gracefully. |

---

## Cross-Cutting Summary

| Algorithm | Rationality Required | Uses Walls in Likelihood | Handles Failed Trajectories | Needs Multiple Trajectories | Learn from Structure vs Frequency |
|---|---|---|---|---|---|
| Visit Freq | None | N/A | Yes (treats all steps equally) | Better with more | Frequency |
| Surprise v2 | Weak (momentum + goal-direction) | No (Manhattan only) | Yes | Better with more | Frequency |
| Inv. Planning | Soft-optimal (β=2) | Yes (MDP transitions) | Only if visit subgoal | Better with more | Structure |
| BNIRL | Soft-optimal (Manhattan) | No (Manhattan only) | Yes | Better with more | Structure |
| BIRL | Fully optimal | Yes (policy iteration) | Hurts (suboptimal acts penalised) | Yes | Structure |
| MaxEnt IRL | Max-entropy optimal | Yes (soft VI) | Hurts | Yes (SVF needs multiple) | Both |
| T-REX | None (proxy ranking) | Yes (BFS efficiency) | Yes (ranked below successful) | Yes (need ≥2 ranks) | Both |

**Key takeaways for LLM agents:**
- Algorithms assuming *full optimality* (BIRL) are most at risk: LLMs frequently take suboptimal actions.
- Algorithms using *Manhattan distance* in the likelihood (BNIRL, Surprise v2) misspecify the geometry when walls matter.
- **MaxEnt IRL** is the most principled choice for LLMs that have some goal-directedness but are not perfectly rational.
- **T-REX** is uniquely suited to LLM data because it does not require a rationality model — only that trajectories can be comparatively ranked, and it actively benefits from unsuccessful runs.
- **Visit Frequency** remains a strong baseline precisely because it makes no agent model assumptions.
