# Formal Definitions: Coin Trajectory Metrics

This document derives the exact equations behind every metric computed by
`coin_trajectory_analysis.py`, grouped into four families:

1. **Accuracy and success at the task** (§2) — did the trajectory actually
   collect the coin and reach the goal?
2. **Action accuracy, optimality, and capability metrics** (§3) — how good
   were the individual decisions and the overall path, relative to optimal?
3. **Uncertainty metrics: overall action distribution** (§4) — how spread
   out / confident was the empirical policy relative to the optimal one?
4. **Movement-pattern metrics** (§5) — behavioral signatures independent of
   success or optimality: wall bumps, revisits, oscillations, and spatial
   region preference (quadrant/corner dwelling).

For a column-by-column glossary of every CSV field (rather than the
underlying math), see
[`coin_trajectory_analysis_metrics.md`](coin_trajectory_analysis_metrics.md).

## 1. Notation

- **Action space** `A = {LEFT, RIGHT, UP, DOWN}`, with per-action grid
  deltas `Δ(UP)=(0,-1)`, `Δ(DOWN)=(0,1)`, `Δ(LEFT)=(-1,0)`, `Δ(RIGHT)=(1,0)`.
- **Trajectory** `τ = (s₀,a₀,s₁,a₁,…,a_{T-1},s_T)` — one rollout on a grid;
  `pos(s_t)` is the agent's `(x,y)` position at step `t`. `L(τ)` = the
  trajectory's recorded step count (`trajectory_length`).
- **Trajectory set** `𝒯 = {τ₁,…,τ_N}` — all `N` trajectories collected on a
  given grid (e.g. repeated LLM rollouts, or a single oracle rollout).
- **Landmarks**: `s_start`, `s_goal`, `s_coin` — the grid's agent-start,
  goal, and coin cells.
- **Coin collection step** `c*(τ)` — the step index at which the coin
  disappears from the observation, attributed to the step whose action
  collected it (`find_coin_collection_step`). If the coin is never collected,
  `c*(τ) := ∞` (every step is phase 1).
- **Phase of step `t` in trajectory `τ`:**
  ```
  phase(t, τ) = 1  if t ≤ c*(τ)
              = 2  if t > c*(τ)
  ```
  (the step that walks onto the coin is itself phase 1 — see
  `get_step_optimal_actions`, `coin_trajectory_analysis.py:575`).
- **Optimal action sets**, computed once per grid by backward Dijkstra
  (`compute_optimal_actions_from_text_grid`, `analysis_utils.py:131`):
  - `π*_coin(s) ⊆ A` — directions from state `s` that strictly reduce
    shortest-path distance to the coin.
  - `π*_goal(s) ⊆ A` — same, toward the goal.
  - `π*_phase(s) := π*_coin(s)` if phase 1, else `π*_goal(s)`.
  - `L* = start_to_goal_via_coin_distance` — the A*-optimal path length
    start→coin→goal, i.e. the length of a trajectory that achieves full
    success as efficiently as possible.

---

## 2. Accuracy and Success at the Task

Outcome-level metrics: did the trajectory actually achieve the instructed
objective, independent of *how* efficiently or confidently it did so (that's
§3-4). All are boolean per trajectory, averaged into grid-level rates.

**Coin collection**, detected from the rendered grid rather than the
environment's internal state — the coin's symbol `C` is present at `s_coin`
in step `i`'s observation until the step whose action walks onto it:
```
present(i) = 𝟙[ grid_state(i) at s_coin shows the coin symbol ]
c*(τ)            = the step i-1 such that present(i-1)=1 ∧ present(i)=0   (first disappearance)
coin_collected(τ) = 𝟙[ c*(τ) is defined ]
```

**Goal reached**, reconstructed by applying the trajectory's *last* recorded
action to its last recorded position (the environment doesn't log the
post-action resting position of the final step directly):
```
s_final(τ)      = pos(s_{T}) + Δ(a_{T-1})
reached_goal(τ) = 𝟙[ s_final(τ) = s_goal ]
```

**Per-trajectory outcome predicates:**
```
full_success(τ) = coin_collected(τ) ∧ reached_goal(τ)     (the instructed objective, in full)
goal_only(τ)     = ¬coin_collected(τ) ∧ reached_goal(τ)    (navigated competently, ignored the coin)
```

**Grid-level rates**, averaged over the `N` trajectories on a grid:
```
coin_collected_rate               = (1/N) Σ_τ coin_collected(τ)
full_success_rate = goal_after_coin_rate = (1/N) Σ_τ full_success(τ)
goal_only_rate                     = (1/N) Σ_τ goal_only(τ)
```
`goal_only_rate` in particular dissociates navigation competence from
instruction-following: a trajectory that reaches the goal without ever
touching the coin is capable but non-compliant.

*Code:* `is_coin_present_at`, `find_coin_collection_step`, `load_coin_trajectory`
(coin_trajectory_analysis.py); grid-level rates computed inline in
`compute_coin_grid_metrics`.

---

## 3. Action Accuracy, Optimality, and Capability Metrics

Given the trajectory actually happened — how good were its individual
decisions, and how efficient was the overall path, relative to optimal?

### 3.1 Action accuracy

**Per-step correctness indicator:**
```
correct(t, τ) = 𝟙[ a_t ∈ π*_phase(t,τ)(s_t) ]
```

**Phase-1 / phase-2 accuracy**, pooled across every trajectory on the grid:
```
Acc₁ = Σ_τ Σ_{t : phase(t,τ)=1} correct(t,τ)  /  Σ_τ |{t : phase(t,τ)=1}|
Acc₂ = Σ_τ Σ_{t : phase(t,τ)=2} correct(t,τ)  /  Σ_τ |{t : phase(t,τ)=2}|
```
(`= 0` if the phase's step count is 0.)

**Combined accuracy** — simple pooling over *all* steps regardless of phase,
equivalently a step-count-weighted average of `Acc₁` and `Acc₂`:
```
Acc = (T₁·Acc₁ + T₂·Acc₂) / (T₁ + T₂)
```
where `T₁, T₂` are the total phase-1 / phase-2 step counts across all
trajectories on the grid. **This combined figure is correctly phase-aware**,
because every individual step is scored against its own phase's optimal set
*before* being pooled — no cross-phase information is lost. This is unlike
the naive combined entropy/JSD in §4, which required a dedicated fix.

*Code:* `compute_phase_accuracy` (`coin_trajectory_analysis.py:738`),
`get_step_optimal_actions` (`:575`); the grid-level pooled version
(`mean_step_accuracy`) is recomputed inline in `compute_coin_grid_metrics`
(`:906`) via the same per-step `get_step_optimal_actions` dispatch, so it
agrees with `Acc` above by construction.

### 3.2 SPL (Success weighted by Path Length)

A single number combining success with path-length optimality — zero for
any failed trajectory, and for successful ones, discounted by how much
longer than optimal the realised path was:
```
SPL(τ) = 𝟙[full_success(τ)] · L* / max(L*, L(τ))
SPL̄    = (1/N) Σ_τ SPL(τ)
```
Bounded in `[0,1]`: `1` only if every trajectory on the grid achieved full
success via a path of exactly the optimal length `L*`; `0` if no trajectory
succeeded at all. A trajectory that succeeds but wanders (`L(τ) > L*`) is
scored strictly between 0 and 1, rather than being counted identically to a
perfectly efficient success (as a bare `full_success_rate` would).

*Code:* `compute_coin_spl` (grid-level mean), and the equivalent per-trajectory
computation inline in `compute_single_trajectory_row`.

---

## 4. Uncertainty Metrics (Overall Action Distribution)

How spread-out or confident the model's *empirical* policy is at each
visited state, relative to the *optimal* policy there — independent of
whether any single action happened to be "correct" (§3 measures that).

### 4.1 Entropy

For a pool of steps `S` (e.g. "every phase-1 step across all trajectories on
this grid"), build the empirical action-count table per visited state:
```
n(s, a) = #{ (t,τ) ∈ S : s_t = s, a_t = a }
p̂(a | s) = n(s, a) / Σ_{a'∈A} n(s, a')
```

**Empirical (Shannon) entropy at state `s`, in bits:**
```
H(s) = − Σ_{a∈A} p̂(a|s) · log₂ p̂(a|s)          (0·log₂0 := 0)
```

**Optimal-policy entropy at `s`** — the ambiguity intrinsic to the *task*,
not the agent, given `k = |π*_phase(s)|` tied-optimal directions:
```
H*(s) = log₂ k          (0 if k = 0, e.g. at the target cell itself)
```

**Grid-level mean** — an *unweighted* mean over the set of distinct states
actually visited in that phase's pool, `𝒱_phase` (a state visited 50 times
contributes exactly one value, the same as a state visited once):
```
H̄_phase = (1 / |𝒱_phase|) · Σ_{s ∈ 𝒱_phase} H(s)
```
`mean_entropy_phase1` uses `S` = phase-1 steps and `π*_coin`;
`mean_entropy_phase2` uses `S` = phase-2 steps and `π*_goal`.

**Combined `mean_entropy` — phase-aware pooling of per-state values.** The
combined figure does *not* merge phase-1/phase-2 raw action counts into one
pool, nor merge `opt_to_coin`/`opt_to_goal` into one optimal-target map
(both of which would silently collapse to `π*_goal` everywhere, since both
cover every reachable cell — confirmed empirically: 100% key overlap on a
real test grid, with 63/81 cells having a genuinely different optimal
direction toward the coin vs. the goal). Instead, each phase's visited
states are scored against its own correct target *first*, and only the
resulting **already-computed per-state values** are pooled:
```
H̄_combined = (1 / (|𝒱_phase1| + |𝒱_phase2|)) ·
              ( Σ_{s∈𝒱_phase1} H(s; π*_coin)  +  Σ_{s∈𝒱_phase2} H(s; π*_goal) )
```
Equivalently, `mean_entropy_phase1`/`mean_entropy_phase2` weighted by the
number of distinct states visited in each phase:
`H̄_combined = (|𝒱_phase1|·H̄_phase1 + |𝒱_phase2|·H̄_phase2) / (|𝒱_phase1| + |𝒱_phase2|)`.
A state visited in *both* phases contributes **two** values to this pool —
never one value from a blended empirical distribution scored against the
wrong target.

*Code:* `StateActionCounts` (`full_obs_trajectory_analysis.py:76`),
`shannon_entropy`/`optimal_entropy` (`metrics.py:36,61`, re-exported through
`analysis_utils.py`), `collect_uncertainty_values`, `build_phase_pools`,
`compute_phase_aware_combined_uncertainty` (`coin_trajectory_analysis.py`).

### 4.2 Jensen-Shannon Divergence (JSD)

Define the **optimal reference distribution** at state `s` — uniform over
the tied-optimal directions, using the same `π*_phase(s)` as above:
```
q(a | s) = 1 / k   if a ∈ π*_phase(s)
         = 0        otherwise           (k = |π*_phase(s)|)
```
and the mixture `m(a|s) = ½·( p̂(a|s) + q(a|s) )`. Then:
```
JSD(s) = ½ · Σ_a q(a|s)·log₂( q(a|s) / m(a|s) )
       + ½ · Σ_a p̂(a|s)·log₂( p̂(a|s) / m(a|s) )
       = ½·KL(q ‖ m) + ½·KL(p̂ ‖ m)
```
Bounded in `[0, 1]` (log base 2): `0` = the empirical policy matches the
optimal one exactly (in a uniform-over-ties sense); `1` = maximal divergence
(empirical mass entirely on non-optimal actions).

**Phase-specific mean** (only over states with `π*_phase(s) ≠ ∅`):
```
J̄_phase = (1 / |𝒱_phase|) · Σ_{s ∈ 𝒱_phase} JSD(s)
```
`mean_jsd_phase1` (vs. `π*_coin`) and `mean_jsd_phase2` (vs. `π*_goal`).

**Combined `mean_jsd`.** Same pooling principle as entropy: phase 1's JSD
values (each `s` scored against `π*_coin`) and phase 2's JSD values (against
`π*_goal`) are computed independently, then concatenated and averaged:
```
J̄_combined = (1 / (|𝒱_phase1| + |𝒱_phase2|)) ·
              ( Σ_{s∈𝒱_phase1} JSD(s; π*_coin)  +  Σ_{s∈𝒱_phase2} JSD(s; π*_goal) )
```

*Code:* `jensen_shannon_divergence` (`metrics.py:78`), same call sites as §4.1.

### 4.3 Expected Calibration Error (ECE)

Uses the same `p̂(a|s)` / `π*_phase(s)` pair: per state, confidence =
`max_a p̂(a|s)`, accuracy = `𝟙[argmax_a p̂(a|s) ∈ π*(s)]`; states are binned
into 10 confidence bins and
```
ECE = Σ_bins (|bin|/n) · |acc(bin) − conf(bin)|
```
The combined `ece` is phase-aware the same way as §4.1-4.2:
`compute_phase_aware_combined_ece` collects `(confidence, accuracy)` pairs
per phase — phase 1 scored against `π*_coin`, phase 2 against `π*_goal` —
concatenates the two phases' pair-lists, and bins/scores the pooled result
once. No phase-split `ece_phase1`/`ece_phase2` columns exist — only the
combined `ece`.

*Code:* `collect_ece_values`, `bin_and_score_ece` (`full_obs_trajectory_analysis.py`),
`compute_phase_aware_combined_ece` (`coin_trajectory_analysis.py`).

### 4.4 Phase-awareness cross-reference

| Quantity | Phase 1 | Phase 2 | Combined | Phase-aware? |
|---|---|---|---|---|
| Action accuracy (§3.1) | `Acc₁` vs. `π*_coin` | `Acc₂` vs. `π*_goal` | `Acc` — step-count-weighted average | ✅ yes |
| Entropy (§4.1) | `H̄_phase1` vs. `π*_coin` | `H̄_phase2` vs. `π*_goal` | `H̄_combined` — pools per-state values, weighted by distinct states visited per phase | ✅ yes |
| JSD (§4.2) | `J̄_phase1` vs. `π*_coin` | `J̄_phase2` vs. `π*_goal` | `J̄_combined` — same pooling principle | ✅ yes |
| ECE (§4.3) | — (not computed per phase) | — | binned confidence/accuracy, each pair scored against its own phase's target before pooling | ✅ yes |

---

## 5. Movement-Pattern Metrics

Behavioral signatures independent of success or optimality — *how* the
agent moved, not whether it succeeded or chose the correct direction.

### 5.1 Invalid actions (wall bumps)

An action is invalid if it targets a wall cell or a cell outside the grid —
the agent stays in place and the step is consumed:
```
invalid(t, τ) = 𝟙[ pos(s_t) + Δ(a_t) is a wall cell, or out of grid bounds ]
num_invalid_actions(τ)  = Σ_t invalid(t, τ)
invalid_action_rate(τ)  = num_invalid_actions(τ) / L(τ)
mean_invalid_actions    = (1/N) Σ_τ num_invalid_actions(τ)
```
When the wall layout isn't available, this falls back to inferring a bump
from the next step's position being unchanged (`pos(s_{t+1}) = pos(s_t)`) —
this fallback cannot classify a trajectory's final step, since there is no
next step to compare against.

*Code:* `is_invalid_action`, `compute_trajectory_movement_stats`.

### 5.2 Revisit and oscillation counts

Let `visited_t = {pos(s_0), …, pos(s_{t-1})}` be the set of cells visited
strictly before step `t`, and `coin_cells` the grid's coin cell(s):
```
num_cell_revisits(τ)      = Σ_t 𝟙[ pos(s_t) ∈ visited_t ]

num_immediate_revisits(τ) = Σ_t 𝟙[ t≥2 ∧ pos(s_t)=pos(s_{t-2}) ∧ pos(s_t)≠pos(s_{t-1}) ]
                             (an A→B→A double-back; excludes consecutive
                              blocked/no-op moves, which never leave A)

num_coin_oscillations(τ)  = Σ_t 𝟙[ (pos(s_t)∈coin_cells ∧ pos(s_t)∈visited_t)
                                  ∨ (pos(s_{t-1})∈coin_cells ∧ pos(s_t)∈visited_t) ]
                             (arriving at, or departing from, a coin cell
                              onto a previously-visited cell — includes
                              post-collection oscillation)
```
Grid-level: `mean_cell_revisits`, `mean_immediate_revisits`,
`mean_coin_oscillations` — each the mean of the corresponding per-trajectory
count over the `N` trajectories on the grid.

*Code:* `compute_trajectory_movement_stats`.

### 5.3 Absolute and relative direction counts

Absolute: `num_actions_up/down/left/right(τ)` — raw counts of each action
string chosen. Relative (heading continuity, from the previous step's
action `a_{t-1}`, for `t ≥ 1`):
```
front(t)      = 𝟙[ a_t = a_{t-1} ]                    (continued straight)
back(t)       = 𝟙[ a_t = opposite(a_{t-1}) ]           (180° reversal)
left_turn(t)  = 𝟙[ a_t = left_of(a_{t-1}) ]
right_turn(t) = 𝟙[ a_t = right_of(a_{t-1}) ]
```
exactly one of the four holds per `t ≥ 1`. Grid-level fields
(`mean_actions_up`, …, `mean_steps_front`, …) are the per-trajectory counts
averaged over the grid's `N` trajectories, same convention as §5.1-5.2.

*Code:* `compute_trajectory_movement_stats`.

### 5.4 Spatial / corner-dwelling metrics (preferred quadrant / corner)

Motivated by an empirical observation that LLM trajectories tend to dwell
heavily in specific interior *corners* that are not the start, goal, or
coin cell (confirmed directly against real data: on one grid, the single
most-visited cell across 10 trajectories was the top-right interior corner,
with 157 visits, despite the coin, goal, and start all being elsewhere).

**Quadrant assignment.** For a grid of side length `N`, the interior
midpoint is `mid = (N-1)/2`. A position `(x,y)` is assigned to one of four
quadrants:
```
vertical   = "u" if y ≤ mid else "d"
horizontal = "l" if x ≤ mid else "r"
quadrant(x,y) = vertical + horizontal   ∈ {ul, ur, dl, dr}
```
The exact center cell (`x = y = mid`, which always exists since the
interior span `N-2` is odd for every grid size in this dataset) resolves to
`ul` by the `≤` convention — an arbitrary but consistent tie-break.

**Corner-region membership.** The corner-region side length scales with
grid size rather than being fixed, so it covers a comparable *fraction* of
the grid regardless of `N`:
```
interior_size = N - 2
r = corner_radius(N) = min( max(1, round(0.25 · interior_size)), interior_size // 2 )
```
The cap at `interior_size // 2` guarantees the four corner boxes can never
overlap — any two corners sharing an edge remain separated by at least one
non-corner cell — so a single action (a Manhattan step of 1) can never move
directly from one corner region into another. Verified both algebraically
and by exhaustive scan over the full interior for `N ∈ {7,9,11}` (radii
1, 2, 2 respectively; fractions ≈ 20%, 29%, 22%). A position is classified
into whichever corner's `r × r` box (anchored at that interior corner) it
falls in, or `None` if it falls in none of the four.

**Quadrant occupancy, preferred quadrant, and quadrant entropy** — computed
per trajectory, from that trajectory's own sequence of visited positions
`s_0, …, s_T`:
```
frac_q = |{t : quadrant(s_t) = q}| / (T+1),   q ∈ {ul, ur, dl, dr}
preferred_quadrant = argmax_q frac_q
quadrant_entropy   = shannon_entropy({frac_ul, frac_ur, frac_dl, frac_dr})
```
(valid input to `shannon_entropy` since the four quadrant fractions always
sum to 1). `quadrant_entropy` ranges `[0, log₂4] = [0, 2]` bits — 0 =
entirely one quadrant, 2 = perfectly even across all four.

**Corner occupancy, preferred corner, and longest dwell run** — analogous
occupancy fractions using `classify_corner` instead of `quadrant`, *except*
that the four corner fractions do **not** sum to 1 (there is an implicit,
unstored "not in any corner" mass) — **never pass `corner_fraction_*`
directly to `shannon_entropy`**. `preferred_corner` is the argmax corner
fraction, or `None` if the trajectory never enters any corner region at all
(all four fractions are 0). The dwell-run metric captures a literal "got
stuck for `k` steps in a row" episode, distinct from aggregate occupancy
time spread across many short visits:
```
max_corner_dwell_run = longest run of consecutive t with classify_corner(s_t) ≠ None
```

**Grid-level aggregation.** All of the above are computed **per trajectory
first, then averaged across the trajectories on a grid** — the same
convention used throughout §5 (`compute_trajectory_movement_stats` →
`move_totals`/`move_means`). This is deliberate, not incidental:
`max_corner_dwell_run` is order-dependent, so concatenating positions across
trajectories before computing it would splice fake multi-hundred-step "runs"
across trajectory boundaries that never occurred in any single trajectory —
there is no valid way to pool it the way phase-aware entropy/JSD are pooled
(§4). Applying the same per-trajectory-then-averaged rule uniformly to *all*
spatial metrics (not just the dwell run) keeps the aggregation model
consistent rather than mixing two different conventions. Grid-level
`preferred_quadrant`/`preferred_corner` are the argmax of the *mean*
fractions across trajectories (equally weighting each trajectory, not each
raw step) — a distinct convention from taking the most-common per-trajectory
label (mode), chosen for consistency with every other grid-level number in
§5 being a mean. `worst_max_corner_dwell_run` (the max, not mean, across
trajectories) is additionally reported to surface the single longest
observed stuck-episode on a grid, alongside the trajectory-averaged
`mean_max_corner_dwell_run`.

**Confound flags vs. task-relevant cells.** For a grid with known
`s_start`, `s_goal`, and `s_coin`, six flags indicate whether the
*preferred* quadrant/corner happens to be the same quadrant/corner
containing each of those three points:
```
preferred_quadrant_contains_X = 𝟙[ quadrant(X) = preferred_quadrant ],   X ∈ {start, goal, coin}
preferred_corner_contains_X   = 𝟙[ preferred_corner ≠ None  and  classify_corner(X) = preferred_corner ]
```
These are exposed as-is (per trajectory, and as a grid-level mean, i.e. the
*fraction* of trajectories whose own preferred region overlapped that
point) rather than folded into a single "adjusted" occupancy number — this
keeps the distinction between "the model wandered somewhere task-irrelevant"
and "the preferred region happens to be where the task already required
visiting" fully transparent and inspectable, rather than baking in an
automatic correction that might not match how a given analysis wants to use
it.

*Code:* `corner_radius`, `classify_quadrant`, `classify_corner`,
`compute_spatial_stats` (`coin_trajectory_analysis.py`), called once per
trajectory from both `compute_single_trajectory_row` (per-trajectory CSV)
and the per-trajectory loop inside `compute_coin_grid_metrics` (grid-level
CSV, averaged).
