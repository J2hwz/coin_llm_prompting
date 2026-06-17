# Coin Trajectory Analysis — Metrics Reference

## Scripts

### `coin_trajectory_analysis.py` — Compute metrics

#### Usage

```bash
# Single model directory
python -m analysis.coin_trajectory_analysis \
  --trajectory-dir path/to/trajectories \
  --output-dir path/to/outputs

# Multiple models (one subdirectory per model)
python -m analysis.coin_trajectory_analysis \
  --trajectory-dir path/to/models_root \
  --output-dir path/to/outputs \
  --multi-model
```

#### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--trajectory-dir` | Yes | — | Directory containing coin trajectory JSON files (or root of model subdirectories when using `--multi-model`) |
| `--output-dir` | No | `src/reveng/analysis/outputs/coin_trajectory_analysis` | Directory to save analysis outputs |
| `--model-name` | No | Derived from directory name | Override the model name used in output filenames and summaries |
| `--batch-size` | No | `40` | Number of grids to process per batch; reduce if RAM is limited |
| `--multi-model` | No | `False` | Process multiple models, one per subdirectory of `--trajectory-dir` |

#### Expected Input Format

Trajectory files must match the pattern:
```
{model}_size{N}_comp{X.X}_grid{N}_coin_{effort}_traj{N}.json
```

Each grid group also requires a layout file in the same directory:
```
# base transform
{model}_size{N}_comp{X.X}_grid{N}_coin_layout.json

# ISO transform
{model}_size{N}_comp{X.X}_grid{N}_coin_{transform}_layout.json
```

#### Outputs

Results are written to `{output-dir}/{model_name}/`:

| File | Description |
|---|---|
| `coin_trajectory_metrics_<model>.csv` | Grid-level metrics (one row per grid × effort group) |
| `coin_per_trajectory_<model>.csv` | Per-trajectory metrics |
| `coin_summary_by_size_complexity.csv` | Metrics averaged by grid size and density |
| `coin_summary_by_distance.csv` | State-level metrics binned by distance to goal |
| `coin_overall_summary.json` | Top-level summary statistics |
| `coin_success_breakdown.png` | Bar chart of success rates |
| `phase_action_accuracy.png` | Phase 1 vs phase 2 action accuracy |
| `phase_uncertainty.png` | Phase 1 vs phase 2 entropy/JSD scatter |
| `capability_vs_uncertainty.png` | Capability vs uncertainty scatter |
| `metrics_by_distance.png` | Metrics as a function of distance to goal |

---

### `reshuffle_walls_from_layouts` — Generate wall permutations and collect trajectories

#### Usage

```bash
coinenv-cli reshuffle-walls-from-layouts \
  --layout-dir path/to/layouts \
  --output-dir path/to/outputs \
  --num-permutations 5 \
  --num-trajectories-per-permutation 5 \
  --reasoning-efforts low \
  --model-names together_ai/openai/gpt-oss-20b
```

For each `*_layout.json` in `--layout-dir`, generates `--num-permutations` structurally distinct mazes that share the same grid size, complexity, agent start, goal, and coin positions but have different internal wall layouts, then collects trajectories on each.

#### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--layout-dir` | Yes | — | Directory containing `*_layout.json` files to permute |
| `--output-dir` | No | `.` (data dir) | Directory to save output JSON files |
| `--num-permutations` | No | `5` | Number of distinct wall layouts to generate per input layout |
| `--num-trajectories-per-permutation` | No | `5` | Trajectories per (permutation, model, effort) |
| `--max-attempts-per-permutation` | No | `200` | Max maze regeneration retries before applying the wall-count fallback |
| `--max-steps-per-trajectory` | No | `50` | Maximum steps per trajectory |
| `--reasoning-efforts` | No | `["low"]` | Reasoning effort levels |
| `--model-names` | No | `["together_ai/openai/gpt-oss-20b"]` | Models in `"provider/model_id"` format |
| `--max-workers` | No | `min(32, total_tasks)` | Parallel worker threads |
| `--verbose` | No | `False` | Print detailed per-trajectory logging |

#### Wall generation strategy

Each permutation re-runs the randomised DFS maze generator at the original complexity. Anchor positions (start, goal, coin) are kept fixed by:
1. **Retry**: if any anchor lands on a wall, discard and regenerate (up to `--max-attempts-per-permutation` times). Complexity preserved exactly.
2. **Fallback** (last resort): clear the conflicting anchor cells and re-wall an equal number of non-anchor empty interior cells, preserving total wall count.

#### Outputs

Files are written to `--output-dir/`:

| File | Description |
|---|---|
| `{stem}_walls{N}_layout.json` | Permuted layout JSON (same schema as source layouts) with added `"variant"` and `"permutation_index"` fields |
| `{stem}_walls{N}_{effort}_traj{traj_id}.json` | Trajectory file for permutation N |

---

### `plot_trajectories.py` — Visualise trajectories

#### Usage

```bash
python analysis/plot_trajectories.py <data_folder>
```

Scans `<data_folder>` for `*_layout.json` files and, for each layout, finds matching `*_low_traj*.json` trajectory files. Outputs are written to `<data_folder>/plots/`.

#### Expected Input Format

```
<data_folder>/
  {prefix}_layout.json          # one per layout
  {prefix}_low_traj0.json       # trajectories for that layout
  {prefix}_low_traj1.json
  ...
```

#### Outputs

Two files are produced per layout:

| File | Description |
|---|---|
| `{prefix}_panels.png` | One panel per trajectory arranged in a 5-column grid; each panel shows the path overlaid on the maze with a blue→red temporal colour gradient, arrowheads every 5 steps, and an outcome badge (✓/✗ for coin and goal) |
| `{prefix}_heatmap.png` | Cell visit-frequency heatmap aggregated across all trajectories for that layout |

---

## Overview

The coin environment has two sequential objectives:
1. Collect the coin (Ball at `coin_pos`)
2. Reach the terminal goal

Optimal actions are computed via backward Dijkstra from each target:
- **Phase 1:** target = `coin_pos` (steps up to and including coin collection)
- **Phase 2:** target = `goal_pos` (steps after coin collection)

This module produces three output tables per model:

| File | Description |
|---|---|
| `coin_trajectory_metrics_<model>.csv` | One row per (grid × effort) group |
| `coin_per_trajectory_<model>.csv` | One row per individual trajectory |
| `coin_summary_by_distance.csv` | State-level metrics binned by distance to goal |

---

## Grid-Level Metrics (`coin_trajectory_metrics_<model>.csv`)

### Metadata / Grouping Keys

| Column | Description |
|---|---|
| `grid_id` | Unique string key: `"size{N}_comp{X}_grid{N}_{transform}_{effort}"` |
| `grid_size` | Grid width/height in cells |
| `density` | Wall density (complexity parameter, 0–1) |
| `instance_id` | Integer grid index within this size/density |
| `reasoning_effort` | LLM reasoning budget: `"low"` or `"medium"` |
| `transform_type` | Coordinate transform applied: `"base"` or ISO variant name |

### Distance Metrics (A* shortest paths on the grid)

| Column | Description |
|---|---|
| `start_to_coin_distance` | A* steps: agent start → coin |
| `coin_to_goal_distance` | A* steps: coin → goal |
| `start_to_goal_via_coin_distance` | A* steps: start → coin → goal (sum of the two above); used as L* in SPL |
| `start_to_goal_distance` | A* steps: start → goal directly, ignoring the coin |
| `coin_detour_distance` | Minimum A* distance from the coin cell to any cell on the optimal start→goal path; measures how far off the direct route the coin lies (0 = coin lies on the direct path, higher = bigger detour required) |

### Raw Count Fields

| Column | Description |
|---|---|
| `num_trajectories` | Number of trajectories in this group |
| `num_coin_collected` | Trajectories where the coin was collected |
| `num_goal_after_coin` | Trajectories with full success (coin + goal) |
| `num_goal_only` | Trajectories that reached the goal without collecting the coin |

### Success Metrics (rates = count / `num_trajectories`)

| Column | Description |
|---|---|
| `coin_collected_rate` | Fraction of trajectories that collected the coin |
| `goal_after_coin_rate` | Fraction that collected coin AND then reached the goal (full success) |
| `full_success_rate` | Alias for `goal_after_coin_rate` |
| `goal_only_rate` | Fraction that reached the goal WITHOUT collecting the coin |

### Capability Metrics

| Column | Description |
|---|---|
| `mean_trajectory_length` | Mean number of steps across trajectories in the group |
| `mean_action_accuracy_phase1` | Fraction of phase-1 steps where the chosen action is in the optimal action set toward the coin (pooled across all phase-1 steps from all trajectories) |
| `mean_action_accuracy_phase2` | Same as above but for phase-2 steps toward the goal |
| `mean_action_accuracy` | Overall accuracy pooled across both phases (weighted by step counts) |
| `mean_step_accuracy` | Identical calculation to `mean_action_accuracy`; retained as a separate field for cross-module compatibility |
| `total_steps` | Total number of steps summed across all trajectories in the group |
| `spl` | Success weighted by Path Length — `int(full_success) × L* / max(L*, L)`, averaged across trajectories; L* = `start_to_goal_via_coin_distance`; ranges 0–1 where 1 = every trajectory succeeded via the optimal path |

### Uncertainty Metrics (pooled empirical action distributions per state)

| Column | Description |
|---|---|
| `mean_entropy` | Mean Shannon entropy (bits) of the empirical action distribution across all visited states, weighted by visit count; higher = more spread/uncertain choices |
| `mean_optimal_entropy` | Mean entropy of the *optimal* action distribution (log₂ of the number of equally-good optimal actions); reflects inherent ambiguity in the optimal policy at each state |
| `mean_jsd` | Mean Jensen-Shannon Divergence between the empirical and optimal action distributions; 0 = perfectly matches optimal, 1 = maximally divergent |
| `mean_entropy_phase1` | `mean_entropy` computed using only phase-1 state visits |
| `mean_jsd_phase1` | `mean_jsd` computed using only phase-1 state visits (vs `opt_to_coin`) |
| `mean_entropy_phase2` | `mean_entropy` computed using only phase-2 state visits |
| `mean_jsd_phase2` | `mean_jsd` computed using only phase-2 state visits (vs `opt_to_goal`) |
| `ece` | Expected Calibration Error across all steps: mean absolute difference between predicted confidence (empirical action frequency) and observed accuracy, binned and weighted by bin size |

### Movement Metrics (mean per trajectory)

**Absolute action counts:**

| Column | Description |
|---|---|
| `mean_actions_up` | Mean number of UP actions per trajectory |
| `mean_actions_down` | Mean number of DOWN actions per trajectory |
| `mean_actions_left` | Mean number of LEFT actions per trajectory |
| `mean_actions_right` | Mean number of RIGHT actions per trajectory |

**Relative direction counts** (relative to the previous step's heading):

| Column | Description |
|---|---|
| `mean_steps_front` | Mean steps continuing in the same direction as the previous step |
| `mean_steps_left_turn` | Mean steps turning left relative to the previous heading |
| `mean_steps_right_turn` | Mean steps turning right relative to the previous heading |
| `mean_steps_back` | Mean steps reversing 180° relative to the previous heading |

**Revisit counts:**

| Column | Description |
|---|---|
| `mean_cell_revisits` | Mean steps landing on any cell previously visited in the same trajectory |
| `mean_immediate_revisits` | Mean steps returning to the cell occupied exactly two steps earlier (A→B→A double-backs, excluding blocked no-op moves) |
| `mean_coin_oscillations` | Mean steps oscillating through the coin cell: either arriving at the coin cell (having been there before) or departing the coin cell to a previously-visited adjacent cell; counts post-collection oscillation |

---

## Per-Trajectory Metrics (`coin_per_trajectory_<model>.csv`)

### Metadata / Grouping Keys

| Column | Description |
|---|---|
| `trajectory_id` | Integer index of the trajectory within its grid group |
| `grid_id` | Integer grid index |
| `grid_key` | Full string key matching the grid-level table |
| `grid_size`, `density`, `reasoning_effort`, `transform_type` | Same definitions as grid-level table |

### Distance Fields

Same definitions as grid-level: `start_to_coin_distance`, `coin_to_goal_distance`, `start_to_goal_via_coin_distance`, `start_to_goal_distance`, `coin_detour_distance`.

### Outcome Fields

| Column | Description |
|---|---|
| `reached_goal` | 1 if the agent reached the goal cell, 0 otherwise |
| `coin_collected` | 1 if the coin was collected during the trajectory, 0 otherwise |
| `coin_collection_step` | Step index (0-based) at which the coin was collected; -1 if never |

### Per-Trajectory Capability

| Column | Description |
|---|---|
| `trajectory_length` | Total number of steps in this trajectory |
| `spl` | Per-trajectory SPL: `(L* / max(L*, L))` if full success else 0 |
| `action_accuracy` | Overall phase-weighted action accuracy for this trajectory |
| `action_accuracy_phase1` | Phase-1 action accuracy for this trajectory |
| `action_accuracy_phase2` | Phase-2 action accuracy for this trajectory (0.0 if no phase 2) |

### Per-Trajectory Movement Counts (raw integers, not means)

Same definitions as grid-level movement metrics but raw counts for the single trajectory:

`num_actions_up`, `num_actions_down`, `num_actions_left`, `num_actions_right`,
`num_steps_front`, `num_steps_left_turn`, `num_steps_right_turn`, `num_steps_back`,
`num_cell_revisits`, `num_immediate_revisits`, `num_coin_oscillations`

---

## State-Level Distance Metrics (`coin_summary_by_distance.csv`)

Aggregated across all states binned by their A* distance to the goal.

| Column | Description |
|---|---|
| `distance_to_goal` | A* distance from the state to the goal cell |
| `grid_size`, `density`, `reasoning_effort` | Grouping keys |
| `entropy` | Shannon entropy of the empirical action distribution at this state |
| `optimal_entropy` | Entropy of the optimal action distribution at this state |
| `jsd` | Jensen-Shannon Divergence at this state |
| `is_optimal` | 1 if the most-frequently-chosen action is in the optimal set |
| `n_observations` | Total number of visits to this state across all trajectories |
