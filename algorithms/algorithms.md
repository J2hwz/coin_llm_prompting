# algorithms/

Self-contained package for subgoal inference from gridworld trajectory data.
Given observed human navigation trajectories and a grid layout, each algorithm
predicts which cell the agent was heading toward as an intermediate subgoal
(before the known terminal goal).

---

## Entry point

```
python run_algorithms.py <data_dir> [--mode {individual,pooled,both}] [--effort EFFORT] [--skip-invalid-actions]
```

`<data_dir>` must contain the layout and trajectory JSON files described below.
Results are written to `<data_dir>/plots/`. `--skip-invalid-actions` discards
wall-bump steps (zero position change) when *loading* trajectories — note the
3 non-baseline algorithms (Inv. Planning, MaxEnt IRL, Surprise v2) already
drop wall-bump steps internally regardless of this flag, so it should
normally be left off: turning it on would also strip wall-bump revisits from
the two visit-frequency baselines, which are supposed to count them.

**Modes**

| Mode | Description |
|---|---|
| `individual` | Run each trajectory separately |
| `pooled` | Pool all successful trajectories and run once |
| `both` | Both (default) |

---

## Algorithms

Five algorithms are run in sequence by default. Each takes `(paths, layout)`
and returns `(score_grid, predicted_json_coords)`.

| Name | File | Score type | Method |
|---|---|---|---|
| Cell Visit Freq | `visit_frequency.py` | `visits` | Baseline: most-visited non-start/non-goal cell, pooled per-step counts |
| Trajectory Visit Freq | `trajectory_visit_frequency.py` | `traj_visits` | Baseline: most-visited non-start/non-goal cell, deduped per trajectory |
| Inv. Planning | `inv_planning.py` | `posterior` | Bayesian inverse planning via soft value iteration (Baker et al. 2011) |
| MaxEnt IRL | `maxent_irl.py` | `reward` | Maximum Causal Entropy IRL (Ziebart 2010) |
| Surprise v2 | `surprise_v2.py` | `surprise` | Shannon surprise relative to terminal-directed prior (Buidze et al. 2025) |

`archive/bnirl.py` (BNIRL, fixed-K=2 Gibbs sampler), `archive/birl_wrapper.py`
(BIRL, PolicyWalk MCMC — Ramachandran & Amir 2007), and `archive/trex.py`
(T-REX neural reward learning, requires PyTorch) also remain in the repo and
are directly importable/callable, but are excluded from `run_algorithms.py`'s
default `ALGORITHMS` suite — see [File structure](#file-structure).

Each algorithm is self-contained: it can be imported and called independently
without going through `run_algorithms.py`.

---

## Input data format

### Layout file

Filename pattern: `*_grid{N}_coin_layout.json`

```json
{
  "grid_layout":      [["#", "_", ...], ...],
  "goal_pos":         [col, row_from_top],
  "coin_pos":         [col, row_from_top],
  "agent_start_pos":  [col, row_from_top]
}
```

- `grid_layout`: 2D list of strings. `"#"` = wall; any other value = open cell
  (`"_"` = empty, `"A"` = agent start, `"G"` = goal, `"C"` = coin).
  Row 0 is the top of the grid.
- All positions are `[col, row_from_top]` (JSON coordinate convention).

### Trajectory files

Filename pattern: `*_grid{N}_coin_low_traj{M}.json`

```json
{
  "steps": [
    {
      "grid_state":      ["header row", "0 # A  ", "1  G  ", ...],
      "agent_action":    "UP",
      "coin_collected":  false
    },
    ...
  ]
}
```

- `grid_state`: list of strings encoding the grid at that step. Row indices
  appear as the first token on each row; `A` marks the agent position.
- `agent_action`: one of `UP`, `DOWN`, `LEFT`, `RIGHT`.
- `coin_collected`: whether the coin has been collected by this step.

Only trajectories that collect the coin **and** reach `goal_pos` are used
(filtered by `is_successful` in `run_algorithms.py`).

---

## Output

All output is written to `<data_dir>/plots/`:

| File | Description |
|---|---|
| `{grid_label}_individual.png` | One panel per (trajectory × algorithm); heatmap of score grid |
| `{grid_label}_aggregated.png` | One panel per algorithm; all trajectories pooled |
| `manhattan_summary.png` | Bar chart of Manhattan distance to true subgoal, by grid and algorithm |
| `results.csv` | One row per (grid, traj/AGG, algorithm) with full score grid as JSON |

`grid_label` is a short, unique-per-directory string like `comp0.0_grid0` (see
[Grid identification](#grid-identification) below) — not a bare int.

### `results.csv` columns

`grid_id`, `traj_id`, `mode_type`, `algorithm`, `score_type`,
`predicted_col`, `predicted_row`, `true_coin_col`, `true_coin_row`,
`manhattan_dist`, `score_grid_json`

`grid_id` holds the same short `grid_label` string used in filenames (e.g.
`comp0.0_grid0`), not a bare int. `score_grid_json` is a 2D JSON array indexed
`[row_from_top][col]`. Wall cells and excluded cells (start, goal) are `null`.

### Grid identification

A directory can contain multiple config variants (e.g. several
`grid_complexity` values) that each re-use the same `grid_id` numbers — a bare
`grid_id` is **not** unique within such a directory. `run_algorithms.py`
identifies each grid by the full filename prefix before `_coin_layout.json`
(e.g. `..._comp0.0_grid0`), and derives the short `comp{X}_grid{N}` label
above purely for display/filenames. `load_grid(data_dir, grid_key, ...)`
takes this full prefix, not a bare `grid_id`.

---

## Coordinate conventions

Two coordinate systems are used internally:

| Convention | Format | Origin |
|---|---|---|
| JSON coords | `(col, row_from_top)` | Row 0 = top of grid |
| MDP coords | `(x, y_mdp)` | `y=0` = bottom row |

Conversion helpers are in `algorithms_util.py` (`to_mdp`, `to_json`).
All public wrappers accept and return JSON coordinates.
Score grids are always indexed `[row_from_top][col]` (display orientation).

---

## File structure

```
algorithms/
├── run_algorithms.py     # Entry point — orchestrates all algorithms, plots, CSV
│
├── visit_frequency.py            # Baseline: pooled cell visit frequency
├── trajectory_visit_frequency.py # Baseline: per-trajectory (deduped) visit frequency
├── surprise_v2.py                # Surprise model (terminal-directed)
├── inv_planning.py               # Bayesian inverse planning
├── maxent_irl.py                 # Maximum Causal Entropy IRL
│
├── gridworld.py          # GridMdpOld class and construct_mdp_old helper
├── mdp_funcs.py          # Generic MDP solvers: policy_iteration, value_function, q_value
├── grid_utils.py         # Grid geometry: orientations, vector_add, turn helpers
├── algorithms_util.py    # Coordinate conversion (to_mdp, to_json) + shared grid/trajectory
│                          # builders (build_gridmdp_old, json_path_to_mdp_traj)
│
├── tests/                 # Notebook tests for surprise_v2.py
│
└── archive/               # Not part of the default 5-algorithm suite; kept for reference
    ├── bnirl.py             # BNIRL Gibbs sampler
    ├── birl_wrapper.py      # BIRL PolicyWalk MCMC
    ├── trex.py              # T-REX neural reward learning (requires torch)
    ├── run_trex.py          # T-REX entry point
    └── irl_comparison.ipynb # Legacy pre-restructure comparison notebook
```

`algorithms_util.py` exports `build_gridmdp_old` and `json_path_to_mdp_traj`,
which are reused by `inv_planning.py`, `maxent_irl.py`, and (from `archive/`)
`birl_wrapper.py`.

---

## Dependencies

Default 5-algorithm suite:

```
numpy
scipy        # logsumexp (inv_planning.py)
matplotlib   # plotting only (run_algorithms.py)
```

`archive/` extras (only needed if running that code directly): `scipy`
(`multivariate_normal`, `birl_wrapper.py`), `tqdm` (BIRL progress bar,
`birl_wrapper.py`), `torch` (`trex.py`). `archive/trex.py` also imports
`analysis.metrics` from the repo root — the only import in this package
that reaches outside `algorithms/`.

---

## Compatibility with coinenv-cli

**Compatible commands** (produce correctly named files):

| Command | Layout file | Trajectory file |
|---|---|---|
| `get_multiple_trajectories_coin_env` | `*_grid{N}_coin_layout.json` | `*_grid{N}_coin_{effort}_traj*.json` |

**Not directly compatible** (layout files lack the `_coin_layout` suffix):
- `augment_from_layouts` — produces `*_iso{type}_layout.json`
- `reshuffle_walls_from_layouts` — produces `*_walls{N}_layout.json`

**Effort level:** use `--effort` to select which trajectory files are loaded (default `low`):

```bash
python run_algorithms.py data/my_run/ --effort low
python run_algorithms.py data/my_run/ --effort medium
```

This matches files named `*_coin_low_traj*.json`, `*_coin_medium_traj*.json`, etc.
