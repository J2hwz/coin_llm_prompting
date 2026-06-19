# algorithms/

Self-contained package for subgoal inference from gridworld trajectory data.
Given observed human navigation trajectories and a grid layout, each algorithm
predicts which cell the agent was heading toward as an intermediate subgoal
(before the known terminal goal).

---

## Entry point

```
python run_algorithms.py <data_dir> [--mode {individual,pooled,both}]
```

`<data_dir>` must contain the layout and trajectory JSON files described below.
Results are written to `<data_dir>/plots/`.

**Modes**

| Mode | Description |
|---|---|
| `individual` | Run each trajectory separately |
| `pooled` | Pool all successful trajectories and run once |
| `both` | Both (default) |

---

## Algorithms

Six algorithms are run in sequence. Each takes `(paths, layout)` and returns
`(score_grid, predicted_json_coords)`.

| Name | File | Score type | Method |
|---|---|---|---|
| Visit Freq | `visit_frequency.py` | `visits` | Baseline: most-visited non-start/non-goal cell |
| Surprise v2 | `surprise_v2.py` | `surprise` | Shannon surprise relative to terminal-directed prior (Buidze et al. 2025) |
| Inv. Planning | `inv_planning.py` | `posterior` | Bayesian inverse planning via soft value iteration (Baker et al. 2011) |
| BNIRL | `bnirl.py` | `samples` | Fixed-K=2 Gibbs sampler (non-parametric Bayesian IRL) |
| BIRL | `birl_wrapper.py` | `reward` | PolicyWalk MCMC (Ramachandran & Amir 2007) |
| MaxEnt IRL | `maxent_irl.py` | `reward` | Maximum Causal Entropy IRL (Ziebart 2010) |

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
| `grid{N}_individual.png` | One panel per (trajectory × algorithm); heatmap of score grid |
| `grid{N}_aggregated.png` | One panel per algorithm; all trajectories pooled |
| `manhattan_summary.png` | Bar chart of Manhattan distance to true subgoal, by grid and algorithm |
| `results.csv` | One row per (grid, traj/AGG, algorithm) with full score grid as JSON |

### `results.csv` columns

`grid_id`, `traj_id`, `mode_type`, `algorithm`, `score_type`,
`predicted_col`, `predicted_row`, `true_coin_col`, `true_coin_row`,
`manhattan_dist`, `score_grid_json`

`score_grid_json` is a 2D JSON array indexed `[row_from_top][col]`.
Wall cells and excluded cells (start, goal) are `null`.

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
├── visit_frequency.py    # Baseline: state visit frequency
├── surprise_v2.py        # Surprise model (terminal-directed)
├── inv_planning.py       # Bayesian inverse planning + shared MDP builder
├── bnirl.py              # BNIRL Gibbs sampler
├── birl_wrapper.py       # BIRL PolicyWalk MCMC
├── maxent_irl.py         # Maximum Causal Entropy IRL
│
├── gridworld.py          # GridMdpOld class and construct_mdp_old helper
├── mdp_funcs.py          # Generic MDP solvers: policy_iteration, value_function, q_value
├── grid_utils.py         # Grid geometry: orientations, vector_add, turn helpers
└── algorithms_util.py    # Coordinate conversion: to_mdp, to_json
```

`inv_planning.py` also exports `build_gridmdp_old` and `json_path_to_mdp_traj`,
which are reused by `birl_wrapper.py` and `maxent_irl.py`.

---

## Dependencies

```
numpy
scipy        # logsumexp (inv_planning), multivariate_normal (birl_wrapper)
matplotlib   # plotting only (run_algorithms.py)
tqdm         # BIRL progress bar (birl_wrapper.py)
```

No other files outside this directory are imported.

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
