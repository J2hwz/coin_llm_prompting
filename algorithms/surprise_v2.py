"""
surprise_v2.py
--------------
Terminal-directed surprise model for subgoal inference.

Algorithm
---------
Instead of evaluating each candidate cell as a possible subgoal (v1), this
model asks: "at which cell did the agent act most surprisingly, given that a
rational agent should be heading toward the known terminal goal?"

For each step (pos, action) in a trajectory:
  1. Compute the combined prior p(action | pos, prev_action, terminal_goal)
     using the same movement × state prior as v1, but with the terminal as
     the reference goal rather than a candidate subgoal.
  2. Compute Shannon surprise: h = −log₂ p(action)
  3. Accumulate h at pos.

The cell with the highest total surprise is the predicted subgoal. For
multiple trajectories, per-cell surprise is summed across all trajectories.

Intuition
---------
- Actions toward the terminal have HIGH prior → LOW surprise.
- Actions away from the terminal (or against momentum) have LOW prior → HIGH surprise.
- The agent should accumulate high surprise at the subgoal, because that is
  where it deviates most from efficient terminal-directed navigation.

Reference
---------
Buidze et al. (2025) — parameters unchanged from the original model.

Public API
----------
    run_surprise_v2(paths, layout, params=DEFAULT_PARAMS)
        -> (score_grid_display, predicted_json_coords)

    infer_subgoal_v2(trajectories, terminal_goal, params, grid_size, exclude=None)
        -> (score_grid, best)

    trajectory_surprise(trajectory, terminal_goal, params, grid_size)
        -> {pos: total_surprise}

    step_surprise(action, pos, prev_action, terminal_goal, params, grid_size)
        -> float

Coordinate conventions
----------------------
- JSON coords : (col, row_from_top)  — row 0 = top of grid image
- MDP coords  : (x, y_mdp)           — y=0 = bottom row
- score_grid from infer_subgoal_v2 is indexed [y_mdp][x].
- score_grid_display from run_surprise_v2 is indexed [row_top][col].
"""

import sys
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_ALGO_DIR = Path(__file__).resolve().parent   # algorithms/
if str(_ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGO_DIR))

from algorithms_util import to_mdp, to_json


# ── Model parameters (Buidze et al., 2025, Table 1) ──────────────────────────

LAMBDA  = 0.10   # backward movement probability          (Eq. 4)
GAMMA   = 2.14   # forward-to-lateral ratio               (Eq. 5)
ALPHA   = 1.84   # state prior decay rate                 (Eq. 8)
TAU     = 6.93   # softmax temperature  (unused in v2 core, kept for API compat)
EPSILON = 0.39   # reward discount      (unused in v2 core, kept for API compat)

DEFAULT_PARAMS = (LAMBDA, GAMMA, ALPHA, TAU, EPSILON)


# ── Action definitions ────────────────────────────────────────────────────────

ACTIONS      = ['N', 'E', 'S', 'W']
ACTION_DELTA = {'N': (0, 1), 'E': (1, 0), 'S': (0, -1), 'W': (-1, 0)}
OPPOSITE     = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
LEFT_OF      = {'N': 'W', 'W': 'S', 'S': 'E', 'E': 'N'}
RIGHT_OF     = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}


# ── Helper ────────────────────────────────────────────────────────────────────

def _manhattan(a, b):
    """Manhattan distance between two (x, y) grid positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ── Prior functions (identical to v1) ────────────────────────────────────────

def movement_prior(prev_action, lambda_param, gamma_param):
    """Compute movement-kinetics prior over the 4 cardinal actions (Eqs. 3–7).

    Parameters
    ----------
    prev_action : str or None
        Previous action ('N', 'E', 'S', 'W'), or None on the first step
        (returns a uniform prior).
    lambda_param : float
        Probability assigned to the backward direction.
    gamma_param : float
        Forward-to-lateral ratio parameter.

    Returns
    -------
    dict[str, float]
        Action → prior probability. Values sum to 1.
    """
    if prev_action is None:
        return {a: 0.25 for a in ACTIONS}

    p_b  = lambda_param
    p_f  = (gamma_param / (gamma_param + 1.0)) * (1.0 - lambda_param)
    p_lr = 0.5 * (1.0 - gamma_param / (gamma_param + 1.0)) * (1.0 - lambda_param)

    backward = OPPOSITE[prev_action]
    return {
        prev_action:           p_f,
        backward:              p_b,
        LEFT_OF[prev_action]:  p_lr,
        RIGHT_OF[prev_action]: p_lr,
    }


def state_prior(pos, goal, alpha_param, grid_size):
    """Compute state prior over actions based on resulting-cell proximity to goal (Eq. 8).

    Parameters
    ----------
    pos : tuple[int, int]
        Current (x, y) position.
    goal : tuple[int, int]
        Reference goal (x, y) — in v2 this is always the terminal goal.
    alpha_param : float
        Decay rate: p(s') ∝ 1/alpha^d where d = Manhattan(s', goal).
    grid_size : int
        Grid side length.

    Returns
    -------
    dict[str, float]
        Action → normalised state prior of resulting cell.
        Out-of-bounds actions receive weight 0 before normalisation.
    """
    raw = {}
    for a in ACTIONS:
        dx, dy = ACTION_DELTA[a]
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < grid_size and 0 <= ny < grid_size:
            d = _manhattan((nx, ny), goal)
            raw[a] = 1.0 / (alpha_param ** d) if d > 0 else 1.0
        else:
            raw[a] = 0.0
    total = sum(raw.values())
    if total == 0:
        valid = [a for a in ACTIONS
                 if 0 <= pos[0] + ACTION_DELTA[a][0] < grid_size
                 and 0 <= pos[1] + ACTION_DELTA[a][1] < grid_size]
        return {a: (1.0 / len(valid) if a in valid else 0.0) for a in ACTIONS}
    return {a: v / total for a, v in raw.items()}


def combined_prior(pos, prev_action, goal, lambda_param, gamma_param,
                   alpha_param, grid_size):
    """Multiplicatively combine movement and state priors, then normalise (Eq. 9).

    Parameters
    ----------
    pos : tuple[int, int]
    prev_action : str or None
    goal : tuple[int, int]
        Reference goal — in v2 this is always the terminal goal.
    lambda_param, gamma_param, alpha_param : float
    grid_size : int

    Returns
    -------
    dict[str, float]
        Normalised combined action probabilities. Sums to 1.
    """
    mp = movement_prior(prev_action, lambda_param, gamma_param)
    sp = state_prior(pos, goal, alpha_param, grid_size)
    combined = {a: mp[a] * sp[a] for a in ACTIONS}
    total = sum(combined.values())
    if total == 0:
        return {a: 0.25 for a in ACTIONS}
    return {a: v / total for a, v in combined.items()}


# ── Core surprise functions ───────────────────────────────────────────────────

def step_surprise(action, pos, prev_action, terminal_goal, params, grid_size):
    """Shannon surprise of a single action relative to the terminal-directed prior.

    Surprise = −log₂ p(action | pos, prev_action, terminal_goal)

    High surprise means the agent took an action that was unexpected given
    rational navigation toward the terminal: either moving away from it
    (violates state prior) or breaking momentum (violates movement prior).

    Parameters
    ----------
    action : str
        Observed action ('N', 'E', 'S', 'W').
    pos : tuple[int, int]
        Current (x, y_mdp) position.
    prev_action : str or None
        Previous action, or None for the first step.
    terminal_goal : tuple[int, int]
        Known terminal goal in MDP coordinates.
    params : tuple[float, ...]
        (lambda, gamma, alpha, tau, epsilon) — only lambda, gamma, alpha used.
    grid_size : int

    Returns
    -------
    float
        Shannon surprise in bits (≥ 0). Higher = more unexpected.
    """
    lambda_p, gamma_p, alpha_p, _, _ = params
    prior = combined_prior(pos, prev_action, terminal_goal,
                           lambda_p, gamma_p, alpha_p, grid_size)
    return -np.log2(max(prior[action], 1e-12))


def trajectory_surprise(trajectory, terminal_goal, params, grid_size):
    """Accumulate per-step surprise at each visited cell for one trajectory.

    For each step (pos, action), compute step_surprise and add it to the
    running total for pos. If the agent returns to a cell (e.g. oscillation),
    surprise from all visits is summed.

    Parameters
    ----------
    trajectory : list of ((x, y_mdp), action_str)
        One trajectory in MDP coordinates.
    terminal_goal : (x, y_mdp)
    params : tuple[float, ...]
    grid_size : int

    Returns
    -------
    dict[(x, y_mdp), float]
        Total surprise accumulated at each cell. Non-negative values.
    """
    scores = {}
    for step, (pos, action) in enumerate(trajectory):
        prev = trajectory[step - 1][1] if step > 0 else None
        h = step_surprise(action, pos, prev, terminal_goal, params, grid_size)
        scores[pos] = scores.get(pos, 0.0) + h
    return scores


# ── Inference ─────────────────────────────────────────────────────────────────

def infer_subgoal_v2(trajectories, terminal_goal, params, grid_size,
                     exclude=None):
    """Infer subgoal as the most surprising cell across all trajectories.

    For each trajectory, per-cell surprise is computed via trajectory_surprise.
    Scores are summed element-wise across trajectories, then argmax is taken
    (excluding specified cells such as start and terminal).

    Parameters
    ----------
    trajectories : list of list of ((x, y_mdp), action_str)
        One or more trajectories in MDP coordinates.
    terminal_goal : (x, y_mdp)
        Known terminal goal (excluded from prediction by default via `exclude`).
    params : tuple[float, ...]
        (lambda, gamma, alpha, tau, epsilon) — only lambda, gamma, alpha used.
    grid_size : int
        Grid side length (score grid is grid_size × grid_size).
    exclude : set of (x, y_mdp), optional
        Cells to exclude from argmax (typically: starts + terminal).

    Returns
    -------
    score_grid : np.ndarray, shape (grid_size, grid_size)
        Total surprise at each cell, indexed [y_mdp][x].
        Cells never visited are NaN.
    best : (x, y_mdp)
        Cell with the highest total surprise, excluding `exclude`.
    """
    total = {}
    for traj in trajectories:
        for pos, h in trajectory_surprise(traj, terminal_goal, params, grid_size).items():
            total[pos] = total.get(pos, 0.0) + h

    score_grid = np.full((grid_size, grid_size), np.nan)
    for (x, y), h in total.items():
        if 0 <= x < grid_size and 0 <= y < grid_size:
            score_grid[y, x] = h

    candidates = {p: h for p, h in total.items()
                  if exclude is None or p not in exclude}
    best = max(candidates, key=candidates.get)
    return score_grid, best


def path_to_surprise_traj(path, n_rows):
    """Convert a JSON-coord path to a list of (pos_mdp, action_str) pairs.

    Wall-bump steps (zero net movement in MDP coords) are dropped.

    Parameters
    ----------
    path : list of (col, row_top)
        Positions in JSON coordinates, as returned by build_path().
    n_rows : int

    Returns
    -------
    list of ((x, y_mdp), action_str)
    """
    _dir_map = {(0, 1): 'N', (0, -1): 'S', (1, 0): 'E', (-1, 0): 'W'}
    traj = []
    for i in range(len(path) - 1):
        col, rt = path[i]
        pos  = to_mdp(col, rt, n_rows)
        dx   = path[i + 1][0] - path[i][0]
        dy_t = path[i + 1][1] - path[i][1]
        dy_m = -dy_t                           # flip: MDP y increases upward
        act  = _dir_map.get((dx, dy_m))
        if act is not None:                    # skip wall-bumps
            traj.append((pos, act))
    return traj


# ── Public wrapper ────────────────────────────────────────────────────────────

def run_surprise_v2(paths, layout, params=DEFAULT_PARAMS):
    """Terminal-directed surprise model for subgoal inference.

    Scores each visited cell by total Shannon surprise of steps taken from it,
    where surprise is measured against the combined prior directed at the
    terminal goal. The predicted subgoal is the highest-scoring cell,
    excluding the terminal and agent start cells.

    Same interface as run_surprise (v1).

    Parameters
    ----------
    paths : list of list of (col, row_top)
        One or more trajectory paths in JSON coordinates.
    layout : dict
        Grid layout dict with keys: grid_layout, goal_pos, coin_pos,
        agent_start_pos.
    params : tuple
        (lambda, gamma, alpha, tau, epsilon). Default: DEFAULT_PARAMS.
        Only lambda, gamma, alpha affect the result.

    Returns
    -------
    score_grid_display : np.ndarray, shape (n_rows, n_cols)
        Total surprise at each cell, indexed [row_top, col].
        Cells never visited, wall cells, and the terminal are NaN.
    predicted_json_coords : (col, row_top)
        Cell with the highest surprise score.
    """
    n_rows = len(layout["grid_layout"])
    n_cols = len(layout["grid_layout"][0])
    goal_j = tuple(layout["goal_pos"])
    goal_m = to_mdp(*goal_j, n_rows)
    grid   = layout["grid_layout"]

    grid_size = max(n_rows, n_cols)

    trajs_s = [path_to_surprise_traj(p, n_rows) for p in paths]
    trajs_s = [t for t in trajs_s if t]

    if not trajs_s:
        return np.full((n_rows, n_cols), np.nan), goal_j

    start_set = {to_mdp(*p[0], n_rows) for p in paths if p}
    exclude   = start_set | {goal_m}

    score_grid, best_m = infer_subgoal_v2(
        trajs_s, goal_m, params, grid_size, exclude=exclude,
    )

    # score_grid[y_mdp][x] → flip to display coords [row_top][col]
    score_display = np.full((n_rows, n_cols), np.nan)
    for y_m in range(n_rows):
        for col in range(n_cols):
            if col < score_grid.shape[1] and y_m < score_grid.shape[0]:
                v = score_grid[y_m, col]
                if not np.isnan(v):
                    score_display[n_rows - 1 - y_m, col] = v

    # Mask wall cells
    for rt in range(n_rows):
        for col in range(n_cols):
            if grid[rt][col] == "#":
                score_display[rt, col] = np.nan

    return score_display, to_json(*best_m, n_rows)
