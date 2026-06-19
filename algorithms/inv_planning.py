"""
inv_planning.py
---------------
Bayesian Inverse Planning (subgoal inference via soft value iteration).

Algorithm
---------
For each candidate subgoal g, compute a soft-optimal policy toward g using
soft value iteration, then evaluate the likelihood of the observed trajectory
under that policy. A Bayesian posterior over subgoals is formed by normalising
these likelihoods. The predicted subgoal is the argmax of the posterior.

Reference
---------
Baker, Saxe & Tenenbaum (2011). Bayesian theory of mind.
Implementation: jupyter_demos/inverse_planning/bayesian_subgoal_inference_demo.py

Public API
----------
    run_inv_planning(paths, layout, beta=2.0)
        -> (score_grid_display, predicted_json_coords)

Coordinate conventions
----------------------
- JSON coords : (col, row_from_top)  — row 0 = top of grid image
- MDP coords  : (x, y_mdp)           — y=0 = bottom row
- Conversion  : y_mdp = n_rows - 1 - row_top
"""

import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

# ── Path setup ────────────────────────────────────────────────────────────────
_ALGO_DIR = Path(__file__).resolve().parent
if str(_ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGO_DIR))

from algorithms_util import to_mdp, to_json
from gridworld import GridMdpOld

# Cardinal actions in MDP (x, y_mdp) space: W, S, E, N
_ACTIONS = [(-1, 0), (0, -1), (1, 0), (0, 1)]


# ── Soft value iteration ──────────────────────────────────────────────────────

def _soft_value_iteration(mdp, goal, beta, max_iter=600, tol=1e-7):
    """Backward pass from goal using soft Bellman equations.

    V(goal) = 0 (absorbing), step cost = −1 for all other states.
    V(s)    = (1/β) · log Σ_a exp(β · Q(s, a))
    Q(s, a) = −1 + γ · Σ_s' T(s, a, s') · V(s')
    """
    V = {s: 0.0 for s in mdp.states}
    for _ in range(max_iter):
        V_new = {}
        for s in mdp.states:
            if s == goal:
                V_new[s] = 0.0
                continue
            qs = np.array([
                -1.0 + mdp.gamma * sum(p * V[sp]
                                       for p, sp in mdp.transitions[s][a])
                for a in _ACTIONS
            ])
            V_new[s] = (1.0 / beta) * logsumexp(beta * qs)
        delta = max(abs(V_new[s] - V[s]) for s in mdp.states)
        V = V_new
        if delta < tol:
            break

    Q = {}
    for s in mdp.states:
        for a in _ACTIONS:
            if s == goal:
                Q[s, a] = 0.0
            else:
                Q[s, a] = (-1.0 + mdp.gamma *
                           sum(p * V[sp] for p, sp in mdp.transitions[s][a]))
    return V, Q


# ── Likelihood helpers ────────────────────────────────────────────────────────

def _step_log_prob(Q, s, a, beta):
    """log P(a | s) under the Boltzmann policy with Q-function Q."""
    qs = np.array([Q[s, a_] for a_ in _ACTIONS])
    return float(beta * Q[s, a] - logsumexp(beta * qs))


def _traj_log_likelihood(traj, Q_c, Q_g, subgoal, beta):
    """log P(trajectory | subgoal = c).

    Splits at t* = first visit to c.
    Leg 1 (0 … t*−1): scored under Q_c (policy toward subgoal).
    Leg 2 (t* … end−1): scored under Q_g (policy toward terminal).
    Returns −∞ if trajectory never visits c.
    """
    states = [s for s, _ in traj]
    try:
        t_star = states.index(subgoal)
    except ValueError:
        return -np.inf

    ll = 0.0
    for i in range(t_star):
        s, a = traj[i]
        ll += _step_log_prob(Q_c, s, a, beta)
    for i in range(t_star, len(traj) - 1):
        s, a = traj[i]
        if a is None:
            continue
        ll += _step_log_prob(Q_g, s, a, beta)
    return ll


# ── Candidate set and posterior ───────────────────────────────────────────────

def candidate_set(trajectories, starts, terminal):
    """Union of visited states across all trajectories,
    excluding known start positions and the terminal."""
    excluded = set(starts) | {terminal}
    sets = [{s for s, _ in traj if s not in excluded} for traj in trajectories]
    if not sets:
        return frozenset()
    result = sets[0]
    for vs in sets[1:]:
        result = result | vs
    return frozenset(result)


def compute_posterior(trajectories, mdp, candidates, terminal, beta):
    """Compute P(c | trajectories) for each candidate c.

    1. Backward pass from terminal (shared across all candidates).
    2. For each c: backward pass from c, sum log-likelihoods.
    3. Log-softmax normalisation (uniform prior).

    Returns dict {candidate → posterior probability}.
    """
    _, Q_g = _soft_value_iteration(mdp, terminal, beta)

    log_liks = {}
    for c in candidates:
        _, Q_c = _soft_value_iteration(mdp, c, beta)
        log_liks[c] = sum(
            _traj_log_likelihood(t, Q_c, Q_g, c, beta) for t in trajectories
        )

    finite = {c: v for c, v in log_liks.items() if np.isfinite(v)}
    if not finite:
        n = len(log_liks)
        return {c: 1.0 / n for c in log_liks}

    ll_arr = np.array(list(finite.values()))
    probs  = np.exp(ll_arr - logsumexp(ll_arr))

    posterior = {c: 0.0 for c in log_liks}
    for c, p in zip(finite.keys(), probs):
        posterior[c] = float(p)
    return posterior


# ── Grid construction ─────────────────────────────────────────────────────────

def build_gridmdp_old(layout):
    """Construct a GridMdpOld from a layout dict loaded from JSON.

    Parameters
    ----------
    layout : dict
        Keys: grid_layout (list[list[str]]), goal_pos [col, row_top],
              coin_pos, agent_start_pos.

    Returns
    -------
    GridMdpOld
        States are (x, y_mdp) tuples. Walls are None cells.
    """
    n_rows = len(layout["grid_layout"])
    goal_col, goal_row_top = layout["goal_pos"]
    goal_y = n_rows - 1 - goal_row_top
    # GridMdpOld reverses row order internally, so pass JSON order (row 0 = top)
    raw = [
        [None if c == "#" else -1.0 for c in row]
        for row in layout["grid_layout"]
    ]
    return GridMdpOld(raw, terminal_locs=[(goal_col, goal_y)], gamma=0.99)


def json_path_to_mdp_traj(path, n_rows, terminal_marker=True):
    """Convert a JSON-coord path to an MDP trajectory.

    Parameters
    ----------
    path : list of (col, row_top)
    n_rows : int
    terminal_marker : bool
        If True, append (final_state_mdp, None) as the terminal step.

    Returns
    -------
    list of (state_mdp, action_mdp)
        Wall-bump steps (zero delta in MDP coords) are dropped.
    """
    if len(path) < 2:
        return []
    traj = []
    for i in range(len(path) - 1):
        col, rt = path[i]
        s    = to_mdp(col, rt, n_rows)
        dx   = path[i + 1][0] - path[i][0]
        dy_t = path[i + 1][1] - path[i][1]
        dy_m = -dy_t                           # flip: MDP y increases upward
        if (dx, dy_m) != (0, 0):
            traj.append((s, (dx, dy_m)))
    if terminal_marker and path:
        col, rt = path[-1]
        traj.append((to_mdp(col, rt, n_rows), None))
    return traj


# ── Main wrapper ──────────────────────────────────────────────────────────────

def run_inv_planning(paths, layout, beta=2.0):
    """Bayesian subgoal inference via soft value iteration.

    For each candidate cell, evaluates how likely the observed trajectory is
    under a soft-optimal policy aimed at that cell, then normalises to a
    posterior. Works with any number of trajectories (one-shot or aggregated).

    Parameters
    ----------
    paths : list of list of (col, row_top)
        One or more trajectory paths in JSON coordinates.
    layout : dict
        Grid layout dict with keys: grid_layout, goal_pos, coin_pos,
        agent_start_pos.
    beta : float
        Softmax temperature for the soft policy (higher = more deterministic).

    Returns
    -------
    score_grid_display : np.ndarray, shape (n_rows, n_cols)
        Posterior probability at each cell, indexed [row_top, col].
        Wall cells and the terminal (goal) cell are NaN.
    predicted_json_coords : (col, row_top)
        Cell with the highest posterior probability.
    """
    n_rows = len(layout["grid_layout"])
    n_cols = len(layout["grid_layout"][0])
    goal_j = tuple(layout["goal_pos"])
    goal_m = to_mdp(*goal_j, n_rows)

    mdp = build_gridmdp_old(layout)

    trajs_mdp = [
        json_path_to_mdp_traj(p, n_rows, terminal_marker=True)
        for p in paths if len(p) >= 2
    ]
    trajs_mdp = [t for t in trajs_mdp if t]

    if not trajs_mdp:
        return np.full((n_rows, n_cols), np.nan), goal_j

    starts = [t[0][0] for t in trajs_mdp]
    cands  = candidate_set(trajs_mdp, starts, goal_m)

    if not cands:
        excluded = set(starts) | {goal_m}
        cands = frozenset(
            s for t in trajs_mdp for s, _ in t if s not in excluded
        )

    if not cands:
        return np.full((n_rows, n_cols), np.nan), goal_j

    posterior = compute_posterior(trajs_mdp, mdp, cands, goal_m, beta)

    score = np.full((n_rows, n_cols), np.nan)
    for (col, y), prob in posterior.items():
        score[n_rows - 1 - y, col] = prob

    best_m = max(posterior, key=posterior.get)
    return score, to_json(*best_m, n_rows)
