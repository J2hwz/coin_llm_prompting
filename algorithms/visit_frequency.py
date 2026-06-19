"""
visit_frequency.py
------------------
State-visit frequency baseline for subgoal inference.

The predicted subgoal is the most-visited non-start, non-goal cell across
all provided trajectories. No model or MDP is needed.

Public API
----------
    run_visit_freq(paths, layout)
        -> (score_grid, predicted_json_coords)

Coordinate conventions
----------------------
- JSON coords : (col, row_from_top) — row 0 = top
- score_grid  : [n_rows, n_cols], indexed [row_top, col]
"""

import sys
from pathlib import Path

import numpy as np

_ALGO_DIR = Path(__file__).resolve().parent
if str(_ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGO_DIR))


def run_visit_freq(paths, layout):
    """
    State-visit frequency baseline.

    Counts how many times each cell is visited across all trajectories,
    then returns the most-visited non-start, non-goal cell.

    Parameters
    ----------
    paths : list of list of (col, row_top)
    layout : dict  — grid_layout, goal_pos, coin_pos, agent_start_pos

    Returns
    -------
    score_grid : np.ndarray [n_rows, n_cols]   visit counts (NaN for excluded)
    predicted  : (col, row_top)
    """
    n_rows = len(layout["grid_layout"])
    n_cols = len(layout["grid_layout"][0])
    grid   = layout["grid_layout"]
    goal_j = tuple(layout["goal_pos"])

    start_set = {tuple(p[0]) for p in paths if p}
    excluded  = start_set | {goal_j}

    counts = np.zeros((n_rows, n_cols))
    for path in paths:
        for col, rt in path:
            if (col, rt) not in excluded and grid[rt][col] != "#":
                counts[rt, col] += 1

    score = np.full((n_rows, n_cols), np.nan)
    best, best_count = goal_j, -1
    for rt in range(n_rows):
        for col in range(n_cols):
            if grid[rt][col] == "#" or (col, rt) in excluded:
                continue
            score[rt, col] = counts[rt, col]
            if counts[rt, col] > best_count:
                best_count = counts[rt, col]
                best = (col, rt)

    return score, best
