"""
algorithms_util.py
------------------
Shared coordinate, grid-construction, and trajectory helpers for the
algorithms package.

Coordinate conventions
----------------------
- JSON coords : (col, row_from_top)  — row 0 = top of grid image
- MDP coords  : (x, y_mdp)           — y=0 = bottom row
"""

import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_ALGO_DIR = Path(__file__).resolve().parent
if str(_ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGO_DIR))

from gridworld import GridMdpOld


def to_mdp(col, row_top, n_rows):
    """Convert JSON (col, row_top) to MDP (col, y_mdp) coordinates."""
    return (col, n_rows - 1 - row_top)


def to_json(col, y_mdp, n_rows):
    """Convert MDP (col, y_mdp) to JSON (col, row_top) coordinates."""
    return (col, n_rows - 1 - y_mdp)


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
