"""
algorithms_util.py
------------------
Shared coordinate helpers for the algorithms package.

Coordinate conventions
----------------------
- JSON coords : (col, row_from_top)  — row 0 = top of grid image
- MDP coords  : (x, y_mdp)           — y=0 = bottom row
"""


def to_mdp(col, row_top, n_rows):
    """Convert JSON (col, row_top) to MDP (col, y_mdp) coordinates."""
    return (col, n_rows - 1 - row_top)


def to_json(col, y_mdp, n_rows):
    """Convert MDP (col, y_mdp) to JSON (col, row_top) coordinates."""
    return (col, n_rows - 1 - y_mdp)
