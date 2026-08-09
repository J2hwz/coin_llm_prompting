"""
Re-exposes inv_planning.py's existing inline smoke tests to pytest (they
were previously only runnable via `python inv_planning.py`), plus new tests
for the argmax-of-score-grid contract and wall-bump handling.
"""

import numpy as np

from inv_planning import (
    run_inv_planning,
    test_sanity_dead_end_argmax,
    test_order_invariance,
    test_missing_visit_not_disqualifying,
    test_wall_bump_steps_collapse_to_next_real_move,
    test_no_underflow_many_trajectories,
)

__all__ = [
    "test_sanity_dead_end_argmax",
    "test_order_invariance",
    "test_missing_visit_not_disqualifying",
    "test_wall_bump_steps_collapse_to_next_real_move",
    "test_no_underflow_many_trajectories",
]


def test_smoke_dead_end_fixture(dead_end_fixture):
    layout, paths, true_coin_json = dead_end_fixture
    score, predicted = run_inv_planning(paths, layout)

    n_rows = len(layout["grid_layout"])
    n_cols = len(layout["grid_layout"][0])
    assert score.shape == (n_rows, n_cols)
    col, row_top = predicted
    assert layout["grid_layout"][row_top][col] != "#"


def test_argmax_matches_predicted(dead_end_fixture):
    layout, paths, _ = dead_end_fixture
    score, predicted = run_inv_planning(paths, layout)
    col, row_top = predicted
    finite = score[~np.isnan(score)]
    assert np.isclose(score[row_top, col], finite.max())


def test_wall_bump_does_not_change_prediction(dead_end_fixture, wall_bump_paths):
    """A wall-bump duplicate must be dropped by json_path_to_mdp_traj, so the
    two paths (identical except for the extra no-op step) must score the same."""
    layout, _, _ = dead_end_fixture
    clean, with_detour = wall_bump_paths

    score_clean, pred_clean = run_inv_planning([clean], layout)
    score_detour, pred_detour = run_inv_planning([with_detour], layout)

    assert pred_clean == pred_detour
    np.testing.assert_allclose(score_clean, score_detour, equal_nan=True)
