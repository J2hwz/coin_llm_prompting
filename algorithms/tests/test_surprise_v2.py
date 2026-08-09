import numpy as np

from surprise_v2 import run_surprise_v2, path_to_surprise_traj
from algorithms_util import to_mdp


def test_smoke_dead_end_fixture(dead_end_fixture):
    layout, paths, true_coin_json = dead_end_fixture
    score, predicted = run_surprise_v2(paths, layout)

    n_rows = len(layout["grid_layout"])
    n_cols = len(layout["grid_layout"][0])
    assert score.shape == (n_rows, n_cols)
    col, row_top = predicted
    assert layout["grid_layout"][row_top][col] != "#"


def test_argmax_matches_predicted(dead_end_fixture):
    layout, paths, _ = dead_end_fixture
    score, predicted = run_surprise_v2(paths, layout)
    col, row_top = predicted
    finite = score[~np.isnan(score)]
    assert np.isclose(score[row_top, col], finite.max())


def test_wall_bump_does_not_change_prediction(dead_end_fixture, wall_bump_paths):
    layout, _, _ = dead_end_fixture
    clean, with_detour = wall_bump_paths

    score_clean, pred_clean = run_surprise_v2([clean], layout)
    score_detour, pred_detour = run_surprise_v2([with_detour], layout)

    assert pred_clean == pred_detour
    np.testing.assert_allclose(score_clean, score_detour, equal_nan=True)


def test_path_to_surprise_traj_drops_wall_bump():
    """The wall-bump step (repeated position) must not appear as an entry."""
    n_rows = 3
    path = [(0, 0), (1, 0), (1, 0), (1, 0), (1, 0), (2, 0)]
    traj = path_to_surprise_traj(path, n_rows)
    assert len(traj) == 2
    assert traj[0] == (to_mdp(0, 0, n_rows), "E")
    assert traj[1] == (to_mdp(1, 0, n_rows), "E")
