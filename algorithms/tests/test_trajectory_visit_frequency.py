import numpy as np

from trajectory_visit_frequency import run_trajectory_visit_freq


def test_smoke_dead_end_fixture(dead_end_fixture):
    layout, paths, true_coin_json = dead_end_fixture
    score, predicted = run_trajectory_visit_freq(paths, layout)

    n_rows = len(layout["grid_layout"])
    n_cols = len(layout["grid_layout"][0])
    assert score.shape == (n_rows, n_cols)
    col, row_top = predicted
    assert layout["grid_layout"][row_top][col] != "#"


def test_argmax_matches_predicted(dead_end_fixture):
    layout, paths, _ = dead_end_fixture
    score, predicted = run_trajectory_visit_freq(paths, layout)
    col, row_top = predicted
    finite = score[~np.isnan(score)]
    assert score[row_top, col] == finite.max()


def test_revisit_within_trajectory_counts_once(dead_end_fixture, wall_bump_paths):
    """A cell visited multiple times by the SAME trajectory (wall-bump or not)
    must still only contribute 1 to that cell's trajectory-count."""
    layout, _, _ = dead_end_fixture
    clean, with_detour = wall_bump_paths

    score_clean, _ = run_trajectory_visit_freq([clean], layout)
    score_detour, _ = run_trajectory_visit_freq([with_detour], layout)

    junction_col, junction_row = 3, 1
    assert score_clean[junction_row, junction_col] == 1
    assert score_detour[junction_row, junction_col] == 1


def test_differs_from_cell_visit_freq_on_revisits(dead_end_fixture):
    """Two trajectories where one revisits a cell 3x and the other visits it
    once must be deduped (score 2), unlike raw cell-level counting (score 4)."""
    from visit_frequency import run_visit_freq

    layout, _, _ = dead_end_fixture
    paths = [
        [[1, 1], [2, 1], [3, 1], [3, 2], [3, 1], [3, 2], [3, 1], [4, 1], [5, 1]],
        [[1, 1], [2, 1], [3, 1], [3, 2], [3, 1], [4, 1], [5, 1]],
    ]
    score_t, _ = run_trajectory_visit_freq(paths, layout)
    score_c, _ = run_visit_freq(paths, layout)

    assert score_t[2, 3] == 2
    assert score_c[2, 3] == 3
