import numpy as np

from visit_frequency import run_visit_freq


def test_smoke_dead_end_fixture(dead_end_fixture):
    layout, paths, true_coin_json = dead_end_fixture
    score, predicted = run_visit_freq(paths, layout)

    n_rows = len(layout["grid_layout"])
    n_cols = len(layout["grid_layout"][0])
    assert score.shape == (n_rows, n_cols)
    assert isinstance(predicted, tuple) and len(predicted) == 2
    col, row_top = predicted
    assert layout["grid_layout"][row_top][col] != "#"


def test_argmax_matches_predicted(dead_end_fixture):
    layout, paths, _ = dead_end_fixture
    score, predicted = run_visit_freq(paths, layout)
    col, row_top = predicted
    finite = score[~np.isnan(score)]
    assert score[row_top, col] == finite.max()


def test_baselines_count_wall_hits(dead_end_fixture, wall_bump_paths):
    """Cell Visit Freq must count a wall-bump revisit as extra evidence for that cell."""
    layout, _, _ = dead_end_fixture
    clean, with_detour = wall_bump_paths

    score_clean, _ = run_visit_freq([clean], layout)
    score_detour, _ = run_visit_freq([with_detour], layout)

    # (3, 1) is the junction cell where the extra wall-bump duplicate sits.
    junction_col, junction_row = 3, 1
    assert score_detour[junction_row, junction_col] > score_clean[junction_row, junction_col]
