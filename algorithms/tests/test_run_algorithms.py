import json
import subprocess
import sys
from pathlib import Path

import pytest

import run_algorithms as ra
from run_algorithms import (
    ALGORITHMS,
    _grid_key,
    _grid_label,
    build_path,
    is_successful,
    load_grid,
)

_REPO_DIR = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _REPO_DIR / "data" / "one_coin" / "low" / "7_low" / "control"

no_data = pytest.mark.skipif(
    not _DATA_DIR.exists(),
    reason="data/one_coin is gitignored and not present in this environment",
)


def _fake_step(col, action, coin_collected, row=0, n_cols=6):
    cells = ["_"] * n_cols
    cells[col] = "A"
    grid_state = ["header", f"{row} " + " ".join(cells)]
    return {"grid_state": grid_state, "agent_action": action, "coin_collected": coin_collected}


# ── build_path ────────────────────────────────────────────────────────────────

def test_build_path_keeps_wall_bump_by_default():
    steps = [
        _fake_step(0, "RIGHT", False),
        _fake_step(1, "RIGHT", False),
        _fake_step(1, "RIGHT", False),  # wall bump — stayed at col 1
        _fake_step(2, "RIGHT", False),
    ]
    path = build_path(steps, skip_invalid_actions=False)
    assert path.count((1, 0)) == 2


def test_build_path_skip_invalid_actions_drops_wall_bump():
    steps = [
        _fake_step(0, "RIGHT", False),
        _fake_step(1, "RIGHT", False),
        _fake_step(1, "RIGHT", False),  # wall bump — stayed at col 1
        _fake_step(2, "RIGHT", False),
    ]
    path = build_path(steps, skip_invalid_actions=True)
    assert path.count((1, 0)) == 1


# ── is_successful ────────────────────────────────────────────────────────────

def test_is_successful_requires_coin_collected():
    layout = {"goal_pos": [2, 0]}
    steps = [
        _fake_step(0, "RIGHT", False),
        _fake_step(1, "RIGHT", False),
        _fake_step(2, "RIGHT", False),  # reaches goal, never collects coin
    ]
    assert is_successful(steps, layout) is False


def test_is_successful_requires_goal_reached():
    layout = {"goal_pos": [5, 5]}  # never visited
    steps = [
        _fake_step(0, "RIGHT", False),
        _fake_step(1, "RIGHT", True),  # collects coin
        _fake_step(2, "RIGHT", True),
    ]
    assert is_successful(steps, layout) is False


def test_is_successful_true_when_both_conditions_met():
    layout = {"goal_pos": [2, 0]}
    steps = [
        _fake_step(0, "RIGHT", False),
        _fake_step(1, "RIGHT", True),
        _fake_step(2, "RIGHT", True),
    ]
    assert is_successful(steps, layout) is True


# ── load_grid corrupted-file handling ────────────────────────────────────────

def _open_row_layout():
    """3x3 grid_layout with an open top row (row_top=0) and walls elsewhere,
    goal/coin/start all on that open row so build_path's synthetic
    (col, row=0) positions never land on a wall."""
    return {
        "grid_layout": [["_", "_", "_"], ["#", "#", "#"], ["#", "#", "#"]],
        "goal_pos": [1, 0], "coin_pos": [1, 0], "agent_start_pos": [0, 0],
    }


def test_load_grid_skips_corrupted_trajectory_file(tmp_path):
    """A JSONDecodeError in one trajectory file must not crash the whole grid
    load — it should be skipped, with the other valid trajectories still used."""
    layout = _open_row_layout()
    (tmp_path / "g_coin_layout.json").write_text(json.dumps(layout))

    # single RIGHT step from col 0 -> path becomes [(0,0), (1,0)], reaching goal (1,0)
    steps = [_fake_step(0, "RIGHT", True, n_cols=2)]
    good_traj = {"steps": steps}
    (tmp_path / "g_coin_low_traj0.json").write_text(json.dumps(good_traj))
    (tmp_path / "g_coin_low_traj1.json").write_text('{"steps": [truncated garbage')

    layout_out, paths, traj_ids = load_grid(tmp_path, "g", effort="low")
    assert traj_ids == [0]
    assert len(paths) == 1


def test_load_grid_skips_trajectory_that_visits_a_wall_cell(tmp_path):
    """A trajectory whose recorded path lands on a wall cell relative to its
    paired layout (mismatched grid data) must be skipped, not crash or get
    silently fed to the algorithms."""
    layout = _open_row_layout()
    layout["goal_pos"] = [1, 1]  # (1, 1) is "#" in this layout -> unreachable goal
    (tmp_path / "g_coin_layout.json").write_text(json.dumps(layout))

    steps = [_fake_step(0, "RIGHT", True, n_cols=2, row=1)]  # walks along the wall row
    (tmp_path / "g_coin_low_traj0.json").write_text(json.dumps({"steps": steps}))

    layout_out, paths, traj_ids = load_grid(tmp_path, "g", effort="low")
    assert paths == []
    assert traj_ids == []


def test_path_visits_wall_detects_wall_and_out_of_bounds():
    from run_algorithms import _path_visits_wall
    grid = [["_", "_"], ["#", "_"]]
    assert _path_visits_wall([(0, 0), (1, 0)], grid) is False
    assert _path_visits_wall([(0, 0), (0, 1)], grid) is True  # (0,1) is "#"
    assert _path_visits_wall([(5, 5)], grid) is True  # out of bounds


# ── ALGORITHMS registry ──────────────────────────────────────────────────────

def test_algorithms_list_is_exactly_the_five():
    names = [name for name, _ in ALGORITHMS]
    assert names == [
        "Cell Visit Freq",
        "Trajectory Visit Freq",
        "Inv. Planning",
        "MaxEnt IRL",
        "Surprise v2",
    ]
    assert len(ra._ALGO_COLORS) == len(ALGORITHMS)
    assert set(ra._ALGO_SCORE_TYPES) == set(names)


# ── grid_key / grid_label ────────────────────────────────────────────────────

def test_grid_key_extracts_prefix():
    assert _grid_key("model_size7_comp0.0_grid3_coin_layout.json") == "model_size7_comp0.0_grid3"
    assert _grid_key("not_a_layout_file.json") is None


def test_grid_label_extracts_short_suffix():
    assert _grid_label("model_size7_comp0.0_grid3") == "comp0.0_grid3"
    assert _grid_label("some_other_prefix") == "some_other_prefix"  # falls back


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_registers_skip_invalid_actions_flag():
    result = subprocess.run(
        [sys.executable, str(_REPO_DIR / "algorithms" / "run_algorithms.py"), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--skip-invalid-actions" in result.stdout


# ── Real-data regression: grid_id collision fix ─────────────────────────────

@no_data
def test_grid_collision_regression_finds_all_40_grids():
    layout_files = sorted(_DATA_DIR.glob("*_coin_layout.json"))
    grid_keys = sorted({gk for lf in layout_files for gk in [_grid_key(lf.name)] if gk})
    assert len(grid_keys) == 40


@no_data
def test_grid_collision_regression_different_complexities_differ():
    """Two grid keys sharing the same trailing grid number but different
    complexity must load to structurally different layouts (not collapsed)."""
    layout_files = sorted(_DATA_DIR.glob("*_coin_layout.json"))
    grid_keys = sorted({gk for lf in layout_files for gk in [_grid_key(lf.name)] if gk})

    grid0_keys = [gk for gk in grid_keys if gk.endswith("_grid0")]
    assert len(grid0_keys) >= 2, "expected multiple complexity variants of grid0"

    layouts = [load_grid(_DATA_DIR, gk, effort="low")[0] for gk in grid0_keys[:2]]
    assert layouts[0]["grid_layout"] != layouts[1]["grid_layout"] or \
        layouts[0]["coin_pos"] != layouts[1]["coin_pos"]


@no_data
def test_real_data_pooled_run_all_five_algorithms():
    layout_files = sorted(_DATA_DIR.glob("*_coin_layout.json"))
    grid_keys = sorted({gk for lf in layout_files for gk in [_grid_key(lf.name)] if gk})

    layout, paths, traj_ids = None, [], []
    for gk in grid_keys:
        layout, paths, traj_ids = load_grid(_DATA_DIR, gk, effort="low")
        if paths:
            break
    assert paths, "expected at least one grid with successful trajectories"

    n_rows = len(layout["grid_layout"])
    n_cols = len(layout["grid_layout"][0])
    for name, algo_fn in ALGORITHMS:
        score, pred = algo_fn(paths, layout)
        col, row_top = pred
        assert 0 <= row_top < n_rows and 0 <= col < n_cols
        assert layout["grid_layout"][row_top][col] != "#", f"{name} predicted a wall cell"
        dist = abs(col - layout["coin_pos"][0]) + abs(row_top - layout["coin_pos"][1])
        assert isinstance(dist, int) and dist >= 0
