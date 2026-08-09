"""
Tests for tune_params.py's CV / grid-search infrastructure.

Uses ThreadPoolExecutor (not ProcessPoolExecutor) here — grid_search /
predict_and_score / cross_validate only call executor.map(), so they're
executor-agnostic; threads sidestep any multiprocessing-pickling concerns
for functions defined in a pytest-collected test module.
"""

import csv
import json
from concurrent.futures import ThreadPoolExecutor

from tune_params import (
    _compute_theta_stability,
    _write_fold_csv,
    _write_summary_json,
    cross_validate,
    grid_search,
    stratified_kfold,
)


def _make_synthetic_grids(n, sizes):
    """n synthetic 'grids' cycling through the given grid_size labels."""
    return [
        {
            "data_dir": "synthetic",
            "grid_key": f"grid{i}",
            "label": f"grid{i}",
            "layout": {"grid_size": sizes[i % len(sizes)]},
            "paths": [],
        }
        for i in range(n)
    ]


def test_stratified_kfold_balanced_and_disjoint():
    grids = _make_synthetic_grids(30, [7, 9, 11])
    splits = stratified_kfold(grids, k=5, seed=42)

    assert len(splits) == 5
    seen_test = set()
    for train_idx, test_idx in splits:
        assert set(train_idx).isdisjoint(test_idx)
        assert set(train_idx) | set(test_idx) == set(range(30))
        assert len(test_idx) == 6  # 30 grids / 5 folds, evenly divisible
        seen_test.update(test_idx)
    assert seen_test == set(range(30))


def test_stratified_kfold_no_grid_in_multiple_test_folds():
    grids = _make_synthetic_grids(23, [7, 9, 11])  # not evenly divisible by k
    splits = stratified_kfold(grids, k=5, seed=1)
    all_test = [i for _, test_idx in splits for i in test_idx]
    assert len(all_test) == len(set(all_test)) == 23


def _synthetic_score(candidate, layout, paths):
    """Deterministic 'loss' with a known minimum at candidate == target."""
    return abs(candidate - layout["target"])


def test_grid_search_selects_known_minimum():
    grids = [{"layout": {"target": 5}, "paths": []} for _ in range(4)]
    candidates = [1, 3, 5, 7, 9]
    with ThreadPoolExecutor(max_workers=2) as ex:
        ranked = grid_search(candidates, grids, _synthetic_score, ex)
    assert ranked[0] == (5, 0.0)


def test_cross_validate_recovers_known_optimum_on_every_fold():
    grids = _make_synthetic_grids(20, [7, 9, 11])
    for g in grids:
        g["layout"]["target"] = 5  # every grid agrees the true optimum is 5
    candidates = [1, 3, 5, 7, 9]

    with ThreadPoolExecutor(max_workers=2) as ex:
        result = cross_validate(candidates, grids, _synthetic_score, ex, k=4, seed=7)

    assert len(result["fold_results"]) == 4
    for r in result["fold_results"]:
        assert r["theta"] in candidates
        assert r["theta"] == 5
        assert r["held_out_score"] == 0.0
    assert result["full_data_best"]["theta"] == 5
    assert result["full_data_best"]["score"] == 0.0
    assert result["n_grids"] == 20

    assert result["held_out_mean"] == 0.0
    assert result["held_out_std"] == 0.0
    assert result["theta_stability"] == {
        "distinct_thetas": [5],
        "all_folds_agree": True,
        "matches_full_data_best": True,
    }


def test_compute_theta_stability_detects_agreement_and_disagreement():
    agree = _compute_theta_stability([2, 2, 2], best_theta=2)
    assert agree == {
        "distinct_thetas": [2],
        "all_folds_agree": True,
        "matches_full_data_best": True,
    }

    disagree = _compute_theta_stability([1, 2, 9], best_theta=2)
    assert disagree["distinct_thetas"] == [1, 2, 9]
    assert disagree["all_folds_agree"] is False
    assert disagree["matches_full_data_best"] is False


def test_write_fold_csv_round_trips(tmp_path):
    grids = _make_synthetic_grids(8, [7, 9])
    for g in grids:
        g["layout"]["target"] = 5
    candidates = [1, 3, 5, 7, 9]

    with ThreadPoolExecutor(max_workers=2) as ex:
        result = cross_validate(candidates, grids, _synthetic_score, ex, k=2, seed=3)

    path = tmp_path / "folds.csv"
    _write_fold_csv(path, ["theta"], result["fold_results"])

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0].keys() == {
        "fold",
        "theta",
        "train_dist",
        "held_out_dist",
        "n_train",
        "n_test",
    }
    assert len(rows) == len(result["fold_results"])
    for row, expected in zip(rows, result["fold_results"]):
        assert int(row["fold"]) == expected["fold"]
        assert float(row["theta"]) == expected["theta"]
        assert float(row["held_out_dist"]) == expected["held_out_score"]


def test_write_summary_json_round_trips(tmp_path):
    grids = _make_synthetic_grids(8, [7, 9])
    for g in grids:
        g["layout"]["target"] = 5
    candidates = [1, 3, 5, 7, 9]

    with ThreadPoolExecutor(max_workers=2) as ex:
        result = cross_validate(candidates, grids, _synthetic_score, ex, k=2, seed=3)

    path = tmp_path / "summary.json"
    _write_summary_json(
        path, "toy_model", ["theta"], result, meta={"k": 2, "seed": 3, "effort": "low"}
    )

    with open(path) as f:
        summary = json.load(f)
    assert summary["model"] == "toy_model"
    assert summary["k"] == 2
    assert summary["effort"] == "low"
    assert summary["held_out_mean"] == result["held_out_mean"]
    assert summary["held_out_std"] == result["held_out_std"]
    assert summary["full_data_best"]["theta"] == result["full_data_best"]["theta"]
