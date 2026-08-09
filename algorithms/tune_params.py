"""
tune_params.py
---------------
Cross-validated grid-search estimation of inv_planning.py's beta and
surprise_v2.py's (lambda, gamma, alpha), fit directly on real LLM trajectory
data instead of the literature-borrowed defaults currently hardcoded there
(beta=2 from Baker et al. 2011; lambda/gamma/alpha from Buidze et al. 2025,
fit to human navigation).

Does NOT modify inv_planning.py / surprise_v2.py — reports results only.

Method
------
CV unit = grid (ground truth coin_pos is per-grid; both algorithms already
pool all successful trajectories for one grid into a single prediction).
Stratified k-fold by grid_size (7/9/11). Per fold:
    fit    — grid-search the candidate minimizing mean Manhattan distance
             over the training folds only.
    predict — apply that one fitted value (not re-searched) to the held-out
              fold's grids.
    score  — mean Manhattan distance on the held-out fold only.
Separately, fit on *all* usable grids for the final reported point estimate.

Usage
-----
    python tune_params.py <data_dir> [<data_dir> ...] \\
        [--model {inv_planning,surprise_v2,both}] [--effort low] \\
        [--k 5] [--seed 42] [--max-workers N] [--output-dir tuning_results]
"""

import argparse
import csv
import itertools
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ALGO_DIR = Path(__file__).resolve().parent
if str(_ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGO_DIR))

import run_algorithms as ra
from inv_planning import run_inv_planning
from surprise_v2 import run_surprise_v2, TAU, EPSILON


# ── Candidate grids ──────────────────────────────────────────────────────────

BETA_CANDIDATES = [0.5, 1, 2, 4, 8, 16]
LAMBDA_CANDIDATES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
GAMMA_CANDIDATES = [1, 2, 4, 6, 8, 10]
ALPHA_CANDIDATES = [1.0, 1.5, 1.84, 2.5, 3.5]


# ── Scoring ──────────────────────────────────────────────────────────────────


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def score_inv_planning(beta, layout, paths):
    _, pred = run_inv_planning(paths, layout, beta=beta)
    return _manhattan(pred, tuple(layout["coin_pos"]))


def score_surprise_v2(params3, layout, paths):
    lam, gam, alpha = params3
    _, pred = run_surprise_v2(paths, layout, params=(lam, gam, alpha, TAU, EPSILON))
    return _manhattan(pred, tuple(layout["coin_pos"]))


def _score_one(args):
    score_fn, candidate, layout, paths = args
    return score_fn(candidate, layout, paths)


# ── Data discovery ───────────────────────────────────────────────────────────


def discover_usable_grids(data_dirs, effort="low"):
    """Every grid (across all given directories) with >=1 successful trajectory."""
    grids = []
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        layout_files = sorted(data_dir.glob("*_coin_layout.json"))
        grid_keys = sorted(
            {gk for lf in layout_files for gk in [ra._grid_key(lf.name)] if gk}
        )
        for gk in grid_keys:
            layout, paths, traj_ids = ra.load_grid(data_dir, gk, effort=effort)
            if paths:
                grids.append(
                    {
                        "data_dir": str(data_dir),
                        "grid_key": gk,
                        "label": ra._grid_label(gk),
                        "layout": layout,
                        "paths": paths,
                    }
                )
    return grids


# ── Cross-validation ─────────────────────────────────────────────────────────


def stratified_kfold(grids, k=5, seed=42):
    """k (train_idx, test_idx) splits, stratified by layout['grid_size']."""
    rng = random.Random(seed)
    by_size = defaultdict(list)
    for i, g in enumerate(grids):
        by_size[g["layout"].get("grid_size")].append(i)

    folds = [[] for _ in range(k)]
    for _, idxs in sorted(by_size.items(), key=lambda kv: str(kv[0])):
        idxs = list(idxs)
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs):
            folds[j % k].append(idx)

    all_idx = set(range(len(grids)))
    splits = []
    for i in range(k):
        test_idx = sorted(folds[i])
        train_idx = sorted(all_idx - set(test_idx))
        splits.append((train_idx, test_idx))
    return splits


def grid_search(candidates, grids, score_fn, executor, indices=None):
    """Fit step: mean Manhattan distance for every candidate over the given
    grids (or a subset by index). Returns [(candidate, mean_score), ...]
    sorted ascending by mean_score (best first)."""
    idxs = indices if indices is not None else list(range(len(grids)))
    tasks = [
        (score_fn, c, grids[i]["layout"], grids[i]["paths"])
        for c in candidates
        for i in idxs
    ]
    flat = list(executor.map(_score_one, tasks, chunksize=4))

    n = len(idxs)
    means = []
    for ci, c in enumerate(candidates):
        scores = flat[ci * n : (ci + 1) * n]
        means.append((c, float(np.mean(scores))))
    means.sort(key=lambda t: t[1])
    return means


def predict_and_score(theta, grids, score_fn, executor, indices):
    """Predict + score step: apply one already-chosen parameter value
    (never re-searched) to the given grids; return mean Manhattan distance."""
    tasks = [(score_fn, theta, grids[i]["layout"], grids[i]["paths"]) for i in indices]
    scores = list(executor.map(_score_one, tasks, chunksize=4))
    return float(np.mean(scores))


def _compute_theta_stability(fold_thetas, best_theta):
    """How much the CV-selected parameter varies across folds, and whether
    it agrees with the separately fitted full-data point estimate."""
    distinct_thetas = sorted(set(fold_thetas), key=str)
    return {
        "distinct_thetas": distinct_thetas,
        "all_folds_agree": len(distinct_thetas) == 1,
        "matches_full_data_best": distinct_thetas == [best_theta],
    }


def cross_validate(candidates, grids, score_fn, executor, k=5, seed=42):
    splits = stratified_kfold(grids, k=k, seed=seed)

    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        fit_ranked = grid_search(
            candidates, grids, score_fn, executor, indices=train_idx
        )
        theta_i, train_score = fit_ranked[0]
        held_out_score = predict_and_score(theta_i, grids, score_fn, executor, test_idx)
        fold_results.append(
            {
                "fold": fold_i,
                "theta": theta_i,
                "train_score": train_score,
                "held_out_score": held_out_score,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )

    full_ranked = grid_search(candidates, grids, score_fn, executor, indices=None)
    best_theta, best_score = full_ranked[0]

    held = [r["held_out_score"] for r in fold_results]
    theta_stability = _compute_theta_stability(
        [r["theta"] for r in fold_results], best_theta
    )

    return {
        "fold_results": fold_results,
        "held_out_mean": float(np.mean(held)),
        "held_out_std": float(np.std(held)),
        "theta_stability": theta_stability,
        "full_data_best": {"theta": best_theta, "score": best_score},
        "full_loss_curve": full_ranked,
        "n_grids": len(grids),
    }


# ── Reporting ─────────────────────────────────────────────────────────────────


def _theta_dict(param_names, theta):
    values = theta if isinstance(theta, tuple) else (theta,)
    return dict(zip(param_names, values))


def _print_report(name, param_names, result):
    print(f"\n=== {name} — {result['n_grids']} usable grids ===")
    print(
        f"{'Fold':>5}  {'theta':<30} {'train_dist':>11} {'held_out_dist':>14} {'n_train':>8} {'n_test':>7}"
    )
    for r in result["fold_results"]:
        print(
            f"{r['fold']:>5}  {str(r['theta']):<30} {r['train_score']:>11.3f} "
            f"{r['held_out_score']:>14.3f} {r['n_train']:>8} {r['n_test']:>7}"
        )
    print(
        f"{'mean':>5}  {'':<30} {'':<11} "
        f"{result['held_out_mean']:>7.3f} ± {result['held_out_std']:<5.3f}"
    )

    stability = result["theta_stability"]
    if stability["all_folds_agree"]:
        print("theta: stable across all folds")
    else:
        print(f"theta: varied across folds — {stability['distinct_thetas']}")

    best = _theta_dict(param_names, result["full_data_best"]["theta"])
    print(
        f"\nFull-data best: {best}  mean_dist={result['full_data_best']['score']:.3f}"
    )


def _write_csv(path, param_names, full_loss_curve):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(param_names) + ["mean_manhattan_dist"])
        for theta, score in full_loss_curve:
            row = list(theta) if isinstance(theta, tuple) else [theta]
            writer.writerow(row + [score])


def _write_fold_csv(path, param_names, fold_results):
    """One row per CV fold: chosen theta, train/held-out score, fold sizes."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["fold"]
            + list(param_names)
            + ["train_dist", "held_out_dist", "n_train", "n_test"]
        )
        for r in fold_results:
            theta = r["theta"]
            theta_row = list(theta) if isinstance(theta, tuple) else [theta]
            writer.writerow(
                [r["fold"]]
                + theta_row
                + [r["train_score"], r["held_out_score"], r["n_train"], r["n_test"]]
            )


def _write_summary_json(path, name, param_names, result, meta):
    """Consolidated, machine-readable per-algorithm summary: the honest
    (held-out) generalisation estimate, theta stability across folds, and
    the full-data point estimate — the natural source for a writeup table."""
    summary = {
        "model": name,
        "param_names": list(param_names),
        "n_grids": result["n_grids"],
        **meta,
        "held_out_mean": result["held_out_mean"],
        "held_out_std": result["held_out_std"],
        "theta_stability": {
            "distinct_thetas": [
                list(t) if isinstance(t, tuple) else t
                for t in result["theta_stability"]["distinct_thetas"]
            ],
            "all_folds_agree": result["theta_stability"]["all_folds_agree"],
            "matches_full_data_best": result["theta_stability"][
                "matches_full_data_best"
            ],
        },
        "full_data_best": {
            "theta": (
                list(result["full_data_best"]["theta"])
                if isinstance(result["full_data_best"]["theta"], tuple)
                else result["full_data_best"]["theta"]
            ),
            "score": result["full_data_best"]["score"],
        },
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def _plot_beta(path, full_loss_curve, fold_results):
    curve = dict(full_loss_curve)
    betas = sorted(curve)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(betas, [curve[b] for b in betas], "o-", label="full-data mean")
    for i, r in enumerate(fold_results):
        ax.scatter(
            [r["theta"]],
            [r["held_out_score"]],
            color="crimson",
            zorder=5,
            label="fold held-out" if i == 0 else None,
        )
    ax.set_xscale("log")
    ax.set_xlabel("beta")
    ax.set_ylabel("mean Manhattan distance")
    ax.set_title("Inv. Planning — beta")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_surprise(path, full_loss_curve, best_theta):
    curve = dict(full_loss_curve)
    best_lam, best_gam, best_alpha = best_theta
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    slices = [
        (0, "lambda", LAMBDA_CANDIDATES, (best_gam, best_alpha)),
        (1, "gamma", GAMMA_CANDIDATES, (best_lam, best_alpha)),
        (2, "alpha", ALPHA_CANDIDATES, (best_lam, best_gam)),
    ]
    for ax, (idx, name, cands, fixed) in zip(axes, slices):
        xs, ys = [], []
        for v in cands:
            if idx == 0:
                theta = (v, fixed[0], fixed[1])
            elif idx == 1:
                theta = (fixed[0], v, fixed[1])
            else:
                theta = (fixed[0], fixed[1], v)
            if theta in curve:
                xs.append(v)
                ys.append(curve[theta])
        ax.plot(xs, ys, "o-")
        ax.axvline(best_theta[idx], color="crimson", ls="--", lw=1)
        ax.set_xlabel(name)
        ax.set_ylabel("mean Manhattan distance")
    fig.suptitle(
        "Surprise v2 — marginal effect of each parameter (others fixed at full-data best)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────


def run_and_report(
    name, param_names, candidates, grids, score_fn, executor, k, seed, effort, out_dir
):
    t0 = time.time()
    result = cross_validate(candidates, grids, score_fn, executor, k=k, seed=seed)
    elapsed = time.time() - t0
    _print_report(name, param_names, result)
    print(f"  ({elapsed:.1f}s)")

    _write_csv(out_dir / f"{name}.csv", param_names, result["full_loss_curve"])
    _write_fold_csv(out_dir / f"{name}_folds.csv", param_names, result["fold_results"])
    _write_summary_json(
        out_dir / f"{name}_summary.json",
        name,
        param_names,
        result,
        meta={"k": k, "seed": seed, "effort": effort},
    )
    if name == "inv_planning":
        _plot_beta(
            out_dir / f"{name}.png", result["full_loss_curve"], result["fold_results"]
        )
    else:
        _plot_surprise(
            out_dir / f"{name}.png",
            result["full_loss_curve"],
            result["full_data_best"]["theta"],
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dirs", nargs="+", help="One or more directories to pool grids from"
    )
    parser.add_argument(
        "--model", choices=["inv_planning", "surprise_v2", "both"], default="both"
    )
    parser.add_argument("--effort", default="low")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--output-dir", default="tuning_results")
    args = parser.parse_args()

    grids = discover_usable_grids(args.data_dirs, effort=args.effort)
    print(
        f"Found {len(grids)} usable grid(s) across {len(args.data_dirs)} director(y/ies)."
    )
    if not grids:
        sys.exit(
            "No usable grids found (no directory had a grid with a successful trajectory)."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        if args.model in ("inv_planning", "both"):
            run_and_report(
                "inv_planning",
                ["beta"],
                BETA_CANDIDATES,
                grids,
                score_inv_planning,
                executor,
                args.k,
                args.seed,
                args.effort,
                out_dir,
            )

        if args.model in ("surprise_v2", "both"):
            candidates = list(
                itertools.product(LAMBDA_CANDIDATES, GAMMA_CANDIDATES, ALPHA_CANDIDATES)
            )
            run_and_report(
                "surprise_v2",
                ["lambda", "gamma", "alpha"],
                candidates,
                grids,
                score_surprise_v2,
                executor,
                args.k,
                args.seed,
                args.effort,
                out_dir,
            )


if __name__ == "__main__":
    main()
