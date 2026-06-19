"""
run_trex.py
-----------
Train T-REX and compare its reward map against MaxEnt IRL.

Usage
-----
    python run_trex.py <data_dir> [options]

<data_dir> must contain layout JSON files (*_coin_layout.json) and trajectory
JSON files (*_coin_{effort}_traj*.json). A plots/ subdirectory is created there.

Options
-------
    --effort      Trajectory reasoning effort level (default: low)
    --clip-len    Clip length for T-REX training pairs (default: 10)
    --n-pairs     Number of training clip pairs (default: 5000)
    --lr          Adam learning rate (default: 1e-3)
    --hidden-dim  Reward network hidden layer size (default: 64)
    --save-model  If set, save PyTorch state_dict to plots/grid{N}_trex_weights.pt

Output
------
    plots/grid{N}_trex_vs_maxent.png   — side-by-side reward heatmaps
    plots/results_trex.csv             — per-grid summary (optional)
    Console summary table              — predicted subgoal, Manhattan dists, Spearman r
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_ALGO_DIR = Path(__file__).resolve().parent
if str(_ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGO_DIR))

from run_algorithms import load_grid          # reused: file discovery + filtering
from maxent_irl import run_maxent             # for comparison
from trex import (
    rank_trajectories,
    load_all_trajectories,
    train_trex,
    extract_reward_map,
    compare_with_maxent,
)


def _score_grid_to_json(score_grid):
    """Serialize a score grid to a JSON string matching results.csv format.

    NaN cells (walls, start, goal) are written as null, matching the null
    convention used by run_algorithms.py's results.csv.
    """
    rows = []
    for r in range(score_grid.shape[0]):
        row = [None if np.isnan(score_grid[r, c]) else float(score_grid[r, c])
               for c in range(score_grid.shape[1])]
        rows.append(row)
    return json.dumps(rows)


def run_grid_trex(data_dir, grid_id, plots_dir, effort="low",
                  clip_len=10, n_pairs=5000, lr=1e-3, hidden_dim=64,
                  save_model=False):
    """Run the full T-REX pipeline on a single grid and compare to MaxEnt IRL.

    Parameters
    ----------
    data_dir : Path
    grid_id  : int
    plots_dir : Path
    effort, clip_len, n_pairs, lr, hidden_dim, save_model : see main()

    Returns
    -------
    dict or None
        Keys: grid_id, trex_pred, maxent_pred, trex_dist, maxent_dist,
              spearman_r, n_trajs, ranking_scores.
        None if no successful trajectories exist.
    """
    # Load layout + successful paths (MaxEnt needs successful-only)
    layout, successful_paths, _ = load_grid(data_dir, grid_id, effort=effort)
    if not successful_paths:
        print(f"  Grid {grid_id}: no successful trajectories — skipping")
        return None

    coin_j = tuple(layout["coin_pos"])

    # Load all trajectories (successful + unsuccessful) for T-REX
    all_paths, success_flags, all_traj_ids = load_all_trajectories(
        data_dir, grid_id, layout, effort=effort
    )
    n_success = sum(success_flags)
    n_fail    = len(all_paths) - n_success
    print(f"  Grid {grid_id}: {n_success} successful + {n_fail} unsuccessful  (coin={coin_j})")

    # ── Step 1: rank ──────────────────────────────────────────────────────────
    ranked_paths, ranked_scores, order = rank_trajectories(
        all_paths, layout, success_flags=success_flags
    )
    score_strs = "  ".join(
        f"traj{all_traj_ids[orig_i]}{'(F)' if not success_flags[orig_i] else ''}: {s:.3f}"
        for orig_i, s in zip(order, ranked_scores)
    )
    print(f"    Ranking (worst→best): {score_strs}")

    # ── Steps 2-4: train T-REX ────────────────────────────────────────────────
    net = train_trex(ranked_paths, layout,
                     clip_len=clip_len, n_pairs=n_pairs,
                     lr=lr, hidden_dim=hidden_dim)

    if save_model:
        import torch
        model_path = plots_dir / f"grid{grid_id}_trex_weights.pt"
        torch.save(net.state_dict(), model_path)
        print(f"    → Saved model weights: {model_path.name}")

    # ── Step 5a: extract T-REX reward map ────────────────────────────────────
    trex_grid, trex_pred = extract_reward_map(net, layout)

    # ── Step 5b: run MaxEnt IRL on successful-only paths (unchanged) ─────────
    print(f"  [MaxEnt IRL] Running on {n_success} successful trajectories ...",
          end=" ", flush=True)
    maxent_grid, maxent_pred = run_maxent(successful_paths, layout)
    print("done")

    # ── Step 5c: compare and save plot ────────────────────────────────────────
    save_path = plots_dir / f"grid{grid_id}_trex_vs_maxent.png"
    results = compare_with_maxent(trex_grid, maxent_grid, layout, save_path)
    print(f"    → Saved comparison plot: {save_path.name}")

    td, md, sr = results["trex_dist"], results["maxent_dist"], results["spearman_r"]
    print(f"    T-REX:  pred={results['trex_pred']}  Manhattan={td}")
    print(f"    MaxEnt: pred={results['maxent_pred']}  Manhattan={md}")
    print(f"    Spearman r (T-REX vs MaxEnt): {sr:.3f}")

    return {
        "grid_id": grid_id,
        "n_trajs": len(all_paths),
        "ranking_scores": ranked_scores,
        "trex_grid": trex_grid,
        "maxent_grid": maxent_grid,
        **results,
    }


def main(data_dir, effort="low", clip_len=10, n_pairs=5000, lr=1e-3,
         hidden_dim=64, save_model=False):
    data_dir = Path(data_dir).resolve()
    plots_dir = data_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    layout_files = sorted(data_dir.glob("*_coin_layout.json"))
    grid_ids = sorted({
        int(m.group(1))
        for lf in layout_files
        for m in [re.search(r"_grid(\d+)_coin_layout", lf.name)]
        if m
    })

    if not grid_ids:
        print(f"No layout files found in {data_dir}")
        return

    print(f"Found {len(grid_ids)} grid(s): {grid_ids}")
    print(f"Output directory: {plots_dir}")
    print(f"Effort: {effort}  clip_len={clip_len}  n_pairs={n_pairs}  "
          f"lr={lr}  hidden_dim={hidden_dim}\n")

    all_results = []
    for gid in grid_ids:
        res = run_grid_trex(
            data_dir, gid, plots_dir,
            effort=effort, clip_len=clip_len, n_pairs=n_pairs,
            lr=lr, hidden_dim=hidden_dim, save_model=save_model,
        )
        if res is not None:
            all_results.append(res)
        print()

    if not all_results:
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    col_w = 14
    print("── Overall Summary " + "─" * 55)
    header = (f"  {'Grid':>5}  {'Trajs':>5}  "
              f"{'T-REX pred':>{col_w}}  {'T-REX d':>7}  "
              f"{'MaxEnt pred':>{col_w}}  {'MaxEnt d':>8}  {'Spearman r':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in all_results:
        print(f"  {r['grid_id']:>5}  {r['n_trajs']:>5}  "
              f"{str(r['trex_pred']):>{col_w}}  {r['trex_dist']:>7}  "
              f"{str(r['maxent_pred']):>{col_w}}  {r['maxent_dist']:>8}  "
              f"{r['spearman_r']:>10.3f}")

    trex_dists = [r["trex_dist"] for r in all_results]
    maxent_dists = [r["maxent_dist"] for r in all_results]
    print(f"\n  Mean T-REX Manhattan:  {np.mean(trex_dists):.2f}")
    print(f"  Mean MaxEnt Manhattan: {np.mean(maxent_dists):.2f}")

    # ── CSV output (matching results.csv format from run_algorithms.py) ──────
    csv_path = plots_dir / "results_trex.csv"
    fieldnames = ["grid_id", "traj_id", "mode_type", "algorithm", "score_type",
                  "predicted_col", "predicted_row",
                  "true_coin_col", "true_coin_row",
                  "manhattan_dist", "score_grid_json"]

    # Reload layouts for coin positions
    layout_map = {}
    for lf in layout_files:
        m = re.search(r"_grid(\d+)_coin_layout", lf.name)
        if m:
            with open(lf) as jf:
                layout_map[int(m.group(1))] = json.load(jf)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            layout = layout_map.get(r["grid_id"], {})
            coin_j = tuple(layout.get("coin_pos", [0, 0]))
            # T-REX row
            writer.writerow({
                "grid_id": r["grid_id"],
                "traj_id": "AGG",
                "mode_type": "pooled",
                "algorithm": "T-REX",
                "score_type": "reward",
                "predicted_col": r["trex_pred"][0],
                "predicted_row": r["trex_pred"][1],
                "true_coin_col": coin_j[0],
                "true_coin_row": coin_j[1],
                "manhattan_dist": r["trex_dist"],
                "score_grid_json": _score_grid_to_json(r["trex_grid"]),
            })
            # MaxEnt row (for direct comparison in the same file)
            writer.writerow({
                "grid_id": r["grid_id"],
                "traj_id": "AGG",
                "mode_type": "pooled",
                "algorithm": "MaxEnt IRL",
                "score_type": "reward",
                "predicted_col": r["maxent_pred"][0],
                "predicted_row": r["maxent_pred"][1],
                "true_coin_col": coin_j[0],
                "true_coin_row": coin_j[1],
                "manhattan_dist": r["maxent_dist"],
                "score_grid_json": _score_grid_to_json(r["maxent_grid"]),
            })
    print(f"\n  → Saved {csv_path.name}  ({len(all_results)} row(s))")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train T-REX and compare against MaxEnt IRL."
    )
    parser.add_argument(
        "data_dir",
        help="Directory containing layout + trajectory JSON files",
    )
    parser.add_argument(
        "--effort", default="low",
        help="Reasoning effort level of the trajectories to load (default: low)",
    )
    parser.add_argument("--clip-len", type=int, default=10,
                        help="Clip length for T-REX training pairs (default: 10)")
    parser.add_argument("--n-pairs", type=int, default=5000,
                        help="Number of training clip pairs per grid (default: 5000)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Reward network hidden size (default: 64)")
    parser.add_argument("--save-model", action="store_true",
                        help="Save trained network weights to plots/")
    args = parser.parse_args()
    main(
        args.data_dir,
        effort=args.effort,
        clip_len=args.clip_len,
        n_pairs=args.n_pairs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        save_model=args.save_model,
    )
