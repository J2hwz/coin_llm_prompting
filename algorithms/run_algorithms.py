"""
run_algorithms.py
-----------------
Run 5 IRL algorithms on trajectory data and produce comparison plots.

Algorithms
----------
1. Surprise v2     — terminal-directed surprise model
2. Inv. Planning   — Bayesian inverse planning (soft value iteration)
3. BNIRL           — Fixed-K=2 Gibbs sampler
4. BIRL            — PolicyWalk MCMC (Ramachandran & Amir 2007)
5. MaxEnt IRL      — Maximum Causal Entropy IRL (Ziebart 2010)

Usage
-----
    python run_algorithms.py <data_dir> [--mode {individual,pooled,both}]

<data_dir> must contain layout JSON files (*_coin_layout.json) and trajectory
JSON files (*_coin_low_traj*.json). A plots/ subdirectory is created there.

Modes
-----
individual  — one row per successful trajectory, one column per algorithm
pooled      — all successful trajectories combined, one column per algorithm
both        — produce both (default)

Output
------
<data_dir>/plots/
    grid{N}_individual.png
    grid{N}_aggregated.png
    manhattan_summary.png
    results.csv            — one row per (grid, traj/AGG, algorithm); score_grid_json
                             column contains the full score grid as a JSON 2D array
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_ALGO_DIR = Path(__file__).resolve().parent
if str(_ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGO_DIR))

from visit_frequency import run_visit_freq
from surprise_v2     import run_surprise_v2
from inv_planning    import run_inv_planning
from bnirl           import run_bnirl
from birl_wrapper    import run_birl
from maxent_irl      import run_maxent

ALGORITHMS = [
    ("Visit Freq",    run_visit_freq),
    ("Surprise v2",   run_surprise_v2),
    ("Inv. Planning", run_inv_planning),
    ("BNIRL",         run_bnirl),
    ("BIRL",          run_birl),
    ("MaxEnt IRL",    run_maxent),
]

_ALGO_COLORS = ["#8C8C8C", "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

_ALGO_SCORE_TYPES = {
    "Visit Freq":    "visits",
    "Surprise v2":   "surprise",
    "Inv. Planning": "posterior",
    "BNIRL":         "samples",
    "BIRL":          "reward",
    "MaxEnt IRL":    "reward",
}

# ── Data helpers ──────────────────────────────────────────────────────────────

_ACTION_DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}


def _find_symbol(grid_state_rows, symbol):
    for row_str in grid_state_rows[1:]:
        parts = row_str.split()
        if len(parts) < 2:
            continue
        try:
            row_idx = int(parts[0])
        except ValueError:
            continue
        for col_idx, cell in enumerate(parts[1:]):
            if cell == symbol:
                return (col_idx, row_idx)
    return None


def build_path(steps, skip_invalid_actions: bool = False):
    """Reconstruct (col, row_from_top) path from JSON steps."""
    path = []
    for step in steps:
        pos = _find_symbol(step.get("grid_state", []), "A")
        if pos is None:
            break
        if skip_invalid_actions and path and pos == path[-1]:
            continue  # wall bump — agent didn't move
        path.append(pos)
    if steps and path:
        last_action = steps[-1].get("agent_action", "").upper()
        dx, dy = _ACTION_DELTA.get(last_action, (0, 0))
        path.append((path[-1][0] + dx, path[-1][1] + dy))
    return path


def is_successful(steps, layout):
    """Return True if the trajectory collects the coin and reaches the terminal."""
    if not any(s.get("coin_collected", False) for s in steps):
        return False
    path = build_path(steps)
    return tuple(layout["goal_pos"]) in path


def load_grid(data_dir, grid_id, effort="low", skip_invalid_actions: bool = False):
    """Load layout and all successful trajectory paths for one grid."""
    data_dir = Path(data_dir)
    layout_files = list(data_dir.glob(f"*_grid{grid_id}_coin_layout.json"))
    if not layout_files:
        raise FileNotFoundError(f"No layout file for grid {grid_id} in {data_dir}")
    with open(layout_files[0]) as f:
        layout = json.load(f)

    traj_files = sorted(data_dir.glob(f"*_grid{grid_id}_coin_{effort}_traj*.json"))
    successful_paths, traj_ids = [], []
    for tf in traj_files:
        m = re.search(r"_traj(\d+)\.json$", tf.name)
        traj_id = int(m.group(1)) if m else -1
        with open(tf) as f:
            data = json.load(f)
        steps = data.get("steps", [])
        if is_successful(steps, layout):
            successful_paths.append(build_path(steps, skip_invalid_actions=skip_invalid_actions))
            traj_ids.append(traj_id)

    return layout, successful_paths, traj_ids


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_result(ax, score_grid, layout, predicted_json, title):
    """Draw one algorithm result panel onto ax."""
    n_rows = len(layout["grid_layout"])
    n_cols = len(layout["grid_layout"][0])
    grid   = layout["grid_layout"]
    coin_j  = tuple(layout["coin_pos"])
    goal_j  = tuple(layout["goal_pos"])
    start_j = tuple(layout["agent_start_pos"])

    wall_mask = np.array(
        [[grid[r][c] == "#" for c in range(n_cols)] for r in range(n_rows)],
        dtype=bool,
    )
    nan_mask = np.isnan(score_grid)
    masked   = np.ma.masked_where(wall_mask | nan_mask, score_grid)

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("#4a4a4a")
    im = ax.imshow(masked, origin="upper", cmap=cmap,
                   interpolation="nearest", aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for r in range(n_rows + 1):
        ax.axhline(r - 0.5, color="#888", lw=0.3, zorder=1)
    for c in range(n_cols + 1):
        ax.axvline(c - 0.5, color="#888", lw=0.3, zorder=1)

    for (cx, cy), lbl in [(goal_j, "G"), (start_j, "S"), (coin_j, "C")]:
        ax.text(cx, cy, lbl, ha="center", va="center",
                fontsize=7, fontweight="bold", color="black", zorder=5)

    ax.add_patch(mpatches.Circle(
        coin_j, 0.35, color="limegreen", fill=False, lw=1.5, zorder=6
    ))
    ax.plot(predicted_json[0], predicted_json[1], "x",
            color="crimson", ms=8, mew=2, zorder=7)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=6, pad=2)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _grid_to_json(score):
    """Serialise score_grid to a JSON string (NaN → null)."""
    def _clean(v):
        return None if (isinstance(v, float) and np.isnan(v)) else float(v)
    return json.dumps([[_clean(v) for v in row] for row in score.tolist()])


# ── Per-grid processing ───────────────────────────────────────────────────────

def run_grid(data_dir, grid_id, plots_dir, mode="both", effort="low", skip_invalid_actions: bool = False):
    """Run all algorithms on one grid, save plots, return (results, csv_rows)."""
    layout, paths, traj_ids = load_grid(data_dir, grid_id, effort=effort, skip_invalid_actions=skip_invalid_actions)

    if not paths:
        print(f"  Grid {grid_id}: no successful trajectories — skipping")
        return None

    coin_j = tuple(layout["coin_pos"])
    n = len(paths)
    n_algos = len(ALGORITHMS)
    print(f"  Grid {grid_id}: {n} successful trajectories  (coin={coin_j})")

    results = {name: {"individual": [], "aggregated": None} for name, _ in ALGORITHMS}
    csv_rows = []

    # ── Individual plot ───────────────────────────────────────────────────────
    if mode in ("individual", "both"):
        fig, axes = plt.subplots(n, n_algos,
                                 figsize=(3 * n_algos, n * 3 + 0.5),
                                 squeeze=False)

        for col, (name, _) in enumerate(ALGORITHMS):
            fig.text(
                (col + 0.5) / n_algos, 0.995,
                name, ha="center", va="top",
                fontsize=9, fontweight="bold", transform=fig.transFigure,
            )

        for row, (path, tid) in enumerate(zip(paths, traj_ids)):
            for col, (name, algo_fn) in enumerate(ALGORITHMS):
                print(f"    traj{tid}  {name} ...", end=" ", flush=True)
                score, pred = algo_fn([path], layout)
                dist = _manhattan(pred, coin_j)
                results[name]["individual"].append(dist)
                plot_result(axes[row, col], score, layout, pred, f"d = {dist}")
                print(f"d={dist}")
                csv_rows.append({
                    "grid_id": grid_id,
                    "traj_id": tid,
                    "mode_type": "individual",
                    "algorithm": name,
                    "score_type": _ALGO_SCORE_TYPES[name],
                    "predicted_col": pred[0],
                    "predicted_row": pred[1],
                    "true_coin_col": coin_j[0],
                    "true_coin_row": coin_j[1],
                    "manhattan_dist": dist,
                    "score_grid_json": _grid_to_json(score),
                })
            axes[row, 0].set_ylabel(f"traj{tid}", fontsize=7,
                                    rotation=0, labelpad=24, va="center")

        fig.suptitle(
            f"Grid {grid_id} — Individual ({n} successful, coin={coin_j})",
            fontsize=10, y=1.0,
        )
        fig.tight_layout()
        out = plots_dir / f"grid{grid_id}_individual.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"    → Saved {out.name}")

    # ── Aggregated plot ───────────────────────────────────────────────────────
    if mode in ("pooled", "both"):
        fig, axes = plt.subplots(1, n_algos,
                                 figsize=(3 * n_algos, 3.4),
                                 squeeze=False)

        for col, (name, algo_fn) in enumerate(ALGORITHMS):
            print(f"    aggregated  {name} ...", end=" ", flush=True)
            score, pred = algo_fn(paths, layout)
            dist = _manhattan(pred, coin_j)
            results[name]["aggregated"] = dist
            plot_result(axes[0, col], score, layout, pred, f"{name}\nd = {dist}")
            print(f"d={dist}")
            csv_rows.append({
                "grid_id": grid_id,
                "traj_id": "AGG",
                "mode_type": "pooled",
                "algorithm": name,
                "score_type": _ALGO_SCORE_TYPES[name],
                "predicted_col": pred[0],
                "predicted_row": pred[1],
                "true_coin_col": coin_j[0],
                "true_coin_row": coin_j[1],
                "manhattan_dist": dist,
                "score_grid_json": _grid_to_json(score),
            })

        fig.suptitle(
            f"Grid {grid_id} — Aggregated ({n} trajectories, coin={coin_j})",
            fontsize=10,
        )
        fig.tight_layout()
        out = plots_dir / f"grid{grid_id}_aggregated.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"    → Saved {out.name}")

    # Fill missing mode results with None
    for name, _ in ALGORITHMS:
        if not results[name]["individual"]:
            results[name]["individual"] = [None] * n
        if results[name]["aggregated"] is None and mode == "individual":
            results[name]["aggregated"] = None

    # Print per-grid distance table
    col_w = 15
    header = f"    {'Traj':>5}  " + "  ".join(f"{nm:>{col_w}}" for nm, _ in ALGORITHMS)
    print(header)
    print("    " + "-" * (7 + (col_w + 2) * n_algos))
    for i, tid in enumerate(traj_ids):
        row_str = f"    {tid:>5}  " + "  ".join(
            f"{results[nm]['individual'][i]:>{col_w}}"
            if results[nm]['individual'][i] is not None else f"{'—':>{col_w}}"
            for nm, _ in ALGORITHMS
        )
        print(row_str)
    if mode in ("pooled", "both"):
        agg_str = f"    {'AGG':>5}  " + "  ".join(
            f"{results[nm]['aggregated']:>{col_w}}"
            if results[nm]['aggregated'] is not None else f"{'—':>{col_w}}"
            for nm, _ in ALGORITHMS
        )
        print(agg_str)

    return results, csv_rows


# ── Summary plot ──────────────────────────────────────────────────────────────

def plot_manhattan_summary(all_results, plots_dir, mode="both"):
    """Two-panel bar chart comparing Manhattan distances across grids and algorithms."""
    grid_ids = sorted(all_results.keys())
    n_grids  = len(grid_ids)
    n_algos  = len(ALGORITHMS)
    x        = np.arange(n_grids)
    width    = 0.7 / n_algos

    fig, axes = plt.subplots(1, 2 if mode == "both" else 1,
                             figsize=(13 if mode == "both" else 7, 5),
                             squeeze=False)
    ax_ind = axes[0, 0]
    ax_agg = axes[0, 1] if mode == "both" else None

    for ai, ((name, _), color) in enumerate(zip(ALGORITHMS, _ALGO_COLORS)):
        offsets = x + (ai - (n_algos - 1) / 2) * width

        if mode in ("individual", "both"):
            ind_means, ind_scatter = [], []
            for gid in grid_ids:
                dists = [d for d in all_results[gid][name]["individual"] if d is not None]
                ind_means.append(np.mean(dists) if dists else 0)
                ind_scatter.append(dists)

            ax_ind.bar(offsets, ind_means, width=width * 0.9,
                       color=color, alpha=0.85, label=name)
            rng = np.random.default_rng(42)
            for xi, dists in zip(offsets, ind_scatter):
                if dists:
                    jitter = rng.uniform(-width * 0.15, width * 0.15, len(dists))
                    ax_ind.scatter(xi + jitter, dists,
                                   color=color, s=18, alpha=0.6, zorder=3)

        if mode in ("pooled", "both") and ax_agg is not None:
            agg_vals = [all_results[gid][name]["aggregated"] for gid in grid_ids
                        if all_results[gid][name]["aggregated"] is not None]
            if agg_vals:
                ax_agg.bar(offsets[:len(agg_vals)], agg_vals, width=width * 0.9,
                           color=color, alpha=0.85, label=name)
                for xi, v in zip(offsets[:len(agg_vals)], agg_vals):
                    ax_agg.text(xi, v + 0.05, str(v), ha="center", va="bottom",
                                fontsize=7, color=color)

    algo_names = " / ".join(n for n, _ in ALGORITHMS)
    for ax, title in [
        (ax_ind, "Individual trajectories  (bars = mean, dots = each traj)"),
        (ax_agg, "Aggregated  (all successful trajectories combined)"),
    ]:
        if ax is None:
            continue
        ax.set_xticks(x)
        ax.set_xticklabels([f"Grid {g}" for g in grid_ids])
        ax.set_ylabel("Manhattan distance to true subgoal")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.suptitle(f"Manhattan Distance — {algo_names}", fontsize=10)
    fig.tight_layout()
    out = plots_dir / "manhattan_summary.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  → Saved {out.name}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(data_dir, mode="both", effort="low", skip_invalid_actions: bool = False):
    data_dir  = Path(data_dir).resolve()
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
    print(f"Mode: {mode}  |  Effort: {effort}\n")

    all_results = {}
    all_csv_rows = []
    for gid in grid_ids:
        result = run_grid(data_dir, gid, plots_dir, mode=mode, effort=effort, skip_invalid_actions=skip_invalid_actions)
        if result is not None:
            res, rows = result
            all_results[gid] = res
            all_csv_rows.extend(rows)
        print()

    if all_csv_rows:
        csv_path = plots_dir / "results.csv"
        fieldnames = [
            "grid_id", "traj_id", "mode_type", "algorithm", "score_type",
            "predicted_col", "predicted_row", "true_coin_col", "true_coin_row",
            "manhattan_dist", "score_grid_json",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_csv_rows)
        print(f"\n  → Saved results.csv ({len(all_csv_rows)} rows)")

    if len(all_results) >= 2:
        plot_manhattan_summary(all_results, plots_dir, mode=mode)

    # Overall summary
    print("\n── Overall mean Manhattan distance ──────────────────────")
    for name, _ in ALGORITHMS:
        if mode in ("individual", "both"):
            ind = [d for res in all_results.values()
                   for d in res[name]["individual"] if d is not None]
            ind_str = f"individual mean = {np.mean(ind):.2f}" if ind else "individual = —"
        else:
            ind_str = ""
        if mode in ("pooled", "both"):
            agg = [res[name]["aggregated"] for res in all_results.values()
                   if res[name]["aggregated"] is not None]
            agg_str = f"aggregated mean = {np.mean(agg):.2f}" if agg else "aggregated = —"
        else:
            agg_str = ""
        parts = "  |  ".join(p for p in [ind_str, agg_str] if p)
        print(f"  {name:<16}  {parts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 5 IRL algorithms on trajectory data."
    )
    parser.add_argument(
        "data_dir",
        help="Directory containing layout + trajectory JSON files",
    )
    parser.add_argument(
        "--mode",
        choices=["individual", "pooled", "both"],
        default="both",
        help="Run per-trajectory, pooled, or both (default: both)",
    )
    parser.add_argument(
        "--effort",
        default="low",
        help="Reasoning effort level of the trajectories to load (default: low)",
    )
    args = parser.parse_args()
    main(args.data_dir, mode=args.mode, effort=args.effort)
