"""
plot_grid_comparison.py
------------------------
One figure: 5 rows (raw trajectory-visit heatmap + the 4 algorithm
predictions) x 3 columns (conditions: low, medium, low_deceptive), all for
the same single grid (default: size 9, density 0.2, grid0) so the same maze
layout/instructions can be compared side by side across conditions.

Row 1 (heat map) is a plain visit-count tally over the pooled successful
trajectories for that grid — every cell (including start/goal) counted
as-is, no candidate-cell exclusions or algorithmic modeling. This contrasts
with rows 2-5, which reuse the already-computed algorithm score grids/
predictions from compare_conditions.py's data sources (same "best available"
mix: Surprise v2 CV-tuned, the other 3 literature-default; Inv. Planning
excluded — see compare_conditions.py docstring).

Column headers and row labels are anchored to each axes' own coordinate
transform (not a fixed figure-fraction position), so they stay aligned to
the actual grid images regardless of per-panel colorbar width.

Usage
-----
    python plot_grid_comparison.py [--grid-size 9] [--density 0.2] [--grid-num 0]
        [--output-dir algorithms/results]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ALGO_DIR = Path(__file__).resolve().parent
if str(_ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGO_DIR))

import run_algorithms as ra
from compare_conditions import (
    CONDITIONS, load_condition_rows, build_grid_index,
    _load_layout, _draw_panel,
)

CONDITION_ORDER = ["low", "medium", "low_deceptive"]
CONDITION_EFFORT = {"low": "low", "medium": "medium", "low_deceptive": "low"}
CONDITION_DISPLAY = {
    "low": "Low reasoning",
    "medium": "Medium reasoning",
    "low_deceptive": "Low - deceptive",
}

# (internal algorithm key or None for the raw heat map, display row label)
ROW_SPECS = [
    (None, "Heat map"),
    ("Cell Visit Freq", "Cell visit freq."),
    ("Trajectory Visit Freq", "Trajectory visit freq."),
    ("MaxEnt IRL", "MaxEnt IRL"),
    ("Surprise v2", "Surprise"),
]


def _raw_visit_counts(layout, paths):
    """Plain visit-count grid over pooled paths — every cell as-is (start
    and goal included, no candidate-cell exclusions), walls NaN."""
    grid = layout["grid_layout"]
    n_rows, n_cols = len(grid), len(grid[0])
    counts = np.full((n_rows, n_cols), np.nan)
    for r in range(n_rows):
        for c in range(n_cols):
            if grid[r][c] != "#":
                counts[r, c] = 0
    for path in paths:
        for col, rt in path:
            if 0 <= rt < n_rows and 0 <= col < n_cols and grid[rt][col] != "#":
                counts[rt, col] += 1
    return counts


def _find_raw_paths(layout_dir, grid_id, effort):
    """Locate the full grid_key for grid_id in layout_dir and load its
    pooled successful trajectory paths (same filter run_algorithms.py uses)."""
    matches = list(Path(layout_dir).glob(f"*{grid_id}_coin_layout.json"))
    if not matches:
        return None
    grid_key = ra._grid_key(matches[0].name)
    _, paths, _ = ra.load_grid(layout_dir, grid_key, effort=effort)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--density", default="0.2")
    parser.add_argument("--grid-num", type=int, default=0)
    parser.add_argument("--output-dir", default=str(_ALGO_DIR / "results"))
    args = parser.parse_args()

    grid_id = f"comp{args.density}_grid{args.grid_num}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for condition, cfg in CONDITIONS.items():
        all_rows.extend(load_condition_rows(condition, cfg))
    grid_index = build_grid_index(all_rows)

    # Precompute per-condition data once (layout, raw visit counts, algo rows).
    per_condition = {}
    for condition in CONDITION_ORDER:
        cfg = CONDITIONS[condition]
        layout_dir = cfg["layout_dirs"][args.grid_size]
        layout = _load_layout(layout_dir, grid_id)
        paths, counts = None, None
        if layout is not None:
            paths = _find_raw_paths(layout_dir, grid_id, CONDITION_EFFORT[condition])
            counts = _raw_visit_counts(layout, paths or [])
        algo_rows = grid_index.get((condition, args.grid_size, grid_id), {})
        per_condition[condition] = {
            "layout": layout, "paths": paths, "counts": counts, "algo_rows": algo_rows,
        }

    n_rows, n_cols = len(ROW_SPECS), len(CONDITION_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.2 * n_cols, 3 * n_rows),
                              squeeze=False)

    for row_i, (algo_key, row_label) in enumerate(ROW_SPECS):
        for col_i, condition in enumerate(CONDITION_ORDER):
            ax = axes[row_i, col_i]
            data = per_condition[condition]

            if data["layout"] is None:
                ax.axis("off")
            elif algo_key is None:
                n = len(data["paths"] or [])
                ra.plot_result(ax, data["counts"], data["layout"], (-1, -1),
                                f"n={n} trajectories")
            else:
                r = data["algo_rows"].get(algo_key)
                if r is None:
                    ax.axis("off")
                    ax.set_title("(no prediction)", fontsize=6)
                else:
                    _draw_panel(ax, r, data["layout"], f"d = {r['manhattan_dist']}")

            # Column header (condition name), anchored to this axes' own
            # transform — stays aligned to the grid regardless of colorbar
            # width. Only drawn on the top row.
            if row_i == 0:
                ax.text(0.5, 1.18, CONDITION_DISPLAY[condition], transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=12, fontweight="bold")

            # Row label (algorithm name), vertical, anchored to this axes'
            # own transform. Only drawn in the first column.
            if col_i == 0:
                ax.text(-0.26, 0.5, row_label, transform=ax.transAxes,
                        ha="center", va="center", rotation=90,
                        fontsize=12, fontweight="bold")

    fig.tight_layout()

    out = out_dir / f"grid_comparison_size{args.grid_size}_{grid_id}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
