"""
plot_grid_layouts.py
---------------------
Renders a fixed panel of one-coin gridworld layouts — actual grids generated
for the experiment (read from `data/one_coin/low/<size>_low/control/*_layout.json`,
grid index 0 by default) — with no trajectories overlaid, just walls, start,
goal, and coin.

Produces two versions of the same 12-grid panel (grid sizes {7, 9, 11} x
wall densities {0.0, 0.2, 0.4, 0.6}):

    grid_layouts_landscape.png   3x4, rows = grid size,   columns = density
    grid_layouts_portrait.png    4x3, rows = density,     columns = grid size

Usage:
    python analysis/plot_grid_layouts.py [--grid-index 0]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "one_coin" / "low"
OUT_DIR = REPO_ROOT / "analysis" / "outputs" / "figures" / "one_coin_low"

SIZES = [7, 9, 11]
DENSITIES = [0.0, 0.2, 0.4, 0.6]
MODEL_PREFIX = "together_ai_openai_gpt-oss-20b"

# Same palette used elsewhere in analysis/ (plot_trajectories.py) so this
# figure reads as part of the same visual family.
WALL, EMPTY, GOAL, COIN = 0, 1, 2, 3
GRID_CMAP = ListedColormap(["#4a4a4a", "#f5f5f5", "#5cb85c", "#FFD700"])


# ── Loading ───────────────────────────────────────────────────────────────────

def layout_path(size: int, density: float, grid_index: int) -> Path:
    fname = f"{MODEL_PREFIX}_size{size}_comp{density}_grid{grid_index}_coin_layout.json"
    return DATA_ROOT / f"{size}_low" / "control" / fname


def load_layout(size: int, density: float, grid_index: int) -> dict:
    path = layout_path(size, density, grid_index)
    if not path.exists():
        raise FileNotFoundError(f"Missing layout file: {path}")
    with open(path) as f:
        return json.load(f)


# ── Panel drawing ─────────────────────────────────────────────────────────────

def draw_panel(ax, layout: dict) -> None:
    grid = layout["grid_layout"]
    rows, cols = len(grid), len(grid[0])
    start_pos = layout["agent_start_pos"]
    goal_pos = layout["goal_pos"]
    coin_pos = layout["coin_pos"]

    cell_img = [[EMPTY] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "#":
                cell_img[r][c] = WALL
    cell_img[goal_pos[1]][goal_pos[0]] = GOAL
    cell_img[coin_pos[1]][coin_pos[0]] = COIN

    ax.imshow(cell_img, origin="upper", cmap=GRID_CMAP, vmin=0, vmax=3,
               interpolation="nearest", aspect="equal")

    for r in range(rows + 1):
        ax.axhline(r - 0.5, color="#aaaaaa", lw=0.3, zorder=1)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color="#aaaaaa", lw=0.3, zorder=1)

    ax.text(start_pos[0], start_pos[1], "S", ha="center", va="center",
             fontsize=8, fontweight="bold", color="black", zorder=5)
    ax.text(goal_pos[0], goal_pos[1], "G", ha="center", va="center",
             fontsize=8, fontweight="bold", color="black", zorder=5)
    ax.text(coin_pos[0], coin_pos[1], "C", ha="center", va="center",
             fontsize=8, fontweight="bold", color="black", zorder=5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#888888")
        spine.set_linewidth(0.6)


# ── Figure assembly ───────────────────────────────────────────────────────────

def build_figure(layouts: dict, orientation: str) -> plt.Figure:
    """orientation: 'landscape' (rows=size, cols=density) or
    'portrait' (rows=density, cols=size)."""
    if orientation == "landscape":
        nrows, ncols = len(SIZES), len(DENSITIES)
        row_vals, col_vals = SIZES, DENSITIES
        row_label, col_label = "Grid size", "Density"
        figsize = (3.0 * ncols, 3.0 * nrows)
    else:
        nrows, ncols = len(DENSITIES), len(SIZES)
        row_vals, col_vals = DENSITIES, SIZES
        row_label, col_label = "Density", "Grid size"
        figsize = (3.0 * ncols, 3.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            size, density = (rv, cv) if row_label == "Grid size" else (cv, rv)
            ax = axes[i][j]
            draw_panel(ax, layouts[(size, density)])
            if i == 0:
                ax.set_title(f"{col_label} = {cv}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{row_label} = {rv}", fontsize=11)

    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main(grid_index: int = 0) -> None:
    layouts = {
        (size, density): load_layout(size, density, grid_index)
        for size in SIZES
        for density in DENSITIES
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = build_figure(layouts, "landscape")
    out_path = OUT_DIR / "grid_layouts_landscape.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    fig = build_figure(layouts, "portrait")
    out_path = OUT_DIR / "grid_layouts_portrait.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-index", type=int, default=0,
                         help="Which grid index (0-9) to draw per (size, density) cell.")
    args = parser.parse_args()
    main(grid_index=args.grid_index)
