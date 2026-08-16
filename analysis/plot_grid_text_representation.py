"""
plot_grid_text_representation.py
----------------------------------
Two-panel figure for a single grid: a rendered maze on the left and, on the
right, the exact text-based representation the LLM agent receives in its
prompt (row/column indices + symbol matrix, as substituted into the
`{{grid_state}}` placeholder of grid_one_coin_control.j2 — see the "worked
example" in that template, which uses this same row/column-indexed format).

Defaults to a 9x9, density-0.4 one-coin grid (grid index 0), read from
data/one_coin/low/9_low/control/*_layout.json.

Usage:
    python analysis/plot_grid_text_representation.py [--grid-size 9] [--density 0.4] [--grid-index 0]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "one_coin" / "low"
OUT_DIR = REPO_ROOT / "analysis" / "outputs" / "figures" / "one_coin_low"
MODEL_PREFIX = "together_ai_openai_gpt-oss-20b"

# Same palette/style as plot_grid_layouts.py and plot_condition_trajectories.py.
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


# ── Left panel: rendered maze ───────────────────────────────────────────────

def draw_rendered_grid(ax, layout: dict) -> None:
    grid = layout["grid_layout"]
    rows, cols = len(grid), len(grid[0])
    goal_pos = layout["goal_pos"]
    coin_pos = layout["coin_pos"]
    start_pos = layout["agent_start_pos"]

    cell_img = np.full((rows, cols), EMPTY, dtype=int)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "#":
                cell_img[r, c] = WALL
    cell_img[goal_pos[1], goal_pos[0]] = GOAL
    cell_img[coin_pos[1], coin_pos[0]] = COIN

    ax.imshow(cell_img, origin="upper", cmap=GRID_CMAP, vmin=0, vmax=3,
              interpolation="nearest", aspect="equal")

    for r in range(rows + 1):
        ax.axhline(r - 0.5, color="#aaaaaa", lw=0.3, zorder=1)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color="#aaaaaa", lw=0.3, zorder=1)

    ax.text(start_pos[0], start_pos[1], "S", ha="center", va="center",
             fontsize=8, fontweight="bold", color="black", zorder=9)
    ax.text(goal_pos[0], goal_pos[1], "G", ha="center", va="center",
             fontsize=8, fontweight="bold", color="black", zorder=9)
    ax.text(coin_pos[0], coin_pos[1], "C", ha="center", va="center",
             fontsize=8, fontweight="bold", color="black", zorder=9)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#888888")
        spine.set_linewidth(0.6)


# ── Right panel: text-table exactly as sent to the LLM ────────────────────

def draw_text_table(ax, layout: dict) -> None:
    grid = layout["grid_layout"]
    rows, cols = len(grid), len(grid[0])

    header = [""] + [str(c) for c in range(cols)]
    cell_text = [header]
    for r in range(rows):
        cell_text.append([str(r)] + [grid[r][c] for c in range(cols)])

    ax.axis("off")
    table = ax.table(cellText=cell_text, cellLoc="center", loc="center",
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#999999")
        cell.set_linewidth(0.6)
        cell.set_text_props(fontfamily="monospace", fontsize=11)
        if r == 0 or c == 0:
            cell.set_text_props(fontfamily="monospace", fontsize=11, color="#555555")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_figure(layout: dict) -> plt.Figure:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 5.2))
    draw_rendered_grid(ax_left, layout)
    draw_text_table(ax_right, layout)

    fig.tight_layout()
    return fig


def main(grid_size: int = 9, density: float = 0.4, grid_index: int = 0) -> None:
    layout = load_layout(grid_size, density, grid_index)
    fig = build_figure(layout)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"grid_text_representation_size{grid_size}_comp{density}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--density", type=float, default=0.4)
    parser.add_argument("--grid-index", type=int, default=0)
    args = parser.parse_args()
    main(grid_size=args.grid_size, density=args.density, grid_index=args.grid_index)
