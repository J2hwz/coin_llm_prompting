"""
plot_trajectories_two_coins.py

CLI: given a folder of two-coin layout + trajectory JSON files, produce two
outputs per layout in <folder>/plots/:

    {prefix}_panels.png   — one panel per trajectory (N panels in a 5-col grid)
    {prefix}_heatmap.png  — cell visit-frequency heatmap across all trajectories

Usage:
    python plot_trajectories_two_coins.py <data_folder> [effort]

effort defaults to "low". Pass "medium" or "high" to match those trajectory files.

Naming convention expected in <data_folder>:
    *_layout.json                 — one per layout (must contain coin_pos_1, coin_pos_2)
    *_{effort}_traj{N}.json       — trajectories for that layout (N = 0, 1, 2, …)
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# ── Constants ─────────────────────────────────────────────────────────────────

ACTION_DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

WALL, EMPTY, GOAL, COIN = 0, 1, 2, 3
GRID_CMAP  = ListedColormap(["#4a4a4a", "#f5f5f5", "#5cb85c", "#FFD700"])
_TRAJ_CMAP = LinearSegmentedColormap.from_list("traj", ["#2166ac", "#d6604d"])

NCOLS = 5


# ── Parsing helpers ───────────────────────────────────────────────────────────

def load_layout(layout_path: Path) -> dict:
    with open(layout_path) as f:
        data = json.load(f)
    return {
        "coin_pos_1":  tuple(data["coin_pos_1"]),
        "coin_pos_2":  tuple(data["coin_pos_2"]),
        "goal_pos":    tuple(data["goal_pos"]),
        "start_pos":   tuple(data["agent_start_pos"]),
        "grid_layout": data["grid_layout"],
        "grid_size":   data.get("grid_size"),
    }


def find_symbol_in_grid_state(grid_state_rows, symbol):
    """Return (col, row) of `symbol` in the text grid_state, or None."""
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


def count_coins_in_grid_state(grid_state_rows):
    """Count how many 'C' symbols appear in the text grid_state."""
    count = 0
    for row_str in grid_state_rows[1:]:
        parts = row_str.split()
        if len(parts) < 2:
            continue
        try:
            int(parts[0])
        except ValueError:
            continue
        count += sum(1 for cell in parts[1:] if cell == "C")
    return count


def find_coin_collection_steps(steps):
    """Return (step_idx_coin1, step_idx_coin2) where each coin disappears.

    Watches the count of 'C' symbols across steps. Returns None for a coin
    if it was never collected during the trajectory.
    """
    prev_count = None
    collection_steps = []
    for i, step in enumerate(steps):
        count = count_coins_in_grid_state(step.get("grid_state", []))
        if prev_count is not None and count < prev_count:
            collection_steps.append(i)
        prev_count = count

    first  = collection_steps[0] if len(collection_steps) > 0 else None
    second = collection_steps[1] if len(collection_steps) > 1 else None
    return first, second


def build_path(steps):
    """Return list of (col, row) positions visited, including final position."""
    path = []
    for step in steps:
        pos = find_symbol_in_grid_state(step.get("grid_state", []), "A")
        if pos is None:
            break
        path.append(pos)
    if steps and path:
        last_action = steps[-1].get("agent_action", "").upper()
        dx, dy = ACTION_DELTA.get(last_action, (0, 0))
        path.append((path[-1][0] + dx, path[-1][1] + dy))
    return path


def detect_reached_goal(steps, goal_pos):
    if not steps:
        return False
    pos = find_symbol_in_grid_state(steps[-1].get("grid_state", []), "A")
    if pos is None:
        return False
    action = steps[-1].get("agent_action", "").upper()
    dx, dy = ACTION_DELTA.get(action, (0, 0))
    return (pos[0] + dx, pos[1] + dy) == tuple(goal_pos)


# ── Panel drawing ─────────────────────────────────────────────────────────────

def draw_panel(ax, layout, path, traj_id, coins_collected, reached_goal,
               cc_step_1, cc_step_2):
    grid       = layout["grid_layout"]
    rows       = len(grid)
    cols       = len(grid[0])
    coin_pos_1 = layout["coin_pos_1"]
    coin_pos_2 = layout["coin_pos_2"]
    goal_pos   = layout["goal_pos"]
    start_pos  = layout["start_pos"]

    ax.set_facecolor("white")

    # Cell-type image
    cell_img = np.ones((rows, cols), dtype=int) * EMPTY
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "#":
                cell_img[r, c] = WALL
    if not reached_goal:
        cell_img[goal_pos[1], goal_pos[0]] = GOAL
    if cc_step_1 is None:  # coin 1 never collected — draw it
        cell_img[coin_pos_1[1], coin_pos_1[0]] = COIN
    if cc_step_2 is None:  # coin 2 never collected — draw it
        cell_img[coin_pos_2[1], coin_pos_2[0]] = COIN

    ax.imshow(cell_img, origin="upper", cmap=GRID_CMAP, vmin=0, vmax=3,
              interpolation="nearest", aspect="equal")

    # Grid lines
    for r in range(rows + 1):
        ax.axhline(r - 0.5, color="#aaaaaa", lw=0.3, zorder=1)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color="#aaaaaa", lw=0.3, zorder=1)

    # ── Temporal-gradient path with perpendicular jitter ──────────────────────
    if len(path) >= 2:
        n = len(path) - 1
        jitter_scale = 0.25
        edge_count = {}
        jsegs = []
        for i in range(n):
            p0, p1 = path[i], path[i + 1]
            edge = (min(p0, p1), max(p0, p1))
            k = edge_count.get(edge, 0)
            edge_count[edge] = k + 1
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            length = math.hypot(dx, dy)
            if length > 0:
                px, py = -dy / length, dx / length
                mag  = (k % 2) * jitter_scale
                ox, oy = mag * px, mag * py
            else:
                ox, oy = 0.0, 0.0
            jsegs.append([(p0[0] + ox, p0[1] + oy), (p1[0] + ox, p1[1] + oy)])

        t_vals = np.linspace(0, 1, n)

        all_segs, all_t = [], []
        for i in range(n):
            all_segs.append(jsegs[i])
            all_t.append(t_vals[i])
            if i < n - 1:
                gap_start = jsegs[i][1]
                gap_end   = jsegs[i + 1][0]
                if gap_start != gap_end:
                    all_segs.append([gap_start, gap_end])
                    all_t.append((t_vals[i] + t_vals[i + 1]) / 2)

        lc = LineCollection(all_segs, array=np.array(all_t), cmap=_TRAJ_CMAP,
                            linewidths=0.9, alpha=0.6, zorder=5)
        lc.set_clim(0, 1)
        ax.add_collection(lc)

        for i in range(4, n, 5):
            p0_j, p1_j = jsegs[i]
            dx = p1_j[0] - p0_j[0]
            dy = p1_j[1] - p0_j[1]
            mid = ((p0_j[0] + p1_j[0]) / 2, (p0_j[1] + p1_j[1]) / 2)
            seg_color = _TRAJ_CMAP(t_vals[i])
            ax.annotate("",
                xy=(mid[0] + dx * 0.001, mid[1] + dy * 0.001),
                xytext=(mid[0] - dx * 0.001, mid[1] - dy * 0.001),
                arrowprops=dict(arrowstyle="-|>", color=seg_color,
                                lw=0.5, mutation_scale=6),
                zorder=7)
            dest = path[i + 1]
            ax.text(dest[0] - 0.42, dest[1] - 0.42, str(i + 1),
                    fontsize=3.5, color=seg_color, fontweight="bold",
                    ha="left", va="top", zorder=8)

    # Final position
    if path:
        if reached_goal:
            ax.plot(path[-1][0], path[-1][1], "^",
                    color="#5cb85c", ms=14, mec="white", mew=0.8, zorder=8)
        else:
            ax.plot(path[-1][0], path[-1][1], "X",
                    color="darkred", ms=7, mec="white", mew=0.7, zorder=8)

    # Coin collection markers — yellow diamond per coin collected
    for cc_step in (cc_step_1, cc_step_2):
        if cc_step is not None:
            idx = min(cc_step, len(path) - 1)
            ax.plot(path[idx][0], path[idx][1], "D",
                    color="#FFD700", ms=9, mec="white", mew=0.8, zorder=8)

    # ── Cell text labels ──────────────────────────────────────────────────────
    ax.text(start_pos[0], start_pos[1], "S",
            ha="center", va="center", fontsize=5, fontweight="bold",
            color="black", zorder=9)
    ax.text(goal_pos[0], goal_pos[1], "G",
            ha="center", va="center", fontsize=5, fontweight="bold",
            color="black", zorder=9)
    ax.text(coin_pos_1[0], coin_pos_1[1], "C1",
            ha="center", va="center", fontsize=4, fontweight="bold",
            color="black", zorder=9)
    ax.text(coin_pos_2[0], coin_pos_2[1], "C2",
            ha="center", va="center", fontsize=4, fontweight="bold",
            color="black", zorder=9)

    # ── Outcome badge ─────────────────────────────────────────────────────────
    goal_str  = "✓G" if reached_goal else "✗G"
    coin_str  = f"C:{coins_collected}/2"
    both_collected = coins_collected == 2
    badge_color = ("#2d8a2d" if (reached_goal and both_collected)
                  else "#cc8800" if (reached_goal or coins_collected > 0)
                  else "#cc2222")
    ax.text(0.98, 0.98, f"{goal_str} {coin_str}",
            transform=ax.transAxes, fontsize=5.5, va="top", ha="right",
            color=badge_color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75))

    n_steps = max(0, len(path) - 1)
    ax.text(0.02, 0.98, f"{n_steps}s",
            transform=ax.transAxes, fontsize=5.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75))

    ax.set_title(f"traj {traj_id}", fontsize=6, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_panels(layout, traj_data, prefix, out_dir):
    """One figure with one panel per trajectory."""
    n     = len(traj_data)
    nrows = max(1, math.ceil(n / NCOLS))

    grid           = layout["grid_layout"]
    g_rows, g_cols = len(grid), len(grid[0])
    panel_w = max(1.8, g_cols * 0.3)
    panel_h = max(1.8, g_rows * 0.3)

    fig, axes = plt.subplots(nrows, NCOLS,
                             figsize=(NCOLS * panel_w + 0.4, nrows * panel_h + 0.6))
    axes = np.array(axes).reshape(nrows, NCOLS)

    for idx in range(nrows * NCOLS):
        r, c = divmod(idx, NCOLS)
        ax = axes[r, c]
        if idx < n:
            td = traj_data[idx]
            draw_panel(ax, layout, td["path"], td["traj_id"],
                       td["coins_collected"], td["reached_goal"],
                       td["cc_step_1"], td["cc_step_2"])
        else:
            ax.set_visible(False)

    fig.suptitle(prefix, fontsize=7, y=1.01)
    fig.tight_layout(rect=[0, 0, 0.93, 1])

    sm = plt.cm.ScalarMappable(cmap=_TRAJ_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.945, 0.1, 0.014, 0.78])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Start", "End"])
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Timestep", fontsize=6, rotation=270, labelpad=10)

    out_path = out_dir / f"{prefix}_panels.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_heatmap(layout, traj_data, prefix, out_dir):
    """Cell visit-frequency heatmap aggregated across all trajectories."""
    grid       = layout["grid_layout"]
    rows, cols = len(grid), len(grid[0])
    goal_pos   = layout["goal_pos"]
    coin_pos_1 = layout["coin_pos_1"]
    coin_pos_2 = layout["coin_pos_2"]
    start_pos  = layout["start_pos"]

    counts = np.zeros((rows, cols), dtype=float)
    for td in traj_data:
        for (c, r) in td["path"]:
            if 0 <= r < rows and 0 <= c < cols:
                counts[r, c] += 1

    wall_mask = np.array(
        [[grid[r][c] == "#" for c in range(cols)] for r in range(rows)]
    )
    heat = np.ma.masked_where(wall_mask, counts)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("#4a4a4a")

    im = ax.imshow(heat, origin="upper", cmap=cmap, aspect="equal",
                   interpolation="nearest", vmin=0)
    plt.colorbar(im, ax=ax, label="Visit count", fraction=0.046)

    for r in range(rows + 1):
        ax.axhline(r - 0.5, color="#777", lw=0.3, zorder=1)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color="#777", lw=0.3, zorder=1)

    ax.text(goal_pos[0], goal_pos[1], "G", ha="center", va="center",
            fontsize=7, fontweight="bold", color="black", zorder=5)
    ax.text(coin_pos_1[0], coin_pos_1[1], "C1", ha="center", va="center",
            fontsize=6, fontweight="bold", color="black", zorder=5)
    ax.text(coin_pos_2[0], coin_pos_2[1], "C2", ha="center", va="center",
            fontsize=6, fontweight="bold", color="black", zorder=5)
    ax.text(start_pos[0], start_pos[1], "S", ha="center", va="center",
            fontsize=7, fontweight="bold", color="black", zorder=5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{prefix}\nVisit frequency ({len(traj_data)} trajectories)",
                 fontsize=7)
    fig.tight_layout()
    out_path = out_dir / f"{prefix}_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def process_layout(layout_path: Path, out_dir: Path, effort: str = "low"):
    layout = load_layout(layout_path)
    prefix = layout_path.stem[: -len("_layout")]

    traj_paths = sorted(
        layout_path.parent.glob(f"{prefix}_{effort}_traj*.json"),
        key=lambda p: int("".join(filter(str.isdigit, p.stem.split("traj")[-1]))),
    )
    if not traj_paths:
        print(f"  WARNING: no trajectory files found for {layout_path.name}, skipping.")
        return

    traj_data = []
    for traj_id, traj_path in enumerate(traj_paths):
        with open(traj_path) as f:
            data = json.load(f)
        steps          = data.get("steps", [])
        path           = build_path(steps)
        cc_step_1, cc_step_2 = find_coin_collection_steps(steps)
        coins_collected = (0 + (cc_step_1 is not None) + (cc_step_2 is not None))
        reached_goal   = detect_reached_goal(steps, layout["goal_pos"])
        traj_data.append(dict(
            path=path,
            coins_collected=coins_collected,
            cc_step_1=cc_step_1,
            cc_step_2=cc_step_2,
            reached_goal=reached_goal,
            traj_id=traj_id,
        ))

    plot_panels(layout, traj_data, prefix, out_dir)
    plot_heatmap(layout, traj_data, prefix, out_dir)


def main(folder: Path, effort: str = "low"):
    layout_files = sorted(folder.glob("*_layout.json"))
    if not layout_files:
        sys.exit(f"No *_layout.json files found in {folder}")

    out_dir = folder / "plots"
    out_dir.mkdir(exist_ok=True)

    print(f"Found {len(layout_files)} layout(s) in {folder}")
    for lf in layout_files:
        print(f"  {lf.name}")
        process_layout(lf, out_dir, effort=effort)

    print(f"\nDone. Plots saved to {out_dir}/")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("Usage: python plot_trajectories_two_coins.py <data_folder> [effort]")
    effort = sys.argv[2] if len(sys.argv) == 3 else "low"
    main(Path(sys.argv[1]), effort=effort)
