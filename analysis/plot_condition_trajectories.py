"""
plot_condition_trajectories.py
--------------------------------
Renders one successful and one unsuccessful trajectory for each of 12
experimental conditions, all on the *same grid size and density* (7x7,
density 0.2) so only the condition (and outcome) varies across panels — the
maze difficulty is held fixed.

For each condition, layout+trajectory JSON files are searched grid-by-grid
(grid index 0, 1, 2, ...) for the first trajectory that satisfies the
condition's success criterion and the first that does not; the panel then
draws that trajectory's path over its own maze (walls, start, goal, and any
coin(s) not collected by episode end).

Conditions (12): control low/medium, fine-tuned control low/medium,
deceptive low, random action, avoid low/medium, collect-all low/medium,
collect-one low/medium.

Produces two versions of the same 24-panel figure (12 conditions x 2
outcomes):

    condition_trajectories_landscape.png   2x12, rows = outcome, columns = condition
    condition_trajectories_portrait.png    12x2, rows = condition, columns = outcome

Usage:
    python analysis/plot_condition_trajectories.py
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "analysis" / "outputs" / "figures" / "overall"

GRID_SIZE = 7
DENSITY = 0.2
GRID_IDS = list(range(10))  # search order when looking for a success/failure example

ACTION_DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
WALL, EMPTY, GOAL, COIN = 0, 1, 2, 3
GRID_CMAP = ListedColormap(["#4a4a4a", "#f5f5f5", "#5cb85c", "#FFD700"])
_TRAJ_CMAP = LinearSegmentedColormap.from_list("traj", ["#2166ac", "#d6604d"])


@dataclass
class Condition:
    key: str
    label: str  # multi-line title
    folder: Path
    effort: str  # trajectory filename effort tag: "low" / "medium" / "random"
    num_coins: int  # 1 or 2
    # (reached_goal, n_coins_collected) -> success
    success_fn: Callable[[bool, int], bool]


CONDITIONS = [
    Condition("control_low", "Collect coin\n(low)",
              REPO_ROOT / "data/one_coin/low/7_low/control", "low", 1,
              lambda g, c: g and c >= 1),
    Condition("control_medium", "Collect coin\n(medium)",
              REPO_ROOT / "data/one_coin/medium/7_medium", "medium", 1,
              lambda g, c: g and c >= 1),
    Condition("ft_control_low", "Fine-tuned\n(low)",
              REPO_ROOT / "data/finetuned_model/control/low/7_low", "low", 1,
              lambda g, c: g and c >= 1),
    Condition("ft_control_medium", "Fine-tuned\n(medium)",
              REPO_ROOT / "data/finetuned_model/control/medium/7_medium", "medium", 1,
              lambda g, c: g and c >= 1),
    Condition("deceptive_low", "Deceptive\n(low)",
              REPO_ROOT / "data/one_coin/low_deceptive/7_low_deceptive_with_steps", "low", 1,
              lambda g, c: g and c >= 1),
    Condition("random", "Random\naction",
              REPO_ROOT / "data/one_coin/random/7_random", "random", 1,
              lambda g, c: g and c >= 1),
    Condition("avoid_low", "Avoid coin\n(low)",
              REPO_ROOT / "data/one_coin_avoid/low/7_low", "low", 1,
              lambda g, c: g and c == 0),
    Condition("avoid_medium", "Avoid coin\n(medium)",
              REPO_ROOT / "data/one_coin_avoid/medium/7_medium", "medium", 1,
              lambda g, c: g and c == 0),
    Condition("collect_all_low", "Collect-all\n(low)",
              REPO_ROOT / "data/two_coins/collect_all/low/7_low", "low", 2,
              lambda g, c: g and c == 2),
    Condition("collect_all_medium", "Collect-all\n(medium)",
              REPO_ROOT / "data/two_coins/collect_all/medium/7_medium", "medium", 2,
              lambda g, c: g and c == 2),
    # Note: "collect one" success requires *exactly* one coin, not "at least
    # one" — the instruction is to collect only one and leave the other, so
    # a trajectory that touches both coins is not a compliant example even
    # though it still reaches the goal.
    Condition("collect_one_low", "Collect-one\n(low)",
              REPO_ROOT / "data/two_coins/collect_one/low/7_low", "low", 2,
              lambda g, c: g and c == 1),
    Condition("collect_one_medium", "Collect-one\n(medium)",
              REPO_ROOT / "data/two_coins/collect_one/medium/7_medium", "medium", 2,
              lambda g, c: g and c == 1),
]


# ── Loading helpers ────────────────────────────────────────────────────────────

def load_layout(layout_path: Path) -> dict:
    with open(layout_path) as f:
        data = json.load(f)
    if "coin_pos_1" in data:
        coin_positions = [tuple(data["coin_pos_1"]), tuple(data["coin_pos_2"])]
    else:
        coin_positions = [tuple(data["coin_pos"])]
    return {
        "coin_positions": coin_positions,
        "goal_pos": tuple(data["goal_pos"]),
        "start_pos": tuple(data["agent_start_pos"]),
        "grid_layout": data["grid_layout"],
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


def build_path(steps: list) -> list:
    """Return list of (col, row) positions visited, including the final
    position reached by applying the last recorded action."""
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


def detect_reached_goal(path: list, goal_pos: tuple) -> bool:
    return bool(path) and path[-1] == tuple(goal_pos)


def count_coins_collected(path: list, coin_positions: list) -> int:
    """A coin is collected iff its cell was ever visited (coins are passable
    and collection is automatic on entry) — checked directly against the
    agent's realised path rather than any per-step info flag, since the
    two-coin trajectory schema doesn't carry one."""
    visited = set(path)
    return sum(1 for cp in coin_positions if cp in visited)


# ── Example selection ─────────────────────────────────────────────────────────

@dataclass
class Example:
    layout: dict
    path: list
    reached_goal: bool
    n_coins_collected: int
    success: bool
    grid_id: int
    traj_id: int


def find_examples(cond: Condition) -> tuple[Optional[Example], Optional[Example]]:
    """Search grids 0..9 for the first successful and first unsuccessful
    trajectory under this condition's success_fn."""
    best: dict[bool, Example] = {}

    for grid_id in GRID_IDS:
        if True in best and False in best:
            break
        layout_matches = sorted(cond.folder.glob(f"*size{GRID_SIZE}_comp{DENSITY}_grid{grid_id}_*_layout.json"))
        if not layout_matches:
            continue
        layout_path = layout_matches[0]
        layout = load_layout(layout_path)
        prefix = layout_path.name[: -len("_layout.json")]

        traj_paths = sorted(
            layout_path.parent.glob(f"{prefix}_{cond.effort}_traj*.json"),
            key=lambda p: int("".join(filter(str.isdigit, p.stem.split("traj")[-1]))),
        )
        for traj_path in traj_paths:
            with open(traj_path) as f:
                data = json.load(f)
            steps = data.get("steps", [])
            path = build_path(steps)
            reached_goal = detect_reached_goal(path, layout["goal_pos"])
            n_collected = count_coins_collected(path, layout["coin_positions"])
            success = bool(cond.success_fn(reached_goal, n_collected))

            if success not in best:
                traj_id = int("".join(filter(str.isdigit, traj_path.stem.split("traj")[-1])))
                best[success] = Example(layout, path, reached_goal, n_collected,
                                          success, grid_id, traj_id)
            if True in best and False in best:
                break

    return best.get(True), best.get(False)


# ── Panel drawing ─────────────────────────────────────────────────────────────

def draw_panel(ax, example: Example) -> None:
    layout = example.layout
    grid = layout["grid_layout"]
    rows, cols = len(grid), len(grid[0])
    goal_pos = layout["goal_pos"]
    start_pos = layout["start_pos"]
    path = example.path
    visited = set(path)

    ax.set_facecolor("white")

    cell_img = np.ones((rows, cols), dtype=int) * EMPTY
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "#":
                cell_img[r, c] = WALL
    if not example.reached_goal:
        cell_img[goal_pos[1], goal_pos[0]] = GOAL
    for cp in layout["coin_positions"]:
        if cp not in visited:  # only draw coins that weren't collected
            cell_img[cp[1], cp[0]] = COIN

    ax.imshow(cell_img, origin="upper", cmap=GRID_CMAP, vmin=0, vmax=3,
               interpolation="nearest", aspect="equal")

    for r in range(rows + 1):
        ax.axhline(r - 0.5, color="#aaaaaa", lw=0.3, zorder=1)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color="#aaaaaa", lw=0.3, zorder=1)

    # ── Temporal-gradient path with perpendicular jitter ────────────────────
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
                mag = (k % 2) * jitter_scale
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
                gap_end = jsegs[i + 1][0]
                if gap_start != gap_end:
                    all_segs.append([gap_start, gap_end])
                    all_t.append((t_vals[i] + t_vals[i + 1]) / 2)

        lc = LineCollection(all_segs, array=np.array(all_t), cmap=_TRAJ_CMAP,
                            linewidths=0.9, alpha=0.6, zorder=5)
        lc.set_clim(0, 1)
        ax.add_collection(lc)

    # Final position — green triangle if goal reached, dark-red X otherwise
    if path:
        if example.reached_goal:
            ax.plot(path[-1][0], path[-1][1], "^",
                    color="#5cb85c", ms=12, mec="white", mew=0.8, zorder=8)
        else:
            ax.plot(path[-1][0], path[-1][1], "X",
                    color="darkred", ms=7, mec="white", mew=0.7, zorder=8)

    # ── Cell text labels ──────────────────────────────────────────────────────
    ax.text(start_pos[0], start_pos[1], "S", ha="center", va="center",
             fontsize=6, fontweight="bold", color="black", zorder=9)
    ax.text(goal_pos[0], goal_pos[1], "G", ha="center", va="center",
             fontsize=6, fontweight="bold", color="black", zorder=9)
    for i, cp in enumerate(layout["coin_positions"]):
        label = "C" if len(layout["coin_positions"]) == 1 else f"C{i + 1}"
        ax.text(cp[0], cp[1], label, ha="center", va="center",
                 fontsize=5, fontweight="bold", color="black", zorder=9)

    # ── Outcome badge + step count ────────────────────────────────────────────
    goal_str = "✓G" if example.reached_goal else "✗G"
    n_coins = len(layout["coin_positions"])
    coin_str = ("✓C" if example.n_coins_collected >= 1 else "✗C") if n_coins == 1 \
        else f"C:{example.n_coins_collected}/{n_coins}"
    badge_color = "#2d8a2d" if example.success else "#cc2222"
    ax.text(0.98, 0.98, f"{goal_str} {coin_str}",
             transform=ax.transAxes, fontsize=5.5, va="top", ha="right",
             color=badge_color,
             bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75))

    n_steps = max(0, len(path) - 1)
    ax.text(0.02, 0.98, f"{n_steps}s",
             transform=ax.transAxes, fontsize=5.5, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#888888")
        spine.set_linewidth(0.6)


# ── Figure assembly ───────────────────────────────────────────────────────────

def draw_empty_panel(ax, outcome_label: str) -> None:
    """Panel shown when no trajectory of the requested outcome exists in the
    dataset for this condition at this grid — e.g. a condition solved 10/10
    at this difficulty has no failure example to show."""
    ax.set_facecolor("#f5f5f5")
    ax.text(0.5, 0.5, f"No {outcome_label.lower()}\nexample in dataset",
             transform=ax.transAxes, ha="center", va="center",
             fontsize=7, color="#888888", style="italic")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.6)


OUTCOMES = [("Successful", True), ("Unsuccessful", False)]
GROUP_SIZE = 4  # conditions per band, so no more than 4 grid panels line up in a row/column


# Portrait: the column gap between bands needs to fit the condition-label
# text printed to the left of each band, so it gets a dedicated spacer track
# (a fraction of a normal panel's width).
PORTRAIT_BAND_SPACER = 0.4

# Landscape: bands stack vertically. A single flat GridSpec with one hspace
# value keeps every row gap (within a band and between bands) exactly equal.
LANDSCAPE_HSPACE = 0.25


def _banded_ratios(n_bands: int, n_outcomes: int, spacer: float) -> list[float]:
    """[1]*n_outcomes per band, with a `spacer`-sized gap track between
    consecutive bands (none trailing after the last one)."""
    ratios = []
    for b in range(n_bands):
        ratios += [1.0] * n_outcomes
        if b < n_bands - 1:
            ratios.append(spacer)
    return ratios


def _place_panel(ax, cond: Condition, outcome_label: str, is_success: bool,
                  examples: dict, orientation: str, is_band_edge: bool, is_first_in_band: bool) -> None:
    example = examples[cond.key][is_success]
    if example is None:
        draw_empty_panel(ax, outcome_label)
    else:
        draw_panel(ax, example)

    if orientation == "landscape":
        if is_band_edge:  # top row of this band
            ax.set_title(cond.label, fontsize=9)
        if is_first_in_band:  # left column of every band
            ax.set_ylabel(outcome_label, fontsize=10)
    else:
        if is_first_in_band:  # top row of every band
            ax.set_title(outcome_label, fontsize=10)
        if is_band_edge:  # left column of this band
            ax.set_ylabel(cond.label, fontsize=8, rotation=0,
                           ha="right", va="center", labelpad=8)


def build_figure(examples: dict, orientation: str) -> plt.Figure:
    """Conditions are wrapped into bands of GROUP_SIZE so that no single row
    (landscape) or column (portrait) holds more than GROUP_SIZE grid panels;
    bands stack in the other direction.

    orientation: 'landscape' (each band = 2 rows x GROUP_SIZE cols, bands
    stacked vertically -> caps panels-per-row at GROUP_SIZE) or 'portrait'
    (each band = GROUP_SIZE rows x 2 cols, bands placed side by side -> caps
    panels-per-column at GROUP_SIZE).
    """
    n_cond = len(CONDITIONS)
    n_bands = math.ceil(n_cond / GROUP_SIZE)
    n_outcomes = len(OUTCOMES)

    if orientation == "landscape":
        nrows = n_bands * n_outcomes
        figsize = (2.3 * GROUP_SIZE, 2.6 * nrows)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows, GROUP_SIZE, hspace=LANDSCAPE_HSPACE, wspace=0.3)

        for ci, cond in enumerate(CONDITIONS):
            band, pos_in_band = divmod(ci, GROUP_SIZE)
            for oi, (outcome_label, is_success) in enumerate(OUTCOMES):
                ax = fig.add_subplot(gs[band * n_outcomes + oi, pos_in_band])
                _place_panel(ax, cond, outcome_label, is_success, examples, orientation,
                             is_band_edge=(oi == 0), is_first_in_band=(pos_in_band == 0))
        return fig

    # Portrait: single flat GridSpec with an explicit spacer column between bands.
    height_ratios = [1.0] * GROUP_SIZE
    width_ratios = _banded_ratios(n_bands, n_outcomes, PORTRAIT_BAND_SPACER)
    band_start = lambda b: b * (n_outcomes + 1)  # n_outcomes panel-tracks + 1 spacer track

    figsize = (2.3 * sum(width_ratios), 2.6 * sum(height_ratios))
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(len(height_ratios), len(width_ratios),
                           height_ratios=height_ratios, width_ratios=width_ratios,
                           hspace=0.45, wspace=0.3)

    for ci, cond in enumerate(CONDITIONS):
        band, pos_in_band = divmod(ci, GROUP_SIZE)
        for oi, (outcome_label, is_success) in enumerate(OUTCOMES):
            ax = fig.add_subplot(gs[pos_in_band, band_start(band) + oi])
            _place_panel(ax, cond, outcome_label, is_success, examples, orientation,
                         is_band_edge=(oi == 0), is_first_in_band=(pos_in_band == 0))

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    examples: dict[str, dict[bool, Optional[Example]]] = {}
    for cond in CONDITIONS:
        success_ex, failure_ex = find_examples(cond)
        examples[cond.key] = {True: success_ex, False: failure_ex}
        for is_success, ex in ((True, success_ex), (False, failure_ex)):
            tag = "success" if is_success else "failure"
            if ex is None:
                print(f"  WARNING: no {tag} example found for {cond.key}")
            else:
                print(f"  {cond.key:<20s} {tag:<8s} -> grid{ex.grid_id} traj{ex.traj_id} "
                      f"({len(ex.path) - 1} steps)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = build_figure(examples, "landscape")
    out_path = OUT_DIR / "condition_trajectories_landscape.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    fig = build_figure(examples, "portrait")
    out_path = OUT_DIR / "condition_trajectories_portrait.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
