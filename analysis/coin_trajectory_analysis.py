"""Analyze coin navigation trajectories with two-phase optimal policy.

Full metrics reference: analysis/coin_trajectory_analysis_metrics.md
"""

import argparse
import gc
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.analysis_utils import (
    ACTION_NAME_TO_ID,
    OptimalActionSet,
    TrajectoryGridParams,
    TrajectoryStep,
    compute_optimal_actions_from_text_grid,
    extract_agent_position_from_grid_state,
    jensen_shannon_divergence,
    optimal_entropy,
    sanitize_label,
    shannon_entropy,
)
from analysis.full_obs_trajectory_analysis import (
    StateActionCounts,
    batch_grid_keys,
    bin_and_score_ece,
    collect_ece_values,
    collect_uncertainty_values,
    compute_empirical_uncertainty_metrics,
    compute_summary_by_distance,
    discover_model_directories,
)
from analysis.visualization import (
    plot_capability_vs_uncertainty,
    plot_metrics_by_distance,
    save_figure,
    setup_paper_style,
)

COIN_SYMBOL = "C"
WALL_SYMBOL = "#"
KNOWN_EFFORTS = {"low", "medium", "random"}

_ACTION_DELTAS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

# Transform token (from filename) → coarse trajectory category
TRANSFORM_TO_CATEGORY = {
    "base": "base",
    "ReflectEnv": "reflect",
    "RotateEnv": "rotate",
    "TransposeEnv": "transpose",
    "StartGoalSwap": "start_goal_swap",
}

# Containing directory → category, fallback for unrecognised transform tokens
DIR_TO_CATEGORY = {
    "control": "base",
    "augmentations": "unknown_transform",
    "random_starts": "random_starts",
    "reshuffled": "reshuffled_walls",
}


def infer_trajectory_category(filepath: Path, transform_type: str) -> str:
    """Categorise a trajectory as base / reflect / rotate / transpose /
    start_goal_swap / random_starts / reshuffled_walls.

    The filename transform token is authoritative; the containing directory
    name is only used as a fallback for unrecognised tokens.
    """
    category = TRANSFORM_TO_CATEGORY.get(transform_type)
    if category:
        return category
    if transform_type.startswith("RandomStart"):
        return "random_starts"
    if transform_type.startswith("walls"):
        return "reshuffled_walls"
    return DIR_TO_CATEGORY.get(filepath.parent.name, "unknown")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CoinTrajectoryGridParams(TrajectoryGridParams):
    """Grid parameters extended with coin position."""

    coin_pos: Optional[tuple[int, int]] = None
    start_to_coin_distance: int = 0       # A* dist: start → coin (-1 if unreachable)
    coin_to_goal_distance: int = 0       # A* dist: coin → goal (-1 if unreachable)
    start_to_goal_via_coin_distance: int = 0      # = start_to_coin_distance + coin_to_goal_distance
    start_to_goal_distance: int = 0    # A* dist: start → goal (ignoring coin, -1 if unreachable)
    coin_detour_distance: int = 0      # min dist from coin to any cell on optimal start→goal path
    coin_reachable_from_start: bool = True
    goal_reachable_from_start: bool = True
    goal_reachable_from_coin: bool = True
    grid_layout: Optional[list[list[str]]] = None  # wall layout, for invalid-action detection


@dataclass
class CoinLightweightTrajectory:
    """Trajectory with coin phase tracking."""

    grid_params: CoinTrajectoryGridParams
    steps: list[TrajectoryStep]
    reached_goal: bool
    coin_collected: bool
    coin_collection_step: Optional[int]  # step index when coin was collected, or None
    reasoning_effort: str = "low"
    transform_type: str = "base"
    trajectory_category: str = "base"
    trajectory_id: int = 0

    @property
    def trajectory_length(self) -> int:
        return len(self.steps)

    @property
    def steps_phase1(self) -> list[TrajectoryStep]:
        """Steps before (and including) coin collection."""
        if self.coin_collection_step is None:
            return self.steps
        return self.steps[: self.coin_collection_step + 1]

    @property
    def steps_phase2(self) -> list[TrajectoryStep]:
        """Steps after coin collection."""
        if self.coin_collection_step is None:
            return []
        return self.steps[self.coin_collection_step + 1 :]


@dataclass
class CoinGridTrajectoryMetrics:
    """Per-grid metrics for coin trajectories."""

    grid_id: str
    grid_size: int
    density: float
    instance_id: int
    reasoning_effort: str
    transform_type: str
    trajectory_category: str
    start_to_coin_distance: int
    coin_to_goal_distance: int
    start_to_goal_via_coin_distance: int
    start_to_goal_distance: int    # direct start→goal (ignoring coin)
    coin_detour_distance: int      # min dist from coin to optimal start→goal path
    coin_reachable_from_start: int
    goal_reachable_from_start: int
    goal_reachable_from_coin: int

    # --- Success breakdown ---
    num_trajectories: int
    num_coin_collected: int
    num_goal_after_coin: int   # coin + goal (full success)
    num_goal_only: int         # goal without coin
    coin_collected_rate: float
    goal_after_coin_rate: float  # full success
    goal_only_rate: float
    full_success_rate: float     # alias for goal_after_coin_rate

    # --- Capability ---
    mean_trajectory_length: float
    mean_action_accuracy: float          # weighted across both phases
    mean_action_accuracy_phase1: float   # toward coin
    mean_action_accuracy_phase2: float   # toward goal after coin
    spl: float

    # --- Uncertainty (empirical distributions) ---
    mean_entropy: float
    mean_optimal_entropy: float
    mean_jsd: float
    mean_entropy_phase1: float
    mean_optimal_entropy_phase1: float
    mean_jsd_phase1: float
    mean_entropy_phase2: float
    mean_optimal_entropy_phase2: float
    mean_jsd_phase2: float
    ece: float

    mean_step_accuracy: float
    total_steps: int

    # --- Absolute action counts (mean per trajectory) ---
    mean_actions_up: float = 0.0
    mean_actions_down: float = 0.0
    mean_actions_left: float = 0.0
    mean_actions_right: float = 0.0

    # --- Relative direction counts (mean per trajectory, orientation from previous step) ---
    mean_steps_front: float = 0.0
    mean_steps_left_turn: float = 0.0
    mean_steps_right_turn: float = 0.0
    mean_steps_back: float = 0.0      # immediate reversal (double-back)

    # --- Revisit counts (mean per trajectory) ---
    mean_cell_revisits: float = 0.0        # steps landing on any previously visited cell
    mean_immediate_revisits: float = 0.0   # steps returning to the immediately preceding cell
    mean_coin_oscillations: float = 0.0    # steps oscillating through the coin cell
    mean_invalid_actions: float = 0.0      # wall-bump actions (agent stays in the same cell)

    # --- Spatial / corner-dwelling metrics ---
    # Each computed per trajectory, then averaged (same convention as the
    # movement metrics above) — never by pooling positions across
    # trajectories first. preferred_quadrant/preferred_corner are the argmax
    # of the mean fractions below, not a per-trajectory mode.
    mean_quadrant_fraction_ul: float = 0.0
    mean_quadrant_fraction_ur: float = 0.0
    mean_quadrant_fraction_dl: float = 0.0
    mean_quadrant_fraction_dr: float = 0.0
    preferred_quadrant: str = ""
    mean_quadrant_entropy: float = 0.0
    mean_corner_fraction_ul: float = 0.0
    mean_corner_fraction_ur: float = 0.0
    mean_corner_fraction_dl: float = 0.0
    mean_corner_fraction_dr: float = 0.0
    preferred_corner: Optional[str] = None
    mean_max_corner_dwell_run: float = 0.0
    worst_max_corner_dwell_run: int = 0
    mean_preferred_quadrant_contains_start: float = 0.0
    mean_preferred_quadrant_contains_coin: float = 0.0
    mean_preferred_corner_contains_start: float = 0.0
    mean_preferred_corner_contains_goal: float = 0.0
    mean_preferred_corner_contains_coin: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_id": self.grid_id,
            "grid_size": self.grid_size,
            "density": self.density,
            "instance_id": self.instance_id,
            "reasoning_effort": self.reasoning_effort,
            "transform_type": self.transform_type,
            "trajectory_category": self.trajectory_category,
            "start_to_coin_distance": self.start_to_coin_distance,
            "coin_to_goal_distance": self.coin_to_goal_distance,
            "start_to_goal_via_coin_distance": self.start_to_goal_via_coin_distance,
            "start_to_goal_distance": self.start_to_goal_distance,
            "coin_detour_distance": self.coin_detour_distance,
            "coin_reachable_from_start": self.coin_reachable_from_start,
            "goal_reachable_from_start": self.goal_reachable_from_start,
            "goal_reachable_from_coin": self.goal_reachable_from_coin,
            "num_trajectories": self.num_trajectories,
            "num_coin_collected": self.num_coin_collected,
            "num_goal_after_coin": self.num_goal_after_coin,
            "num_goal_only": self.num_goal_only,
            "coin_collected_rate": self.coin_collected_rate,
            "goal_after_coin_rate": self.goal_after_coin_rate,
            "goal_only_rate": self.goal_only_rate,
            "full_success_rate": self.full_success_rate,
            "mean_trajectory_length": self.mean_trajectory_length,
            "mean_action_accuracy": self.mean_action_accuracy,
            "mean_action_accuracy_phase1": self.mean_action_accuracy_phase1,
            "mean_action_accuracy_phase2": self.mean_action_accuracy_phase2,
            "spl": self.spl,
            "mean_entropy": self.mean_entropy,
            "mean_optimal_entropy": self.mean_optimal_entropy,
            "mean_jsd": self.mean_jsd,
            "mean_entropy_phase1": self.mean_entropy_phase1,
            "mean_optimal_entropy_phase1": self.mean_optimal_entropy_phase1,
            "mean_jsd_phase1": self.mean_jsd_phase1,
            "mean_entropy_phase2": self.mean_entropy_phase2,
            "mean_optimal_entropy_phase2": self.mean_optimal_entropy_phase2,
            "mean_jsd_phase2": self.mean_jsd_phase2,
            "ece": self.ece,
            "mean_step_accuracy": self.mean_step_accuracy,
            "total_steps": self.total_steps,
            "mean_actions_up": self.mean_actions_up,
            "mean_actions_down": self.mean_actions_down,
            "mean_actions_left": self.mean_actions_left,
            "mean_actions_right": self.mean_actions_right,
            "mean_steps_front": self.mean_steps_front,
            "mean_steps_left_turn": self.mean_steps_left_turn,
            "mean_steps_right_turn": self.mean_steps_right_turn,
            "mean_steps_back": self.mean_steps_back,
            "mean_cell_revisits": self.mean_cell_revisits,
            "mean_immediate_revisits": self.mean_immediate_revisits,
            "mean_coin_oscillations": self.mean_coin_oscillations,
            "mean_invalid_actions": self.mean_invalid_actions,
            "mean_quadrant_fraction_ul": self.mean_quadrant_fraction_ul,
            "mean_quadrant_fraction_ur": self.mean_quadrant_fraction_ur,
            "mean_quadrant_fraction_dl": self.mean_quadrant_fraction_dl,
            "mean_quadrant_fraction_dr": self.mean_quadrant_fraction_dr,
            "preferred_quadrant": self.preferred_quadrant,
            "mean_quadrant_entropy": self.mean_quadrant_entropy,
            "mean_corner_fraction_ul": self.mean_corner_fraction_ul,
            "mean_corner_fraction_ur": self.mean_corner_fraction_ur,
            "mean_corner_fraction_dl": self.mean_corner_fraction_dl,
            "mean_corner_fraction_dr": self.mean_corner_fraction_dr,
            "preferred_corner": self.preferred_corner,
            "mean_max_corner_dwell_run": self.mean_max_corner_dwell_run,
            "worst_max_corner_dwell_run": self.worst_max_corner_dwell_run,
            "mean_preferred_quadrant_contains_start": self.mean_preferred_quadrant_contains_start,
            "mean_preferred_quadrant_contains_coin": self.mean_preferred_quadrant_contains_coin,
            "mean_preferred_corner_contains_start": self.mean_preferred_corner_contains_start,
            "mean_preferred_corner_contains_goal": self.mean_preferred_corner_contains_goal,
            "mean_preferred_corner_contains_coin": self.mean_preferred_corner_contains_coin,
        }


@dataclass
class CoinModelTrajectoryResults:
    """Results for a single model across all grids."""

    model_name: str
    df: pd.DataFrame
    state_df: pd.DataFrame
    summary_by_size_density: pd.DataFrame
    summary_by_distance: pd.DataFrame
    overall_summary: dict[str, Any]
    per_trajectory_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# =============================================================================
# Filename Parsing
# =============================================================================


def parse_coin_trajectory_filename(filename: str) -> Optional[dict[str, Any]]:
    """
    Parse coin trajectory filename to extract metadata.
    Expected format: {model}_size{N}_comp{X.X}_grid{N}_coin_{effort}_traj{N}.json
    """
    pattern = r"(.+)_size(\d+)_comp([\d.]+)_grid(\d+)_coin_(\w+)_traj(\d+)\.json"
    match = re.match(pattern, filename)
    if not match:
        return None

    model, size, comp, grid_id, effort_field, traj_id = match.groups()

    # Splitting iso_transform from parsed effort 
    if effort_field in KNOWN_EFFORTS:
        transform_type = "base"
        reasoning_effort = effort_field
    else:
        parts = effort_field.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in KNOWN_EFFORTS:
            transform_type, reasoning_effort = parts
        else:
            transform_type = effort_field
            reasoning_effort = "unknown"

    return {
        "model": model,
        "grid_size": int(size),
        "density": float(comp),
        "grid_id": int(grid_id),
        "transform_type": transform_type,
        "reasoning_effort": reasoning_effort,
        "trajectory_id": int(traj_id),
    }


def discover_coin_trajectory_files(
    trajectory_dir: Path,
) -> dict[str, list[Path]]:
    """Discover coin trajectory files grouped by (grid_key, effort).

    Recurses into subdirectories (e.g. control/, augmentations/, random_starts/,
    reshuffled/), so a parent dir containing several variants — or several
    sizes — is combined into one analysis.

    Returns:
        Dict mapping "{size}_{comp}_{grid_id}_{transform}_{effort}" to list of
        trajectory paths.
    """
    grouped: dict[str, list[Path]] = defaultdict(list)

    for filepath in sorted(trajectory_dir.rglob("*_coin_*_traj*.json")):
        parsed = parse_coin_trajectory_filename(filepath.name)
        if parsed:
            key = (
                f"size{parsed['grid_size']}_"
                f"comp{parsed['density']}_"
                f"grid{parsed['grid_id']}_"
                f"{parsed['transform_type']}_"
                f"{parsed['reasoning_effort']}"
            )
            grouped[key].append(filepath)

    return dict(grouped)


# =============================================================================
# Grid Layout Loading
# =============================================================================


def load_coin_grid_layout(
    layout_file: Path,
) -> Optional[tuple[list[list[str]], tuple[int, int], tuple[int, int]]]:
    """Load the coin grid layout file.

    Returns:
        (grid_layout, coin_pos, goal_pos) or None on failure.
        coin_pos and goal_pos are (x, y) tuples.
    """
    try:
        with open(layout_file, "r") as f:
            data = json.load(f)

        grid_layout = data.get("grid_layout")
        raw_coin = data.get("coin_pos")
        raw_goal = data.get("goal_pos")

        if grid_layout is None or raw_coin is None or raw_goal is None:
            return None

        coin_pos = (int(raw_coin[0]), int(raw_coin[1]))
        goal_pos = (int(raw_goal[0]), int(raw_goal[1]))
        return grid_layout, coin_pos, goal_pos

    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        return None


def find_coin_layout_file(traj_file: Path, grid_id: int) -> Optional[Path]:
    """Locate the *_coin_layout.json for a given trajectory file."""
    parsed = parse_coin_trajectory_filename(traj_file.name)
    if parsed is None:
        return None

    transform = parsed["transform_type"]
    if transform == "base":
        layout_name = (
            f"{parsed['model']}_size{parsed['grid_size']}_"
            f"comp{parsed['density']}_grid{grid_id}_coin_layout.json"
        )
    else:
        layout_name = (
            f"{parsed['model']}_size{parsed['grid_size']}_"
            f"comp{parsed['density']}_grid{grid_id}_coin_{transform}_layout.json"
        )
    candidate = traj_file.parent / layout_name
    return candidate if candidate.exists() else None


# =============================================================================
# Coin Detection
# =============================================================================


def is_coin_present_at(grid_state: list[str], coin_pos: tuple[int, int]) -> bool:
    """Return True if the coin symbol 'C' is at coin_pos in the grid state text.

    Grid state format (rows are y, parts[0] is row number, rest are column symbols):
        ['  0 1 2 ...', '0 # _ _ ...', '1 # A C ...', ...]
    coin_pos is (x, y).
    """
    cx, cy = coin_pos
    row_idx = cy + 1  # skip header row
    if row_idx >= len(grid_state):
        return False
    parts = grid_state[row_idx].split()
    col_idx = cx + 1  # skip leading row-number
    if col_idx >= len(parts):
        return False
    return parts[col_idx] == COIN_SYMBOL


def find_coin_collection_step(
    raw_steps: list[dict[str, Any]],
    coin_pos: tuple[int, int],
) -> Optional[int]:
    """Return the step index at which the coin disappears from the grid, or None."""
    coin_was_present = None

    for i, step in enumerate(raw_steps):
        grid_state = step.get("grid_state", [])
        present = is_coin_present_at(grid_state, coin_pos)

        if coin_was_present is None:
            coin_was_present = present

        if coin_was_present and not present:
            # Coin was there, now gone — collected at step i-1 (the step that moved onto it)
            return max(0, i - 1)

    return None  # coin never collected


# =============================================================================
# Trajectory Loading
# =============================================================================


def load_coin_trajectory(
    filepath: Path,
    coin_pos: tuple[int, int],
    goal_pos: tuple[int, int],
    start_to_coin_distance: int,
    coin_to_goal_distance: int,
    start_to_goal_distance: int = 0,
    coin_detour_distance: int = 0,
    trajectory_category: str = "base",
    coin_reachable_from_start: bool = True,
    goal_reachable_from_start: bool = True,
    goal_reachable_from_coin: bool = True,
    grid_layout: Optional[list[list[str]]] = None,
) -> Optional[CoinLightweightTrajectory]:
    """Load a coin trajectory file and detect coin collection from grid states."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        gp = data.get("grid_params", {})
        start_coords = gp.get("agent_start_coordinates", [0, 0])
        agent_start = (int(start_coords[1]), int(start_coords[0]))

        parsed = parse_coin_trajectory_filename(filepath.name)
        grid_id = parsed["grid_id"] if parsed else 0
        effort = parsed["reasoning_effort"] if parsed else "unknown"
        transform_type = parsed["transform_type"] if parsed else "base"
        trajectory_id = parsed["trajectory_id"] if parsed else 0

        # If either leg is unreachable (-1), the via-coin total is unreachable too.
        via_coin_distance = (
            start_to_coin_distance + coin_to_goal_distance
            if start_to_coin_distance >= 0 and coin_to_goal_distance >= 0
            else -1
        )

        grid_params = CoinTrajectoryGridParams(
            grid_size=gp.get("grid_width", 0),
            complexity=gp.get("grid_complexity", 0.0),
            grid_id=grid_id,
            astar_distance=via_coin_distance,
            agent_start=agent_start,
            goal=goal_pos,
            coin_pos=coin_pos,
            start_to_coin_distance=start_to_coin_distance,
            coin_to_goal_distance=coin_to_goal_distance,
            start_to_goal_via_coin_distance=via_coin_distance,
            start_to_goal_distance=start_to_goal_distance,
            coin_detour_distance=coin_detour_distance,
            coin_reachable_from_start=coin_reachable_from_start,
            goal_reachable_from_start=goal_reachable_from_start,
            goal_reachable_from_coin=goal_reachable_from_coin,
            grid_layout=grid_layout,
        )

        raw_steps = data.get("steps", [])

        # Detect coin collection step from grid states
        coin_collection_step = find_coin_collection_step(raw_steps, coin_pos)
        coin_collected = coin_collection_step is not None

        # Build lightweight steps
        steps = []
        for i, step in enumerate(raw_steps):
            grid_state = step.get("grid_state", [])
            agent_pos = extract_agent_position_from_grid_state(grid_state)
            agent_action = step.get("agent_action", "")
            steps.append(TrajectoryStep(step_id=i, agent_position=agent_pos, agent_action=agent_action))

        # Detect goal reached: apply last action to last position
        reached_goal = False
        if steps:
            pos = steps[-1].agent_position
            last_action = steps[-1].agent_action.upper()
            dx, dy = _ACTION_DELTAS.get(last_action, (0, 0))
            final_pos = (pos[0] + dx, pos[1] + dy)
            reached_goal = (final_pos == goal_pos)

        return CoinLightweightTrajectory(
            grid_params=grid_params,
            steps=steps,
            reached_goal=reached_goal,
            coin_collected=coin_collected,
            coin_collection_step=coin_collection_step,
            reasoning_effort=effort,
            transform_type=transform_type,
            trajectory_category=trajectory_category,
            trajectory_id=trajectory_id,
        )

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Warning: Error loading {filepath.name}: {e}")
        return None


# =============================================================================
# Two-Phase Optimal Actions
# =============================================================================


def compute_two_phase_optimal_actions(
    grid_layout: list[list[str]],
    coin_pos: tuple[int, int],
    goal_pos: tuple[int, int],
) -> tuple[
    dict[tuple[int, int], OptimalActionSet],
    dict[tuple[int, int], int],
    dict[tuple[int, int], OptimalActionSet],
    dict[tuple[int, int], int],
]:
    """Compute optimal actions for both phases via backward Dijkstra.

    Returns:
        (opt_to_coin, dist_to_coin, opt_to_goal, dist_to_goal)
    """
    # Phase 1: navigate to coin. Treat coin cell as passable (it is, just a ball).
    opt_to_coin, dist_to_coin = compute_optimal_actions_from_text_grid(grid_layout, coin_pos)

    # Phase 2: navigate from coin to goal. Use same layout (coin cell now empty, still passable).
    opt_to_goal, dist_to_goal = compute_optimal_actions_from_text_grid(grid_layout, goal_pos)

    return opt_to_coin, dist_to_coin, opt_to_goal, dist_to_goal


def get_step_optimal_actions(
    step: TrajectoryStep,
    traj: CoinLightweightTrajectory,
    opt_to_coin: dict[tuple[int, int], OptimalActionSet],
    opt_to_goal: dict[tuple[int, int], OptimalActionSet],
) -> OptimalActionSet:
    """Return the correct optimal action set for a step given which phase we're in."""
    in_phase2 = (
        traj.coin_collection_step is not None
        and step.step_id > traj.coin_collection_step
    )
    return opt_to_goal.get(step.agent_position, set()) if in_phase2 else opt_to_coin.get(step.agent_position, set())


def compute_coin_detour_distance(
    grid_layout: list[list[str]],
    agent_start: tuple[int, int],
    dist_to_coin: dict[tuple[int, int], int],
    dist_to_goal: dict[tuple[int, int], int],
) -> int:
    """Min distance from the coin to any cell on an optimal start→goal path.

    A cell c lies on an optimal path iff
        dist_from_start[c] + dist_to_goal[c] == dist_to_goal[agent_start].
    Returns -1 if the goal is unreachable from start.
    """
    start_to_goal = dist_to_goal.get(agent_start, -1)
    if start_to_goal < 0:
        return -1

    # Backward Dijkstra from agent_start gives dist_from_start (undirected grid).
    _, dist_from_start = compute_optimal_actions_from_text_grid(grid_layout, agent_start)

    min_detour = float("inf")
    for cell, d_from_start in dist_from_start.items():
        d_to_goal = dist_to_goal.get(cell, float("inf"))
        if d_from_start + d_to_goal == start_to_goal:
            d_to_coin = dist_to_coin.get(cell, float("inf"))
            if d_to_coin < min_detour:
                min_detour = d_to_coin

    return int(min_detour) if min_detour != float("inf") else -1


# =============================================================================
# Metrics Computation
# =============================================================================


_OPPOSITE  = {"UP": "DOWN", "DOWN": "UP",   "LEFT": "RIGHT", "RIGHT": "LEFT"}
_LEFT_TURN = {"UP": "LEFT", "DOWN": "RIGHT", "LEFT": "DOWN",  "RIGHT": "UP"}


def is_invalid_action(
    pos: tuple[int, int],
    action: str,
    grid_layout: list[list[str]],
) -> bool:
    """True if taking `action` from `pos` runs into a wall or out of bounds.

    Unrecognised action strings are not wall bumps (they are already penalised
    in action accuracy).
    """
    delta = _ACTION_DELTAS.get(action)
    if delta is None:
        return False
    tx, ty = pos[0] + delta[0], pos[1] + delta[1]
    if ty < 0 or ty >= len(grid_layout) or tx < 0 or tx >= len(grid_layout[ty]):
        return True
    return grid_layout[ty][tx] == WALL_SYMBOL


def compute_trajectory_movement_stats(
    steps: list[TrajectoryStep],
    grid_layout: Optional[list[list[str]]],
    coin_positions: set[tuple[int, int]],
) -> dict[str, int]:
    """Absolute action counts, relative direction counts, and revisit counts.

    Relative directions are computed from the perspective of the agent's heading
    (direction of the previous step).

    Revisit counts:
    - num_cell_revisits:     steps landing on any previously visited cell
    - num_immediate_revisits: steps returning to the immediately preceding cell (double-back)
    - num_coin_oscillations: steps oscillating through a coin cell — either arriving at a
                             coin cell (previously visited) or departing a coin cell to a
                             previously-visited adjacent cell
    - num_invalid_actions:   actions that run into a wall / out of bounds, leaving the agent
                             in the same cell (checked against the grid layout; falls back to
                             same-cell comparison when no layout is available)
    """
    action_counts: dict[str, int] = {"UP": 0, "DOWN": 0, "LEFT": 0, "RIGHT": 0}
    rel_counts: dict[str, int] = {"front": 0, "left_turn": 0, "right_turn": 0, "back": 0}

    visited: set[tuple[int, int]] = set()
    num_cell_revisits = 0
    num_immediate_revisits = 0
    num_coin_oscillations = 0
    num_invalid_actions = 0

    for i, step in enumerate(steps):
        action = step.agent_action.upper()
        pos = step.agent_position

        if action in action_counts:
            action_counts[action] += 1

        if grid_layout is not None:
            if is_invalid_action(pos, action, grid_layout):
                num_invalid_actions += 1
        elif i + 1 < len(steps) and steps[i + 1].agent_position == pos:
            # No layout: a blocked move shows as the next observation having the
            # same agent position. The final step cannot be classified this way.
            num_invalid_actions += 1

        if i > 0:
            prev_pos = steps[i - 1].agent_position
            prev_action = steps[i - 1].agent_action.upper()

            # Relative direction
            if prev_action in action_counts and action in action_counts:
                if action == prev_action:
                    rel_counts["front"] += 1
                elif action == _OPPOSITE[prev_action]:
                    rel_counts["back"] += 1
                elif action == _LEFT_TURN[prev_action]:
                    rel_counts["left_turn"] += 1
                else:
                    rel_counts["right_turn"] += 1

            # Double-back: returned to the cell occupied two steps ago, having actually
            # moved away in between (excludes consecutive blocked/no-op moves)
            if i >= 2 and pos == steps[i - 2].agent_position and pos != prev_pos:
                num_immediate_revisits += 1

            # Coin oscillation: move to/from a coin cell landing on a previously visited cell
            if (pos in coin_positions and pos in visited) or (
                prev_pos in coin_positions and pos in visited
            ):
                num_coin_oscillations += 1

        # General cell revisit
        if pos in visited:
            num_cell_revisits += 1
        visited.add(pos)

    return {
        "num_actions_up":           action_counts["UP"],
        "num_actions_down":         action_counts["DOWN"],
        "num_actions_left":         action_counts["LEFT"],
        "num_actions_right":        action_counts["RIGHT"],
        "num_steps_front":          rel_counts["front"],
        "num_steps_left_turn":      rel_counts["left_turn"],
        "num_steps_right_turn":     rel_counts["right_turn"],
        "num_steps_back":           rel_counts["back"],
        "num_cell_revisits":        num_cell_revisits,
        "num_immediate_revisits":   num_immediate_revisits,
        "num_coin_oscillations":    num_coin_oscillations,
        "num_invalid_actions":      num_invalid_actions,
    }


# =============================================================================
# Spatial / Corner-Dwelling Metrics
# =============================================================================


_QUADRANT_ORDER = ["ul", "ur", "dl", "dr"]


def corner_radius(grid_size: int) -> int:
    """Corner-region side length, scaled to ~20-25% of the interior span.

    interior_size = grid_size - 2 (interior spans 1..grid_size-2). Targets
    round(0.25 * interior_size), floored to >=1, capped at interior_size // 2
    so the four corner boxes can never overlap — this guarantees at least
    one empty cell between any two corners sharing an edge, so a single
    action (a Manhattan step of 1) can never jump directly from one corner
    region into another. Returns 0 for degenerate grids too small to fit
    non-overlapping corners (interior_size < 2).
    """
    interior_size = grid_size - 2
    if interior_size < 2:
        return 0
    r = max(1, round(0.25 * interior_size))
    return min(r, interior_size // 2)


def classify_quadrant(pos: tuple[int, int], grid_size: int) -> Optional[str]:
    """ul/ur/dl/dr, split at the interior midpoint (grid_size-1)/2, or None
    if pos falls exactly on the middle row or middle column.

    mid is always an exact integer (interior_size is odd for every grid
    size in this dataset), so there is a genuine middle row (y == mid) and
    middle column (x == mid). Positions on either are excluded from all
    four quadrants rather than tie-broken into one, since they aren't
    unambiguously in any quadrant.
    """
    x, y = pos
    mid = (grid_size - 1) / 2
    if x == mid or y == mid:
        return None
    vertical = "u" if y < mid else "d"
    horizontal = "l" if x < mid else "r"
    return vertical + horizontal


def classify_corner(pos: tuple[int, int], grid_size: int) -> Optional[str]:
    """ul/ur/dl/dr if pos falls in that corner's corner_radius(grid_size)
    box, else None."""
    r = corner_radius(grid_size)
    if r <= 0:
        return None
    x, y = pos
    lo, hi = 1, grid_size - 2
    near_top = y <= lo + r - 1
    near_bottom = y >= hi - r + 1
    near_left = x <= lo + r - 1
    near_right = x >= hi - r + 1

    if near_top and near_left:
        return "ul"
    if near_top and near_right:
        return "ur"
    if near_bottom and near_left:
        return "dl"
    if near_bottom and near_right:
        return "dr"
    return None


def quadrant_matches_preferred(pos: tuple[int, int], grid_size: int, preferred_quadrant: str) -> bool:
    """True if pos's quadrant equals preferred_quadrant.

    preferred_quadrant is stored uppercase (e.g. "UL") while
    classify_quadrant returns lowercase (e.g. "ul") — this does the
    case-insensitive comparison correctly. A position on the midline
    (classify_quadrant returns None) never matches."""
    q = classify_quadrant(pos, grid_size)
    return q is not None and q.upper() == preferred_quadrant


def quadrant_label(pos: tuple[int, int], grid_size: int) -> str:
    """Quadrant of pos as a lowercase label (ul/ur/dl/dr), or "midline" if
    pos falls on the exact middle row or middle column (classify_quadrant
    returns None for those)."""
    q = classify_quadrant(pos, grid_size)
    return q if q is not None else "midline"


def compute_spatial_stats(
    positions: list[tuple[int, int]],
    grid_size: int,
) -> dict[str, Any]:
    """Quadrant/corner occupancy, preferred region, quadrant entropy, and
    longest any-corner dwell run for ONE trajectory's own ordered position
    list.

    Positions on the exact middle row or middle column are excluded from
    all four quadrants (classify_quadrant returns None for them) rather
    than tie-broken into one — quadrant_count_ul/ur/dl/dr therefore do not
    sum to len(positions); the gap is quadrant_count_midline.
    quadrant_fraction_ul/ur/dl/dr are computed over classified (non-midline)
    positions only, so they still sum to 1.0 and quadrant_entropy (which
    assumes a proper distribution) stays valid; quadrant_fraction_midline
    is computed over the full trajectory instead, since it isn't part of
    that same partition.

    Always call this per-trajectory — never on a pooled multi-trajectory
    list. max_corner_dwell_run is order-dependent; concatenating separate
    trajectories' positions would splice fake dwell runs across trajectory
    boundaries. Grid-level statistics are obtained by averaging this
    function's per-trajectory outputs, not by pooling positions first (see
    compute_coin_grid_metrics).
    """
    n = len(positions)
    quadrant_counts = {q: 0 for q in _QUADRANT_ORDER}
    corner_counts = {q: 0 for q in _QUADRANT_ORDER}
    midline_count = 0

    max_dwell = 0
    current_dwell = 0

    for pos in positions:
        quadrant = classify_quadrant(pos, grid_size)
        if quadrant is None:
            midline_count += 1
        else:
            quadrant_counts[quadrant] += 1
        corner = classify_corner(pos, grid_size)
        if corner is not None:
            corner_counts[corner] += 1
            current_dwell += 1
            max_dwell = max(max_dwell, current_dwell)
        else:
            current_dwell = 0

    n_classified = n - midline_count
    quadrant_fractions = {
        q: (quadrant_counts[q] / n_classified if n_classified > 0 else 0.0) for q in _QUADRANT_ORDER
    }
    corner_fractions = {
        q: (corner_counts[q] / n if n > 0 else 0.0) for q in _QUADRANT_ORDER
    }

    preferred_quadrant = max(quadrant_fractions, key=quadrant_fractions.get).upper()
    preferred_corner = (
        max(corner_fractions, key=corner_fractions.get).upper()
        if max(corner_fractions.values()) > 0
        else None
    )
    quadrant_entropy = shannon_entropy({i: f for i, f in enumerate(quadrant_fractions.values())})

    return {
        "preferred_quadrant": preferred_quadrant,
        "quadrant_entropy": quadrant_entropy,
        "quadrant_count_ul": quadrant_counts["ul"],
        "quadrant_count_ur": quadrant_counts["ur"],
        "quadrant_count_dl": quadrant_counts["dl"],
        "quadrant_count_dr": quadrant_counts["dr"],
        "quadrant_count_midline": midline_count,
        "quadrant_fraction_ul": quadrant_fractions["ul"],
        "quadrant_fraction_ur": quadrant_fractions["ur"],
        "quadrant_fraction_dl": quadrant_fractions["dl"],
        "quadrant_fraction_dr": quadrant_fractions["dr"],
        "quadrant_fraction_midline": (midline_count / n if n > 0 else 0.0),
        "preferred_corner": preferred_corner,
        "corner_fraction_ul": corner_fractions["ul"],
        "corner_fraction_ur": corner_fractions["ur"],
        "corner_fraction_dl": corner_fractions["dl"],
        "corner_fraction_dr": corner_fractions["dr"],
        "max_corner_dwell_run": max_dwell,
    }


def compute_phase_accuracy(
    trajectories: list[CoinLightweightTrajectory],
    opt_to_coin: dict[tuple[int, int], OptimalActionSet],
    opt_to_goal: dict[tuple[int, int], OptimalActionSet],
) -> tuple[float, float, float]:
    """Compute action accuracy for phase 1, phase 2, and combined.

    Returns:
        (accuracy_phase1, accuracy_phase2, accuracy_combined)
    """
    p1_correct, p1_total = 0, 0
    p2_correct, p2_total = 0, 0

    for traj in trajectories:
        for step in traj.steps:
            action_id = ACTION_NAME_TO_ID.get(step.agent_action.upper())
            in_phase2 = (
                traj.coin_collection_step is not None
                and step.step_id > traj.coin_collection_step
            )
            optimal_set = (
                opt_to_goal.get(step.agent_position, set())
                if in_phase2
                else opt_to_coin.get(step.agent_position, set())
            )
            is_correct = int(action_id is not None and action_id in optimal_set)

            if in_phase2:
                p2_correct += is_correct
                p2_total += 1
            else:
                p1_correct += is_correct
                p1_total += 1

    acc1 = p1_correct / p1_total if p1_total > 0 else 0.0
    acc2 = p2_correct / p2_total if p2_total > 0 else 0.0
    total = p1_total + p2_total
    combined = (p1_correct + p2_correct) / total if total > 0 else 0.0
    return acc1, acc2, combined


def build_phase_pools(
    trajectories: list[CoinLightweightTrajectory],
) -> tuple[StateActionCounts, StateActionCounts]:
    """Split every step into its phase-1 (pre-coin) / phase-2 (post-coin) pool.

    Uses the same in_phase2 rule as get_step_optimal_actions/compute_phase_accuracy.

    Returns:
        (counts_p1, counts_p2)
    """
    counts_p1: StateActionCounts = StateActionCounts()
    counts_p2: StateActionCounts = StateActionCounts()

    for traj in trajectories:
        for step in traj.steps:
            in_phase2 = (
                traj.coin_collection_step is not None
                and step.step_id > traj.coin_collection_step
            )
            if in_phase2:
                counts_p2.add(step.agent_position, step.agent_action)
            else:
                counts_p1.add(step.agent_position, step.agent_action)

    return counts_p1, counts_p2


def compute_phase_uncertainty(
    counts_p1: StateActionCounts,
    opt_to_coin: dict[tuple[int, int], OptimalActionSet],
    counts_p2: StateActionCounts,
    opt_to_goal: dict[tuple[int, int], OptimalActionSet],
) -> tuple[float, float, float, float, float, float]:
    """Compute mean entropy, optimal entropy, and JSD split by phase.

    Returns:
        (mean_entropy_p1, mean_optimal_entropy_p1, mean_jsd_p1,
         mean_entropy_p2, mean_optimal_entropy_p2, mean_jsd_p2)
    """
    ent1, opt_ent1, jsd1 = compute_empirical_uncertainty_metrics(counts_p1, opt_to_coin)
    ent2, opt_ent2, jsd2 = compute_empirical_uncertainty_metrics(counts_p2, opt_to_goal)
    return ent1, opt_ent1, jsd1, ent2, opt_ent2, jsd2


def compute_phase_aware_combined_uncertainty(
    counts_p1: StateActionCounts,
    opt_to_coin: dict[tuple[int, int], OptimalActionSet],
    counts_p2: StateActionCounts,
    opt_to_goal: dict[tuple[int, int], OptimalActionSet],
) -> tuple[float, float, float]:
    """Phase-correct combined entropy / optimal-entropy / JSD.

    Each phase's visited states are scored against its OWN optimal target,
    and only the resulting per-state VALUES are concatenated and averaged —
    never raw action counts (would blend phase-1/phase-2 choices at shared
    cells) and never the optimal-target dicts (would collapse to
    opt_to_goal, since both cover every reachable cell). Equivalent to
    weighting mean_entropy_phase1/phase2 by the number of distinct states
    visited in each phase.

    Returns:
        (mean_entropy, mean_optimal_entropy, mean_jsd)
    """
    ent1, opt_ent1, jsd1 = collect_uncertainty_values(counts_p1, opt_to_coin)
    ent2, opt_ent2, jsd2 = collect_uncertainty_values(counts_p2, opt_to_goal)

    entropies = ent1 + ent2
    optimal_entropies = opt_ent1 + opt_ent2
    jsds = jsd1 + jsd2

    mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    mean_optimal_entropy = (
        sum(optimal_entropies) / len(optimal_entropies) if optimal_entropies else 0.0
    )
    mean_jsd = sum(jsds) / len(jsds) if jsds else 0.0

    return mean_entropy, mean_optimal_entropy, mean_jsd


def compute_phase_aware_combined_ece(
    counts_p1: StateActionCounts,
    opt_to_coin: dict[tuple[int, int], OptimalActionSet],
    counts_p2: StateActionCounts,
    opt_to_goal: dict[tuple[int, int], OptimalActionSet],
    n_bins: int = 10,
) -> float:
    """Phase-correct combined ECE — see compute_phase_aware_combined_uncertainty
    for why pooling happens at the (confidence, accuracy) value level."""
    conf1, acc1 = collect_ece_values(counts_p1, opt_to_coin)
    conf2, acc2 = collect_ece_values(counts_p2, opt_to_goal)
    return bin_and_score_ece(conf1 + conf2, acc1 + acc2, n_bins)


def compute_coin_spl(
    trajectories: list[CoinLightweightTrajectory],
    optimal_total_distance: int,
) -> float:
    """SPL using full_success (coin + goal) and total optimal path length."""
    if not trajectories or optimal_total_distance <= 0:
        return 0.0

    spls = []
    for traj in trajectories:
        success = traj.reached_goal and traj.coin_collected
        L = traj.trajectory_length
        L_star = optimal_total_distance
        spls.append((1 if success else 0) * L_star / max(L_star, L))

    return sum(spls) / len(spls)


def compute_single_trajectory_row(
    traj: CoinLightweightTrajectory,
    opt_to_coin: dict[tuple[int, int], OptimalActionSet],
    opt_to_goal: dict[tuple[int, int], OptimalActionSet],
    grid_key: str,
) -> dict[str, Any]:
    """Build one CSV row for a single trajectory.

    Entropy/JSD/ECE are omitted — they require pooled empirical distributions
    and are computed at grid level only.
    """
    gp = traj.grid_params

    # Phase-aware action accuracy
    p1_correct, p1_total = 0, 0
    p2_correct, p2_total = 0, 0
    for step in traj.steps:
        action_id = ACTION_NAME_TO_ID.get(step.agent_action.upper())
        in_phase2 = (
            traj.coin_collection_step is not None
            and step.step_id > traj.coin_collection_step
        )
        optimal_set = (
            opt_to_goal.get(step.agent_position, set())
            if in_phase2
            else opt_to_coin.get(step.agent_position, set())
        )
        is_correct = int(action_id is not None and action_id in optimal_set)
        if in_phase2:
            p2_correct += is_correct
            p2_total += 1
        else:
            p1_correct += is_correct
            p1_total += 1

    acc1 = p1_correct / p1_total if p1_total > 0 else 0.0
    acc2 = p2_correct / p2_total if p2_total > 0 else 0.0
    total_steps = p1_total + p2_total
    acc_combined = (p1_correct + p2_correct) / total_steps if total_steps > 0 else 0.0

    # Per-trajectory SPL: int(success) * L* / max(L*, L)
    L_star = gp.start_to_goal_via_coin_distance
    L = traj.trajectory_length
    success = traj.reached_goal and traj.coin_collected
    spl = (L_star / max(L_star, L)) if (success and L_star > 0) else 0.0

    move = compute_trajectory_movement_stats(
        traj.steps, traj.grid_params.grid_layout, {traj.grid_params.coin_pos}
    )

    positions = [step.agent_position for step in traj.steps]
    spatial = compute_spatial_stats(positions, gp.grid_size)
    preferred_corner_flag = spatial["preferred_corner"] is not None

    return {
        "trajectory_id":          traj.trajectory_id,
        "grid_id":                gp.grid_id,
        "grid_key":               grid_key,
        "grid_size":              gp.grid_size,
        "density":                gp.complexity,
        "reasoning_effort":       traj.reasoning_effort,
        "transform_type":         traj.transform_type,
        "trajectory_category":    traj.trajectory_category,
        "start_to_coin_distance":    gp.start_to_coin_distance,
        "coin_to_goal_distance":    gp.coin_to_goal_distance,
        "start_to_goal_via_coin_distance":   gp.start_to_goal_via_coin_distance,
        "start_to_goal_distance": gp.start_to_goal_distance,
        "coin_detour_distance":   gp.coin_detour_distance,
        "coin_reachable_from_start": int(gp.coin_reachable_from_start),
        "goal_reachable_from_start": int(gp.goal_reachable_from_start),
        "goal_reachable_from_coin":  int(gp.goal_reachable_from_coin),
        "reached_goal":           int(traj.reached_goal),
        "coin_collected":         int(traj.coin_collected),
        "coin_collection_step":   traj.coin_collection_step if traj.coin_collection_step is not None else -1,
        "trajectory_length":      L,
        "spl":                    spl,
        "action_accuracy":        acc_combined,
        "action_accuracy_phase1": acc1,
        "action_accuracy_phase2": acc2,
        **move,
        "invalid_action_rate":    move["num_invalid_actions"] / L if L > 0 else 0.0,
        "preferred_quadrant":     spatial["preferred_quadrant"],
        "quadrant_entropy":       spatial["quadrant_entropy"],
        "quadrant_count_ul":      spatial["quadrant_count_ul"],
        "quadrant_count_ur":      spatial["quadrant_count_ur"],
        "quadrant_count_dl":      spatial["quadrant_count_dl"],
        "quadrant_count_dr":      spatial["quadrant_count_dr"],
        "quadrant_count_midline": spatial["quadrant_count_midline"],
        "quadrant_fraction_ul":   spatial["quadrant_fraction_ul"],
        "quadrant_fraction_ur":   spatial["quadrant_fraction_ur"],
        "quadrant_fraction_dl":   spatial["quadrant_fraction_dl"],
        "quadrant_fraction_dr":   spatial["quadrant_fraction_dr"],
        "quadrant_fraction_midline": spatial["quadrant_fraction_midline"],
        "preferred_corner":       spatial["preferred_corner"],
        "corner_fraction_ul":     spatial["corner_fraction_ul"],
        "corner_fraction_ur":     spatial["corner_fraction_ur"],
        "corner_fraction_dl":     spatial["corner_fraction_dl"],
        "corner_fraction_dr":     spatial["corner_fraction_dr"],
        "max_corner_dwell_run":   spatial["max_corner_dwell_run"],
        "start_quadrant":         quadrant_label(gp.agent_start, gp.grid_size),
        "terminal_quadrant":      quadrant_label(gp.goal, gp.grid_size),
        "coin_quadrant":          quadrant_label(gp.coin_pos, gp.grid_size),
        "preferred_quadrant_contains_start": int(quadrant_matches_preferred(gp.agent_start, gp.grid_size, spatial["preferred_quadrant"])),
        "preferred_quadrant_contains_coin":  int(quadrant_matches_preferred(gp.coin_pos, gp.grid_size, spatial["preferred_quadrant"])),
        "preferred_corner_contains_start":   int(preferred_corner_flag and classify_corner(gp.agent_start, gp.grid_size) == spatial["preferred_corner"]),
        "preferred_corner_contains_goal":    int(preferred_corner_flag and classify_corner(gp.goal, gp.grid_size) == spatial["preferred_corner"]),
        "preferred_corner_contains_coin":    int(preferred_corner_flag and classify_corner(gp.coin_pos, gp.grid_size) == spatial["preferred_corner"]),
    }


def compute_coin_grid_metrics(
    trajectories: list[CoinLightweightTrajectory],
    opt_to_coin: dict[tuple[int, int], OptimalActionSet],
    dist_to_coin: dict[tuple[int, int], int],
    opt_to_goal: dict[tuple[int, int], OptimalActionSet],
    dist_to_goal: dict[tuple[int, int], int],
    grid_key: str,
) -> tuple[Optional[CoinGridTrajectoryMetrics], list[dict[str, Any]]]:
    """Compute all metrics for a coin grid from its trajectories."""
    if not trajectories:
        return None, []

    pattern = re.match(r"size(\d+)_comp([\d.]+)_grid(\d+)_", grid_key)
    if not pattern:
        return None, []

    grid_size = int(pattern.group(1))
    density = float(pattern.group(2))
    instance_id = int(pattern.group(3))
    # Transform/effort come from the parsed filenames, not the grid_key regex —
    # transforms like RandomStart_4_2 contain underscores the regex can't split.
    transform_type = trajectories[0].transform_type
    effort = trajectories[0].reasoning_effort
    trajectory_category = trajectories[0].trajectory_category

    gp = trajectories[0].grid_params
    astar_coin = gp.start_to_coin_distance
    astar_goal = gp.coin_to_goal_distance
    astar_total = gp.start_to_goal_via_coin_distance
    start_to_goal = gp.start_to_goal_distance
    coin_detour = gp.coin_detour_distance

    # Success breakdown
    num_coin = sum(1 for t in trajectories if t.coin_collected)
    num_full = sum(1 for t in trajectories if t.coin_collected and t.reached_goal)
    num_goal_only = sum(1 for t in trajectories if not t.coin_collected and t.reached_goal)
    n = len(trajectories)

    # Capability
    mean_traj_len = sum(t.trajectory_length for t in trajectories) / n
    acc1, acc2, acc_combined = compute_phase_accuracy(trajectories, opt_to_coin, opt_to_goal)
    spl = compute_coin_spl(trajectories, astar_total)

    # Per-step accuracy — each step is scored against its own phase's optimal
    # set via get_step_optimal_actions, so this is already phase-correct.
    total_steps = 0
    total_correct = 0

    for traj in trajectories:
        for step in traj.steps:
            total_steps += 1
            optimal_set = get_step_optimal_actions(step, traj, opt_to_coin, opt_to_goal)
            action_id = ACTION_NAME_TO_ID.get(step.agent_action.upper())
            if action_id is not None and action_id in optimal_set:
                total_correct += 1

    mean_step_accuracy = total_correct / total_steps if total_steps > 0 else 0.0

    # Phase-split pools feed both the phase-specific and the phase-aware
    # combined uncertainty/ECE computations below. Pooling happens at the
    # level of already-correctly-scored per-state values (never by merging
    # raw action counts across phases, and never by merging opt_to_coin/
    # opt_to_goal into one dict — the latter collapses to opt_to_goal
    # entirely, since both cover every reachable cell of the grid).
    counts_p1, counts_p2 = build_phase_pools(trajectories)

    mean_ent, mean_opt_ent, mean_jsd = compute_phase_aware_combined_uncertainty(
        counts_p1, opt_to_coin, counts_p2, opt_to_goal
    )
    ece = compute_phase_aware_combined_ece(counts_p1, opt_to_coin, counts_p2, opt_to_goal)
    ent1, opt_ent1, jsd1, ent2, opt_ent2, jsd2 = compute_phase_uncertainty(
        counts_p1, opt_to_coin, counts_p2, opt_to_goal
    )

    # Movement stats — averaged across trajectories
    move_keys = [
        "num_actions_up", "num_actions_down", "num_actions_left", "num_actions_right",
        "num_steps_front", "num_steps_left_turn", "num_steps_right_turn", "num_steps_back",
        "num_cell_revisits", "num_immediate_revisits", "num_coin_oscillations",
        "num_invalid_actions",
    ]
    move_totals: dict[str, float] = {k: 0.0 for k in move_keys}
    for traj in trajectories:
        stats = compute_trajectory_movement_stats(
            traj.steps, traj.grid_params.grid_layout, {traj.grid_params.coin_pos}
        )
        for k, v in stats.items():
            move_totals[k] += v
    move_means = {k: v / n for k, v in move_totals.items()}

    # Spatial / corner-dwelling stats. Computed per trajectory, then averaged
    # — never by pooling positions across trajectories first (max_corner_dwell_run
    # is order-dependent; pooling would splice fake runs across trajectory
    # boundaries), so every one of these is aggregated the same way movement
    # stats already are (move_totals/move_means above), for consistency.
    quadrant_fraction_totals = {q: 0.0 for q in _QUADRANT_ORDER}
    corner_fraction_totals = {q: 0.0 for q in _QUADRANT_ORDER}
    quadrant_entropy_total = 0.0
    dwell_runs: list[int] = []
    confound_totals = {
        "preferred_quadrant_contains_start": 0,
        "preferred_quadrant_contains_coin": 0,
        "preferred_corner_contains_start": 0,
        "preferred_corner_contains_goal": 0,
        "preferred_corner_contains_coin": 0,
    }

    for traj in trajectories:
        spatial = compute_spatial_stats(
            [s.agent_position for s in traj.steps], traj.grid_params.grid_size
        )
        for q in _QUADRANT_ORDER:
            quadrant_fraction_totals[q] += spatial[f"quadrant_fraction_{q}"]
            corner_fraction_totals[q] += spatial[f"corner_fraction_{q}"]
        quadrant_entropy_total += spatial["quadrant_entropy"]
        dwell_runs.append(spatial["max_corner_dwell_run"])

        t_gp = traj.grid_params
        preferred_corner_flag = spatial["preferred_corner"] is not None
        confound_totals["preferred_quadrant_contains_start"] += int(
            quadrant_matches_preferred(t_gp.agent_start, t_gp.grid_size, spatial["preferred_quadrant"])
        )
        confound_totals["preferred_quadrant_contains_coin"] += int(
            quadrant_matches_preferred(t_gp.coin_pos, t_gp.grid_size, spatial["preferred_quadrant"])
        )
        confound_totals["preferred_corner_contains_start"] += int(
            preferred_corner_flag and classify_corner(t_gp.agent_start, t_gp.grid_size) == spatial["preferred_corner"]
        )
        confound_totals["preferred_corner_contains_goal"] += int(
            preferred_corner_flag and classify_corner(t_gp.goal, t_gp.grid_size) == spatial["preferred_corner"]
        )
        confound_totals["preferred_corner_contains_coin"] += int(
            preferred_corner_flag and classify_corner(t_gp.coin_pos, t_gp.grid_size) == spatial["preferred_corner"]
        )

    mean_quadrant_fraction = {q: v / n for q, v in quadrant_fraction_totals.items()}
    mean_corner_fraction = {q: v / n for q, v in corner_fraction_totals.items()}
    mean_quadrant_entropy = quadrant_entropy_total / n
    mean_max_corner_dwell_run = sum(dwell_runs) / n
    worst_max_corner_dwell_run = max(dwell_runs) if dwell_runs else 0
    mean_confound = {k: v / n for k, v in confound_totals.items()}

    # Grid-level preferred quadrant/corner = argmax of the mean fractions
    # (equally weights each trajectory, rather than each raw step).
    grid_preferred_quadrant = max(mean_quadrant_fraction, key=mean_quadrant_fraction.get).upper()
    grid_preferred_corner = (
        max(mean_corner_fraction, key=mean_corner_fraction.get).upper()
        if max(mean_corner_fraction.values()) > 0
        else None
    )

    # Per-state metrics for distance analysis (use dist_to_goal as reference
    # distance for both phases — this file studies uncertainty vs. distance
    # to the final goal specifically, regardless of phase). Each phase's
    # pool is scored against its own optimal target; a cell visited in both
    # phases correctly yields two rows (tagged by "phase") rather than one
    # row blended across phases and scored against the wrong target.
    state_metrics: list[dict[str, Any]] = []
    for phase, counts, optimal_map in (
        (1, counts_p1, opt_to_coin),
        (2, counts_p2, opt_to_goal),
    ):
        for pos, action_counts in counts.counts.items():
            total = sum(action_counts.values())
            if total == 0:
                continue

            empirical_dist = counts.get_empirical_distribution(pos)
            optimal_set = optimal_map.get(pos, set())
            distance = dist_to_goal.get(pos, -1)

            if distance < 0:
                continue

            entropy = shannon_entropy(empirical_dist)
            opt_ent = optimal_entropy(len(optimal_set)) if optimal_set else 0.0
            jsd = jensen_shannon_divergence(optimal_set, empirical_dist) if optimal_set else None
            most_likely = max(empirical_dist, key=lambda a: empirical_dist[a])
            is_optimal = 1 if most_likely in optimal_set else 0

            state_metrics.append({
                "grid_size": grid_size,
                "density": density,
                "reasoning_effort": effort,
                "distance_to_goal": distance,
                "phase": phase,
                "entropy": entropy,
                "optimal_entropy": opt_ent,
                "jsd": jsd,
                "is_optimal": is_optimal,
                "n_observations": total,
            })

    metrics = CoinGridTrajectoryMetrics(
        grid_id=grid_key,
        grid_size=grid_size,
        density=density,
        instance_id=instance_id,
        reasoning_effort=effort,
        transform_type=transform_type,
        trajectory_category=trajectory_category,
        start_to_coin_distance=astar_coin,
        coin_to_goal_distance=astar_goal,
        start_to_goal_via_coin_distance=astar_total,
        start_to_goal_distance=start_to_goal,
        coin_detour_distance=coin_detour,
        coin_reachable_from_start=int(gp.coin_reachable_from_start),
        goal_reachable_from_start=int(gp.goal_reachable_from_start),
        goal_reachable_from_coin=int(gp.goal_reachable_from_coin),
        num_trajectories=n,
        num_coin_collected=num_coin,
        num_goal_after_coin=num_full,
        num_goal_only=num_goal_only,
        coin_collected_rate=num_coin / n,
        goal_after_coin_rate=num_full / n,
        goal_only_rate=num_goal_only / n,
        full_success_rate=num_full / n,
        mean_trajectory_length=mean_traj_len,
        mean_action_accuracy=acc_combined,
        mean_action_accuracy_phase1=acc1,
        mean_action_accuracy_phase2=acc2,
        spl=spl,
        mean_entropy=mean_ent,
        mean_optimal_entropy=mean_opt_ent,
        mean_jsd=mean_jsd,
        mean_entropy_phase1=ent1,
        mean_optimal_entropy_phase1=opt_ent1,
        mean_jsd_phase1=jsd1,
        mean_entropy_phase2=ent2,
        mean_optimal_entropy_phase2=opt_ent2,
        mean_jsd_phase2=jsd2,
        ece=ece,
        mean_step_accuracy=mean_step_accuracy,
        total_steps=total_steps,
        mean_actions_up=move_means["num_actions_up"],
        mean_actions_down=move_means["num_actions_down"],
        mean_actions_left=move_means["num_actions_left"],
        mean_actions_right=move_means["num_actions_right"],
        mean_steps_front=move_means["num_steps_front"],
        mean_steps_left_turn=move_means["num_steps_left_turn"],
        mean_steps_right_turn=move_means["num_steps_right_turn"],
        mean_steps_back=move_means["num_steps_back"],
        mean_cell_revisits=move_means["num_cell_revisits"],
        mean_immediate_revisits=move_means["num_immediate_revisits"],
        mean_coin_oscillations=move_means["num_coin_oscillations"],
        mean_invalid_actions=move_means["num_invalid_actions"],
        mean_quadrant_fraction_ul=mean_quadrant_fraction["ul"],
        mean_quadrant_fraction_ur=mean_quadrant_fraction["ur"],
        mean_quadrant_fraction_dl=mean_quadrant_fraction["dl"],
        mean_quadrant_fraction_dr=mean_quadrant_fraction["dr"],
        preferred_quadrant=grid_preferred_quadrant,
        mean_quadrant_entropy=mean_quadrant_entropy,
        mean_corner_fraction_ul=mean_corner_fraction["ul"],
        mean_corner_fraction_ur=mean_corner_fraction["ur"],
        mean_corner_fraction_dl=mean_corner_fraction["dl"],
        mean_corner_fraction_dr=mean_corner_fraction["dr"],
        preferred_corner=grid_preferred_corner,
        mean_max_corner_dwell_run=mean_max_corner_dwell_run,
        worst_max_corner_dwell_run=worst_max_corner_dwell_run,
        mean_preferred_quadrant_contains_start=mean_confound["preferred_quadrant_contains_start"],
        mean_preferred_quadrant_contains_coin=mean_confound["preferred_quadrant_contains_coin"],
        mean_preferred_corner_contains_start=mean_confound["preferred_corner_contains_start"],
        mean_preferred_corner_contains_goal=mean_confound["preferred_corner_contains_goal"],
        mean_preferred_corner_contains_coin=mean_confound["preferred_corner_contains_coin"],
    )

    return metrics, state_metrics


# =============================================================================
# Main Processing Pipeline
# =============================================================================


def process_model_coin_trajectories(
    trajectory_dir: Path,
    model_name: Optional[str] = None,
    batch_size: int = 20,
) -> CoinModelTrajectoryResults:
    """Process all coin trajectories for a model.

    Trajectories are grouped by (grid_key, reasoning_effort). The coin and goal
    positions are loaded from the corresponding *_coin_layout.json file.
    """
    if model_name is None:
        model_name = sanitize_label(trajectory_dir.name)

    print(f"\nProcessing coin model: {model_name}")
    print(f"Trajectory directory: {trajectory_dir}")

    grouped = discover_coin_trajectory_files(trajectory_dir)
    print(f"Found {len(grouped)} (grid × effort) combinations")

    if not grouped:
        raise ValueError(f"No coin trajectory files found in {trajectory_dir}")

    all_metrics: list[CoinGridTrajectoryMetrics] = []
    all_state_metrics: list[dict[str, Any]] = []
    all_traj_rows: list[dict[str, Any]] = []

    grid_keys = sorted(grouped.keys())
    total_batches = (len(grid_keys) + batch_size - 1) // batch_size

    for batch_idx, batch_keys in enumerate(batch_grid_keys(grid_keys, batch_size)):
        print(f"\n  Batch {batch_idx + 1}/{total_batches}: {len(batch_keys)} grids...")

        for grid_key in tqdm(batch_keys, desc=f"Batch {batch_idx + 1}", leave=False):
            traj_files = grouped[grid_key]
            if not traj_files:
                continue

            # Find grid layout file
            parsed = parse_coin_trajectory_filename(traj_files[0].name)
            if parsed is None:
                continue

            grid_id = parsed["grid_id"]
            layout_file = find_coin_layout_file(traj_files[0], grid_id)

            if layout_file is None:
                print(f"  Warning: no layout file for {grid_key}, skipping.")
                continue

            layout_result = load_coin_grid_layout(layout_file)
            if layout_result is None:
                print(f"  Warning: could not parse layout for {grid_key}, skipping.")
                continue

            grid_layout, coin_pos, goal_pos = layout_result

            # Compute two-phase optimal actions
            opt_to_coin, dist_to_coin, opt_to_goal, dist_to_goal = (
                compute_two_phase_optimal_actions(grid_layout, coin_pos, goal_pos)
            )

            # A* distance from coin to goal (start→coin loaded per trajectory below).
            # -1 = unreachable (cell absent from the Dijkstra distance map).
            astar_goal_from_coin = dist_to_goal.get(coin_pos, -1)
            goal_reachable_from_coin = coin_pos in dist_to_goal

            trajectory_category = infer_trajectory_category(
                traj_files[0], parsed["transform_type"]
            )

            # Grid-level distances computed once from the first trajectory's agent_start
            # (all trajectories for a grid share the same start position).
            _grid_start_to_goal: int = 0
            _grid_coin_detour: int = 0
            _grid_distances_computed = False

            # Load trajectories
            trajectories: list[CoinLightweightTrajectory] = []
            for traj_file in traj_files:
                # Get agent start to compute proper A* distances
                try:
                    with open(traj_file, "r") as f:
                        raw_data = json.load(f)
                    gp_raw = raw_data.get("grid_params", {})
                    start_coords = gp_raw.get("agent_start_coordinates", [0, 0])
                    agent_start = (int(start_coords[1]), int(start_coords[0]))
                    astar_coin_dist = dist_to_coin.get(agent_start, -1)
                except Exception:
                    agent_start = (0, 0)
                    astar_coin_dist = -1

                coin_reachable_from_start = agent_start in dist_to_coin
                goal_reachable_from_start = agent_start in dist_to_goal
                if not (coin_reachable_from_start and goal_reachable_from_start and goal_reachable_from_coin):
                    print(
                        f"  Warning: unreachable target in {grid_key} "
                        f"(start={agent_start}, coin_reachable={coin_reachable_from_start}, "
                        f"goal_reachable={goal_reachable_from_start}, "
                        f"goal_from_coin={goal_reachable_from_coin})"
                    )

                if not _grid_distances_computed:
                    _grid_start_to_goal = dist_to_goal.get(agent_start, -1)
                    _grid_coin_detour = compute_coin_detour_distance(
                        grid_layout, agent_start, dist_to_coin, dist_to_goal
                    )
                    _grid_distances_computed = True

                traj = load_coin_trajectory(
                    traj_file,
                    coin_pos=coin_pos,
                    goal_pos=goal_pos,
                    start_to_coin_distance=astar_coin_dist,
                    coin_to_goal_distance=astar_goal_from_coin,
                    start_to_goal_distance=_grid_start_to_goal,
                    coin_detour_distance=_grid_coin_detour,
                    trajectory_category=trajectory_category,
                    coin_reachable_from_start=coin_reachable_from_start,
                    goal_reachable_from_start=goal_reachable_from_start,
                    goal_reachable_from_coin=goal_reachable_from_coin,
                    grid_layout=grid_layout,
                )
                if traj is not None:
                    trajectories.append(traj)
                    all_traj_rows.append(
                        compute_single_trajectory_row(traj, opt_to_coin, opt_to_goal, grid_key)
                    )

            if not trajectories:
                continue

            metrics, state_metrics = compute_coin_grid_metrics(
                trajectories,
                opt_to_coin, dist_to_coin,
                opt_to_goal, dist_to_goal,
                grid_key,
            )
            if metrics:
                all_metrics.append(metrics)
                all_state_metrics.extend(state_metrics)

        gc.collect()

    df = pd.DataFrame([m.to_dict() for m in all_metrics])
    state_df = pd.DataFrame(all_state_metrics)
    per_traj_df = pd.DataFrame(all_traj_rows)

    summary_df = _compute_coin_summary_by_size_density(df)
    distance_df = compute_summary_by_distance(state_df) if not state_df.empty else pd.DataFrame()
    overall = _compute_coin_overall_summary(df)

    return CoinModelTrajectoryResults(
        model_name=model_name,
        df=df,
        state_df=state_df,
        summary_by_size_density=summary_df,
        summary_by_distance=distance_df,
        overall_summary=overall,
        per_trajectory_df=per_traj_df,
    )


# =============================================================================
# Summaries
# =============================================================================


def _compute_coin_summary_by_size_density(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics grouped by grid_size, density, and reasoning_effort."""
    if df.empty:
        return pd.DataFrame()

    group_cols = ["grid_size", "density"]
    if "reasoning_effort" in df.columns:
        group_cols.append("reasoning_effort")

    return (
        df.groupby(group_cols)
        .agg(
            n_grids=("grid_id", "count"),
            mean_coin_collected_rate=("coin_collected_rate", "mean"),
            se_coin_collected_rate=("coin_collected_rate", "sem"),
            mean_full_success_rate=("full_success_rate", "mean"),
            se_full_success_rate=("full_success_rate", "sem"),
            mean_goal_only_rate=("goal_only_rate", "mean"),
            se_goal_only_rate=("goal_only_rate", "sem"),
            mean_action_accuracy=("mean_action_accuracy", "mean"),
            se_action_accuracy=("mean_action_accuracy", "sem"),
            mean_action_accuracy_phase1=("mean_action_accuracy_phase1", "mean"),
            mean_action_accuracy_phase2=("mean_action_accuracy_phase2", "mean"),
            mean_spl=("spl", "mean"),
            se_spl=("spl", "sem"),
            mean_entropy=("mean_entropy", "mean"),
            mean_jsd=("mean_jsd", "mean"),
            mean_entropy_phase1=("mean_entropy_phase1", "mean"),
            mean_entropy_phase2=("mean_entropy_phase2", "mean"),
            mean_optimal_entropy_phase1=("mean_optimal_entropy_phase1", "mean"),
            mean_optimal_entropy_phase2=("mean_optimal_entropy_phase2", "mean"),
            mean_jsd_phase1=("mean_jsd_phase1", "mean"),
            mean_jsd_phase2=("mean_jsd_phase2", "mean"),
            mean_ece=("ece", "mean"),
            mean_start_to_goal_distance=("start_to_goal_distance", "mean"),
            mean_coin_detour_distance=("coin_detour_distance", "mean"),
            mean_actions_up=("mean_actions_up", "mean"),
            mean_actions_down=("mean_actions_down", "mean"),
            mean_actions_left=("mean_actions_left", "mean"),
            mean_actions_right=("mean_actions_right", "mean"),
            mean_steps_front=("mean_steps_front", "mean"),
            mean_steps_left_turn=("mean_steps_left_turn", "mean"),
            mean_steps_right_turn=("mean_steps_right_turn", "mean"),
            mean_steps_back=("mean_steps_back", "mean"),
            mean_cell_revisits=("mean_cell_revisits", "mean"),
            mean_immediate_revisits=("mean_immediate_revisits", "mean"),
            mean_coin_oscillations=("mean_coin_oscillations", "mean"),
            mean_quadrant_fraction_ul=("mean_quadrant_fraction_ul", "mean"),
            mean_quadrant_fraction_ur=("mean_quadrant_fraction_ur", "mean"),
            mean_quadrant_fraction_dl=("mean_quadrant_fraction_dl", "mean"),
            mean_quadrant_fraction_dr=("mean_quadrant_fraction_dr", "mean"),
            mean_quadrant_entropy=("mean_quadrant_entropy", "mean"),
            mean_corner_fraction_ul=("mean_corner_fraction_ul", "mean"),
            mean_corner_fraction_ur=("mean_corner_fraction_ur", "mean"),
            mean_corner_fraction_dl=("mean_corner_fraction_dl", "mean"),
            mean_corner_fraction_dr=("mean_corner_fraction_dr", "mean"),
            mean_max_corner_dwell_run=("mean_max_corner_dwell_run", "mean"),
            worst_max_corner_dwell_run=("worst_max_corner_dwell_run", "mean"),
            mean_preferred_quadrant_contains_start=("mean_preferred_quadrant_contains_start", "mean"),
            mean_preferred_quadrant_contains_coin=("mean_preferred_quadrant_contains_coin", "mean"),
            mean_preferred_corner_contains_start=("mean_preferred_corner_contains_start", "mean"),
            mean_preferred_corner_contains_goal=("mean_preferred_corner_contains_goal", "mean"),
            mean_preferred_corner_contains_coin=("mean_preferred_corner_contains_coin", "mean"),
        )
        .reset_index()
    )


def _compute_coin_overall_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}

    summary: dict[str, Any] = {
        "n_grids": len(df),
        "overall_coin_collected_rate": float(df["coin_collected_rate"].mean()),
        "overall_full_success_rate": float(df["full_success_rate"].mean()),
        "overall_goal_only_rate": float(df["goal_only_rate"].mean()),
        "overall_mean_action_accuracy": float(df["mean_action_accuracy"].mean()),
        "overall_mean_action_accuracy_phase1": float(df["mean_action_accuracy_phase1"].mean()),
        "overall_mean_action_accuracy_phase2": float(df["mean_action_accuracy_phase2"].mean()),
        "overall_spl": float(df["spl"].mean()),
        "overall_mean_entropy": float(df["mean_entropy"].mean()),
        "overall_mean_jsd": float(df["mean_jsd"].mean()),
        "overall_ece": float(df["ece"].mean()),
        "overall_mean_start_to_goal_distance": float(df["start_to_goal_distance"].mean()),
        "overall_mean_coin_detour_distance": float(df["coin_detour_distance"].mean()),
        "overall_mean_cell_revisits": float(df["mean_cell_revisits"].mean()),
        "overall_mean_immediate_revisits": float(df["mean_immediate_revisits"].mean()),
        "overall_mean_coin_oscillations": float(df["mean_coin_oscillations"].mean()),
        "overall_mean_steps_back": float(df["mean_steps_back"].mean()),
        "overall_mean_quadrant_entropy": float(df["mean_quadrant_entropy"].mean()),
        "overall_mean_max_corner_dwell_run": float(df["mean_max_corner_dwell_run"].mean()),
        "overall_mean_preferred_quadrant_contains_coin_rate": float(df["mean_preferred_quadrant_contains_coin"].mean()),
        "overall_mean_preferred_corner_contains_coin_rate": float(df["mean_preferred_corner_contains_coin"].mean()),
    }

    if "reasoning_effort" in df.columns:
        by_effort = (
            df.groupby("reasoning_effort")[
                ["coin_collected_rate", "full_success_rate", "mean_action_accuracy", "spl"]
            ]
            .mean()
            .to_dict(orient="index")
        )
        summary["by_reasoning_effort"] = by_effort

    return summary


# =============================================================================
# Plots
# =============================================================================


def plot_coin_success_breakdown(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> Path:
    """Bar chart of success breakdown: coin only, full success, goal only."""
    setup_paper_style()

    group_cols = ["reasoning_effort"] if "reasoning_effort" in df.columns else []
    if group_cols:
        summary = df.groupby(group_cols)[
            ["coin_collected_rate", "full_success_rate", "goal_only_rate"]
        ].mean()
    else:
        summary = df[["coin_collected_rate", "full_success_rate", "goal_only_rate"]].mean().to_frame().T

    n_groups = len(summary)
    fig_width = max(4, n_groups * 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    x = np.arange(n_groups)
    width = 0.25

    ax.bar(x - width, summary["coin_collected_rate"], width, label="Coin collected")
    ax.bar(x, summary["full_success_rate"], width, label="Coin + goal")
    ax.bar(x + width, summary["goal_only_rate"], width, label="Goal only")

    ax.set_xticks(x)
    ax.set_xticklabels(summary.index if group_cols else ["all"], rotation=30, ha="right", fontsize=9)
    ax.set_xlim(-0.6, n_groups - 0.4)
    ax.set_ylabel("Rate", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Success Breakdown — {model_name}", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8, framealpha=0.9)
    plt.tight_layout()

    output_path = save_figure(fig, output_dir, "coin_success_breakdown")
    plt.close(fig)
    return output_path


def plot_phase_accuracy(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> Path:
    """Compare action accuracy in phase 1 (→ coin) vs phase 2 (→ goal)."""
    setup_paper_style()

    group_cols = ["reasoning_effort"] if "reasoning_effort" in df.columns else []
    if group_cols:
        summary = df.groupby(group_cols)[
            ["mean_action_accuracy_phase1", "mean_action_accuracy_phase2", "mean_action_accuracy"]
        ].mean()
    else:
        summary = df[["mean_action_accuracy_phase1", "mean_action_accuracy_phase2", "mean_action_accuracy"]].mean().to_frame().T

    n_groups = len(summary)
    fig_width = max(4, n_groups * 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    x = np.arange(n_groups)
    width = 0.25

    ax.bar(x - width, summary["mean_action_accuracy_phase1"], width, label="Phase 1 (→ coin)")
    ax.bar(x, summary["mean_action_accuracy_phase2"], width, label="Phase 2 (→ goal)")
    ax.bar(x + width, summary["mean_action_accuracy"], width, label="Combined")

    ax.set_xticks(x)
    ax.set_xticklabels(summary.index if group_cols else ["all"], rotation=30, ha="right", fontsize=9)
    ax.set_xlim(-0.6, n_groups - 0.4)
    ax.set_ylabel("Action Accuracy", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Phase Action Accuracy — {model_name}", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8, framealpha=0.9)
    plt.tight_layout()

    output_path = save_figure(fig, output_dir, "phase_action_accuracy")
    plt.close(fig)
    return output_path


def plot_phase_uncertainty(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> Path:
    """Compare entropy and JSD between phase 1 and phase 2."""
    setup_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (col1, col2, label) in zip(
        axes,
        [
            ("mean_entropy_phase1", "mean_entropy_phase2", "Entropy (bits)"),
            ("mean_jsd_phase1", "mean_jsd_phase2", "JSD"),
        ],
    ):
        ax.scatter(df[col1], df[col2], alpha=0.5, s=15)
        lim = max(df[col1].max(), df[col2].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="Phase 1 = Phase 2")
        ax.set_xlabel(f"Phase 1 {label}", fontsize=9)
        ax.set_ylabel(f"Phase 2 {label}", fontsize=9)
        ax.set_title(f"{label} by Phase", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    plt.suptitle(f"Phase Uncertainty Comparison — {model_name}", fontweight="bold")
    plt.tight_layout()

    output_path = save_figure(fig, output_dir, "phase_uncertainty")
    plt.close(fig)
    return output_path


# =============================================================================
# Save Results
# =============================================================================


def save_coin_results(
    results: CoinModelTrajectoryResults,
    output_dir: Path,
) -> dict[str, Path]:
    model_dir = output_dir / results.model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}

    grid_path = model_dir / "coin_trajectory_metrics.csv"
    results.df.to_csv(grid_path, index=False)
    output_paths["grid_metrics"] = grid_path
    print(f"  Saved: {grid_path}")

    if not results.per_trajectory_df.empty:
        traj_path = model_dir / "coin_per_trajectory.csv"
        results.per_trajectory_df.to_csv(traj_path, index=False)
        output_paths["per_trajectory"] = traj_path
        print(f"  Saved: {traj_path}")

    summary_path = model_dir / "coin_summary_by_size_complexity.csv"
    results.summary_by_size_density.to_csv(summary_path, index=False)
    output_paths["summary"] = summary_path
    print(f"  Saved: {summary_path}")

    if not results.summary_by_distance.empty:
        dist_path = model_dir / "coin_summary_by_distance.csv"
        results.summary_by_distance.to_csv(dist_path, index=False)
        output_paths["distance_summary"] = dist_path
        print(f"  Saved: {dist_path}")

    overall_path = model_dir / "coin_overall_summary.json"
    with open(overall_path, "w") as f:
        json.dump(results.overall_summary, f, indent=2)
    output_paths["overall"] = overall_path
    print(f"  Saved: {overall_path}")

    print("  Generating visualizations...")
    if not results.df.empty:
        plot_coin_success_breakdown(results.df, model_dir, results.model_name)
        plot_phase_accuracy(results.df, model_dir, results.model_name)
        plot_phase_uncertainty(results.df, model_dir, results.model_name)
        plot_capability_vs_uncertainty(results.df, model_dir, results.model_name)

    if not results.state_df.empty and not results.summary_by_distance.empty:
        plot_metrics_by_distance(results.summary_by_distance, model_dir, results.model_name)

    return output_paths


def print_coin_summary(results: CoinModelTrajectoryResults) -> None:
    s = results.overall_summary
    print(f"\n{'='*60}")
    print(f"Model: {results.model_name}  |  Grids: {s.get('n_grids', 0)}")
    print(f"  Coin collected rate:   {s.get('overall_coin_collected_rate', 0):.3f}")
    print(f"  Full success rate:     {s.get('overall_full_success_rate', 0):.3f}  (coin + goal)")
    print(f"  Goal only rate:        {s.get('overall_goal_only_rate', 0):.3f}  (no coin)")
    print(f"  Action accuracy:       {s.get('overall_mean_action_accuracy', 0):.3f}  "
          f"(p1={s.get('overall_mean_action_accuracy_phase1', 0):.3f}, "
          f"p2={s.get('overall_mean_action_accuracy_phase2', 0):.3f})")
    print(f"  SPL:                   {s.get('overall_spl', 0):.3f}")
    print(f"  Mean entropy:          {s.get('overall_mean_entropy', 0):.3f} bits")
    print(f"  Mean JSD:              {s.get('overall_mean_jsd', 0):.3f}")
    print(f"  ECE:                   {s.get('overall_ece', 0):.3f}")
    if "by_reasoning_effort" in s:
        print("  By reasoning effort:")
        for effort, vals in s["by_reasoning_effort"].items():
            print(f"    {effort}: coin={vals.get('coin_collected_rate', 0):.3f}, "
                  f"full={vals.get('full_success_rate', 0):.3f}, "
                  f"acc={vals.get('mean_action_accuracy', 0):.3f}")
    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze coin navigation trajectories with two-phase optimal policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        required=True,
        help="Directory containing coin trajectory JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/outputs",
        help="Base directory to save analysis outputs — a per-model subfolder "
             "(results.model_name) is created underneath it by save_coin_results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Grids to process per batch (limits RAM usage)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model name (default: derived from directory name)",
    )
    parser.add_argument(
        "--multi-model",
        action="store_true",
        help="Process multiple models from subdirectories of trajectory-dir",
    )

    args = parser.parse_args()
    traj_path = Path(args.trajectory_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.multi_model:
        model_dirs = discover_model_directories(traj_path)
        print(f"Found {len(model_dirs)} model directories")
        for model_dir in model_dirs:
            try:
                results = process_model_coin_trajectories(model_dir, batch_size=args.batch_size)
                save_coin_results(results, output_path)
                print_coin_summary(results)
            except Exception as e:
                print(f"Error processing {model_dir.name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        results = process_model_coin_trajectories(
            traj_path,
            model_name=args.model_name,
            batch_size=args.batch_size,
        )
        save_coin_results(results, output_path)
        print_coin_summary(results)


if __name__ == "__main__":
    main()
