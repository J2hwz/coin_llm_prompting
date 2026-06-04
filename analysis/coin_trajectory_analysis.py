"""Analyze coin navigation trajectories with two-phase optimal policy.

The coin environment has two sequential objectives:
  1. Collect the coin (Ball at coin_pos)
  2. Reach the terminal goal

This module computes metrics that reflect both phases:

Success Metrics:
- coin_collected_rate:   fraction of trajectories that collected the coin
- goal_after_coin_rate:  fraction that collected coin AND then reached the goal (full success)
- goal_only_rate:        fraction that reached the goal WITHOUT collecting the coin
- full_success_rate:     alias for goal_after_coin_rate

Capability Metrics (phase-aware):
- mean_action_accuracy_phase1: accuracy toward coin (steps before coin collected)
- mean_action_accuracy_phase2: accuracy toward goal (steps after coin collected)
- mean_action_accuracy:        weighted average across both phases
- spl:                         SPL using optimal_coin_distance + optimal_goal_distance as L*. Essentially calculates the ratio of optimal path to actual path taken. If 1 = fully optimal.

Uncertainty Metrics (empirical distributions, phase-split):
- mean_entropy_phase1 / mean_entropy_phase2
- mean_jsd_phase1 / mean_jsd_phase2
- mean_entropy / mean_jsd: weighted averages
- ece: Expected Calibration Error across all steps

Movement Metrics (mean per trajectory):
  Absolute action counts:
  - mean_actions_up / mean_actions_down / mean_actions_left / mean_actions_right

  Relative direction counts (relative to the previous step's heading):
  - mean_steps_front:      continued in same direction
  - mean_steps_left_turn:  turned left relative to heading
  - mean_steps_right_turn: turned right relative to heading
  - mean_steps_back:       immediate reversal (180-degree turn)

  Revisit counts:
  - mean_cell_revisits:      steps landing on any previously visited cell
  - mean_immediate_revisits: steps returning to the immediately preceding cell (double-back)
  - mean_coin_oscillations:  steps oscillating through the coin cell — arriving at coin (revisit)
                             or departing coin to a previously-visited adjacent cell

Distance Metrics (A* shortest paths on the grid):
- start_to_coin_distance:          start → coin
- coin_to_goal_distance:           coin → goal
- start_to_goal_via_coin_distance: start → coin → goal (= sum of the two above)
- start_to_goal_distance:          start → goal directly (ignoring coin)
- coin_detour_distance:            min distance from the coin to any cell on the optimal
                                   start → goal path (how far off the direct route the coin lies)

Optimal actions are computed via backward Dijkstra:
  Phase 1: target = coin_pos  (before coin collected)
  Phase 2: target = goal_pos  (after coin collected)
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
    compute_ece,
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
KNOWN_EFFORTS = {"low", "medium"}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CoinTrajectoryGridParams(TrajectoryGridParams):
    """Grid parameters extended with coin position."""

    coin_pos: Optional[tuple[int, int]] = None
    start_to_coin_distance: int = 0       # A* dist: start → coin
    coin_to_goal_distance: int = 0       # A* dist: coin → goal
    start_to_goal_via_coin_distance: int = 0      # = start_to_coin_distance + coin_to_goal_distance
    start_to_goal_distance: int = 0    # A* dist: start → goal (ignoring coin)
    coin_detour_distance: int = 0      # min dist from coin to any cell on optimal start→goal path


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
    start_to_coin_distance: int
    coin_to_goal_distance: int
    start_to_goal_via_coin_distance: int
    start_to_goal_distance: int    # direct start→goal (ignoring coin)
    coin_detour_distance: int      # min dist from coin to optimal start→goal path

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
    mean_jsd_phase1: float
    mean_entropy_phase2: float
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_id": self.grid_id,
            "grid_size": self.grid_size,
            "density": self.density,
            "instance_id": self.instance_id,
            "reasoning_effort": self.reasoning_effort,
            "transform_type": self.transform_type,
            "start_to_coin_distance": self.start_to_coin_distance,
            "coin_to_goal_distance": self.coin_to_goal_distance,
            "start_to_goal_via_coin_distance": self.start_to_goal_via_coin_distance,
            "start_to_goal_distance": self.start_to_goal_distance,
            "coin_detour_distance": self.coin_detour_distance,
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
            "mean_jsd_phase1": self.mean_jsd_phase1,
            "mean_entropy_phase2": self.mean_entropy_phase2,
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

    Returns:
        Dict mapping "{size}_{comp}_{grid_id}_{effort}" to list of trajectory paths.
    """
    grouped: dict[str, list[Path]] = defaultdict(list)

    for filepath in sorted(trajectory_dir.glob("*_coin_*_traj*.json")):
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

        grid_params = CoinTrajectoryGridParams(
            grid_size=gp.get("grid_width", 0),
            complexity=gp.get("grid_complexity", 0.0),
            grid_id=grid_id,
            astar_distance=start_to_coin_distance + coin_to_goal_distance,
            agent_start=agent_start,
            goal=goal_pos,
            coin_pos=coin_pos,
            start_to_coin_distance=start_to_coin_distance,
            coin_to_goal_distance=coin_to_goal_distance,
            start_to_goal_via_coin_distance=start_to_coin_distance + coin_to_goal_distance,
            start_to_goal_distance=start_to_goal_distance,
            coin_detour_distance=coin_detour_distance,
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
            dx, dy = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}.get(last_action, (0, 0))
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


def compute_trajectory_movement_stats(
    traj: CoinLightweightTrajectory,
) -> dict[str, int]:
    """Absolute action counts, relative direction counts, and revisit counts.

    Relative directions are computed from the perspective of the agent's heading
    (direction of the previous step).

    Revisit counts:
    - num_cell_revisits:     steps landing on any previously visited cell
    - num_immediate_revisits: steps returning to the immediately preceding cell (double-back)
    - num_coin_oscillations: steps oscillating through the coin cell — either arriving at the
                             coin cell (previously visited) or departing the coin cell to a
                             previously-visited adjacent cell
    """
    action_counts: dict[str, int] = {"UP": 0, "DOWN": 0, "LEFT": 0, "RIGHT": 0}
    rel_counts: dict[str, int] = {"front": 0, "left_turn": 0, "right_turn": 0, "back": 0}

    visited: set[tuple[int, int]] = set()
    num_cell_revisits = 0
    num_immediate_revisits = 0
    num_coin_oscillations = 0
    coin_pos = traj.grid_params.coin_pos

    for i, step in enumerate(traj.steps):
        action = step.agent_action.upper()
        pos = step.agent_position

        if action in action_counts:
            action_counts[action] += 1

        if i > 0:
            prev_pos = traj.steps[i - 1].agent_position
            prev_action = traj.steps[i - 1].agent_action.upper()

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
            if i >= 2 and pos == traj.steps[i - 2].agent_position and pos != prev_pos:
                num_immediate_revisits += 1

            # Coin oscillation: move to/from the coin cell landing on a previously visited cell
            if (pos == coin_pos and pos in visited) or (prev_pos == coin_pos and pos in visited):
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


def compute_phase_uncertainty(
    trajectories: list[CoinLightweightTrajectory],
    opt_to_coin: dict[tuple[int, int], OptimalActionSet],
    opt_to_goal: dict[tuple[int, int], OptimalActionSet],
) -> tuple[float, float, float, float]:
    """Compute mean JSD and entropy split by phase.

    Returns:
        (mean_entropy_p1, mean_jsd_p1, mean_entropy_p2, mean_jsd_p2)
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

    ent1, _, jsd1 = compute_empirical_uncertainty_metrics(counts_p1, opt_to_coin)
    ent2, _, jsd2 = compute_empirical_uncertainty_metrics(counts_p2, opt_to_goal)
    return ent1, jsd1, ent2, jsd2


def compute_coin_spl(
    trajectories: list[CoinLightweightTrajectory],
    optimal_total_distance: int,
) -> float:
    """SPL using full_success (coin + goal) and total optimal path length."""
    if not trajectories or optimal_total_distance == 0:
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

    move = compute_trajectory_movement_stats(traj)

    return {
        "trajectory_id":          traj.trajectory_id,
        "grid_id":                gp.grid_id,
        "grid_key":               grid_key,
        "grid_size":              gp.grid_size,
        "density":                gp.complexity,
        "reasoning_effort":       traj.reasoning_effort,
        "transform_type":         traj.transform_type,
        "start_to_coin_distance":    gp.start_to_coin_distance,
        "coin_to_goal_distance":    gp.coin_to_goal_distance,
        "start_to_goal_via_coin_distance":   gp.start_to_goal_via_coin_distance,
        "start_to_goal_distance": gp.start_to_goal_distance,
        "coin_detour_distance":   gp.coin_detour_distance,
        "reached_goal":           int(traj.reached_goal),
        "coin_collected":         int(traj.coin_collected),
        "coin_collection_step":   traj.coin_collection_step if traj.coin_collection_step is not None else -1,
        "trajectory_length":      L,
        "spl":                    spl,
        "action_accuracy":        acc_combined,
        "action_accuracy_phase1": acc1,
        "action_accuracy_phase2": acc2,
        **move,
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

    pattern = re.match(r"size(\d+)_comp([\d.]+)_grid(\d+)_([A-Za-z][A-Za-z0-9]*)_(\w+)", grid_key)
    if not pattern:
        return None, []

    grid_size = int(pattern.group(1))
    density = float(pattern.group(2))
    instance_id = int(pattern.group(3))
    transform_type = pattern.group(4)
    effort = pattern.group(5)

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

    # Aggregate state-action counts (all steps combined for ECE / overall metrics)
    all_counts: StateActionCounts = StateActionCounts()
    total_steps = 0
    total_correct = 0

    for traj in trajectories:
        for step in traj.steps:
            all_counts.add(step.agent_position, step.agent_action)
            total_steps += 1
            optimal_set = get_step_optimal_actions(step, traj, opt_to_coin, opt_to_goal)
            action_id = ACTION_NAME_TO_ID.get(step.agent_action.upper())
            if action_id is not None and action_id in optimal_set:
                total_correct += 1

    # Build combined optimal_actions map for ECE (phase 1 entries, phase 2 overwrite)
    combined_optimal: dict[tuple[int, int], OptimalActionSet] = {}
    combined_optimal.update(opt_to_coin)
    combined_optimal.update(opt_to_goal)

    mean_ent, mean_opt_ent, mean_jsd = compute_empirical_uncertainty_metrics(all_counts, combined_optimal)
    ece = compute_ece(all_counts, combined_optimal)
    ent1, jsd1, ent2, jsd2 = compute_phase_uncertainty(trajectories, opt_to_coin, opt_to_goal)

    mean_step_accuracy = total_correct / total_steps if total_steps > 0 else 0.0

    # Movement stats — averaged across trajectories
    move_keys = [
        "num_actions_up", "num_actions_down", "num_actions_left", "num_actions_right",
        "num_steps_front", "num_steps_left_turn", "num_steps_right_turn", "num_steps_back",
        "num_cell_revisits", "num_immediate_revisits", "num_coin_oscillations",
    ]
    move_totals: dict[str, float] = {k: 0.0 for k in move_keys}
    for traj in trajectories:
        for k, v in compute_trajectory_movement_stats(traj).items():
            move_totals[k] += v
    move_means = {k: v / n for k, v in move_totals.items()}

    # Per-state metrics for distance analysis (use dist_to_goal as reference distance)
    state_metrics: list[dict[str, Any]] = []
    for pos, action_counts in all_counts.counts.items():
        total = sum(action_counts.values())
        if total == 0:
            continue

        empirical_dist = all_counts.get_empirical_distribution(pos)
        optimal_set = combined_optimal.get(pos, set())
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
        start_to_coin_distance=astar_coin,
        coin_to_goal_distance=astar_goal,
        start_to_goal_via_coin_distance=astar_total,
        start_to_goal_distance=start_to_goal,
        coin_detour_distance=coin_detour,
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
        mean_jsd_phase1=jsd1,
        mean_entropy_phase2=ent2,
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

            # A* distance from coin to goal (start→coin loaded per trajectory below)
            astar_goal_from_coin = dist_to_goal.get(coin_pos, 0)

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
                    astar_coin_dist = dist_to_coin.get(agent_start, 0)
                except Exception:
                    agent_start = (0, 0)
                    astar_coin_dist = 0

                if not _grid_distances_computed:
                    _grid_start_to_goal = dist_to_goal.get(agent_start, 0)
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

    grid_path = model_dir / f"coin_trajectory_metrics_{results.model_name}.csv"
    results.df.to_csv(grid_path, index=False)
    output_paths["grid_metrics"] = grid_path
    print(f"  Saved: {grid_path}")

    if not results.per_trajectory_df.empty:
        traj_path = model_dir / f"coin_per_trajectory_{results.model_name}.csv"
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
        default="src/reveng/analysis/outputs/coin_trajectory_analysis",
        help="Directory to save analysis outputs",
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
