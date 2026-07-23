"""Analyze two-coin navigation trajectories with a three-phase optimal policy.

Extends coin_trajectory_analysis.py to trajectories with two coins
(data/two_coins/), where the success criterion depends on the collection mode
encoded in the directory structure (not in the filename or trajectory JSON):
    collect_one/  — reach the goal after collecting >=1 of the two coins
    collect_all/  — reach the goal after collecting both coins

Per-step action accuracy is scored against whichever coin the agent actually
collected first (retrofit from the observed collection event), mirroring how
coin_trajectory_analysis.py derives its phase split from the observed
coin_collection_step rather than a hypothesized target. See
compute_optimal_total_distance / get_step_phase_and_optimal_actions for the
resulting three-phase model.

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

import pandas as pd
from tqdm import tqdm

from analysis.analysis_utils import (
    ACTION_NAME_TO_ID,
    OptimalActionSet,
    TrajectoryStep,
    compute_optimal_actions_from_text_grid,
    extract_agent_position_from_grid_state,
    sanitize_label,
)
from analysis.coin_trajectory_analysis import (
    KNOWN_EFFORTS,
    _ACTION_DELTAS,
    compute_trajectory_movement_stats,
    is_coin_present_at,
)
from analysis.full_obs_trajectory_analysis import (
    StateActionCounts,
    batch_grid_keys,
    compute_ece,
    compute_empirical_uncertainty_metrics,
    discover_model_directories,
)

COLLECT_MODES = {"collect_one", "collect_all"}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TwoCoinTrajectoryGridParams:
    """Grid parameters for a two-coin grid."""

    grid_size: int
    complexity: float
    grid_id: int
    agent_start: tuple[int, int]
    goal: tuple[int, int]
    coin_pos_1: tuple[int, int]
    coin_pos_2: tuple[int, int]
    collect_mode: str
    grid_layout: Optional[list[list[str]]] = None

    start_to_coin1_distance: int = 0
    start_to_coin2_distance: int = 0
    coin1_to_coin2_distance: int = 0
    coin2_to_coin1_distance: int = 0
    coin1_to_goal_distance: int = 0
    coin2_to_goal_distance: int = 0
    optimal_total_distance: int = 0

    coin1_reachable_from_start: bool = True
    coin2_reachable_from_start: bool = True
    goal_reachable_from_start: bool = True
    goal_reachable_from_coin1: bool = True
    goal_reachable_from_coin2: bool = True


@dataclass
class TwoCoinLightweightTrajectory:
    """Trajectory with independent per-coin collection tracking."""

    grid_params: TwoCoinTrajectoryGridParams
    steps: list[TrajectoryStep]
    reached_goal: bool
    coin1_collected: bool
    coin2_collected: bool
    coin1_collection_step: Optional[int]
    coin2_collection_step: Optional[int]
    reasoning_effort: str = "low"
    transform_type: str = "base"
    trajectory_category: str = "base"
    trajectory_id: int = 0

    @property
    def trajectory_length(self) -> int:
        return len(self.steps)

    @property
    def num_coins_collected(self) -> int:
        return int(self.coin1_collected) + int(self.coin2_collected)

    @property
    def success(self) -> bool:
        if not self.reached_goal:
            return False
        if self.grid_params.collect_mode == "collect_all":
            return self.num_coins_collected == 2
        return self.num_coins_collected >= 1

    @property
    def first_collection_step(self) -> Optional[int]:
        """Step index of whichever coin was collected first, or None if neither was."""
        steps = [s for s in (self.coin1_collection_step, self.coin2_collection_step) if s is not None]
        return min(steps) if steps else None

    @property
    def second_collection_step(self) -> Optional[int]:
        """Step index of the second coin collected — only set if both coins were collected."""
        if self.coin1_collection_step is None or self.coin2_collection_step is None:
            return None
        return max(self.coin1_collection_step, self.coin2_collection_step)


@dataclass
class TwoCoinGridTrajectoryMetrics:
    """Per-grid metrics for two-coin trajectories."""

    grid_id: str
    grid_size: int
    density: float
    instance_id: int
    reasoning_effort: str
    transform_type: str
    trajectory_category: str
    collect_mode: str

    start_to_coin1_distance: int
    start_to_coin2_distance: int
    coin1_to_coin2_distance: int
    coin2_to_coin1_distance: int
    coin1_to_goal_distance: int
    coin2_to_goal_distance: int
    optimal_total_distance: int

    coin1_reachable_from_start: int
    coin2_reachable_from_start: int
    goal_reachable_from_start: int
    goal_reachable_from_coin1: int
    goal_reachable_from_coin2: int

    # --- Success breakdown ---
    num_trajectories: int
    num_trajectories_0_coins: int
    num_trajectories_1_coin: int
    num_trajectories_2_coins: int
    mean_coins_collected: float
    success_rate: float  # mode-aware: collect_one >=1 coin, collect_all both coins

    # --- Capability ---
    mean_trajectory_length: float
    mean_action_accuracy: float
    mean_action_accuracy_phase1: float  # toward the first-collected (or nearer) coin
    mean_action_accuracy_phase2: float  # toward goal (collect_one) or remaining coin (collect_all)
    mean_action_accuracy_phase3: float  # collect_all only: after both coins, toward goal
    spl: float

    # --- Uncertainty (empirical distributions, pooled across phases) ---
    mean_entropy: float
    mean_optimal_entropy: float
    mean_jsd: float
    ece: float

    mean_step_accuracy: float
    total_steps: int

    # --- Movement stats (mean per trajectory) ---
    mean_actions_up: float = 0.0
    mean_actions_down: float = 0.0
    mean_actions_left: float = 0.0
    mean_actions_right: float = 0.0
    mean_steps_front: float = 0.0
    mean_steps_left_turn: float = 0.0
    mean_steps_right_turn: float = 0.0
    mean_steps_back: float = 0.0
    mean_cell_revisits: float = 0.0
    mean_immediate_revisits: float = 0.0
    mean_coin_oscillations: float = 0.0
    mean_invalid_actions: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_id": self.grid_id,
            "grid_size": self.grid_size,
            "density": self.density,
            "instance_id": self.instance_id,
            "reasoning_effort": self.reasoning_effort,
            "transform_type": self.transform_type,
            "trajectory_category": self.trajectory_category,
            "collect_mode": self.collect_mode,
            "start_to_coin1_distance": self.start_to_coin1_distance,
            "start_to_coin2_distance": self.start_to_coin2_distance,
            "coin1_to_coin2_distance": self.coin1_to_coin2_distance,
            "coin2_to_coin1_distance": self.coin2_to_coin1_distance,
            "coin1_to_goal_distance": self.coin1_to_goal_distance,
            "coin2_to_goal_distance": self.coin2_to_goal_distance,
            "optimal_total_distance": self.optimal_total_distance,
            "coin1_reachable_from_start": self.coin1_reachable_from_start,
            "coin2_reachable_from_start": self.coin2_reachable_from_start,
            "goal_reachable_from_start": self.goal_reachable_from_start,
            "goal_reachable_from_coin1": self.goal_reachable_from_coin1,
            "goal_reachable_from_coin2": self.goal_reachable_from_coin2,
            "num_trajectories": self.num_trajectories,
            "num_trajectories_0_coins": self.num_trajectories_0_coins,
            "num_trajectories_1_coin": self.num_trajectories_1_coin,
            "num_trajectories_2_coins": self.num_trajectories_2_coins,
            "mean_coins_collected": self.mean_coins_collected,
            "success_rate": self.success_rate,
            "mean_trajectory_length": self.mean_trajectory_length,
            "mean_action_accuracy": self.mean_action_accuracy,
            "mean_action_accuracy_phase1": self.mean_action_accuracy_phase1,
            "mean_action_accuracy_phase2": self.mean_action_accuracy_phase2,
            "mean_action_accuracy_phase3": self.mean_action_accuracy_phase3,
            "spl": self.spl,
            "mean_entropy": self.mean_entropy,
            "mean_optimal_entropy": self.mean_optimal_entropy,
            "mean_jsd": self.mean_jsd,
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
        }


@dataclass
class TwoCoinModelTrajectoryResults:
    """Results for a single model across all two-coin grids."""

    model_name: str
    df: pd.DataFrame
    summary_by_size_density: pd.DataFrame
    overall_summary: dict[str, Any]
    per_trajectory_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# =============================================================================
# Filename Parsing / Discovery
# =============================================================================


def parse_two_coin_trajectory_filename(filename: str) -> Optional[dict[str, Any]]:
    """Parse two-coin trajectory filename to extract metadata.

    Expected format: {model}_size{N}_comp{X.X}_grid{N}_twocoin_{effort}_traj{N}.json
    """
    pattern = r"(.+)_size(\d+)_comp([\d.]+)_grid(\d+)_twocoin_(\w+)_traj(\d+)\.json"
    match = re.match(pattern, filename)
    if not match:
        return None

    model, size, comp, grid_id, effort_field, traj_id = match.groups()

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


def infer_collect_mode(filepath: Path) -> Optional[str]:
    """Determine collect_one / collect_all from the containing directory path."""
    for part in filepath.parts:
        if part in COLLECT_MODES:
            return part
    return None


def discover_two_coin_trajectory_files(
    trajectory_dir: Path,
) -> dict[str, list[Path]]:
    """Discover two-coin trajectory files grouped by (collect_mode, grid_key, effort).

    Recurses into subdirectories (collect_one/collect_all x low/medium x size_effort).

    Returns:
        Dict mapping "{collect_mode}_size{N}_comp{X}_grid{id}_{transform}_{effort}"
        to list of trajectory paths.
    """
    grouped: dict[str, list[Path]] = defaultdict(list)

    for filepath in sorted(trajectory_dir.rglob("*_twocoin_*_traj*.json")):
        parsed = parse_two_coin_trajectory_filename(filepath.name)
        if parsed is None:
            continue
        collect_mode = infer_collect_mode(filepath)
        if collect_mode is None:
            print(f"  Warning: could not determine collect_mode for {filepath}, skipping.")
            continue
        key = (
            f"{collect_mode}_"
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


def load_two_coin_grid_layout(
    layout_file: Path,
) -> Optional[tuple[list[list[str]], tuple[int, int], tuple[int, int], tuple[int, int]]]:
    """Load a two-coin grid layout file.

    Returns:
        (grid_layout, coin_pos_1, coin_pos_2, goal_pos) or None on failure.
        Positions are (x, y) tuples.
    """
    try:
        with open(layout_file, "r") as f:
            data = json.load(f)

        grid_layout = data.get("grid_layout")
        raw_c1 = data.get("coin_pos_1")
        raw_c2 = data.get("coin_pos_2")
        raw_goal = data.get("goal_pos")

        if grid_layout is None or raw_c1 is None or raw_c2 is None or raw_goal is None:
            return None

        coin_pos_1 = (int(raw_c1[0]), int(raw_c1[1]))
        coin_pos_2 = (int(raw_c2[0]), int(raw_c2[1]))
        goal_pos = (int(raw_goal[0]), int(raw_goal[1]))
        return grid_layout, coin_pos_1, coin_pos_2, goal_pos

    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        return None


def find_two_coin_layout_file(traj_file: Path, grid_id: int) -> Optional[Path]:
    """Locate the *_twocoin_layout.json for a given trajectory file."""
    parsed = parse_two_coin_trajectory_filename(traj_file.name)
    if parsed is None:
        return None

    transform = parsed["transform_type"]
    if transform == "base":
        layout_name = (
            f"{parsed['model']}_size{parsed['grid_size']}_"
            f"comp{parsed['density']}_grid{grid_id}_twocoin_layout.json"
        )
    else:
        layout_name = (
            f"{parsed['model']}_size{parsed['grid_size']}_"
            f"comp{parsed['density']}_grid{grid_id}_twocoin_{transform}_layout.json"
        )
    candidate = traj_file.parent / layout_name
    return candidate if candidate.exists() else None


# =============================================================================
# Coin Collection Detection
# =============================================================================


def find_two_coin_collection_steps(
    raw_steps: list[dict[str, Any]],
    coin_pos_1: tuple[int, int],
    coin_pos_2: tuple[int, int],
) -> tuple[Optional[int], Optional[int]]:
    """Detect each coin's collection step independently from the grid-state text.

    Both coins render as the same 'C' symbol, but collection is tracked per
    *position* (coin_pos_1 vs coin_pos_2), not by symbol — so this correctly
    attributes a collection event to a specific coin regardless of pickup order.
    """

    def _find(coin_pos: tuple[int, int]) -> Optional[int]:
        was_present = None
        for i, step in enumerate(raw_steps):
            grid_state = step.get("grid_state", [])
            present = is_coin_present_at(grid_state, coin_pos)
            if was_present is None:
                was_present = present
            if was_present and not present:
                return max(0, i - 1)
        return None

    return _find(coin_pos_1), _find(coin_pos_2)


# =============================================================================
# Trajectory Loading
# =============================================================================


def load_two_coin_trajectory(
    filepath: Path,
    coin_pos_1: tuple[int, int],
    coin_pos_2: tuple[int, int],
    goal_pos: tuple[int, int],
    collect_mode: str,
    start_to_coin1_distance: int,
    start_to_coin2_distance: int,
    coin1_to_coin2_distance: int,
    coin2_to_coin1_distance: int,
    coin1_to_goal_distance: int,
    coin2_to_goal_distance: int,
    optimal_total_distance: int,
    trajectory_category: str = "base",
    coin1_reachable_from_start: bool = True,
    coin2_reachable_from_start: bool = True,
    goal_reachable_from_start: bool = True,
    goal_reachable_from_coin1: bool = True,
    goal_reachable_from_coin2: bool = True,
    grid_layout: Optional[list[list[str]]] = None,
) -> Optional[TwoCoinLightweightTrajectory]:
    """Load a two-coin trajectory file and detect coin collection from grid states."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        gp = data.get("grid_params", {})
        start_coords = gp.get("agent_start_coordinates", [0, 0])
        agent_start = (int(start_coords[1]), int(start_coords[0]))

        parsed = parse_two_coin_trajectory_filename(filepath.name)
        grid_id = parsed["grid_id"] if parsed else 0
        effort = parsed["reasoning_effort"] if parsed else "unknown"
        transform_type = parsed["transform_type"] if parsed else "base"
        trajectory_id = parsed["trajectory_id"] if parsed else 0

        grid_params = TwoCoinTrajectoryGridParams(
            grid_size=gp.get("grid_width", 0),
            complexity=gp.get("grid_complexity", 0.0),
            grid_id=grid_id,
            agent_start=agent_start,
            goal=goal_pos,
            coin_pos_1=coin_pos_1,
            coin_pos_2=coin_pos_2,
            collect_mode=collect_mode,
            grid_layout=grid_layout,
            start_to_coin1_distance=start_to_coin1_distance,
            start_to_coin2_distance=start_to_coin2_distance,
            coin1_to_coin2_distance=coin1_to_coin2_distance,
            coin2_to_coin1_distance=coin2_to_coin1_distance,
            coin1_to_goal_distance=coin1_to_goal_distance,
            coin2_to_goal_distance=coin2_to_goal_distance,
            optimal_total_distance=optimal_total_distance,
            coin1_reachable_from_start=coin1_reachable_from_start,
            coin2_reachable_from_start=coin2_reachable_from_start,
            goal_reachable_from_start=goal_reachable_from_start,
            goal_reachable_from_coin1=goal_reachable_from_coin1,
            goal_reachable_from_coin2=goal_reachable_from_coin2,
        )

        raw_steps = data.get("steps", [])

        coin1_collection_step, coin2_collection_step = find_two_coin_collection_steps(
            raw_steps, coin_pos_1, coin_pos_2
        )

        steps = []
        for i, step in enumerate(raw_steps):
            grid_state = step.get("grid_state", [])
            agent_pos = extract_agent_position_from_grid_state(grid_state)
            agent_action = step.get("agent_action", "")
            steps.append(TrajectoryStep(step_id=i, agent_position=agent_pos, agent_action=agent_action))

        reached_goal = False
        if steps:
            pos = steps[-1].agent_position
            last_action = steps[-1].agent_action.upper()
            dx, dy = _ACTION_DELTAS.get(last_action, (0, 0))
            final_pos = (pos[0] + dx, pos[1] + dy)
            reached_goal = final_pos == goal_pos

        return TwoCoinLightweightTrajectory(
            grid_params=grid_params,
            steps=steps,
            reached_goal=reached_goal,
            coin1_collected=coin1_collection_step is not None,
            coin2_collected=coin2_collection_step is not None,
            coin1_collection_step=coin1_collection_step,
            coin2_collection_step=coin2_collection_step,
            reasoning_effort=effort,
            transform_type=transform_type,
            trajectory_category=trajectory_category,
            trajectory_id=trajectory_id,
        )

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Warning: Error loading {filepath.name}: {e}")
        return None


# =============================================================================
# Three-Phase Optimal Actions
# =============================================================================


def compute_three_target_optimal_actions(
    grid_layout: list[list[str]],
    coin_pos_1: tuple[int, int],
    coin_pos_2: tuple[int, int],
    goal_pos: tuple[int, int],
) -> tuple[
    dict[tuple[int, int], OptimalActionSet],
    dict[tuple[int, int], int],
    dict[tuple[int, int], OptimalActionSet],
    dict[tuple[int, int], int],
    dict[tuple[int, int], OptimalActionSet],
    dict[tuple[int, int], int],
]:
    """Compute optimal actions/distances toward each of the three targets via backward Dijkstra.

    Returns:
        (opt_to_c1, dist_to_c1, opt_to_c2, dist_to_c2, opt_to_goal, dist_to_goal)
    """
    opt_to_c1, dist_to_c1 = compute_optimal_actions_from_text_grid(grid_layout, coin_pos_1)
    opt_to_c2, dist_to_c2 = compute_optimal_actions_from_text_grid(grid_layout, coin_pos_2)
    opt_to_goal, dist_to_goal = compute_optimal_actions_from_text_grid(grid_layout, goal_pos)
    return opt_to_c1, dist_to_c1, opt_to_c2, dist_to_c2, opt_to_goal, dist_to_goal


def compute_optimal_total_distance(
    collect_mode: str,
    agent_start: tuple[int, int],
    coin_pos_1: tuple[int, int],
    coin_pos_2: tuple[int, int],
    dist_to_c1: dict[tuple[int, int], int],
    dist_to_c2: dict[tuple[int, int], int],
    dist_to_goal: dict[tuple[int, int], int],
) -> int:
    """Optimal total path length for the given collect mode, or -1 if unreachable.

    collect_one: min(start->c1->goal, start->c2->goal).
    collect_all: min over the two visiting orders (c1 then c2, or c2 then c1).
    """
    d_start_c1 = dist_to_c1.get(agent_start, -1)
    d_start_c2 = dist_to_c2.get(agent_start, -1)
    d_c1_goal = dist_to_goal.get(coin_pos_1, -1)
    d_c2_goal = dist_to_goal.get(coin_pos_2, -1)

    if collect_mode == "collect_all":
        d_c1_c2 = dist_to_c2.get(coin_pos_1, -1)
        d_c2_c1 = dist_to_c1.get(coin_pos_2, -1)
        order_a = (
            d_start_c1 + d_c1_c2 + d_c2_goal
            if min(d_start_c1, d_c1_c2, d_c2_goal) >= 0
            else -1
        )
        order_b = (
            d_start_c2 + d_c2_c1 + d_c1_goal
            if min(d_start_c2, d_c2_c1, d_c1_goal) >= 0
            else -1
        )
        candidates = [d for d in (order_a, order_b) if d >= 0]
        return min(candidates) if candidates else -1

    via_c1 = d_start_c1 + d_c1_goal if min(d_start_c1, d_c1_goal) >= 0 else -1
    via_c2 = d_start_c2 + d_c2_goal if min(d_start_c2, d_c2_goal) >= 0 else -1
    candidates = [d for d in (via_c1, via_c2) if d >= 0]
    return min(candidates) if candidates else -1


def get_step_phase_and_optimal_actions(
    step: TrajectoryStep,
    traj: TwoCoinLightweightTrajectory,
    opt_to_c1: dict[tuple[int, int], OptimalActionSet],
    opt_to_c2: dict[tuple[int, int], OptimalActionSet],
    opt_to_goal: dict[tuple[int, int], OptimalActionSet],
) -> tuple[int, OptimalActionSet]:
    """Return (phase_id, optimal_action_set) for a step.

    Phase target is scored against whichever coin the agent actually collected
    first (retrofit from the observed event), not a hypothesized optimal choice:

    - Phase 1: start -> first coin actually collected. If neither coin is ever
      collected, scores against the globally nearer coin for the whole trajectory.
    - Phase 2: after the first coin -> goal directly (collect_one, second coin not
      required) or -> the remaining coin (collect_all; continues scoring against it
      even if never collected, since it's still required).
    - Phase 3 (collect_all only): after both coins are collected -> goal.
    """
    gp = traj.grid_params
    first_step = traj.first_collection_step
    second_step = traj.second_collection_step

    if first_step is None:
        target_is_c1 = gp.start_to_coin1_distance <= gp.start_to_coin2_distance
        actions = opt_to_c1 if target_is_c1 else opt_to_c2
        return 1, actions.get(step.agent_position, set())

    first_is_c1 = traj.coin1_collection_step == first_step

    if step.step_id <= first_step:
        actions = opt_to_c1 if first_is_c1 else opt_to_c2
        return 1, actions.get(step.agent_position, set())

    if gp.collect_mode == "collect_one":
        return 2, opt_to_goal.get(step.agent_position, set())

    if second_step is not None and step.step_id > second_step:
        return 3, opt_to_goal.get(step.agent_position, set())

    actions = opt_to_c2 if first_is_c1 else opt_to_c1
    return 2, actions.get(step.agent_position, set())


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_phase_accuracy(
    trajectories: list[TwoCoinLightweightTrajectory],
    opt_to_c1: dict[tuple[int, int], OptimalActionSet],
    opt_to_c2: dict[tuple[int, int], OptimalActionSet],
    opt_to_goal: dict[tuple[int, int], OptimalActionSet],
) -> tuple[float, float, float, float]:
    """Compute action accuracy for phase 1, 2, 3, and combined.

    Returns:
        (accuracy_phase1, accuracy_phase2, accuracy_phase3, accuracy_combined)
    """
    correct = {1: 0, 2: 0, 3: 0}
    total = {1: 0, 2: 0, 3: 0}

    for traj in trajectories:
        for step in traj.steps:
            phase, optimal_set = get_step_phase_and_optimal_actions(
                step, traj, opt_to_c1, opt_to_c2, opt_to_goal
            )
            action_id = ACTION_NAME_TO_ID.get(step.agent_action.upper())
            is_correct = int(action_id is not None and action_id in optimal_set)
            correct[phase] += is_correct
            total[phase] += 1

    acc1 = correct[1] / total[1] if total[1] > 0 else 0.0
    acc2 = correct[2] / total[2] if total[2] > 0 else 0.0
    acc3 = correct[3] / total[3] if total[3] > 0 else 0.0
    total_all = sum(total.values())
    acc_combined = sum(correct.values()) / total_all if total_all > 0 else 0.0

    return acc1, acc2, acc3, acc_combined


def compute_two_coin_spl(
    trajectories: list[TwoCoinLightweightTrajectory],
    optimal_total_distance: int,
) -> float:
    """SPL using the mode-aware success flag and total optimal path length."""
    if not trajectories or optimal_total_distance <= 0:
        return 0.0

    spls = []
    for traj in trajectories:
        L_star = optimal_total_distance
        L = traj.trajectory_length
        spls.append((1 if traj.success else 0) * L_star / max(L_star, L))

    return sum(spls) / len(spls)


def compute_single_two_coin_trajectory_row(
    traj: TwoCoinLightweightTrajectory,
    opt_to_c1: dict[tuple[int, int], OptimalActionSet],
    opt_to_c2: dict[tuple[int, int], OptimalActionSet],
    opt_to_goal: dict[tuple[int, int], OptimalActionSet],
    grid_key: str,
) -> dict[str, Any]:
    """Build one CSV row for a single two-coin trajectory."""
    gp = traj.grid_params

    correct = {1: 0, 2: 0, 3: 0}
    total = {1: 0, 2: 0, 3: 0}
    for step in traj.steps:
        phase, optimal_set = get_step_phase_and_optimal_actions(
            step, traj, opt_to_c1, opt_to_c2, opt_to_goal
        )
        action_id = ACTION_NAME_TO_ID.get(step.agent_action.upper())
        is_correct = int(action_id is not None and action_id in optimal_set)
        correct[phase] += is_correct
        total[phase] += 1

    acc1 = correct[1] / total[1] if total[1] > 0 else 0.0
    acc2 = correct[2] / total[2] if total[2] > 0 else 0.0
    acc3 = correct[3] / total[3] if total[3] > 0 else 0.0
    total_all = sum(total.values())
    acc_combined = sum(correct.values()) / total_all if total_all > 0 else 0.0

    L_star = gp.optimal_total_distance
    L = traj.trajectory_length
    spl = (L_star / max(L_star, L)) if (traj.success and L_star > 0) else 0.0

    move = compute_trajectory_movement_stats(
        traj.steps, gp.grid_layout, {gp.coin_pos_1, gp.coin_pos_2}
    )

    return {
        "trajectory_id": traj.trajectory_id,
        "grid_id": gp.grid_id,
        "grid_key": grid_key,
        "grid_size": gp.grid_size,
        "density": gp.complexity,
        "reasoning_effort": traj.reasoning_effort,
        "transform_type": traj.transform_type,
        "trajectory_category": traj.trajectory_category,
        "collect_mode": gp.collect_mode,
        "start_to_coin1_distance": gp.start_to_coin1_distance,
        "start_to_coin2_distance": gp.start_to_coin2_distance,
        "coin1_to_coin2_distance": gp.coin1_to_coin2_distance,
        "coin2_to_coin1_distance": gp.coin2_to_coin1_distance,
        "coin1_to_goal_distance": gp.coin1_to_goal_distance,
        "coin2_to_goal_distance": gp.coin2_to_goal_distance,
        "optimal_total_distance": gp.optimal_total_distance,
        "coin1_reachable_from_start": int(gp.coin1_reachable_from_start),
        "coin2_reachable_from_start": int(gp.coin2_reachable_from_start),
        "goal_reachable_from_start": int(gp.goal_reachable_from_start),
        "goal_reachable_from_coin1": int(gp.goal_reachable_from_coin1),
        "goal_reachable_from_coin2": int(gp.goal_reachable_from_coin2),
        "reached_goal": int(traj.reached_goal),
        "num_coins_collected": traj.num_coins_collected,
        "coin1_collected": int(traj.coin1_collected),
        "coin2_collected": int(traj.coin2_collected),
        "coin1_collection_step": traj.coin1_collection_step if traj.coin1_collection_step is not None else -1,
        "coin2_collection_step": traj.coin2_collection_step if traj.coin2_collection_step is not None else -1,
        "success": int(traj.success),
        "trajectory_length": L,
        "spl": spl,
        "action_accuracy": acc_combined,
        "action_accuracy_phase1": acc1,
        "action_accuracy_phase2": acc2,
        "action_accuracy_phase3": acc3,
        **move,
        "invalid_action_rate": move["num_invalid_actions"] / L if L > 0 else 0.0,
    }


def compute_two_coin_grid_metrics(
    trajectories: list[TwoCoinLightweightTrajectory],
    opt_to_c1: dict[tuple[int, int], OptimalActionSet],
    opt_to_c2: dict[tuple[int, int], OptimalActionSet],
    opt_to_goal: dict[tuple[int, int], OptimalActionSet],
    grid_key: str,
) -> Optional[TwoCoinGridTrajectoryMetrics]:
    """Compute all metrics for a two-coin grid from its trajectories."""
    if not trajectories:
        return None

    pattern = re.match(r"(collect_\w+?)_size(\d+)_comp([\d.]+)_grid(\d+)_", grid_key)
    if not pattern:
        return None

    collect_mode = pattern.group(1)
    grid_size = int(pattern.group(2))
    density = float(pattern.group(3))
    instance_id = int(pattern.group(4))
    transform_type = trajectories[0].transform_type
    effort = trajectories[0].reasoning_effort
    trajectory_category = trajectories[0].trajectory_category

    gp = trajectories[0].grid_params
    n = len(trajectories)

    num_0 = sum(1 for t in trajectories if t.num_coins_collected == 0)
    num_1 = sum(1 for t in trajectories if t.num_coins_collected == 1)
    num_2 = sum(1 for t in trajectories if t.num_coins_collected == 2)
    mean_coins = sum(t.num_coins_collected for t in trajectories) / n
    success_rate = sum(1 for t in trajectories if t.success) / n

    mean_traj_len = sum(t.trajectory_length for t in trajectories) / n
    acc1, acc2, acc3, acc_combined = compute_phase_accuracy(trajectories, opt_to_c1, opt_to_c2, opt_to_goal)
    spl = compute_two_coin_spl(trajectories, gp.optimal_total_distance)

    all_counts: StateActionCounts = StateActionCounts()
    total_steps = 0
    total_correct = 0
    for traj in trajectories:
        for step in traj.steps:
            all_counts.add(step.agent_position, step.agent_action)
            total_steps += 1
            _, optimal_set = get_step_phase_and_optimal_actions(step, traj, opt_to_c1, opt_to_c2, opt_to_goal)
            action_id = ACTION_NAME_TO_ID.get(step.agent_action.upper())
            if action_id is not None and action_id in optimal_set:
                total_correct += 1

    # Combined optimal-action map for pooled entropy/JSD/ECE (grid-level approximation:
    # goal target wins over remaining-coin target, which wins over first-coin target,
    # at any position visited under more than one phase across trajectories).
    combined_optimal: dict[tuple[int, int], OptimalActionSet] = {}
    combined_optimal.update(opt_to_c1)
    combined_optimal.update(opt_to_c2)
    combined_optimal.update(opt_to_goal)

    mean_ent, mean_opt_ent, mean_jsd = compute_empirical_uncertainty_metrics(all_counts, combined_optimal)
    ece = compute_ece(all_counts, combined_optimal)
    mean_step_accuracy = total_correct / total_steps if total_steps > 0 else 0.0

    move_keys = [
        "num_actions_up", "num_actions_down", "num_actions_left", "num_actions_right",
        "num_steps_front", "num_steps_left_turn", "num_steps_right_turn", "num_steps_back",
        "num_cell_revisits", "num_immediate_revisits", "num_coin_oscillations",
        "num_invalid_actions",
    ]
    move_totals: dict[str, float] = {k: 0.0 for k in move_keys}
    for traj in trajectories:
        stats = compute_trajectory_movement_stats(
            traj.steps, gp.grid_layout, {gp.coin_pos_1, gp.coin_pos_2}
        )
        for k, v in stats.items():
            move_totals[k] += v
    move_means = {k: v / n for k, v in move_totals.items()}

    return TwoCoinGridTrajectoryMetrics(
        grid_id=grid_key,
        grid_size=grid_size,
        density=density,
        instance_id=instance_id,
        reasoning_effort=effort,
        transform_type=transform_type,
        trajectory_category=trajectory_category,
        collect_mode=collect_mode,
        start_to_coin1_distance=gp.start_to_coin1_distance,
        start_to_coin2_distance=gp.start_to_coin2_distance,
        coin1_to_coin2_distance=gp.coin1_to_coin2_distance,
        coin2_to_coin1_distance=gp.coin2_to_coin1_distance,
        coin1_to_goal_distance=gp.coin1_to_goal_distance,
        coin2_to_goal_distance=gp.coin2_to_goal_distance,
        optimal_total_distance=gp.optimal_total_distance,
        coin1_reachable_from_start=int(gp.coin1_reachable_from_start),
        coin2_reachable_from_start=int(gp.coin2_reachable_from_start),
        goal_reachable_from_start=int(gp.goal_reachable_from_start),
        goal_reachable_from_coin1=int(gp.goal_reachable_from_coin1),
        goal_reachable_from_coin2=int(gp.goal_reachable_from_coin2),
        num_trajectories=n,
        num_trajectories_0_coins=num_0,
        num_trajectories_1_coin=num_1,
        num_trajectories_2_coins=num_2,
        mean_coins_collected=mean_coins,
        success_rate=success_rate,
        mean_trajectory_length=mean_traj_len,
        mean_action_accuracy=acc_combined,
        mean_action_accuracy_phase1=acc1,
        mean_action_accuracy_phase2=acc2,
        mean_action_accuracy_phase3=acc3,
        spl=spl,
        mean_entropy=mean_ent,
        mean_optimal_entropy=mean_opt_ent,
        mean_jsd=mean_jsd,
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
    )


# =============================================================================
# Main Processing Pipeline
# =============================================================================


def process_model_two_coin_trajectories(
    trajectory_dir: Path,
    model_name: Optional[str] = None,
    batch_size: int = 20,
) -> TwoCoinModelTrajectoryResults:
    """Process all two-coin trajectories for a model.

    Trajectories are grouped by (collect_mode, grid_key, reasoning_effort). Coin
    and goal positions are loaded from the corresponding *_twocoin_layout.json.
    """
    if model_name is None:
        model_name = sanitize_label(trajectory_dir.name)

    print(f"\nProcessing two-coin model: {model_name}")
    print(f"Trajectory directory: {trajectory_dir}")

    grouped = discover_two_coin_trajectory_files(trajectory_dir)
    print(f"Found {len(grouped)} (collect_mode x grid x effort) combinations")

    if not grouped:
        raise ValueError(f"No two-coin trajectory files found in {trajectory_dir}")

    all_metrics: list[TwoCoinGridTrajectoryMetrics] = []
    all_traj_rows: list[dict[str, Any]] = []

    grid_keys = sorted(grouped.keys())
    total_batches = (len(grid_keys) + batch_size - 1) // batch_size

    for batch_idx, batch_keys in enumerate(batch_grid_keys(grid_keys, batch_size)):
        print(f"\n  Batch {batch_idx + 1}/{total_batches}: {len(batch_keys)} grids...")

        for grid_key in tqdm(batch_keys, desc=f"Batch {batch_idx + 1}", leave=False):
            traj_files = grouped[grid_key]
            if not traj_files:
                continue

            parsed = parse_two_coin_trajectory_filename(traj_files[0].name)
            if parsed is None:
                continue

            collect_mode = infer_collect_mode(traj_files[0])
            if collect_mode is None:
                continue

            grid_id = parsed["grid_id"]
            layout_file = find_two_coin_layout_file(traj_files[0], grid_id)

            if layout_file is None:
                print(f"  Warning: no layout file for {grid_key}, skipping.")
                continue

            layout_result = load_two_coin_grid_layout(layout_file)
            if layout_result is None:
                print(f"  Warning: could not parse layout for {grid_key}, skipping.")
                continue

            grid_layout, coin_pos_1, coin_pos_2, goal_pos = layout_result

            opt_to_c1, dist_to_c1, opt_to_c2, dist_to_c2, opt_to_goal, dist_to_goal = (
                compute_three_target_optimal_actions(grid_layout, coin_pos_1, coin_pos_2, goal_pos)
            )

            coin1_to_coin2_distance = dist_to_c2.get(coin_pos_1, -1)
            coin2_to_coin1_distance = dist_to_c1.get(coin_pos_2, -1)
            coin1_to_goal_distance = dist_to_goal.get(coin_pos_1, -1)
            coin2_to_goal_distance = dist_to_goal.get(coin_pos_2, -1)
            goal_reachable_from_coin1 = coin_pos_1 in dist_to_goal
            goal_reachable_from_coin2 = coin_pos_2 in dist_to_goal

            trajectory_category = "base"  # no transform variants exist yet for two-coin data

            _grid_optimal_total: int = -1
            _grid_distances_computed = False

            trajectories: list[TwoCoinLightweightTrajectory] = []
            for traj_file in traj_files:
                try:
                    with open(traj_file, "r") as f:
                        raw_data = json.load(f)
                    gp_raw = raw_data.get("grid_params", {})
                    start_coords = gp_raw.get("agent_start_coordinates", [0, 0])
                    agent_start = (int(start_coords[1]), int(start_coords[0]))
                    astar_c1_dist = dist_to_c1.get(agent_start, -1)
                    astar_c2_dist = dist_to_c2.get(agent_start, -1)
                except Exception:
                    agent_start = (0, 0)
                    astar_c1_dist = -1
                    astar_c2_dist = -1

                coin1_reachable_from_start = agent_start in dist_to_c1
                coin2_reachable_from_start = agent_start in dist_to_c2
                goal_reachable_from_start = agent_start in dist_to_goal
                if not (
                    coin1_reachable_from_start
                    and coin2_reachable_from_start
                    and goal_reachable_from_start
                    and goal_reachable_from_coin1
                    and goal_reachable_from_coin2
                ):
                    print(
                        f"  Warning: unreachable target in {grid_key} "
                        f"(start={agent_start}, coin1_reachable={coin1_reachable_from_start}, "
                        f"coin2_reachable={coin2_reachable_from_start}, "
                        f"goal_reachable={goal_reachable_from_start})"
                    )

                if not _grid_distances_computed:
                    _grid_optimal_total = compute_optimal_total_distance(
                        collect_mode, agent_start, coin_pos_1, coin_pos_2,
                        dist_to_c1, dist_to_c2, dist_to_goal,
                    )
                    _grid_distances_computed = True

                traj = load_two_coin_trajectory(
                    traj_file,
                    coin_pos_1=coin_pos_1,
                    coin_pos_2=coin_pos_2,
                    goal_pos=goal_pos,
                    collect_mode=collect_mode,
                    start_to_coin1_distance=astar_c1_dist,
                    start_to_coin2_distance=astar_c2_dist,
                    coin1_to_coin2_distance=coin1_to_coin2_distance,
                    coin2_to_coin1_distance=coin2_to_coin1_distance,
                    coin1_to_goal_distance=coin1_to_goal_distance,
                    coin2_to_goal_distance=coin2_to_goal_distance,
                    optimal_total_distance=_grid_optimal_total,
                    trajectory_category=trajectory_category,
                    coin1_reachable_from_start=coin1_reachable_from_start,
                    coin2_reachable_from_start=coin2_reachable_from_start,
                    goal_reachable_from_start=goal_reachable_from_start,
                    goal_reachable_from_coin1=goal_reachable_from_coin1,
                    goal_reachable_from_coin2=goal_reachable_from_coin2,
                    grid_layout=grid_layout,
                )
                if traj is not None:
                    trajectories.append(traj)
                    all_traj_rows.append(
                        compute_single_two_coin_trajectory_row(traj, opt_to_c1, opt_to_c2, opt_to_goal, grid_key)
                    )

            if not trajectories:
                continue

            metrics = compute_two_coin_grid_metrics(trajectories, opt_to_c1, opt_to_c2, opt_to_goal, grid_key)
            if metrics:
                all_metrics.append(metrics)

        gc.collect()

    df = pd.DataFrame([m.to_dict() for m in all_metrics])
    per_traj_df = pd.DataFrame(all_traj_rows)

    summary_df = _compute_two_coin_summary_by_size_density(df)
    overall = _compute_two_coin_overall_summary(df)

    return TwoCoinModelTrajectoryResults(
        model_name=model_name,
        df=df,
        summary_by_size_density=summary_df,
        overall_summary=overall,
        per_trajectory_df=per_traj_df,
    )


# =============================================================================
# Summaries
# =============================================================================


def _compute_two_coin_summary_by_size_density(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics grouped by collect_mode, grid_size, density, and effort."""
    if df.empty:
        return pd.DataFrame()

    group_cols = ["collect_mode", "grid_size", "density"]
    if "reasoning_effort" in df.columns:
        group_cols.append("reasoning_effort")

    return (
        df.groupby(group_cols)
        .agg(
            n_grids=("grid_id", "count"),
            mean_success_rate=("success_rate", "mean"),
            se_success_rate=("success_rate", "sem"),
            mean_coins_collected=("mean_coins_collected", "mean"),
            mean_action_accuracy=("mean_action_accuracy", "mean"),
            se_action_accuracy=("mean_action_accuracy", "sem"),
            mean_action_accuracy_phase1=("mean_action_accuracy_phase1", "mean"),
            mean_action_accuracy_phase2=("mean_action_accuracy_phase2", "mean"),
            mean_action_accuracy_phase3=("mean_action_accuracy_phase3", "mean"),
            mean_spl=("spl", "mean"),
            se_spl=("spl", "sem"),
            mean_entropy=("mean_entropy", "mean"),
            mean_jsd=("mean_jsd", "mean"),
            mean_ece=("ece", "mean"),
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
            mean_invalid_actions=("mean_invalid_actions", "mean"),
        )
        .reset_index()
    )


def _compute_two_coin_overall_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}

    summary: dict[str, Any] = {
        "n_grids": len(df),
        "overall_success_rate": float(df["success_rate"].mean()),
        "overall_mean_coins_collected": float(df["mean_coins_collected"].mean()),
        "overall_mean_action_accuracy": float(df["mean_action_accuracy"].mean()),
        "overall_mean_action_accuracy_phase1": float(df["mean_action_accuracy_phase1"].mean()),
        "overall_mean_action_accuracy_phase2": float(df["mean_action_accuracy_phase2"].mean()),
        "overall_mean_action_accuracy_phase3": float(df["mean_action_accuracy_phase3"].mean()),
        "overall_spl": float(df["spl"].mean()),
        "overall_mean_entropy": float(df["mean_entropy"].mean()),
        "overall_mean_jsd": float(df["mean_jsd"].mean()),
        "overall_ece": float(df["ece"].mean()),
        "overall_mean_cell_revisits": float(df["mean_cell_revisits"].mean()),
        "overall_mean_immediate_revisits": float(df["mean_immediate_revisits"].mean()),
        "overall_mean_coin_oscillations": float(df["mean_coin_oscillations"].mean()),
        "overall_mean_invalid_actions": float(df["mean_invalid_actions"].mean()),
    }

    if "collect_mode" in df.columns:
        by_mode = (
            df.groupby("collect_mode")[["success_rate", "mean_coins_collected", "mean_action_accuracy", "spl"]]
            .mean()
            .to_dict(orient="index")
        )
        summary["by_collect_mode"] = by_mode

    if "reasoning_effort" in df.columns:
        by_effort = (
            df.groupby("reasoning_effort")[["success_rate", "mean_action_accuracy", "spl"]]
            .mean()
            .to_dict(orient="index")
        )
        summary["by_reasoning_effort"] = by_effort

    return summary


# =============================================================================
# Save Results
# =============================================================================


def save_two_coin_results(
    results: TwoCoinModelTrajectoryResults,
    output_dir: Path,
) -> dict[str, Path]:
    model_dir = output_dir / results.model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}

    grid_path = model_dir / f"twocoin_trajectory_metrics_{results.model_name}.csv"
    results.df.to_csv(grid_path, index=False)
    output_paths["grid_metrics"] = grid_path
    print(f"  Saved: {grid_path}")

    if not results.per_trajectory_df.empty:
        traj_path = model_dir / f"twocoin_per_trajectory_{results.model_name}.csv"
        results.per_trajectory_df.to_csv(traj_path, index=False)
        output_paths["per_trajectory"] = traj_path
        print(f"  Saved: {traj_path}")

    summary_path = model_dir / "twocoin_summary_by_size_complexity.csv"
    results.summary_by_size_density.to_csv(summary_path, index=False)
    output_paths["summary"] = summary_path
    print(f"  Saved: {summary_path}")

    overall_path = model_dir / "twocoin_overall_summary.json"
    with open(overall_path, "w") as f:
        json.dump(results.overall_summary, f, indent=2)
    output_paths["overall"] = overall_path
    print(f"  Saved: {overall_path}")

    return output_paths


def print_two_coin_summary(results: TwoCoinModelTrajectoryResults) -> None:
    s = results.overall_summary
    print(f"\n{'='*60}")
    print(f"Model: {results.model_name}  |  Grids: {s.get('n_grids', 0)}")
    print(f"  Success rate:          {s.get('overall_success_rate', 0):.3f}  (mode-aware)")
    print(f"  Mean coins collected:  {s.get('overall_mean_coins_collected', 0):.3f}  (of 2)")
    print(f"  Action accuracy:       {s.get('overall_mean_action_accuracy', 0):.3f}  "
          f"(p1={s.get('overall_mean_action_accuracy_phase1', 0):.3f}, "
          f"p2={s.get('overall_mean_action_accuracy_phase2', 0):.3f}, "
          f"p3={s.get('overall_mean_action_accuracy_phase3', 0):.3f})")
    print(f"  SPL:                   {s.get('overall_spl', 0):.3f}")
    print(f"  Mean entropy:          {s.get('overall_mean_entropy', 0):.3f} bits")
    print(f"  Mean JSD:              {s.get('overall_mean_jsd', 0):.3f}")
    print(f"  ECE:                   {s.get('overall_ece', 0):.3f}")
    if "by_collect_mode" in s:
        print("  By collect mode:")
        for mode, vals in s["by_collect_mode"].items():
            print(f"    {mode}: success={vals.get('success_rate', 0):.3f}, "
                  f"coins={vals.get('mean_coins_collected', 0):.3f}, "
                  f"acc={vals.get('mean_action_accuracy', 0):.3f}")
    if "by_reasoning_effort" in s:
        print("  By reasoning effort:")
        for effort, vals in s["by_reasoning_effort"].items():
            print(f"    {effort}: success={vals.get('success_rate', 0):.3f}, "
                  f"acc={vals.get('mean_action_accuracy', 0):.3f}")
    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze two-coin navigation trajectories with a three-phase optimal policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        required=True,
        help="Directory containing two-coin trajectory JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/outputs/two_coin_trajectory_analysis",
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
                results = process_model_two_coin_trajectories(model_dir, batch_size=args.batch_size)
                save_two_coin_results(results, output_path)
                print_two_coin_summary(results)
            except Exception as e:
                print(f"Error processing {model_dir.name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        results = process_model_two_coin_trajectories(
            traj_path,
            model_name=args.model_name,
            batch_size=args.batch_size,
        )
        save_two_coin_results(results, output_path)
        print_two_coin_summary(results)


if __name__ == "__main__":
    main()
