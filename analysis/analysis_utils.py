"""Core shared trajectory utilities.

Defines:
- Action constants and type aliases
- Trajectory data classes (TrajectoryGridParams, TrajectoryStep, LightweightTrajectory)
- Trajectory parsing (Dijkstra on text grids, action accuracy, SPL)
- File-naming utilities (parse_filename, sanitize_label, etc.)
- Logprob parsing (distribution_from_logprobs)

Re-exports from metrics.py for backward compatibility:
- All entropy / information-theory functions
- All statistical analysis classes and functions
- CalibrationMetrics, UncertaintyAccuracyMetrics and related helpers
"""

import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Re-export everything from metrics so that
# `from analysis.analysis_utils import shannon_entropy` still works.
from analysis.metrics import (  # noqa: F401
    LOGPROB_EPS,
    ActionDist,
    ActionID,
    CalibrationMetrics,
    ControlledAnalysisResult,
    CorrelationResult,
    OptimalActionSet,
    RegressionResult,
    UncertaintyAccuracyMetrics,
    _compute_auroc_manual,
    _compute_ece,
    compute_calibration_metrics,
    compute_correlation,
    compute_correlations_for_columns,
    compute_optimal_mass,
    compute_partial_correlations,
    compute_selective_prediction_curve,
    compute_stratified_summary,
    compute_uncertainty_accuracy_metrics,
    compute_within_stratum_correlations,
    cross_entropy,
    format_correlation_report,
    jensen_shannon_divergence,
    kl_divergence,
    optimal_entropy,
    residualize,
    run_controlled_analysis,
    run_ols_regression,
    shannon_entropy,
)

# =============================================================================
# Constants
# =============================================================================

ACTION_NAME_TO_ID = {"LEFT": 0, "RIGHT": 1, "UP": 2, "DOWN": 3}
ACTION_ID_TO_NAME = {v: k for k, v in ACTION_NAME_TO_ID.items()}
ACTION_NAMES_UPPER = set(ACTION_NAME_TO_ID.keys())

GridCoord = tuple[int, int]


# =============================================================================
# Trajectory Data Classes
# =============================================================================


@dataclass
class TrajectoryGridParams:
    """Essential grid parameters extracted from a trajectory file."""

    grid_size: int
    complexity: float
    grid_id: int
    astar_distance: int
    agent_start: tuple[int, int]
    goal: tuple[int, int]


@dataclass
class TrajectoryStep:
    """A single step in a trajectory."""

    step_id: int
    agent_position: tuple[int, int]
    agent_action: str  # "UP", "DOWN", "LEFT", "RIGHT"


@dataclass
class LightweightTrajectory:
    """Memory-efficient trajectory with only essential fields."""

    grid_params: TrajectoryGridParams
    steps: list[TrajectoryStep]
    reached_goal: bool
    transform_type: str = "base"

    @property
    def trajectory_length(self) -> int:
        return len(self.steps)


# =============================================================================
# Trajectory Parsing Utilities
# =============================================================================


def extract_agent_position_from_grid_state(grid_state: list[str]) -> tuple[int, int]:
    """Extract agent (x, y) from grid-state text lines.

    Format: ['  0 1 2 ...', '0 # # # ...', '1 # A _ ...', ...]
    Row index is Y, column index is X.
    """
    for row_idx, row in enumerate(grid_state):
        if row_idx == 0:
            continue
        parts = row.split()
        if len(parts) < 2:
            continue
        y = int(parts[0])
        for col_idx, cell in enumerate(parts[1:], start=0):
            if cell == "A":
                return (col_idx, y)
    return (-1, -1)


def compute_optimal_actions_from_text_grid(
    grid_layout: list[list[str]],
    goal: tuple[int, int],
) -> tuple[dict[tuple[int, int], OptimalActionSet], dict[tuple[int, int], int]]:
    """Backward Dijkstra on a text grid layout.

    Args:
        grid_layout: 2D grid where '#' is wall, anything else is passable.
        goal: Goal position (x, y).

    Returns:
        (optimal_actions dict, distances dict)
    """
    height = len(grid_layout)
    width = len(grid_layout[0]) if height > 0 else 0

    def is_passable(x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        return grid_layout[y][x] != "#"

    neighbors = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]

    distances: dict[tuple[int, int], int] = {goal: 0}
    heap_queue: list[tuple[int, tuple[int, int]]] = [(0, goal)]

    while heap_queue:
        dist, (x, y) = heapq.heappop(heap_queue)
        if dist > distances.get((x, y), float("inf")):
            continue
        for dx, dy, _ in neighbors:
            nx, ny = x + dx, y + dy
            if is_passable(nx, ny):
                new_dist = dist + 1
                if new_dist < distances.get((nx, ny), float("inf")):
                    distances[(nx, ny)] = new_dist
                    heapq.heappush(heap_queue, (new_dist, (nx, ny)))

    optimal_actions: dict[tuple[int, int], OptimalActionSet] = {}

    for y in range(height):
        for x in range(width):
            if not is_passable(x, y):
                continue
            current_dist = distances.get((x, y), float("inf"))
            if current_dist == float("inf"):
                continue
            optimal_set: OptimalActionSet = set()
            for dx, dy, action in neighbors:
                nx, ny = x + dx, y + dy
                if is_passable(nx, ny):
                    if distances.get((nx, ny), float("inf")) == current_dist - 1:
                        optimal_set.add(action)
            optimal_actions[(x, y)] = optimal_set

    optimal_actions[goal] = set()
    return optimal_actions, distances


def compute_trajectory_action_accuracy(
    trajectory: LightweightTrajectory,
    optimal_actions: dict[tuple[int, int], OptimalActionSet],
) -> float:
    """Acc(τ) = (1/T) * Σ 1(a_t ∈ π*(s_t))."""
    if not trajectory.steps:
        return 0.0

    correct = 0
    for step in trajectory.steps:
        action_id = ACTION_NAME_TO_ID.get(step.agent_action.upper())
        if action_id is not None and action_id in optimal_actions.get(step.agent_position, set()):
            correct += 1

    return correct / len(trajectory.steps)


def compute_spl(
    trajectories: list[LightweightTrajectory],
    optimal_path_length: int,
) -> float:
    """Success weighted by Path Length: (1/N) * Σ (1_S * L*) / max(L*, L)."""
    if not trajectories or optimal_path_length <= 0:
        return 0.0

    total = 0.0
    for traj in trajectories:
        if traj.reached_goal:
            total += optimal_path_length / max(optimal_path_length, traj.trajectory_length)

    return total / len(trajectories)


# =============================================================================
# File Parsing Utilities
# =============================================================================


def sanitize_label(value: str) -> str:
    """Sanitize a string for use as a file/directory label."""
    if not value:
        return "model"
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def parse_filename(filepath: Path) -> tuple[int, float, int]:
    """Extract (grid_size, complexity, instance_id) from filename.

    Expected format: grid_size{N}_complexity{X.XX}_{NNNN}_metadata.json
    """
    name = filepath.stem
    parts = name.split("_")
    try:
        grid_size = int(parts[1].replace("size", ""))
        complexity = float(parts[2].replace("complexity", ""))
        instance_id = int(parts[3])
        return grid_size, complexity, instance_id
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid filename format: {filepath.name}") from e


def grid_id_from_parts(grid_size: int, complexity: float, instance_id: int) -> str:
    return f"grid_size{grid_size}_complexity{complexity:.2f}_{instance_id:04d}"


def parse_filename_isotransform(filepath: Path) -> tuple[int, float, int, str]:
    """Extract (grid_size, complexity, instance_id, transform_type) from filename.

    Baseline files: grid_size{N}_complexity{X.XX}_{NNNN}_metadata.json → "baseline"
    Transform files: grid_size{N}_complexity{X.XX}_{NNNN}_{Transform}_metadata.json
    """
    name = filepath.stem
    parts = name.split("_")
    try:
        grid_size = int(parts[1].replace("size", ""))
        complexity = float(parts[2].replace("complexity", ""))
        instance_id = int(parts[3])
        transform_type = parts[4] if len(parts) > 5 and parts[4] != "metadata" else "baseline"
        return grid_size, complexity, instance_id, transform_type
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid isotransform filename format: {filepath.name}") from e


def grid_id_from_parts_isotransform(
    grid_size: int, complexity: float, instance_id: int, transform_type: str
) -> str:
    base_id = f"grid_size{grid_size}_complexity{complexity:.2f}_{instance_id:04d}"
    return base_id if transform_type == "baseline" else f"{base_id}_{transform_type}"


# =============================================================================
# Logprob Parsing
# =============================================================================


def normalize_action_token(token: Optional[str]) -> str:
    """Normalize an action token string to uppercase."""
    if token is None:
        return ""
    return token.strip().strip('"').strip("'").upper()


def is_action_token(token: str) -> bool:
    return normalize_action_token(token) in ACTION_NAMES_UPPER


def find_action_token_entry(logprobs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Find the token entry for the action value in a logprobs list."""
    if not logprobs:
        return None

    n = len(logprobs)

    # Strategy 1: structured search for "action": "<VALUE>" pattern
    for i, entry in enumerate(logprobs):
        token = entry.get("token", "")
        if "action" not in token.lower():
            continue

        colon_idx = None
        for j in range(i + 1, min(i + 5, n)):
            if ":" in logprobs[j].get("token", ""):
                colon_idx = j
                break

        if colon_idx is None:
            continue

        for k in range(colon_idx + 1, min(colon_idx + 5, n)):
            candidate_token = logprobs[k].get("token", "")
            if candidate_token.strip() in ("", '"', "'"):
                continue
            if is_action_token(candidate_token):
                return logprobs[k]
            if "}" in candidate_token:
                break

    # Strategy 2: fallback — find any high-confidence standalone action token
    for entry in logprobs:
        token = entry.get("token", "")
        if is_action_token(token):
            logprob = entry.get("logprob")
            if logprob is not None and logprob > -1.0:
                return entry

    return None


def distribution_from_logprobs(
    logprobs: Optional[list[dict[str, Any]]],
) -> Optional[ActionDist]:
    """Extract action probability distribution from a logprobs list."""
    if not logprobs:
        return None

    token_entry = find_action_token_entry(logprobs)
    if not token_entry:
        return None

    entries: dict[ActionID, float] = {}

    def register_action(token_value: Optional[str], logprob: Optional[float]) -> None:
        action = normalize_action_token(token_value)
        if action in ACTION_NAME_TO_ID and logprob is not None:
            action_id = ACTION_NAME_TO_ID[action]
            entries[action_id] = max(entries.get(action_id, -math.inf), logprob)

    register_action(token_entry.get("token"), token_entry.get("logprob"))
    for candidate in token_entry.get("top_logprobs") or []:
        register_action(candidate.get("token"), candidate.get("logprob"))

    if not entries:
        return None

    max_logprob = max(entries.values())
    probs = {aid: math.exp(lp - max_logprob) for aid, lp in entries.items()}
    total = sum(probs.values())
    if total <= 0:
        return None

    return {aid: probs.get(aid, 0.0) / total for aid in ACTION_ID_TO_NAME}
