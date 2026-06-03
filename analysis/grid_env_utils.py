"""MiniGrid environment utilities.

Contains functions that depend on MiniGrid environment objects or the project's
specific file-naming conventions:
- Optimal action computation (env-based Dijkstra)
- Cell and grid processing (logprob → metrics)
- Metadata loading and data classes
- Distance-to-goal analysis
- Isotransform analysis utilities
"""

import heapq
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.metrics import (
    ActionDist,
    OptimalActionSet,
    cross_entropy,
    jensen_shannon_divergence,
    compute_optimal_mass,
    optimal_entropy,
    shannon_entropy,
)
from analysis.analysis_utils import (
    ACTION_ID_TO_NAME,
    GridCoord,
    distribution_from_logprobs,
    grid_id_from_parts,
    grid_id_from_parts_isotransform,
    parse_filename,
    parse_filename_isotransform,
)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class GridMetadata:
    """Metadata for a single grid instance."""

    grid_size: int
    complexity: float
    instance_id: int
    policy_metadata: list[list[dict[str, Any]]]


@dataclass
class CellMetrics:
    """Computed metrics for a single grid cell."""

    grid_id: str
    grid_size: int
    complexity: float
    instance_id: int
    x: int
    y: int
    llm_action: int
    num_optimal_actions: int
    entropy_bits: float
    optimal_entropy_bits: float
    cross_entropy_bits: Optional[float]
    jsd: Optional[float]
    optimal_mass: float
    is_action_optimal: int
    action_probs: dict[str, float]
    distance_to_goal: int = 0

    def to_dict(self) -> dict[str, Any]:
        base = {
            "grid_id": self.grid_id,
            "grid_size": self.grid_size,
            "complexity": self.complexity,
            "instance_id": self.instance_id,
            "x": self.x,
            "y": self.y,
            "llm_action": self.llm_action,
            "num_optimal_actions": self.num_optimal_actions,
            "entropy_bits": self.entropy_bits,
            "optimal_entropy_bits": self.optimal_entropy_bits,
            "cross_entropy_bits": self.cross_entropy_bits,
            "jsd": self.jsd,
            "optimal_mass": self.optimal_mass,
            "is_action_optimal": self.is_action_optimal,
            "distance_to_goal": self.distance_to_goal,
        }
        base.update({f"p_{name}": prob for name, prob in self.action_probs.items()})
        return base


# =============================================================================
# Optimal Actions Computation (MiniGrid env-based)
# =============================================================================


def compute_optimal_actions(env: Any) -> list[list[OptimalActionSet]]:
    """Optimal actions for each cell via backward Dijkstra on a MiniGrid env."""
    base_env = getattr(env, "unwrapped", env)
    grid = base_env.grid
    goal: GridCoord = tuple(base_env.goal_pos)
    width, height = grid.width, grid.height

    def is_passable(x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        cell = grid.get(x, y)
        return (cell is None) or (getattr(cell, "can_overlap", lambda: False)())

    neighbors = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]

    distances: dict[GridCoord, float] = {goal: 0}
    heap: list[tuple[float, GridCoord]] = [(0, goal)]

    while heap:
        dist, (x, y) = heapq.heappop(heap)
        if dist > distances.get((x, y), float("inf")):
            continue
        for dx, dy, _ in neighbors:
            nx, ny = x + dx, y + dy
            if is_passable(nx, ny):
                new_dist = dist + 1
                if new_dist < distances.get((nx, ny), float("inf")):
                    distances[(nx, ny)] = new_dist
                    heapq.heappush(heap, (new_dist, (nx, ny)))

    optimal_actions: list[list[OptimalActionSet]] = [
        [set() for _ in range(width)] for _ in range(height)
    ]

    for y in range(height):
        for x in range(width):
            if not is_passable(x, y):
                continue
            current_dist = distances.get((x, y), float("inf"))
            if current_dist == float("inf"):
                continue
            for dx, dy, action in neighbors:
                nx, ny = x + dx, y + dy
                if is_passable(nx, ny):
                    if distances.get((nx, ny), float("inf")) == current_dist - 1:
                        optimal_actions[y][x].add(action)

    gx, gy = goal
    optimal_actions[gy][gx] = set()
    return optimal_actions


def compute_optimal_actions_and_distances(
    env: Any,
) -> tuple[list[list[OptimalActionSet]], list[list[int]]]:
    """Optimal actions and shortest-path distances for each cell in a MiniGrid env."""
    base_env = getattr(env, "unwrapped", env)
    grid = base_env.grid
    goal: GridCoord = tuple(base_env.goal_pos)
    width, height = grid.width, grid.height

    def is_passable(x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        cell = grid.get(x, y)
        return (cell is None) or (getattr(cell, "can_overlap", lambda: False)())

    neighbors = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]

    distances: dict[GridCoord, int] = {goal: 0}
    heap: list[tuple[int, GridCoord]] = [(0, goal)]

    while heap:
        dist, (x, y) = heapq.heappop(heap)
        if dist > distances.get((x, y), float("inf")):
            continue
        for dx, dy, _ in neighbors:
            nx, ny = x + dx, y + dy
            if is_passable(nx, ny):
                new_dist = dist + 1
                if new_dist < distances.get((nx, ny), float("inf")):
                    distances[(nx, ny)] = new_dist
                    heapq.heappush(heap, (new_dist, (nx, ny)))

    optimal_actions: list[list[OptimalActionSet]] = [
        [set() for _ in range(width)] for _ in range(height)
    ]
    distance_grid: list[list[int]] = [[-1 for _ in range(width)] for _ in range(height)]

    for y in range(height):
        for x in range(width):
            if not is_passable(x, y):
                continue
            current_dist = distances.get((x, y), float("inf"))
            if current_dist == float("inf"):
                continue
            distance_grid[y][x] = int(current_dist)
            for dx, dy, action in neighbors:
                nx, ny = x + dx, y + dy
                if is_passable(nx, ny):
                    if distances.get((nx, ny), float("inf")) == current_dist - 1:
                        optimal_actions[y][x].add(action)

    gx, gy = goal
    optimal_actions[gy][gx] = set()
    distance_grid[gy][gx] = 0
    return optimal_actions, distance_grid


def compute_optimal_distribution(optimal_actions: OptimalActionSet) -> ActionDist:
    """Uniform probability distribution over optimal actions."""
    if not optimal_actions:
        return {aid: 0.0 for aid in ACTION_ID_TO_NAME}
    prob = 1.0 / len(optimal_actions)
    return {aid: prob if aid in optimal_actions else 0.0 for aid in ACTION_ID_TO_NAME}


# =============================================================================
# Data Loading
# =============================================================================


def load_environments(dataset_path: str) -> dict[str, Any]:
    """Load grid environments from a pickle file."""
    with open(dataset_path, "rb") as f:
        return pickle.load(f)


def discover_metadata_files(metadata_dir: Path) -> list[Path]:
    """Find all metadata JSON files in a directory."""
    return sorted(metadata_dir.glob("*_metadata.json"))


def _load_single_metadata(fpath: Path) -> Optional[tuple[str, GridMetadata]]:
    try:
        grid_size, complexity, instance_id = parse_filename(fpath)
        key = grid_id_from_parts(grid_size, complexity, instance_id)

        with open(fpath, "r") as f:
            policy_metadata = json.load(f)

        return key, GridMetadata(
            grid_size=grid_size,
            complexity=complexity,
            instance_id=instance_id,
            policy_metadata=policy_metadata,
        )
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Warning: Skipping invalid file {fpath.name}: {e}")
        return None


def load_metadata_batch(
    metadata_files: list[Path], max_workers: int = 8, show_progress: bool = False
) -> dict[str, GridMetadata]:
    """Load multiple metadata files in parallel."""
    metadata_dict: dict[str, GridMetadata] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_load_single_metadata, f): f for f in metadata_files}
        completed = as_completed(futures)
        if show_progress:
            completed = tqdm(completed, total=len(futures), desc="Loading metadata", leave=False)
        for future in completed:
            result = future.result()
            if result is not None:
                key, metadata = result
                metadata_dict[key] = metadata

    return metadata_dict


def load_single_grid_metadata(
    grid_id: str, metadata_dir: Path
) -> Optional[GridMetadata]:
    """Load metadata for a specific grid by ID string."""
    metadata_file = metadata_dir / f"{grid_id}_metadata.json"
    if not metadata_file.exists():
        return None
    result = _load_single_metadata(metadata_file)
    return result[1] if result else None


# =============================================================================
# Isotransform Metadata Loading
# =============================================================================


@dataclass
class IsotransformGridMetadata:
    """Metadata for a single isotransform grid."""

    grid_size: int
    complexity: float
    instance_id: int
    transform_type: str
    policy_metadata: list[list[dict]]


def _load_single_metadata_isotransform(
    fpath: Path,
) -> Optional[tuple[str, IsotransformGridMetadata]]:
    try:
        grid_size, complexity, instance_id, transform_type = parse_filename_isotransform(fpath)
        key = grid_id_from_parts_isotransform(grid_size, complexity, instance_id, transform_type)

        with open(fpath, "r") as f:
            policy_metadata = json.load(f)

        return key, IsotransformGridMetadata(
            grid_size=grid_size,
            complexity=complexity,
            instance_id=instance_id,
            transform_type=transform_type,
            policy_metadata=policy_metadata,
        )
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Warning: Skipping invalid isotransform file {fpath.name}: {e}")
        return None


def load_metadata_batch_isotransform(
    metadata_files: list[Path], max_workers: int = 8, show_progress: bool = False
) -> dict[str, IsotransformGridMetadata]:
    """Load multiple isotransform metadata files in parallel."""
    metadata_dict: dict[str, IsotransformGridMetadata] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_load_single_metadata_isotransform, f): f for f in metadata_files
        }
        completed = as_completed(futures)
        if show_progress:
            completed = tqdm(completed, total=len(futures), desc="Loading metadata", leave=False)
        for future in completed:
            result = future.result()
            if result is not None:
                key, metadata = result
                metadata_dict[key] = metadata

    return metadata_dict


# =============================================================================
# Cell and Grid Processing
# =============================================================================


def process_cell(
    cell: Any,
    optimal_set: OptimalActionSet,
    grid_id: str,
    metadata: GridMetadata,
    x: int,
    y: int,
    distance_to_goal: int = 0,
) -> Optional[CellMetrics]:
    """Process a single cell's logprobs and compute its metrics."""
    if not isinstance(cell, dict) or not optimal_set:
        return None

    dist = distribution_from_logprobs(cell.get("logprobs"))
    if dist is None:
        return None

    num_optimal = len(optimal_set)
    action_probs = {
        ACTION_ID_TO_NAME[aid].lower(): dist.get(aid, 0.0) for aid in ACTION_ID_TO_NAME
    }

    return CellMetrics(
        grid_id=grid_id,
        grid_size=metadata.grid_size,
        complexity=metadata.complexity,
        instance_id=metadata.instance_id,
        x=x,
        y=y,
        llm_action=cell.get("llm_response", -1),
        num_optimal_actions=num_optimal,
        entropy_bits=shannon_entropy(dist),
        optimal_entropy_bits=optimal_entropy(num_optimal),
        cross_entropy_bits=cross_entropy(optimal_set, dist),
        jsd=jensen_shannon_divergence(optimal_set, dist),
        optimal_mass=compute_optimal_mass(optimal_set, dist),
        is_action_optimal=int(cell.get("llm_response", -1) in optimal_set),
        action_probs=action_probs,
        distance_to_goal=distance_to_goal,
    )


def process_grid(grid_id: str, env: Any, metadata: GridMetadata) -> list[CellMetrics]:
    """Process all cells in a grid and return their metrics."""
    results: list[CellMetrics] = []
    optimal_actions, distance_grid = compute_optimal_actions_and_distances(env)

    for y, row in enumerate(metadata.policy_metadata):
        for x, cell in enumerate(row):
            cell_result = process_cell(
                cell=cell,
                optimal_set=optimal_actions[y][x],
                grid_id=grid_id,
                metadata=metadata,
                x=x,
                y=y,
                distance_to_goal=distance_grid[y][x],
            )
            if cell_result:
                results.append(cell_result)

    return results


def compute_grid_mean_cross_entropy(
    grid_id: str, env: Any, metadata: GridMetadata
) -> Optional[float]:
    """Mean cross-entropy across all valid cells in a grid."""
    cell_metrics = process_grid(grid_id, env, metadata)
    valid_ce = [m.cross_entropy_bits for m in cell_metrics if m.cross_entropy_bits is not None]
    return sum(valid_ce) / len(valid_ce) if valid_ce else None


def compute_grid_mean_jsd(
    grid_id: str, env: Any, metadata: GridMetadata
) -> Optional[float]:
    """Mean Jensen-Shannon divergence across all valid cells in a grid."""
    optimal_actions_grid = compute_optimal_actions(env)
    jsd_values: list[float] = []

    for y, row in enumerate(metadata.policy_metadata):
        for x, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue
            optimal_set = optimal_actions_grid[y][x]
            if not optimal_set:
                continue
            dist = distribution_from_logprobs(cell.get("logprobs"))
            if dist is None:
                continue
            jsd = jensen_shannon_divergence(optimal_set, dist)
            if jsd is not None:
                jsd_values.append(jsd)

    return sum(jsd_values) / len(jsd_values) if jsd_values else None


# =============================================================================
# Distance-to-Goal Analysis
# =============================================================================


@dataclass
class DistanceToGoalMetrics:
    """Metrics relating uncertainty to distance from goal."""

    correlation_entropy_distance: float
    correlation_divergence_distance: float
    correlation_accuracy_distance: float
    mean_entropy_by_distance: dict[int, float]
    accuracy_by_distance: dict[int, float]
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "correlation_entropy_distance": self.correlation_entropy_distance,
            "correlation_divergence_distance": self.correlation_divergence_distance,
            "correlation_accuracy_distance": self.correlation_accuracy_distance,
            "n_samples": self.n_samples,
        }


def compute_distance_to_goal_metrics(
    df: pd.DataFrame,
    entropy_col: str = "entropy_bits",
    divergence_col: str = "jsd",
    correct_col: str = "is_action_optimal",
    distance_col: str = "distance_to_goal",
) -> DistanceToGoalMetrics:
    """Correlations and per-distance summaries relating uncertainty to goal distance."""
    required_cols = [entropy_col, divergence_col, correct_col, distance_col]
    df_clean = df[required_cols].dropna()
    df_clean = df_clean[df_clean[distance_col] >= 0]

    if len(df_clean) == 0:
        return DistanceToGoalMetrics(
            correlation_entropy_distance=0.0,
            correlation_divergence_distance=0.0,
            correlation_accuracy_distance=0.0,
            mean_entropy_by_distance={},
            accuracy_by_distance={},
            n_samples=0,
        )

    grouped = df_clean.groupby(distance_col).agg(
        {entropy_col: "mean", correct_col: "mean"}
    )

    return DistanceToGoalMetrics(
        correlation_entropy_distance=df_clean[entropy_col].corr(df_clean[distance_col]),
        correlation_divergence_distance=df_clean[divergence_col].corr(df_clean[distance_col]),
        correlation_accuracy_distance=df_clean[correct_col].corr(df_clean[distance_col]),
        mean_entropy_by_distance=grouped[entropy_col].to_dict(),
        accuracy_by_distance=grouped[correct_col].to_dict(),
        n_samples=len(df_clean),
    )


def compute_distance_summary(
    df: pd.DataFrame,
    entropy_col: str = "entropy_bits",
    divergence_col: str = "jsd",
    correct_col: str = "is_action_optimal",
    distance_col: str = "distance_to_goal",
    model_col: Optional[str] = None,
) -> pd.DataFrame:
    """Summary statistics by distance to goal (and optionally by model)."""
    df_clean = df.copy()
    df_clean = df_clean[df_clean[distance_col] >= 0]

    group_cols = [distance_col]
    if model_col and model_col in df_clean.columns:
        group_cols = [model_col, distance_col]

    return (
        df_clean.groupby(group_cols)
        .agg(
            mean_entropy=(entropy_col, "mean"),
            std_entropy=(entropy_col, "std"),
            mean_divergence=(divergence_col, "mean"),
            accuracy=(correct_col, "mean"),
            n_samples=(entropy_col, "count"),
        )
        .reset_index()
    )


# =============================================================================
# Isotransform Analysis Utilities
# =============================================================================

TRANSFORM_TYPES = [
    "baseline",
    "ReflectEnv",
    "RotateEnv",
    "StartGoalSwap",
    "TransposeEnv",
]


@dataclass
class TransformRegressionResult:
    """OLS regression result for transform effects."""

    outcome_variable: str
    coefficients: dict[str, float]
    std_errors: dict[str, float]
    p_values: dict[str, float]
    r_squared: float
    n_samples: int
    baseline_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome_variable": self.outcome_variable,
            "coefficients": self.coefficients,
            "std_errors": self.std_errors,
            "p_values": self.p_values,
            "r_squared": self.r_squared,
            "n_samples": self.n_samples,
            "baseline_mean": self.baseline_mean,
        }

    def get_significant_transforms(self, alpha: float = 0.05) -> list[str]:
        return [
            t for t, p in self.p_values.items()
            if t.startswith("transform_type") and p < alpha
        ]


def run_transform_regression(
    df: pd.DataFrame,
    outcome_col: str,
    transform_col: str = "transform_type",
    control_cols: Optional[list[str]] = None,
) -> Optional[TransformRegressionResult]:
    """OLS regression: outcome ~ transform_type + controls (baseline as reference)."""
    from scipy import stats

    if control_cols is None:
        control_cols = ["grid_size", "complexity"]

    df = df.copy()
    if "baseline" not in df[transform_col].values:
        return None

    transform_dummies = pd.get_dummies(df[transform_col], prefix="transform_type", drop_first=False)
    if "transform_type_baseline" in transform_dummies.columns:
        transform_dummies = transform_dummies.drop("transform_type_baseline", axis=1)

    X_cols = list(transform_dummies.columns) + control_cols
    X = pd.concat([transform_dummies, df[control_cols]], axis=1).dropna()
    X_values = X.values.astype(np.float64)
    X_with_intercept = np.column_stack([np.ones(len(X)), X_values])
    y = df.loc[X.index, outcome_col].values.astype(np.float64)

    if len(y) < len(X_cols) + 2:
        return None

    n, k = X_with_intercept.shape

    try:
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        y_pred = X_with_intercept @ beta
        residuals = y - y_pred

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        mse = ss_res / (n - k)
        try:
            var_beta = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            se_beta = np.sqrt(np.diag(var_beta))
            t_stats = beta / se_beta
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))
        except np.linalg.LinAlgError:
            se_beta = np.full(k, np.nan)
            p_values = np.full(k, np.nan)

        coef_names = ["intercept"] + X_cols
        return TransformRegressionResult(
            outcome_variable=outcome_col,
            coefficients=dict(zip(coef_names, beta)),
            std_errors=dict(zip(coef_names, se_beta)),
            p_values=dict(zip(coef_names, p_values)),
            r_squared=r_squared,
            n_samples=n,
            baseline_mean=df[df[transform_col] == "baseline"][outcome_col].mean(),
        )

    except np.linalg.LinAlgError:
        return None


def compute_transform_summary(
    df: pd.DataFrame,
    transform_col: str = "transform_type",
    metrics: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Summary statistics by transform type."""
    if metrics is None:
        metrics = ["entropy_bits", "jsd", "is_action_optimal"]

    agg_dict = {}
    for m in metrics:
        if m in df.columns:
            agg_dict[f"{m}_mean"] = (m, "mean")
            agg_dict[f"{m}_std"] = (m, "std")
    agg_dict["n_samples"] = (metrics[0], "count")

    return df.groupby(transform_col).agg(**agg_dict).reset_index()


def compute_transform_diff_from_baseline(
    df: pd.DataFrame,
    metric_col: str,
    transform_col: str = "transform_type",
    group_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Difference from baseline for each transform, matched within group_cols."""
    if group_cols is None:
        group_cols = ["grid_size", "complexity", "instance_id"]

    baseline = df[df[transform_col] == "baseline"].rename(
        columns={metric_col: "baseline_value"}
    )[group_cols + ["baseline_value"]]

    df_merged = df.merge(baseline, on=group_cols, how="inner")
    df_merged["diff_from_baseline"] = df_merged[metric_col] - df_merged["baseline_value"]
    return df_merged


def compute_distance_summary_by_transform(
    df: pd.DataFrame,
    entropy_col: str = "entropy_bits",
    divergence_col: str = "jsd",
    correct_col: str = "is_action_optimal",
    distance_col: str = "distance_to_goal",
    transform_col: str = "transform_type",
) -> pd.DataFrame:
    """Summary statistics grouped by transform type and distance to goal."""
    df_clean = df[df[distance_col] >= 0].copy()

    return (
        df_clean.groupby([transform_col, distance_col])
        .agg(
            mean_entropy=(entropy_col, "mean"),
            std_entropy=(entropy_col, "std"),
            mean_divergence=(divergence_col, "mean"),
            std_divergence=(divergence_col, "std"),
            accuracy=(correct_col, "mean"),
            error_rate=(correct_col, lambda x: 1 - x.mean()),
            n_samples=(entropy_col, "count"),
        )
        .reset_index()
    )
