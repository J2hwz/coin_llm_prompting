"""Shared matplotlib helpers for publication-quality trajectory analysis figures.

No dependencies on project analysis modules — safe to import from anywhere.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Style Constants
# =============================================================================

PAPER_RC = {
    "font.family": "serif",
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,
    "figure.titlesize": 20,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
}

MODEL_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
]


def setup_paper_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(PAPER_RC)


# =============================================================================
# Figure Saving
# =============================================================================


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    """Save figure to PNG (300 dpi) and PDF subfolders under output_dir."""
    png_dir = output_dir / "png"
    pdf_dir = output_dir / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    png_path = png_dir / f"{filename}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_dir / f"{filename}.pdf", bbox_inches="tight")

    return png_path


# =============================================================================
# Trajectory Analysis Plots
# =============================================================================


def plot_metrics_by_size_density(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> dict[str, Path]:
    """Line plots of key metrics by grid size and density."""
    setup_paper_style()
    output_paths = {}

    metrics = [
        ("goal_success_rate", "Goal Success Rate"),
        ("mean_action_accuracy", "Action Accuracy"),
        ("spl", "SPL"),
        ("mean_jsd", "Mean JSD"),
        ("ece", "ECE"),
    ]

    for metric_col, metric_label in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        size_summary = df.groupby("grid_size")[metric_col].agg(["mean", "sem"])
        axes[0].errorbar(
            size_summary.index, size_summary["mean"], yerr=size_summary["sem"],
            marker="o", capsize=3, color=MODEL_COLORS[0],
        )
        axes[0].set_xlabel("Grid Size")
        axes[0].set_ylabel(metric_label)
        axes[0].set_title(f"{metric_label} by Grid Size")
        axes[0].grid(True, alpha=0.3)

        density_summary = df.groupby("density")[metric_col].agg(["mean", "sem"])
        axes[1].errorbar(
            density_summary.index, density_summary["mean"], yerr=density_summary["sem"],
            marker="o", capsize=3, color=MODEL_COLORS[1],
        )
        axes[1].set_xlabel("Density")
        axes[1].set_ylabel(metric_label)
        axes[1].set_title(f"{metric_label} by Density")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        output_path = save_figure(fig, output_dir, f"{metric_col}_by_size_complexity")
        plt.close(fig)
        output_paths[metric_col] = output_path

    # Entropy: plot model vs optimal together
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, group_col, xlabel in zip(
        axes, ["grid_size", "density"], ["Grid Size", "Density"]
    ):
        ent = df.groupby(group_col)["mean_entropy"].agg(["mean", "sem"])
        opt = df.groupby(group_col)["mean_optimal_entropy"].agg(["mean", "sem"])

        ax.errorbar(ent.index, ent["mean"], yerr=ent["sem"],
                    marker="o", capsize=3, color=MODEL_COLORS[0], label="Model")
        ax.errorbar(opt.index, opt["mean"], yerr=opt["sem"],
                    marker="s", capsize=3, color=MODEL_COLORS[2], linestyle="--", label="Optimal")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean Entropy (bits)")
        ax.set_title(f"Entropy by {xlabel}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, frameon=False)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path = save_figure(fig, output_dir, "mean_entropy_by_size_complexity")
    plt.close(fig)
    output_paths["mean_entropy"] = output_path

    return output_paths


def plot_metrics_by_distance(
    distance_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    smoothing_window: int = 5,
    max_distance: int = 50,
) -> Path:
    """Accuracy and JSD vs distance to goal with rolling-average smoothing."""
    if distance_df.empty:
        return output_dir / "png" / "metrics_by_distance.png"

    setup_paper_style()

    df_sorted = distance_df.sort_values("distance_to_goal").copy()
    for col, new_col in [
        ("mean_entropy", "entropy_smooth"),
        ("mean_jsd", "jsd_smooth"),
        ("accuracy", "accuracy_smooth"),
    ]:
        df_sorted[new_col] = df_sorted[col].rolling(window=smoothing_window, center=True).mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(df_sorted["distance_to_goal"], df_sorted["accuracy"],
                    alpha=0.3, s=15, color=MODEL_COLORS[2], label="Raw")
    axes[0].plot(df_sorted["distance_to_goal"], df_sorted["accuracy_smooth"],
                 linewidth=2, color=MODEL_COLORS[2],
                 label=f"Smoothed (window={smoothing_window})")
    axes[0].set_xlabel("Distance to Goal")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower left", frameon=False)

    axes[1].scatter(df_sorted["distance_to_goal"], df_sorted["mean_jsd"],
                    alpha=0.3, s=15, color=MODEL_COLORS[1], label="Raw")
    axes[1].plot(df_sorted["distance_to_goal"], df_sorted["jsd_smooth"],
                 linewidth=2, color=MODEL_COLORS[1],
                 label=f"Smoothed (window={smoothing_window})")
    axes[1].set_xlabel("Distance to Goal")
    axes[1].set_ylabel("Mean JSD")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, loc="upper left", frameon=False)

    tick_values = list(range(0, max_distance + 1, 10))
    tick_labels = [str(t) if t < max_distance else f"{max_distance}+" for t in tick_values]
    for ax in axes:
        ax.set_xlim(-1, max_distance + 3)
        ax.set_xticks(tick_values)
        ax.set_xticklabels(tick_labels)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path = save_figure(fig, output_dir, "metrics_by_distance")
    plt.close(fig)
    return output_path


def plot_capability_vs_uncertainty(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> Path:
    """Scatter plots: accuracy vs entropy, accuracy vs JSD, SPL vs ECE."""
    setup_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].scatter(df["mean_entropy"], df["mean_action_accuracy"],
                    alpha=0.5, s=20, color=MODEL_COLORS[0])
    axes[0].set_xlabel("Mean Entropy (bits)")
    axes[0].set_ylabel("Action Accuracy")
    axes[0].set_title("Accuracy vs Entropy")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(df["mean_jsd"], df["mean_action_accuracy"],
                    alpha=0.5, s=20, color=MODEL_COLORS[1])
    axes[1].set_xlabel("Mean JSD")
    axes[1].set_ylabel("Action Accuracy")
    axes[1].set_title("Accuracy vs JSD")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(df["ece"], df["spl"],
                    alpha=0.5, s=20, color=MODEL_COLORS[2])
    axes[2].set_xlabel("ECE")
    axes[2].set_ylabel("SPL")
    axes[2].set_title("SPL vs ECE")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Capability vs Uncertainty", fontweight="bold")
    plt.tight_layout()

    output_path = save_figure(fig, output_dir, "capability_vs_uncertainty")
    plt.close(fig)
    return output_path


def plot_heatmaps(
    summary_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> Path:
    """2×2 heatmaps of goal success, accuracy, SPL, and JSD by grid_size × density."""
    setup_paper_style()

    metrics = [
        ("mean_goal_success", "Goal Success Rate"),
        ("mean_action_accuracy", "Action Accuracy"),
        ("mean_spl", "SPL"),
        ("mean_jsd", "Mean JSD"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, (metric_col, metric_label) in enumerate(metrics):
        pivot = summary_df.pivot(index="density", columns="grid_size", values=metric_col)
        im = axes[idx].imshow(pivot.values, cmap="RdYlGn", aspect="auto")
        axes[idx].set_xticks(range(len(pivot.columns)))
        axes[idx].set_xticklabels(pivot.columns)
        axes[idx].set_yticks(range(len(pivot.index)))
        axes[idx].set_yticklabels([f"{c:.1f}" for c in pivot.index])
        axes[idx].set_xlabel("Grid Size")
        axes[idx].set_ylabel("Density")
        axes[idx].set_title(metric_label)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    axes[idx].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        plt.colorbar(im, ax=axes[idx])

    plt.tight_layout()
    output_path = save_figure(fig, output_dir, "metrics_heatmaps")
    plt.close(fig)
    return output_path


def plot_distance_density_heatmap(
    state_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    metric: str = "is_optimal",
    metric_label: str = "Action Accuracy",
    n_distance_bins: int = 10,
    max_distance: int = 50,
) -> Path:
    """Multi-panel heatmap: metric by (distance bin × density) for each grid size.

    One column per unique grid size. X-axis = density, Y-axis = binned distance.
    """
    if state_df.empty or metric not in state_df.columns:
        return output_dir / "png" / f"heatmap_{metric}_by_distance_complexity.png"

    setup_paper_style()

    grid_sizes = sorted(state_df["grid_size"].unique())
    n_sizes = len(grid_sizes)

    if n_sizes == 0:
        return output_dir / "png" / f"heatmap_{metric}_by_distance_complexity.png"

    df = state_df.copy()
    df["distance_capped"] = df["distance_to_goal"].clip(upper=max_distance + 0.5)

    bin_width = max_distance // n_distance_bins
    distance_bins = list(range(0, max_distance, bin_width)) + [max_distance, max_distance + 1]

    distance_labels = []
    for i in range(len(distance_bins) - 1):
        start, end = distance_bins[i], distance_bins[i + 1]
        distance_labels.append(f"{max_distance}+" if start == max_distance else f"{start}-{end}")

    df["distance_bin"] = pd.cut(
        df["distance_capped"], bins=distance_bins, labels=distance_labels, include_lowest=True
    )

    density_values = sorted(df["density"].unique())

    fig, axes = plt.subplots(1, n_sizes, figsize=(2.5 * n_sizes, 6), sharey=True)
    if n_sizes == 1:
        axes = [axes]

    pivots = []
    all_values = []
    for size in grid_sizes:
        pivot = (
            df[df["grid_size"] == size]
            .groupby(["distance_bin", "density"], observed=False)[metric]
            .mean()
            .unstack()
            .reindex(columns=density_values)
            .reindex(distance_labels)
        )
        pivots.append(pivot)
        valid_vals = pivot.values[~np.isnan(pivot.values)]
        if len(valid_vals) > 0:
            all_values.extend(valid_vals)

    vmin, vmax = (np.min(all_values), np.max(all_values)) if all_values else (0, 1)

    for idx, (size, pivot) in enumerate(zip(grid_sizes, pivots)):
        ax = axes[idx]
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                       vmin=vmin, vmax=vmax, origin="lower")

        ax.set_xticks(range(len(density_values)))
        ax.set_xticklabels([f"{c:.1f}" for c in density_values], fontsize=10)
        ax.set_title(f"{size}x{size}", fontsize=20, fontweight="bold")
        ax.set_xticks(np.arange(-0.5, len(density_values), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(distance_labels), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=3)

        if idx == 0:
            ax.set_ylabel("Distance to Goal", fontsize=20)
        else:
            ax.tick_params(labelleft=False)

        if idx == 2:
            ax.set_xlabel("Density", fontsize=20)

    axes[0].set_yticks(range(len(distance_labels)))
    axes[0].set_yticklabels(distance_labels, fontsize=14)

    plt.subplots_adjust(top=0.92, wspace=0.08, left=0.06, right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(metric_label, fontsize=20)

    output_path = save_figure(fig, output_dir, f"heatmap_{metric}_by_distance_complexity")
    plt.close(fig)
    return output_path
