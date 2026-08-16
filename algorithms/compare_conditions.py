"""
compare_conditions.py
----------------------
Cross-condition comparison of subgoal-inference algorithm results, combining
the "best available" prediction per algorithm: CV-tuned (held-out) for
Surprise v2 (from tune_params.py), literature-default for the other 3
algorithms (from run_algorithms.py). Reads only already-computed results.csv
files plus the original layout JSONs (for rendering only, not computation)
— never re-runs any algorithm.

Inv. Planning is deliberately excluded: its Bayesian posterior sums (not
averages) per-trajectory log-likelihoods across all pooled trajectories that
visit a candidate, which structurally biases it against the true, heavily-
visited coin cell as the number of pooled trajectories grows — confirmed to
explain its anomalous behavior on `medium` (~9.66 trajectories/grid pooled,
vs. ~4 and ~2.6 for the other two conditions). See conversation / git history
for the diagnosis; not fixed here.

Conditions
----------
low            reasoning_effort=low,  control instructions
medium         reasoning_effort=medium, control instructions
low_deceptive  reasoning_effort=low,  deceptive instructions

Usage
-----
    python compare_conditions.py [--output-dir algorithms/results]

Output (--output-dir)
----------------------
combined_results.csv    long format: every (condition, grid_size, grid_id,
                         algorithm) row used in the comparison
summary_table.csv       mean/SEM/n Manhattan distance by condition x algorithm
comparison_plot.png     grouped bar chart of the above
modal_parameters.csv    literature-default vs. full-data-best vs. modal
                         CV-fold theta, by condition, for the 1 tuned algorithm
RESULTS.md              short interpretation write-up generated from the
                         tables above
grids/
    {condition}_{grid_size}_{grid_id}_aggregated.png       one 1x4 panel per grid
    composite_{condition}_{grid_size}_comp{density}.png    up to 10x4 panels
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ALGO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ALGO_DIR.parent
if str(_ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGO_DIR))

import run_algorithms as ra
from surprise_v2 import LAMBDA, GAMMA, ALPHA

# ── Condition registry ──────────────────────────────────────────────────────

_D = _REPO_ROOT / "data" / "one_coin"

CONDITIONS = {
    "low": {
        "run_algorithms_csvs": [
            (_D / "low/results/run_algorithms/7_low/results.csv", 7),
            (_D / "low/results/run_algorithms/9_low/results.csv", 9),
            (_D / "low/results/run_algorithms/11_low/results.csv", 11),
        ],
        "tune_params_csv": _D / "low/results/tune_params/results.csv",
        "tune_params_dir": _D / "low/results/tune_params",
        "layout_dirs": {
            7: _D / "low/7_low/control",
            9: _D / "low/9_low/control",
            11: _D / "low/11_low/control",
        },
    },
    "medium": {
        "run_algorithms_csvs": [
            (_D / "medium/results/run_algorithms/7_medium/results.csv", 7),
            (_D / "medium/results/run_algorithms/9_medium/results.csv", 9),
            (_D / "medium/results/run_algorithms/11_medium/results.csv", 11),
        ],
        "tune_params_csv": _D / "medium/results/tune_params/results.csv",
        "tune_params_dir": _D / "medium/results/tune_params",
        "layout_dirs": {
            7: _D / "medium/7_medium",
            9: _D / "medium/9_medium",
            11: _D / "medium/11_medium",
        },
    },
    "low_deceptive": {
        "run_algorithms_csvs": [
            (_D / "low_deceptive/results/run_algorithms/7_low_deceptive_with_steps/results.csv", 7),
            (_D / "low_deceptive/results/run_algorithms/9_low_deceptive_with_steps/results.csv", 9),
            (_D / "low_deceptive/results/run_algorithms/11_low_deceptive_with_steps/results.csv", 11),
        ],
        "tune_params_csv": _D / "low_deceptive/results/tune_params/results.csv",
        "tune_params_dir": _D / "low_deceptive/results/tune_params",
        "layout_dirs": {
            7: _D / "low_deceptive/7_low_deceptive_with_steps",
            9: _D / "low_deceptive/9_low_deceptive_with_steps",
            11: _D / "low_deceptive/11_low_deceptive_with_steps",
        },
    },
}

EXCLUDED_ALGOS = {"Inv. Planning"}  # see module docstring for why
ALGO_ORDER = [name for name, _ in ra.ALGORITHMS if name not in EXCLUDED_ALGOS]
TUNED_ALGOS = {"Surprise v2"}
UNTUNED_ALGOS = set(ALGO_ORDER) - TUNED_ALGOS

# display name -> (tune_params.py slug, param_names, literature-default theta)
TUNED_ALGO_INFO = {
    "Surprise v2": {"slug": "surprise_v2", "param_names": ["lambda", "gamma", "alpha"],
                     "literature_default": [LAMBDA, GAMMA, ALPHA]},
}

COMBINED_FIELDNAMES = [
    "condition", "grid_size", "density", "grid_id", "traj_id", "algorithm",
    "param_source", "score_type", "predicted_col", "predicted_row",
    "true_coin_col", "true_coin_row", "manhattan_dist", "fold", "theta_json",
    "score_grid_json",
]


# ── Parsing helpers ──────────────────────────────────────────────────────────


def _parse_density(grid_id):
    m = re.search(r"comp([\d.]+)_grid", grid_id)
    return m.group(1) if m else "unknown"


def _parse_grid_num(grid_id):
    m = re.search(r"grid(\d+)$", grid_id)
    return int(m.group(1)) if m else -1


def _parse_grid_size_from_data_dir(data_dir):
    m = re.search(r"/(\d+)_", data_dir)
    return int(m.group(1)) if m else None


# ── Load & combine ───────────────────────────────────────────────────────────


def load_condition_rows(condition, cfg):
    rows = []

    for csv_path, size in cfg["run_algorithms_csvs"]:
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                if r["mode_type"] != "pooled" or r["algorithm"] not in UNTUNED_ALGOS:
                    continue
                row = dict(r)
                row["condition"] = condition
                row["grid_size"] = size
                row["density"] = _parse_density(row["grid_id"])
                row["param_source"] = "literature_default"
                row.setdefault("fold", "")
                row.setdefault("theta_json", "")
                rows.append(row)

    with open(cfg["tune_params_csv"]) as f:
        for r in csv.DictReader(f):
            if r["mode_type"] != "cv_held_out" or r["algorithm"] not in TUNED_ALGOS:
                continue
            row = dict(r)
            row["condition"] = condition
            row["grid_size"] = _parse_grid_size_from_data_dir(row["data_dir"])
            row["density"] = _parse_density(row["grid_id"])
            row["param_source"] = "cv_tuned"
            rows.append(row)

    return rows


def write_combined_csv(path, all_rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMBINED_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


# ── Summary table + plot ─────────────────────────────────────────────────────


def build_summary_table(df):
    summary = (
        df.groupby(["condition", "algorithm"])["manhattan_dist"]
        .agg(mean="mean", sem="sem", n="count")
        .reset_index()
    )
    return summary


def plot_comparison(summary, path):
    conditions = ["low", "medium", "low_deceptive"]
    algos = ALGO_ORDER
    n_algos = len(algos)
    x = np.arange(n_algos)
    width = 0.8 / len(conditions)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(conditions)))
    for ci, cond in enumerate(conditions):
        means, sems = [], []
        for algo in algos:
            row = summary[(summary["condition"] == cond) & (summary["algorithm"] == algo)]
            means.append(row["mean"].iloc[0] if len(row) else 0)
            sems.append(row["sem"].iloc[0] if len(row) else 0)
        offsets = x + (ci - (len(conditions) - 1) / 2) * width
        ax.bar(offsets, means, yerr=sems, width=width * 0.9, label=cond,
               color=colors[ci], capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=15, ha="right")
    ax.set_ylabel("Mean Manhattan distance to true coin (± SEM)")
    ax.set_title("Subgoal-inference accuracy by algorithm and condition")
    ax.legend(title="condition")
    ax.set_ylim(bottom=0)
    fig.text(
        0.01, 0.01,
        "Surprise v2: CV-tuned parameters (held-out predictions).  "
        "Other 3 algorithms: literature-default parameters.  "
        "Inv. Planning excluded (see module docstring).",
        fontsize=7, color="#555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ── Modal parameters ─────────────────────────────────────────────────────────


def _read_folds_csv(path, param_names):
    thetas = []
    with open(path) as f:
        for r in csv.DictReader(f):
            thetas.append(tuple(float(r[p]) for p in param_names))
    return thetas


def build_modal_parameters(output_dir):
    rows = []
    for condition, cfg in CONDITIONS.items():
        for algo, info in TUNED_ALGO_INFO.items():
            slug, param_names = info["slug"], info["param_names"]
            tp_dir = cfg["tune_params_dir"]
            fold_thetas = _read_folds_csv(tp_dir / f"{slug}_folds.csv", param_names)
            with open(tp_dir / f"{slug}_summary.json") as f:
                summary = json.load(f)
            full_data_best = summary["full_data_best"]["theta"]
            full_data_best = full_data_best if isinstance(full_data_best, list) else [full_data_best]

            counts = Counter(fold_thetas)
            modal_theta, modal_count = counts.most_common(1)[0]

            rows.append({
                "condition": condition,
                "algorithm": algo,
                "param_names": json.dumps(param_names),
                "literature_default": json.dumps(info["literature_default"]),
                "full_data_best": json.dumps(full_data_best),
                "modal_fold_theta": json.dumps(list(modal_theta)),
                "modal_fold_count": modal_count,
                "k_folds": len(fold_thetas),
                "matches_full_data_best": list(modal_theta) == full_data_best,
            })

    path = output_dir / "modal_parameters.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return pd.DataFrame(rows)


# ── Grid visualizations ──────────────────────────────────────────────────────


def _json_to_grid(s):
    arr = json.loads(s)
    return np.array([[np.nan if v is None else v for v in row] for row in arr], dtype=float)


_layout_cache = {}


def _load_layout(layout_dir, grid_id):
    key = (str(layout_dir), grid_id)
    if key not in _layout_cache:
        matches = list(Path(layout_dir).glob(f"*{grid_id}_coin_layout.json"))
        if not matches:
            _layout_cache[key] = None
        else:
            with open(matches[0]) as f:
                _layout_cache[key] = json.load(f)
    return _layout_cache[key]


def build_grid_index(all_rows):
    """(condition, grid_size, grid_id) -> {algorithm: row}"""
    index = defaultdict(dict)
    for row in all_rows:
        key = (row["condition"], row["grid_size"], row["grid_id"])
        index[key][row["algorithm"]] = row
    return index


def _draw_panel(ax, row, layout, title):
    score = _json_to_grid(row["score_grid_json"])
    pred = (int(row["predicted_col"]), int(row["predicted_row"]))
    ra.plot_result(ax, score, layout, pred, title)


def make_per_grid_plots(grid_index, cfg_by_condition, out_dir):
    n = 0
    for (condition, grid_size, grid_id), algo_rows in grid_index.items():
        layout = _load_layout(cfg_by_condition[condition]["layout_dirs"][grid_size], grid_id)
        if layout is None:
            continue
        fig, axes = plt.subplots(1, len(ALGO_ORDER), figsize=(3 * len(ALGO_ORDER), 3.4), squeeze=False)
        for col, algo in enumerate(ALGO_ORDER):
            row = algo_rows.get(algo)
            if row is None:
                axes[0, col].axis("off")
                axes[0, col].set_title(f"{algo}\n(no prediction)", fontsize=6)
                continue
            _draw_panel(axes[0, col], row, layout, f"{algo}\nd = {row['manhattan_dist']}")
        fig.suptitle(f"{condition} — size{grid_size} — {grid_id}", fontsize=10)
        fig.tight_layout()
        out = out_dir / f"{condition}_{grid_size}_{grid_id}_aggregated.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        n += 1
    return n


def make_composite_plots(grid_index, cfg_by_condition, out_dir):
    groups = defaultdict(list)  # (condition, grid_size, density) -> [grid_id, ...]
    for (condition, grid_size, grid_id) in grid_index:
        density = _parse_density(grid_id)
        groups[(condition, grid_size, density)].append(grid_id)

    n = 0
    for (condition, grid_size, density), grid_ids in groups.items():
        grid_ids = sorted(grid_ids, key=_parse_grid_num)
        n_rows = len(grid_ids)
        if n_rows == 0:
            continue
        fig, axes = plt.subplots(n_rows, len(ALGO_ORDER),
                                  figsize=(3 * len(ALGO_ORDER), n_rows * 3 + 0.5),
                                  squeeze=False)
        for col, algo in enumerate(ALGO_ORDER):
            fig.text((col + 0.5) / len(ALGO_ORDER), 0.995, algo, ha="center", va="top",
                      fontsize=9, fontweight="bold", transform=fig.transFigure)

        for row_i, grid_id in enumerate(grid_ids):
            algo_rows = grid_index[(condition, grid_size, grid_id)]
            layout = _load_layout(cfg_by_condition[condition]["layout_dirs"][grid_size], grid_id)
            for col, algo in enumerate(ALGO_ORDER):
                ax = axes[row_i, col]
                row = algo_rows.get(algo)
                if layout is None or row is None:
                    ax.axis("off")
                    continue
                _draw_panel(ax, row, layout, f"d = {row['manhattan_dist']}")
            axes[row_i, 0].set_ylabel(grid_id, fontsize=7, rotation=0, labelpad=30, va="center")

        fig.suptitle(f"{condition} — size{grid_size} — comp{density}", fontsize=11, y=1.0)
        fig.tight_layout()
        out = out_dir / f"composite_{condition}_{grid_size}_comp{density}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        n += 1
    return n


# ── Write-up ──────────────────────────────────────────────────────────────────

PARAM_NOTATION_MD = """\
## Parameter notation

Inv. Planning is excluded from this comparison — its Bayesian posterior
sums (rather than averages) per-trajectory log-likelihoods across all
pooled trajectories that visit a candidate, which structurally biases it
against the true, heavily-visited coin cell as the number of pooled
trajectories grows. This was confirmed to explain its anomalous behaviour
on `medium` (~9.66 trajectories/grid pooled there, vs. ~4 and ~2.6 for
`low`/`low_deceptive`) — the one condition where it alone got *worse*
while every other algorithm improved. See git history for the diagnosis.

**Surprise v2 — λ (lambda), γ (gamma), α (alpha)**, in `surprise_v2.py`'s
movement/state prior (Eqs. 3–8), combined into a per-step "surprise" score
= −log₂ P(action). Each parameter sets what counts as *normal* (low-surprise)
movement, so it shapes which steps generate a surprise spike:
- **λ** — prior probability mass assigned to immediately reversing the
  previous action (backtracking). Higher λ → backtracking is expected, so
  real backtracking barely registers as surprising; lower λ → any
  backtrack produces a larger surprise spike.
- **γ** — forward-to-lateral ratio: how strongly continuing straight is
  favoured over turning. Higher γ → straight-line continuation is the
  default expectation (turns are more surprising); lower γ → turning is
  treated as roughly as normal as continuing straight.
- **α** — decay rate of the goal-directed distance prior,
  `p(next cell) ∝ 1/α^d` where `d` = Manhattan distance from the terminal
  goal after the move. Higher α → steps that reduce distance-to-goal are
  strongly expected (so goal-directed behaviour is unsurprising and
  detours/wandering spike hard); α near 1 → little goal pull is assumed, so
  wandering isn't penalised much regardless of whether it heads toward the
  goal.

In `modal_parameters.csv`/the table above, values are listed in the fixed
order given by `param_names` — `[lambda, gamma, alpha]` for Surprise v2.
"""


def write_results_md(path, summary, modal_df, combined_df):
    pivot = summary.pivot(index="algorithm", columns="condition", values="mean").reindex(ALGO_ORDER)
    pivot = pivot[["low", "medium", "low_deceptive"]]

    lines = ["# Subgoal-inference algorithm comparison across conditions", ""]
    lines.append(
        "Mean Manhattan distance (predicted vs. true coin cell) by algorithm and "
        "condition. `low`/`medium` = reasoning_effort under control instructions; "
        "`low_deceptive` = reasoning_effort=low under deceptive instructions. "
        "Surprise v2 uses CV-tuned (held-out) parameters; the other 3 algorithms "
        "use literature-default parameters. Inv. Planning is excluded — see "
        "Parameter notation below."
    )
    lines.append("")
    lines.append("## Summary table")
    lines.append("")
    lines.append(pivot.round(2).to_markdown())
    lines.append("")

    lines.append("## Modal fitted parameters (CV-tuned algorithms only)")
    lines.append("")
    lines.append(
        modal_df[["condition", "algorithm", "literature_default", "full_data_best",
                   "modal_fold_theta", "modal_fold_count", "k_folds"]]
        .to_markdown(index=False)
    )
    lines.append("")

    lines.append(PARAM_NOTATION_MD)

    lines.append("## Observations")
    lines.append("")
    for algo in ALGO_ORDER:
        row = pivot.loc[algo]
        best_cond = row.idxmin()
        worst_cond = row.idxmax()
        lines.append(
            f"- **{algo}**: most accurate in `{best_cond}` ({row[best_cond]:.2f}), "
            f"least accurate in `{worst_cond}` ({row[worst_cond]:.2f})."
        )
    lines.append("")

    for algo in TUNED_ALGO_INFO:
        sub = modal_df[modal_df["algorithm"] == algo]
        vals = ", ".join(f"{r.condition}: {r.full_data_best}" for r in sub.itertuples())
        lines.append(f"- **{algo}** full-data-best parameter by condition — {vals}.")
    lines.append("")

    overall_rank = summary.groupby("algorithm")["mean"].mean().sort_values()
    lines.append(
        "- Averaged across all 3 conditions, algorithms ranked most-to-least "
        "accurate: " + ", ".join(f"{a} ({v:.2f})" for a, v in overall_rank.items()) + "."
    )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(_ALGO_DIR / "results"),
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    grids_dir = out_dir / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for condition, cfg in CONDITIONS.items():
        rows = load_condition_rows(condition, cfg)
        print(f"{condition}: {len(rows)} rows")
        all_rows.extend(rows)

    write_combined_csv(out_dir / "combined_results.csv", all_rows)
    print(f"→ combined_results.csv ({len(all_rows)} rows)")

    df = pd.DataFrame(all_rows)
    df["manhattan_dist"] = df["manhattan_dist"].astype(float)

    summary = build_summary_table(df)
    summary.to_csv(out_dir / "summary_table.csv", index=False)
    print(f"→ summary_table.csv ({len(summary)} rows)")

    plot_comparison(summary, out_dir / "comparison_plot.png")
    print("→ comparison_plot.png")

    modal_df = build_modal_parameters(out_dir)
    print(f"→ modal_parameters.csv ({len(modal_df)} rows)")

    grid_index = build_grid_index(all_rows)
    n_grid_plots = make_per_grid_plots(grid_index, CONDITIONS, grids_dir)
    print(f"→ {n_grid_plots} per-grid plots in grids/")

    n_composite = make_composite_plots(grid_index, CONDITIONS, grids_dir)
    print(f"→ {n_composite} composite plots in grids/")

    write_results_md(out_dir / "RESULTS.md", summary, modal_df, df)
    print("→ RESULTS.md")


if __name__ == "__main__":
    main()
