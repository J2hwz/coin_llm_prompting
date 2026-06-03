"""Pure information-theoretic and statistical metrics.

No dependencies on project modules — safe to import from anywhere.

Sections:
- Type aliases and constants
- Entropy and information theory (shannon, JSD, KL, cross-entropy)
- Statistical analysis (correlation, regression, controlled analysis)
- Calibration metrics
- Uncertainty-accuracy analysis
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

# =============================================================================
# Constants and Type Aliases
# =============================================================================

LOGPROB_EPS = 1e-12

ActionID = int
ActionDist = dict[ActionID, float]
OptimalActionSet = set[ActionID]


# =============================================================================
# Entropy and Information Theory
# =============================================================================


def shannon_entropy(dist: ActionDist) -> float:
    """Compute Shannon entropy in bits: H(X) = -sum_i p_i * log2(p_i)."""
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def cross_entropy(
    optimal_actions: OptimalActionSet, dist: ActionDist, eps: float = LOGPROB_EPS
) -> Optional[float]:
    """Cross-entropy between uniform-optimal and model distributions (bits)."""
    if not optimal_actions:
        return None

    weight = 1.0 / len(optimal_actions)
    ce = 0.0
    for action in optimal_actions:
        model_prob = max(dist.get(action, 0.0), eps)
        ce -= weight * math.log2(model_prob)
    return ce


def compute_optimal_mass(optimal_actions: OptimalActionSet, dist: ActionDist) -> float:
    """Total probability mass assigned to optimal actions."""
    return sum(dist.get(action, 0.0) for action in optimal_actions)


def optimal_entropy(num_optimal_actions: int) -> float:
    """Entropy of uniform distribution over k optimal actions: log2(k)."""
    if num_optimal_actions <= 0:
        return 0.0
    return math.log2(num_optimal_actions)


def kl_divergence(
    optimal_actions: OptimalActionSet, dist: ActionDist, eps: float = LOGPROB_EPS
) -> Optional[float]:
    """KL divergence from uniform-optimal to model: H(opt, model) - H(opt)."""
    ce = cross_entropy(optimal_actions, dist, eps)
    if ce is None:
        return None
    return ce - optimal_entropy(len(optimal_actions))


def jensen_shannon_divergence(
    optimal_actions: OptimalActionSet, dist: ActionDist, eps: float = LOGPROB_EPS
) -> Optional[float]:
    """Jensen-Shannon divergence between uniform-optimal and model distributions.

    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), M = 0.5*(P+Q).
    Bounded in [0, 1] with log base 2. 0 = identical, 1 = completely different.
    """
    if not optimal_actions:
        return None

    p_opt = 1.0 / len(optimal_actions)
    kl_p_m = 0.0
    kl_q_m = 0.0

    # Only need to visit action IDs where either distribution is non-zero
    for action_id in set(optimal_actions) | set(dist.keys()):
        p = p_opt if action_id in optimal_actions else 0.0
        q = dist.get(action_id, 0.0)
        m = 0.5 * (p + q)

        if p > 0 and m > eps:
            kl_p_m += p * math.log2(p / m)
        if q > 0 and m > eps:
            kl_q_m += q * math.log2(q / m)

    return 0.5 * kl_p_m + 0.5 * kl_q_m


# =============================================================================
# Statistical Analysis
# =============================================================================


@dataclass
class CorrelationResult:
    """Pearson correlation with p-value."""

    r: float
    n: int
    p_value: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {"r": self.r, "n": self.n, "p_value": self.p_value}


@dataclass
class RegressionResult:
    """OLS regression result."""

    coefficients: dict[str, float]
    p_values: dict[str, float]
    r_squared: float
    adj_r_squared: float
    n: int
    residuals: Optional[np.ndarray] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "coefficients": self.coefficients,
            "p_values": self.p_values,
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "n": self.n,
        }


@dataclass
class ControlledAnalysisResult:
    """Container for raw, within-stratum, and partial correlations plus regression."""

    raw_correlations: dict[str, CorrelationResult]
    within_stratum_correlations: dict[str, CorrelationResult]
    partial_correlations: dict[str, CorrelationResult]
    regression: Optional[RegressionResult] = None
    stratified_summary: Optional[pd.DataFrame] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_correlations": {k: v.to_dict() for k, v in self.raw_correlations.items()},
            "within_stratum_correlations": {
                k: v.to_dict() for k, v in self.within_stratum_correlations.items()
            },
            "partial_correlations": {
                k: v.to_dict() for k, v in self.partial_correlations.items()
            },
            "regression": self.regression.to_dict() if self.regression else None,
        }


def compute_correlation(
    x: pd.Series, y: pd.Series, min_samples: int = 10
) -> Optional[CorrelationResult]:
    """Pearson correlation with p-value; returns None if fewer than min_samples rows."""
    from scipy import stats

    mask = x.notna() & y.notna()
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < min_samples:
        return None

    r, p = stats.pearsonr(x_clean, y_clean)
    return CorrelationResult(r=r, n=len(x_clean), p_value=p)


def compute_correlations_for_columns(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    min_samples: int = 10,
) -> dict[str, CorrelationResult]:
    """Correlations between x_col and each column in y_cols."""
    results = {}
    for y_col in y_cols:
        if y_col not in df.columns:
            continue
        corr = compute_correlation(df[x_col], df[y_col], min_samples)
        if corr is not None:
            results[y_col] = corr
    return results


def compute_within_stratum_correlations(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    strata_cols: list[str],
    min_stratum_size: int = 10,
    min_strata: int = 3,
) -> dict[str, CorrelationResult]:
    """Within-stratum correlations, weighted-averaged across strata.

    Controls for confounding by computing correlations separately within each
    unique combination of strata_cols, then weighting by stratum size.
    """
    stratum_correlations: dict[str, list[tuple[float, int]]] = {y: [] for y in y_cols}

    for _, group in df.groupby(strata_cols):
        if len(group) < min_stratum_size:
            continue

        for y_col in y_cols:
            if y_col not in group.columns:
                continue
            corr = compute_correlation(group[x_col], group[y_col], min_stratum_size)
            if corr is not None and not np.isnan(corr.r):
                stratum_correlations[y_col].append((corr.r, corr.n))

    results = {}
    for y_col, corr_list in stratum_correlations.items():
        if len(corr_list) < min_strata:
            continue
        total_n = sum(n for _, n in corr_list)
        weighted_r = sum(r * n for r, n in corr_list) / total_n
        results[y_col] = CorrelationResult(r=weighted_r, n=total_n, p_value=None)

    return results


def residualize(y: np.ndarray, X: np.ndarray, add_intercept: bool = True) -> np.ndarray:
    """OLS residuals of y regressed on X."""
    if add_intercept:
        X = np.column_stack([np.ones(len(X)), X])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return y - X @ beta
    except np.linalg.LinAlgError:
        return y


def compute_partial_correlations(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    control_cols: list[str],
    min_samples: int = 30,
) -> dict[str, CorrelationResult]:
    """Partial correlations via residualization on control_cols."""
    from scipy import stats

    all_cols = [x_col] + y_cols + control_cols
    df_clean = df[all_cols].dropna()

    if len(df_clean) < min_samples:
        return {}

    controls = df_clean[control_cols].values
    x_resid = residualize(df_clean[x_col].values, controls)

    results = {}
    for y_col in y_cols:
        y_resid = residualize(df_clean[y_col].values, controls)
        r, p = stats.pearsonr(x_resid, y_resid)
        results[y_col] = CorrelationResult(r=r, n=len(df_clean), p_value=p)

    return results


def run_ols_regression(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    min_samples: int = 30,
) -> Optional[RegressionResult]:
    """OLS regression with multiple predictors; returns None if insufficient data."""
    from scipy import stats

    all_cols = [y_col] + x_cols
    df_clean = df[all_cols].dropna()

    if len(df_clean) < min_samples:
        return None

    y = df_clean[y_col].values
    X = df_clean[x_cols].values
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    n, k = X_with_intercept.shape

    try:
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        y_pred = X_with_intercept @ beta
        residuals = y - y_pred

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)

        mse = ss_res / (n - k)
        try:
            var_beta = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            se_beta = np.sqrt(np.diag(var_beta))
            t_stats = beta / se_beta
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))
        except np.linalg.LinAlgError:
            p_values = np.full(k, np.nan)

        coef_names = ["intercept"] + x_cols
        return RegressionResult(
            coefficients=dict(zip(coef_names, beta)),
            p_values=dict(zip(coef_names, p_values)),
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            n=n,
            residuals=residuals,
        )

    except np.linalg.LinAlgError:
        return None


def compute_stratified_summary(
    df: pd.DataFrame,
    strata_cols: list[str],
    agg_config: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Summary statistics grouped by strata_cols.

    agg_config maps output column name to (input_col, agg_func) tuples,
    e.g. {"mean_entropy": ("entropy_bits", "mean")}.
    """
    agg_dict = {
        out_col: (in_col, func) for out_col, (in_col, func) in agg_config.items()
    }
    return df.groupby(strata_cols).agg(**agg_dict).reset_index()


def run_controlled_analysis(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    control_cols: list[str],
    min_samples: int = 30,
    min_stratum_size: int = 10,
) -> ControlledAnalysisResult:
    """Run raw, within-stratum, and partial correlations plus OLS regression."""
    raw_corrs = compute_correlations_for_columns(df, x_col, y_cols)
    within_stratum_corrs = compute_within_stratum_correlations(
        df, x_col, y_cols, control_cols, min_stratum_size=min_stratum_size
    )
    partial_corrs = compute_partial_correlations(
        df, x_col, y_cols, control_cols, min_samples=min_samples
    )
    regression = None
    if y_cols:
        regression = run_ols_regression(
            df, y_cols[0], [x_col] + control_cols, min_samples=min_samples
        )

    return ControlledAnalysisResult(
        raw_correlations=raw_corrs,
        within_stratum_correlations=within_stratum_corrs,
        partial_correlations=partial_corrs,
        regression=regression,
    )


def format_correlation_report(
    result: ControlledAnalysisResult,
    x_label: str = "X",
    control_labels: Optional[list[str]] = None,
) -> str:
    """Format a ControlledAnalysisResult as a readable text report."""
    lines = []
    control_str = ", ".join(control_labels) if control_labels else "controls"

    lines.append(f"CORRELATION ANALYSIS: {x_label}")
    lines.append("=" * 60)

    lines.append("\n1. RAW CORRELATIONS (no controls):")
    for y_col, corr in result.raw_correlations.items():
        p_str = f", p={corr.p_value:.4f}" if corr.p_value is not None else ""
        lines.append(f"   {y_col}: r={corr.r:.4f} (n={corr.n}){p_str}")

    lines.append(f"\n2. WITHIN-STRATUM CORRELATIONS (stratified by {control_str}):")
    for y_col, corr in result.within_stratum_correlations.items():
        lines.append(f"   {y_col}: r={corr.r:.4f} (n={corr.n})")

    lines.append(f"\n3. PARTIAL CORRELATIONS (controlling for {control_str}):")
    for y_col, corr in result.partial_correlations.items():
        p_str = f", p={corr.p_value:.4f}" if corr.p_value is not None else ""
        lines.append(f"   {y_col}: r={corr.r:.4f} (n={corr.n}){p_str}")

    if result.regression:
        lines.append(f"\n4. OLS REGRESSION (with {control_str} as controls):")
        lines.append(
            f"   R² = {result.regression.r_squared:.4f}, "
            f"Adj R² = {result.regression.adj_r_squared:.4f}, "
            f"n = {result.regression.n}"
        )
        lines.append("   Coefficients:")
        for var, coef in result.regression.coefficients.items():
            p_val = result.regression.p_values.get(var)
            p_str = f", p={p_val:.4f}" if p_val is not None else ""
            sig = "*" if p_val is not None and p_val < 0.05 else ""
            lines.append(f"      {var}: {coef:.4f}{p_str}{sig}")

    return "\n".join(lines)


# =============================================================================
# Calibration Metrics
# =============================================================================


@dataclass
class CalibrationMetrics:
    """Calibration metrics comparing LLM uncertainty to optimal uncertainty."""

    mean_entropy: float
    mean_optimal_entropy: float
    mean_divergence: float
    calibration_error: float   # mean |H_llm - H_opt|
    calibration_bias: float    # mean (H_llm - H_opt), positive = under-confident
    entropy_correlation: float
    n_samples: int

    def to_dict(self) -> dict[str, float]:
        return {
            "mean_entropy": self.mean_entropy,
            "mean_optimal_entropy": self.mean_optimal_entropy,
            "mean_divergence": self.mean_divergence,
            "calibration_error": self.calibration_error,
            "calibration_bias": self.calibration_bias,
            "entropy_correlation": self.entropy_correlation,
            "n_samples": self.n_samples,
        }


def compute_calibration_metrics(
    df: pd.DataFrame,
    entropy_col: str = "entropy_bits",
    optimal_entropy_col: str = "optimal_entropy_bits",
    divergence_col: str = "cross_entropy_bits",
) -> CalibrationMetrics:
    """Compute calibration metrics from a DataFrame with entropy columns."""
    df_clean = df[[entropy_col, optimal_entropy_col, divergence_col]].dropna()

    if len(df_clean) == 0:
        return CalibrationMetrics(
            mean_entropy=0.0,
            mean_optimal_entropy=0.0,
            mean_divergence=0.0,
            calibration_error=0.0,
            calibration_bias=0.0,
            entropy_correlation=0.0,
            n_samples=0,
        )

    h_llm = df_clean[entropy_col]
    h_opt = df_clean[optimal_entropy_col]
    divergence = df_clean[divergence_col]

    return CalibrationMetrics(
        mean_entropy=h_llm.mean(),
        mean_optimal_entropy=h_opt.mean(),
        mean_divergence=divergence.mean(),
        calibration_error=(h_llm - h_opt).abs().mean(),
        calibration_bias=(h_llm - h_opt).mean(),
        entropy_correlation=h_llm.corr(h_opt),
        n_samples=len(df_clean),
    )


# =============================================================================
# Uncertainty-Accuracy Analysis
# =============================================================================


@dataclass
class UncertaintyAccuracyMetrics:
    """Metrics relating LLM uncertainty to prediction accuracy."""

    accuracy: float
    mean_entropy_correct: float
    mean_entropy_incorrect: float
    entropy_gap: float           # incorrect - correct entropy
    auroc: Optional[float]       # can entropy predict errors?
    ece: float
    n_correct: int
    n_incorrect: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "mean_entropy_correct": self.mean_entropy_correct,
            "mean_entropy_incorrect": self.mean_entropy_incorrect,
            "entropy_gap": self.entropy_gap,
            "auroc": self.auroc,
            "ece": self.ece,
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
        }


def compute_uncertainty_accuracy_metrics(
    df: pd.DataFrame,
    entropy_col: str = "entropy_bits",
    correct_col: str = "is_action_optimal",
    n_bins: int = 10,
) -> UncertaintyAccuracyMetrics:
    """Compute metrics answering: does the model know what it doesn't know?"""
    df_clean = df[[entropy_col, correct_col]].dropna()

    if len(df_clean) == 0:
        return UncertaintyAccuracyMetrics(
            accuracy=0.0,
            mean_entropy_correct=0.0,
            mean_entropy_incorrect=0.0,
            entropy_gap=0.0,
            auroc=None,
            ece=0.0,
            n_correct=0,
            n_incorrect=0,
        )

    correct_mask = df_clean[correct_col] == 1
    n_correct = correct_mask.sum()
    n_incorrect = (~correct_mask).sum()
    accuracy = n_correct / len(df_clean)

    mean_entropy_correct = (
        df_clean.loc[correct_mask, entropy_col].mean() if n_correct > 0 else 0.0
    )
    mean_entropy_incorrect = (
        df_clean.loc[~correct_mask, entropy_col].mean() if n_incorrect > 0 else 0.0
    )

    auroc = None
    if n_correct > 0 and n_incorrect > 0:
        try:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(1 - df_clean[correct_col], df_clean[entropy_col])
        except ImportError:
            auroc = _compute_auroc_manual(
                df_clean[entropy_col].values, df_clean[correct_col].values
            )

    # Normalize entropy to [0,1]: max entropy for 4 actions is log2(4) = 2 bits
    max_entropy = 2.0
    confidence = 1 - (df_clean[entropy_col] / max_entropy).clip(0, 1)
    ece = _compute_ece(confidence.values, df_clean[correct_col].values, n_bins)

    return UncertaintyAccuracyMetrics(
        accuracy=accuracy,
        mean_entropy_correct=mean_entropy_correct,
        mean_entropy_incorrect=mean_entropy_incorrect,
        entropy_gap=mean_entropy_incorrect - mean_entropy_correct,
        auroc=auroc,
        ece=ece,
        n_correct=int(n_correct),
        n_incorrect=int(n_incorrect),
    )


def _compute_auroc_manual(scores: np.ndarray, labels: np.ndarray) -> float:
    """Fallback AUROC without sklearn; treats high entropy as predicting errors."""
    labels = 1 - labels
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]
    tp_cumsum = np.cumsum(sorted_labels)
    return float(np.sum((1 - sorted_labels) * tp_cumsum) / (n_pos * n_neg))


def _compute_ece(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error: gap between confidence and accuracy across bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidence >= bin_boundaries[i]) & (confidence < bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            ece += prop_in_bin * abs(correct[in_bin].mean() - confidence[in_bin].mean())

    return float(ece)


def compute_selective_prediction_curve(
    df: pd.DataFrame,
    entropy_col: str = "entropy_bits",
    correct_col: str = "is_action_optimal",
    n_thresholds: int = 20,
) -> pd.DataFrame:
    """Accuracy vs coverage curve: predict only when entropy <= threshold."""
    df_clean = df[[entropy_col, correct_col]].dropna()
    n_total = len(df_clean)

    if n_total == 0:
        return pd.DataFrame(columns=["threshold", "coverage", "accuracy", "n_samples"])

    thresholds = np.linspace(df_clean[entropy_col].min(), df_clean[entropy_col].max(), n_thresholds)

    rows = []
    for thresh in thresholds:
        mask = df_clean[entropy_col] <= thresh
        n_selected = mask.sum()
        rows.append({
            "threshold": thresh,
            "coverage": n_selected / n_total,
            "accuracy": df_clean.loc[mask, correct_col].mean() if n_selected > 0 else 0.0,
            "n_samples": n_selected,
        })

    return pd.DataFrame(rows)
