"""
Effect size calculations for continuous, ordinal, and binary outcomes.

All functions return DataFrames for agent-readable output.
"""

import numpy as np
import pandas as pd
from scipy import stats


def cohens_d_from_model(beta: float, sigma_resid: float) -> float:
    """Cohen's d = model coefficient / residual SD."""
    if sigma_resid == 0:
        return np.nan
    return beta / sigma_resid


def hedges_g(d: float, n: int) -> float:
    """Hedges' g — bias-corrected Cohen's d for small samples."""
    if n <= 3:
        return np.nan
    j = 1 - (3 / (4 * (n - 2) - 1))
    return d * j


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Cliff's delta — non-parametric effect size for ordinal data.

    delta = P(X > Y) - P(X < Y), range [-1, 1].
    """
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return {"delta": np.nan, "interpretation": "insufficient data"}

    greater = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    delta = (greater - less) / (n_x * n_y)

    interpretation = _interpret_cliffs(abs(delta))
    return {"delta": delta, "interpretation": interpretation}


def glass_delta(x: np.ndarray, y_control: np.ndarray) -> float:
    """Glass's delta — using the control group SD."""
    sd_control = np.std(y_control, ddof=1)
    if sd_control == 0:
        return np.nan
    return (np.mean(x) - np.mean(y_control)) / sd_control


def odds_ratio_ci(log_or: float, se: float, alpha: float = 0.05) -> dict:
    """Convert log-OR from GLMM/GEE to OR with CI."""
    z = stats.norm.ppf(1 - alpha / 2)
    return {
        "or": np.exp(log_or),
        "or_ci_lower": np.exp(log_or - z * se),
        "or_ci_upper": np.exp(log_or + z * se),
    }


def risk_difference(p1: float, p0: float) -> dict:
    """Absolute risk difference and NNT."""
    rd = p1 - p0
    nnt = 1 / abs(rd) if abs(rd) > 0 else np.inf
    label = "NNT" if rd < 0 else "NNH"
    return {
        "risk_difference": rd,
        "nnt_or_nnh": nnt,
        "nnt_label": label,
    }


def interpret_cohens_d(d: float) -> str:
    """Standard interpretation thresholds."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    if d_abs < 0.5:
        return "small"
    if d_abs < 0.8:
        return "medium"
    return "large"


def compute_all_effect_sizes(
    df: pd.DataFrame,
    outcome: str,
    condition_col: str = "condition",
) -> pd.DataFrame:
    """
    Compute a full suite of effect sizes for a continuous/ordinal outcome.

    Returns one-row DataFrame with all metrics.
    """
    g0 = df[df[condition_col] == 0][outcome].values
    g1 = df[df[condition_col] == 1][outcome].values

    n0, n1 = len(g0), len(g1)
    mean0, mean1 = np.mean(g0), np.mean(g1)
    sd0, sd1 = np.std(g0, ddof=1), np.std(g1, ddof=1)

    pooled_sd = np.sqrt(((n0 - 1) * sd0**2 + (n1 - 1) * sd1**2) / (n0 + n1 - 2))
    d = (mean1 - mean0) / pooled_sd if pooled_sd > 0 else np.nan
    g = hedges_g(d, n0 + n1)
    glass = glass_delta(g1, g0)
    cliff = cliffs_delta(g1, g0)

    return pd.DataFrame([{
        "outcome": outcome,
        "n_cond0": n0,
        "n_cond1": n1,
        "mean_cond0": mean0,
        "mean_cond1": mean1,
        "sd_cond0": sd0,
        "sd_cond1": sd1,
        "mean_diff": mean1 - mean0,
        "pooled_sd": pooled_sd,
        "cohens_d": d,
        "cohens_d_interpretation": interpret_cohens_d(d) if not np.isnan(d) else "N/A",
        "hedges_g": g,
        "glass_delta": glass,
        "cliffs_delta": cliff["delta"],
        "cliffs_interpretation": cliff["interpretation"],
    }])


def _interpret_cliffs(d: float) -> str:
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"
