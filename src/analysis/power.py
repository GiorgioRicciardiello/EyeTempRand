"""
Power analysis for the EyeTempRand study.

Post-hoc observed power, minimum detectable effect size,
and prospective sample size curves.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from src.config import load_config
from src.exploratory.style import (
    set_publication_style,
    CONDITION_COLORS,
    FIG_SINGLE,
    ANNOT_SIZE,
    format_ax,
)


def post_hoc_power(
    effect_size_d: float,
    n1: int,
    n2: int,
    alpha: float = 0.05,
    icc: float = 0.0,
    obs_per_cluster: int = 1,
) -> float:
    """
    Post-hoc power for a two-sample t-test with design effect correction.

    Parameters
    ----------
    effect_size_d : float
        Cohen's d.
    n1, n2 : int
        Number of clusters (patients) per group.
    alpha : float
    icc : float
        Intraclass correlation (within-patient).
    obs_per_cluster : int
        Observations per patient.
    """
    design_effect = 1 + (obs_per_cluster - 1) * icc
    n1_eff = n1 * obs_per_cluster / design_effect
    n2_eff = n2 * obs_per_cluster / design_effect

    se = np.sqrt(1 / n1_eff + 1 / n2_eff)
    ncp = abs(effect_size_d) / se
    crit = stats.norm.ppf(1 - alpha / 2)
    power = 1 - stats.norm.cdf(crit - ncp) + stats.norm.cdf(-crit - ncp)
    return power


def min_detectable_effect(
    n1: int,
    n2: int,
    alpha: float = 0.05,
    power: float = 0.80,
    icc: float = 0.0,
    obs_per_cluster: int = 1,
) -> float:
    """
    Minimum detectable Cohen's d given sample size and desired power.
    """
    design_effect = 1 + (obs_per_cluster - 1) * icc
    n1_eff = n1 * obs_per_cluster / design_effect
    n2_eff = n2 * obs_per_cluster / design_effect

    se = np.sqrt(1 / n1_eff + 1 / n2_eff)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return (z_alpha + z_beta) * se


def sample_size_for_power(
    effect_size_d: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
    icc: float = 0.0,
    obs_per_cluster: int = 1,
) -> int:
    """
    Required n per group (clusters) for desired power.
    """
    design_effect = 1 + (obs_per_cluster - 1) * icc
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    if abs(effect_size_d) < 1e-10:
        return 999999

    n1 = ((z_alpha + z_beta) / effect_size_d) ** 2 * (1 + 1 / ratio) * design_effect / obs_per_cluster
    return int(np.ceil(n1))


def run_power_analysis(
    df: pd.DataFrame,
    outcomes: list[str],
    out_dir: Path,
) -> pd.DataFrame:
    """
    Full power analysis for all outcomes.

    Returns a summary DataFrame and saves power curve plots.
    """
    cfg = load_config()
    alpha = cfg["analysis"]["alpha"]
    set_publication_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    n0 = df[df["condition"] == 0]["patient"].nunique()
    n1 = df[df["condition"] == 1]["patient"].nunique()
    obs_per = len(df) // (n0 + n1)  # ~16 per patient

    rows = []
    for outcome in outcomes:
        g0 = df[df["condition"] == 0].groupby("patient")[outcome].mean()
        g1 = df[df["condition"] == 1].groupby("patient")[outcome].mean()

        # Patient-level means for power calc
        pooled_sd = np.sqrt(
            ((len(g0) - 1) * g0.std() ** 2 + (len(g1) - 1) * g1.std() ** 2)
            / (len(g0) + len(g1) - 2)
        )
        d = (g1.mean() - g0.mean()) / pooled_sd if pooled_sd > 0 else 0.0

        # Estimate ICC from patient-level variance
        grand_mean = df[outcome].mean()
        patient_means = df.groupby("patient")[outcome].mean()
        var_between = patient_means.var()
        var_total = df[outcome].var()
        icc_est = var_between / var_total if var_total > 0 else 0.0

        power_obs = post_hoc_power(d, n0, n1, alpha, icc=0.0, obs_per_cluster=1)
        power_obs_clustered = post_hoc_power(d, n0, n1, alpha, icc=icc_est, obs_per_cluster=obs_per)
        mde = min_detectable_effect(n0, n1, alpha, power=0.80)
        mde_score = mde * pooled_sd
        n_needed = sample_size_for_power(d, alpha, power=0.80, ratio=n1 / n0)

        rows.append({
            "outcome": outcome,
            "n_cond0": n0,
            "n_cond1": n1,
            "obs_per_patient": obs_per,
            "observed_d": d,
            "observed_mean_diff": g1.mean() - g0.mean(),
            "pooled_sd_patient": pooled_sd,
            "icc_estimate": icc_est,
            "power_patient_level": power_obs,
            "power_clustered": power_obs_clustered,
            "mde_d_80pct": mde,
            "mde_score_units": mde_score,
            "n_per_group_for_80pct": n_needed,
            "alpha": alpha,
        })

        # Power curve
        _plot_power_curve(d, n0, n1, pooled_sd, alpha, outcome, out_dir)

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "power_analysis.csv", index=False)
    return result


def _plot_power_curve(
    d: float,
    n0: int,
    n1: int,
    pooled_sd: float,
    alpha: float,
    outcome: str,
    out_dir: Path,
) -> None:
    """Plot power vs sample size for the observed effect size."""
    n_range = np.arange(5, 201)
    ratio = n1 / n0 if n0 > 0 else 1.0
    powers = [
        post_hoc_power(d, n, int(n * ratio), alpha)
        for n in n_range
    ]

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.plot(n_range, powers, color="#7EC8C8", linewidth=2)
    ax.axhline(0.80, color="#E8998D", linewidth=1, linestyle="--", label="80% power")
    ax.axvline(n0, color="#F4A582", linewidth=1, linestyle=":", label=f"Current n₀={n0}")

    current_power = post_hoc_power(d, n0, n1, alpha)
    ax.scatter([n0], [current_power], s=60, color="#F4A582", zorder=5, edgecolors="#333333")
    ax.text(
        n0 + 3, current_power,
        f"{current_power:.1%}",
        fontsize=ANNOT_SIZE, va="center",
    )

    format_ax(
        ax,
        title=f"Power Curve — {outcome.title()} (d={d:.3f})",
        xlabel="n per group (smaller group)",
        ylabel="Power",
    )
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")

    fig.savefig(out_dir / f"power_curve_{outcome}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"power_curve_{outcome}.pdf", bbox_inches="tight")
    plt.close(fig)
