"""
Model diagnostic plots and tests.

Residual analysis, QQ plots, influence measures, and collinearity checks
for every fitted model. All plots use publication style from exploratory.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from src.exploratory.style import (
    set_publication_style,
    save_figure,
    save_dataframe,
    FIG_SINGLE,
    ANNOT_SIZE,
    format_ax,
)


def run_lmm_diagnostics(
    result,
    model_name: str,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Full diagnostic suite for a statsmodels MixedLMResults object.

    Parameters
    ----------
    result : MixedLMResults
    model_name : str
        Used for filenames (e.g. 'lmm_depth').
    out_dir : Path
        Directory to save plots and CSVs.

    Returns
    -------
    pd.DataFrame
        Summary of diagnostic test results.
    """
    set_publication_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    resid = result.resid
    fitted = result.fittedvalues
    std_resid = (resid - resid.mean()) / resid.std()

    diag_results = {}

    # --- Residuals vs Fitted ---
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.scatter(fitted, resid, s=10, alpha=0.4, color="#7EC8C8", edgecolors="none")
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    outliers = np.abs(std_resid) > 2
    if outliers.any():
        ax.scatter(
            fitted[outliers], resid[outliers],
            s=20, color="#E8998D", edgecolors="#333333", linewidth=0.5, zorder=3,
        )
    format_ax(ax, title=f"Residuals vs Fitted — {model_name}", xlabel="Fitted values", ylabel="Residuals")
    fig.savefig(out_dir / f"diag_{model_name}_residuals_vs_fitted.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"diag_{model_name}_residuals_vs_fitted.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- QQ plot of residuals ---
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    (osm, osr), (slope, intercept, r) = stats.probplot(std_resid, dist="norm")
    ax.scatter(osm, osr, s=12, alpha=0.5, color="#7EC8C8", edgecolors="none")
    ax.plot(osm, slope * np.array(osm) + intercept, "--", color="#333333", linewidth=1)
    format_ax(ax, title=f"QQ Plot — {model_name}", xlabel="Theoretical quantiles", ylabel="Standardized residuals")
    ax.text(
        0.05, 0.95, f"r = {r:.3f}",
        transform=ax.transAxes, fontsize=ANNOT_SIZE, va="top",
    )
    fig.savefig(out_dir / f"diag_{model_name}_qq_residuals.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"diag_{model_name}_qq_residuals.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Scale-location ---
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    sqrt_abs_resid = np.sqrt(np.abs(std_resid))
    ax.scatter(fitted, sqrt_abs_resid, s=10, alpha=0.4, color="#8DA0CB", edgecolors="none")
    # Lowess trend
    try:
        import statsmodels.nonparametric.smoothers_lowess as lowess
        smoothed = lowess.lowess(sqrt_abs_resid, fitted, frac=0.6)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="#E8998D", linewidth=2)
    except Exception:
        pass
    format_ax(
        ax, title=f"Scale-Location — {model_name}",
        xlabel="Fitted values", ylabel="\u221A|Standardized residuals|",
    )
    fig.savefig(out_dir / f"diag_{model_name}_scale_location.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"diag_{model_name}_scale_location.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Histogram of residuals ---
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.hist(std_resid, bins=25, density=True, color="#A8D5E2", edgecolor="white", alpha=0.8)
    x_range = np.linspace(std_resid.min() - 0.5, std_resid.max() + 0.5, 200)
    ax.plot(x_range, stats.norm.pdf(x_range), "--", color="#333333", linewidth=1.5, label="Normal")
    format_ax(ax, title=f"Residual Distribution — {model_name}", xlabel="Standardized residual", ylabel="Density")
    ax.legend()
    fig.savefig(out_dir / f"diag_{model_name}_hist_residuals.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"diag_{model_name}_hist_residuals.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Shapiro-Wilk test ---
    n_test = min(len(std_resid), 5000)
    sample = std_resid[:n_test] if len(std_resid) > 5000 else std_resid
    sw_stat, sw_p = stats.shapiro(sample)
    diag_results["shapiro_w"] = sw_stat
    diag_results["shapiro_p"] = sw_p
    diag_results["normality_ok"] = sw_p > 0.05

    # --- Random effects QQ ---
    try:
        re = result.random_effects
        blups = np.array([v.iloc[0] if hasattr(v, "iloc") else list(v.values())[0] for v in re.values()])
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        (osm_re, osr_re), (sl_re, int_re, r_re) = stats.probplot(blups, dist="norm")
        ax.scatter(osm_re, osr_re, s=20, alpha=0.7, color="#F4A582", edgecolors="#333333", linewidth=0.5)
        ax.plot(osm_re, sl_re * np.array(osm_re) + int_re, "--", color="#333333", linewidth=1)
        format_ax(
            ax, title=f"QQ — Random Effects — {model_name}",
            xlabel="Theoretical quantiles", ylabel="Random intercepts",
        )
        fig.savefig(out_dir / f"diag_{model_name}_qq_random_effects.png", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / f"diag_{model_name}_qq_random_effects.pdf", bbox_inches="tight")
        plt.close(fig)

        sw_re_stat, sw_re_p = stats.shapiro(blups)
        diag_results["re_shapiro_w"] = sw_re_stat
        diag_results["re_shapiro_p"] = sw_re_p
        diag_results["re_normality_ok"] = sw_re_p > 0.05
        diag_results["n_random_groups"] = len(blups)
    except Exception:
        diag_results["re_shapiro_w"] = np.nan
        diag_results["re_shapiro_p"] = np.nan
        diag_results["re_normality_ok"] = None

    # --- Summary stats ---
    diag_results["n_obs"] = len(resid)
    diag_results["n_outliers_2sd"] = int(outliers.sum())
    diag_results["resid_mean"] = resid.mean()
    diag_results["resid_sd"] = resid.std()
    diag_results["resid_skew"] = stats.skew(resid)
    diag_results["resid_kurtosis"] = stats.kurtosis(resid)
    diag_results["model_name"] = model_name

    diag_df = pd.DataFrame([diag_results])
    diag_df.to_csv(out_dir / f"diagnostics_{model_name}.csv", index=False)
    return diag_df


def collinearity_check(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """
    Cross-tabulate categorical predictors to check balance.

    For this study: condition × procedure_code.
    """
    if len(predictors) < 2:
        return pd.DataFrame()

    crosstab = pd.crosstab(df[predictors[0]], df[predictors[1]], margins=True)
    return crosstab.reset_index()


def influence_on_condition(
    df: pd.DataFrame,
    outcome: str,
    model_name: str,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Leave-one-patient-out analysis for the condition coefficient.

    Refits LMM dropping each patient, records change in condition beta.
    """
    import statsmodels.formula.api as smf

    set_publication_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    formula = f"{outcome} ~ C(condition) + C(procedure_code)"
    full_model = smf.mixedlm(formula=formula, data=df, groups=df["patient"])
    full_result = full_model.fit(reml=True)
    full_beta = full_result.fe_params.get("C(condition)[T.1]", np.nan)

    patients = sorted(df["patient"].unique())
    rows = []
    for pid in patients:
        subset = df[df["patient"] != pid]
        try:
            loo_model = smf.mixedlm(formula=formula, data=subset, groups=subset["patient"])
            loo_result = loo_model.fit(reml=True)
            loo_beta = loo_result.fe_params.get("C(condition)[T.1]", np.nan)
        except Exception:
            loo_beta = np.nan
        rows.append({
            "patient_dropped": pid,
            "condition_beta": loo_beta,
            "beta_change": loo_beta - full_beta,
            "pct_change": 100 * (loo_beta - full_beta) / abs(full_beta) if full_beta != 0 else np.nan,
        })

    result_df = pd.DataFrame(rows)
    result_df["full_beta"] = full_beta

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E8998D" if abs(r["pct_change"]) > 10 else "#7EC8C8" for _, r in result_df.iterrows()]
    ax.barh(range(len(result_df)), result_df["beta_change"], color=colors, edgecolor="#333333", linewidth=0.3)
    ax.set_yticks(range(len(result_df)))
    ax.set_yticklabels([f"Pt {int(p)}" for p in result_df["patient_dropped"]], fontsize=7)
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    format_ax(
        ax, title=f"Influence on Condition Effect — {model_name}",
        xlabel="\u0394\u03B2 (condition)", ylabel="Patient dropped",
    )
    fig.savefig(out_dir / f"diag_{model_name}_influence_condition.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"diag_{model_name}_influence_condition.pdf", bbox_inches="tight")
    plt.close(fig)

    result_df.to_csv(out_dir / f"influence_{model_name}.csv", index=False)
    return result_df
