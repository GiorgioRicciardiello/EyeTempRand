"""
Sensitivity analyses and robustness checks.

Tests whether primary findings are robust to:
- Different random effect structures
- Interaction with procedure code
- Grader handling
- Zero-inflation (hurdle model)
- Non-parametric alternatives
- Multiple comparisons correction
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from pathlib import Path

from src.config import load_config
from src.analysis.models import fit_lmm, summarize_lmm
from src.analysis.effect_size import cliffs_delta


# ---------------------------------------------------------------------------
# Random structure comparison
# ---------------------------------------------------------------------------

def compare_random_structures(
    df: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    """
    Fit models with different random effect specifications.

    1. (1 | patient) — primary
    2. No random effects (OLS) — naive baseline, expect anti-conservative SE
    3. (1 | patient) + quadrant random slope (if convergence allows)
    """
    results = []

    # Model 1: patient RE (primary)
    lmm = fit_lmm(df, outcome, include_grader=True)
    fe_lmm = summarize_lmm(lmm, model_name=f"{outcome}_patient_re")
    cond_row = fe_lmm[fe_lmm["term"].str.contains("condition")]
    results.append({
        "outcome": outcome,
        "model": "patient_re",
        "condition_beta": cond_row["estimate"].values[0] if len(cond_row) > 0 else np.nan,
        "condition_se": cond_row["se"].values[0] if len(cond_row) > 0 else np.nan,
        "condition_p": cond_row["pvalue"].values[0] if len(cond_row) > 0 else np.nan,
        "aic": lmm.aic if hasattr(lmm, "aic") else np.nan,
    })

    # Model 2: naive OLS
    try:
        formula = f"{outcome} ~ C(condition) + C(procedure_code) + C(grader)"
        ols = smf.ols(formula, data=df).fit()
        cond_beta = ols.params.get("C(condition)[T.1]", np.nan)
        cond_se = ols.bse.get("C(condition)[T.1]", np.nan)
        cond_p = ols.pvalues.get("C(condition)[T.1]", np.nan)
        results.append({
            "outcome": outcome,
            "model": "naive_ols",
            "condition_beta": cond_beta,
            "condition_se": cond_se,
            "condition_p": cond_p,
            "aic": ols.aic,
        })
    except Exception:
        results.append({
            "outcome": outcome, "model": "naive_ols",
            "condition_beta": np.nan, "condition_se": np.nan,
            "condition_p": np.nan, "aic": np.nan,
        })

    # Model 3: patient RE + quadrant nested RE
    try:
        formula = f"{outcome} ~ C(condition) + C(procedure_code) + C(grader)"
        vc = {"quadrant": "0 + C(quadrant)"}
        lmm2 = smf.mixedlm(formula=formula, data=df, groups=df["patient"], vc_formula=vc)
        lmm2_result = lmm2.fit(reml=True)
        fe2 = summarize_lmm(lmm2_result)
        cond_row2 = fe2[fe2["term"].str.contains("condition")]
        results.append({
            "outcome": outcome,
            "model": "patient_re_quadrant_vc",
            "condition_beta": cond_row2["estimate"].values[0] if len(cond_row2) > 0 else np.nan,
            "condition_se": cond_row2["se"].values[0] if len(cond_row2) > 0 else np.nan,
            "condition_p": cond_row2["pvalue"].values[0] if len(cond_row2) > 0 else np.nan,
            "aic": lmm2_result.aic if hasattr(lmm2_result, "aic") else np.nan,
        })
    except Exception:
        results.append({
            "outcome": outcome, "model": "patient_re_quadrant_vc",
            "condition_beta": np.nan, "condition_se": np.nan,
            "condition_p": np.nan, "aic": np.nan,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Interaction: condition × procedure_code
# ---------------------------------------------------------------------------

def test_interaction(
    df: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    """
    Test condition × procedure_code interaction and report stratified effects.
    """
    from src.analysis.models import fit_lmm_interaction

    result = fit_lmm_interaction(df, outcome, include_grader=True)
    fe = summarize_lmm(result, model_name=f"{outcome}_interaction")

    # Stratified estimates
    strat_rows = []
    for proc in sorted(df["procedure_code"].unique()):
        subset = df[df["procedure_code"] == proc]
        n_patients = subset["patient"].nunique()
        if n_patients < 3:
            strat_rows.append({
                "outcome": outcome,
                "procedure_code": proc,
                "n_patients": n_patients,
                "condition_beta": np.nan,
                "condition_p": np.nan,
                "note": "too few patients for stratified model",
            })
            continue
        try:
            strat_lmm = fit_lmm(subset, outcome, include_grader=True)
            strat_fe = summarize_lmm(strat_lmm)
            cond = strat_fe[strat_fe["term"].str.contains("condition")]
            strat_rows.append({
                "outcome": outcome,
                "procedure_code": proc,
                "n_patients": n_patients,
                "condition_beta": cond["estimate"].values[0] if len(cond) > 0 else np.nan,
                "condition_se": cond["se"].values[0] if len(cond) > 0 else np.nan,
                "condition_p": cond["pvalue"].values[0] if len(cond) > 0 else np.nan,
                "note": "",
            })
        except Exception as e:
            strat_rows.append({
                "outcome": outcome, "procedure_code": proc,
                "n_patients": n_patients,
                "condition_beta": np.nan, "condition_p": np.nan,
                "note": str(e),
            })

    return pd.DataFrame(strat_rows)


# ---------------------------------------------------------------------------
# Grader sensitivity
# ---------------------------------------------------------------------------

def grader_sensitivity(
    df: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    """
    Compare: grader as fixed effect vs. averaged scores.
    """
    from src.analysis.icc import average_grader_scores

    rows = []

    # With grader as fixed
    lmm_grader = fit_lmm(df, outcome, include_grader=True)
    fe1 = summarize_lmm(lmm_grader)
    cond1 = fe1[fe1["term"].str.contains("condition")]
    rows.append({
        "outcome": outcome,
        "grader_handling": "fixed_effect",
        "n_obs": len(df),
        "condition_beta": cond1["estimate"].values[0] if len(cond1) > 0 else np.nan,
        "condition_se": cond1["se"].values[0] if len(cond1) > 0 else np.nan,
        "condition_p": cond1["pvalue"].values[0] if len(cond1) > 0 else np.nan,
    })

    # Averaged across graders
    avg_df = average_grader_scores(df)
    lmm_avg = fit_lmm(avg_df, outcome, include_grader=False)
    fe2 = summarize_lmm(lmm_avg)
    cond2 = fe2[fe2["term"].str.contains("condition")]
    rows.append({
        "outcome": outcome,
        "grader_handling": "averaged",
        "n_obs": len(avg_df),
        "condition_beta": cond2["estimate"].values[0] if len(cond2) > 0 else np.nan,
        "condition_se": cond2["se"].values[0] if len(cond2) > 0 else np.nan,
        "condition_p": cond2["pvalue"].values[0] if len(cond2) > 0 else np.nan,
    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Non-parametric tests
# ---------------------------------------------------------------------------

def nonparametric_tests(
    df: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    """
    Non-parametric alternatives on patient-level mean scores.

    - Mann-Whitney U (Wilcoxon rank-sum)
    - Permutation test (10,000 shuffles)
    - Cliff's delta
    """
    cfg = load_config()
    n_perm = cfg["analysis"]["n_permutations"]
    seed = cfg["analysis"]["random_seed"]

    patient_means = df.groupby(["patient", "condition"])[outcome].mean().reset_index()
    g0 = patient_means[patient_means["condition"] == 0][outcome].values
    g1 = patient_means[patient_means["condition"] == 1][outcome].values

    # Mann-Whitney
    u_stat, mw_p = stats.mannwhitneyu(g0, g1, alternative="two-sided")

    # Permutation test
    observed_diff = np.mean(g1) - np.mean(g0)
    combined = np.concatenate([g0, g1])
    rng = np.random.default_rng(seed)
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(combined)
        perm_diffs[i] = np.mean(combined[len(g0):]) - np.mean(combined[:len(g0)])

    perm_p = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    # Cliff's delta
    cliff = cliffs_delta(g1, g0)

    return pd.DataFrame([{
        "outcome": outcome,
        "test": "patient_level_means",
        "n_cond0": len(g0),
        "n_cond1": len(g1),
        "mean_diff": observed_diff,
        "mann_whitney_u": u_stat,
        "mann_whitney_p": mw_p,
        "permutation_p": perm_p,
        "n_permutations": n_perm,
        "cliffs_delta": cliff["delta"],
        "cliffs_interpretation": cliff["interpretation"],
    }])


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------

def adjust_pvalues(p_values: list[float], labels: list[str]) -> pd.DataFrame:
    """
    Apply Bonferroni and Benjamini-Hochberg corrections.
    """
    from statsmodels.stats.multitest import multipletests

    p_arr = np.array(p_values)
    n_tests = len(p_arr)

    # Bonferroni
    bonf = np.minimum(p_arr * n_tests, 1.0)

    # BH-FDR
    _, fdr, _, _ = multipletests(p_arr, method="fdr_bh")

    return pd.DataFrame({
        "test": labels,
        "p_raw": p_arr,
        "p_bonferroni": bonf,
        "p_fdr_bh": fdr,
        "significant_raw": p_arr < 0.05,
        "significant_bonferroni": bonf < 0.05,
        "significant_fdr": fdr < 0.05,
        "n_tests": n_tests,
    })


# ---------------------------------------------------------------------------
# Bias assessment
# ---------------------------------------------------------------------------

def assess_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check balance of procedure_code across conditions.

    Reports counts and standardized mean differences.
    """
    rows = []
    for proc in sorted(df["procedure_code"].unique()):
        n0 = len(df[(df["condition"] == 0) & (df["procedure_code"] == proc)])
        n1 = len(df[(df["condition"] == 1) & (df["procedure_code"] == proc)])
        p0 = df[df["condition"] == 0]["patient"].nunique()
        p1 = df[df["condition"] == 1]["patient"].nunique()
        rows.append({
            "procedure_code": proc,
            "n_obs_cond0": n0,
            "n_obs_cond1": n1,
            "n_patients_cond0": len(df[(df["condition"] == 0) & (df["procedure_code"] == proc)]["patient"].unique()),
            "n_patients_cond1": len(df[(df["condition"] == 1) & (df["procedure_code"] == proc)]["patient"].unique()),
        })

    result = pd.DataFrame(rows)
    result["pct_cond0"] = result["n_patients_cond0"] / result["n_patients_cond0"].sum() * 100
    result["pct_cond1"] = result["n_patients_cond1"] / result["n_patients_cond1"].sum() * 100
    return result
