"""
Inter-rater reliability analysis.

Computes ICC and weighted Cohen's kappa between Grader 1 and Grader 2
for depth and extent scores. Must be run before primary analysis to
determine whether grader scores should be averaged or modeled separately.
"""

import pandas as pd
import pingouin as pg
from src.config import load_config


def compute_icc(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """
    Compute two-way mixed ICC (absolute agreement) for a single outcome.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with columns: patient, quadrant, grader, and the outcome column.
    outcome : str
        Column name ('depth' or 'extent').

    Returns
    -------
    pd.DataFrame
        Pingouin ICC results table.
    """
    icc_data = df[["patient", "quadrant", "grader", outcome]].copy()
    result = pg.intraclass_corr(
        data=icc_data,
        targets="quadrant",   # each image is a target
        raters="grader",
        ratings=outcome,
    )
    return result


def compute_weighted_kappa(df: pd.DataFrame, outcome: str) -> dict:
    """
    Compute weighted Cohen's kappa between graders for a single outcome.

    Parameters
    ----------
    df : pd.DataFrame
    outcome : str

    Returns
    -------
    dict with keys: kappa, ci_lower, ci_upper, pvalue
    """
    from sklearn.metrics import cohen_kappa_score

    pivot = (
        df.pivot_table(
            index=["patient", "quadrant"],
            columns="grader",
            values=outcome,
            aggfunc="first",
        )
        .reset_index()
    )

    g1 = pivot["Grader 1"].values
    g2 = pivot["Grader 2"].values
    kappa = cohen_kappa_score(g1, g2, weights="quadratic")
    return {"outcome": outcome, "weighted_kappa": round(kappa, 4)}


def run_icc_analysis(df: pd.DataFrame) -> dict:
    """
    Run ICC and kappa for both depth and extent.

    Returns
    -------
    dict with keys 'icc' and 'kappa', each a dict keyed by outcome name.
    """
    cfg = load_config()
    outcomes = cfg["data"]["outcome_columns"]
    threshold = cfg["analysis"]["icc_threshold_acceptable"]

    results = {"icc": {}, "kappa": {}, "recommendation": {}}

    for outcome in outcomes:
        icc_table = compute_icc(df, outcome)
        kappa_result = compute_weighted_kappa(df, outcome)

        icc2 = icc_table.loc[icc_table["Type"] == "ICC2", "ICC"].values[0]
        results["icc"][outcome] = icc_table
        results["kappa"][outcome] = kappa_result

        if icc2 >= threshold:
            results["recommendation"][outcome] = (
                f"ICC2={icc2:.3f} >= {threshold}: scores may be averaged across graders."
            )
        else:
            results["recommendation"][outcome] = (
                f"ICC2={icc2:.3f} < {threshold}: retain grader as a model factor."
            )

    return results


def average_grader_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average depth and extent scores across graders per image.

    Use only after confirming ICC >= threshold from run_icc_analysis().

    Returns
    -------
    pd.DataFrame
        One row per (patient, quadrant) with averaged scores.
        Grader column is dropped.
    """
    return (
        df.groupby(["patient", "quadrant", "condition", "procedure_code",
                    "Randomized_Filename", "Original_Filename"],
                   as_index=False)
        [["depth", "extent"]]
        .mean()
    )
