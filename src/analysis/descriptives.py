"""
Descriptive statistics.

Generates summary tables for outcomes by condition, grader, and procedure code.
All output uses condition 0/1 labels — never cold/warm — until apply_unblinding()
is called at the final reporting stage.
"""

import pandas as pd


def score_summary(df: pd.DataFrame, by: list[str], outcomes: list[str] | None = None) -> pd.DataFrame:
    """
    Summary statistics (n, mean, sd, median, IQR) for outcome scores.

    Parameters
    ----------
    df : pd.DataFrame
    by : list of str
        Grouping columns, e.g. ['condition'] or ['condition', 'grader'].
    outcomes : list of str, optional
        Defaults to ['depth', 'extent'].

    Returns
    -------
    pd.DataFrame
    """
    if outcomes is None:
        outcomes = ["depth", "extent"]

    rows = []
    for outcome in outcomes:
        grp = df.groupby(by)[outcome]
        summary = grp.agg(
            n="count",
            mean="mean",
            sd="std",
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
        ).reset_index()
        summary.insert(0, "outcome", outcome)
        rows.append(summary)

    return pd.concat(rows, ignore_index=True)


def score_distribution(df: pd.DataFrame, outcome: str, by: list[str]) -> pd.DataFrame:
    """
    Frequency table: count and proportion of each score value (0–4) by group.

    Parameters
    ----------
    df : pd.DataFrame
    outcome : str
    by : list of str

    Returns
    -------
    pd.DataFrame
    """
    counts = (
        df.groupby(by + [outcome])
        .size()
        .reset_index(name="n")
    )
    totals = df.groupby(by).size().reset_index(name="total")
    result = counts.merge(totals, on=by)
    result["proportion"] = result["n"] / result["total"]
    return result


def any_bruising_rate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """
    Proportion of images with any bruising (score > 0) for depth and extent.

    Parameters
    ----------
    df : pd.DataFrame
    by : list of str

    Returns
    -------
    pd.DataFrame
    """
    result = df.copy()
    result["depth_any"] = (result["depth"] > 0).astype(int)
    result["extent_any"] = (result["extent"] > 0).astype(int)

    return (
        result.groupby(by)[["depth_any", "extent_any"]]
        .agg(["sum", "mean"])
        .reset_index()
    )
