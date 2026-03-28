"""
Summary DataFrames for exploratory analysis.

Each function computes the tabular data underlying a figure,
so an agent or reviewer can inspect exact values without reading images.
All condition references use 0/1 only — blinding safe.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def score_frequency(
    df: pd.DataFrame,
    outcome: str,
    by: str | None = None,
) -> pd.DataFrame:
    """
    Count and proportion of each score (0–4), optionally grouped.

    Companion to: plot_score_distribution, plot_score_proportions.
    """
    scores = list(range(5))

    if by is None:
        counts = df[outcome].value_counts().reindex(scores, fill_value=0)
        total = len(df)
        return pd.DataFrame({
            "score": scores,
            "count": counts.values,
            "proportion": (counts / total).values,
            "outcome": outcome,
        })

    rows = []
    for grp, subset in df.groupby(by, sort=True):
        counts = subset[outcome].value_counts().reindex(scores, fill_value=0)
        total = len(subset)
        for score in scores:
            rows.append({
                by: grp,
                "score": score,
                "count": counts[score],
                "proportion": counts[score] / total if total else 0,
                "outcome": outcome,
            })
    return pd.DataFrame(rows)


def combined_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Score frequencies for depth and extent by condition."""
    parts = [
        score_frequency(df, outcome, by="condition")
        for outcome in ["depth", "extent"]
    ]
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def summary_by_group(
    df: pd.DataFrame,
    outcome: str,
    by: str,
) -> pd.DataFrame:
    """
    n, mean, sd, median, q1, q3, min, max per group.

    Companion to: violin, box, raincloud plots.
    """
    agg = (
        df.groupby(by)[outcome]
        .agg(
            n="count",
            mean="mean",
            sd="std",
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            min="min",
            max="max",
        )
        .reset_index()
    )
    agg.insert(0, "outcome", outcome)
    return agg


def summary_by_procedure_condition(
    df: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    """
    Stats grouped by procedure_code × condition.

    Companion to: plot_box_by_procedure.
    """
    agg = (
        df.groupby(["procedure_code", "condition"])[outcome]
        .agg(
            n="count",
            mean="mean",
            sd="std",
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            min="min",
            max="max",
        )
        .reset_index()
    )
    agg.insert(0, "outcome", outcome)
    return agg


def condition_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Stats for depth and extent by condition. Companion to: panel plot."""
    parts = [
        summary_by_group(df, outcome, by="condition")
        for outcome in ["depth", "extent"]
    ]
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def patient_quadrant_scores(
    df: pd.DataFrame,
    outcome: str,
    grader: str | None = None,
) -> pd.DataFrame:
    """
    Mean score per patient × quadrant (long format).

    Companion to: plot_patient_quadrant_heatmap.
    """
    work = df.copy()
    if grader is not None:
        work = work[work["grader"] == grader]

    result = (
        work.groupby(["patient", "quadrant"])[outcome]
        .mean()
        .reset_index()
        .rename(columns={outcome: "mean_score"})
    )
    result.insert(0, "outcome", outcome)
    result = result.sort_values(["patient", "quadrant"]).reset_index(drop=True)
    return result


def grader_difference(
    df: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    """
    Absolute difference between graders per patient × quadrant.

    Companion to: plot_grader_difference_heatmap.
    """
    graders = sorted(df["grader"].unique())
    g1 = df[df["grader"] == graders[0]].set_index(["patient", "quadrant"])[outcome]
    g2 = df[df["grader"] == graders[1]].set_index(["patient", "quadrant"])[outcome]

    diff = (g1 - g2).reset_index()
    diff.columns = ["patient", "quadrant", "diff"]
    diff["abs_diff"] = diff["diff"].abs()
    diff.insert(0, "outcome", outcome)
    diff.insert(3, "grader_1", graders[0])
    diff.insert(4, "grader_2", graders[1])
    return diff.sort_values(["patient", "quadrant"]).reset_index(drop=True)


def condition_quadrant_scores(
    df: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    """
    Mean score per condition × patient × quadrant.

    Companion to: plot_condition_heatmap_pair.
    """
    result = (
        df.groupby(["condition", "patient", "quadrant"])[outcome]
        .mean()
        .reset_index()
        .rename(columns={outcome: "mean_score"})
    )
    result.insert(0, "outcome", outcome)
    return result.sort_values(["condition", "patient", "quadrant"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------

def grader_pair_scores(
    df: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    """
    Wide-format: one row per image with both graders' scores.

    Companion to: plot_grader_scatter, plot_grader_bland_altman.
    """
    graders = sorted(df["grader"].unique())
    wide = df.pivot_table(
        index=["patient", "quadrant", "condition"],
        columns="grader",
        values=outcome,
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide["mean_score"] = (wide[graders[0]] + wide[graders[1]]) / 2
    wide["diff"] = wide[graders[0]] - wide[graders[1]]
    wide["abs_diff"] = wide["diff"].abs()
    wide.insert(0, "outcome", outcome)
    return wide


def agreement_matrix(
    df: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    """
    Cross-tabulation of Grader 1 × Grader 2 scores with counts and proportions.

    Companion to: plot_agreement_confusion.
    """
    graders = sorted(df["grader"].unique())
    wide = df.pivot_table(
        index=["patient", "quadrant"],
        columns="grader",
        values=outcome,
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    rows = []
    total = len(wide)
    for g1_score in range(5):
        for g2_score in range(5):
            n = ((wide[graders[0]] == g1_score) & (wide[graders[1]] == g2_score)).sum()
            rows.append({
                "outcome": outcome,
                f"{graders[0]}_score": g1_score,
                f"{graders[1]}_score": g2_score,
                "count": n,
                "proportion": n / total if total else 0,
                "exact_agreement": g1_score == g2_score,
            })
    return pd.DataFrame(rows)


def bland_altman_summary(
    df: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    """
    Bland-Altman summary statistics: mean diff, SD, limits of agreement.

    Companion to: plot_grader_bland_altman.
    """
    pair = grader_pair_scores(df, outcome)
    mean_diff = pair["diff"].mean()
    sd_diff = pair["diff"].std()

    return pd.DataFrame([{
        "outcome": outcome,
        "n_images": len(pair),
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "upper_loa": mean_diff + 1.96 * sd_diff,
        "lower_loa": mean_diff - 1.96 * sd_diff,
        "min_diff": pair["diff"].min(),
        "max_diff": pair["diff"].max(),
        "pct_exact_agreement": (pair["diff"] == 0).mean() * 100,
        "pct_within_1": (pair["abs_diff"] <= 1).mean() * 100,
    }])


# ---------------------------------------------------------------------------
# Patient Profiles
# ---------------------------------------------------------------------------

def patient_score_summary(
    df: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    """
    Per-patient summary: mean, sd, min, max, n across all quadrants/graders.

    Companion to: plot_patient_score_summary.
    """
    result = (
        df.groupby(["patient", "condition"])[outcome]
        .agg(
            n="count",
            mean="mean",
            sd="std",
            median="median",
            min="min",
            max="max",
        )
        .reset_index()
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)
    )
    result.insert(0, "outcome", outcome)
    return result


def quadrant_radar_data(
    df: pd.DataFrame,
    patient_id: int,
    grader: str | None = None,
) -> pd.DataFrame:
    """
    Scores per quadrant for one patient (both outcomes).

    Companion to: plot_quadrant_radar.
    """
    work = df[df["patient"] == patient_id].copy()
    if grader is not None:
        work = work[work["grader"] == grader]

    rows = []
    for outcome in ["depth", "extent"]:
        means = work.groupby("quadrant")[outcome].mean()
        for quad, val in means.items():
            rows.append({
                "patient": patient_id,
                "condition": work["condition"].iloc[0],
                "quadrant": quad,
                "outcome": outcome,
                "mean_score": val,
            })
    return pd.DataFrame(rows)


def top_bottom_patients(
    df: pd.DataFrame,
    outcome: str,
    n: int = 5,
) -> pd.DataFrame:
    """
    Top-n and bottom-n patients by mean score.

    Companion to: plot_top_bottom_patients.
    """
    stats = patient_score_summary(df, outcome)
    top = stats.head(n).assign(group="top")
    bottom = stats.tail(n).assign(group="bottom")
    return pd.concat([top, bottom], ignore_index=True)
