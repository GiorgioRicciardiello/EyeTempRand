"""
Per-patient score summaries.

Identify outliers, unusual scoring profiles, and individual variation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.exploratory.style import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    SCORE_COLOR_LIST,
    FIG_SINGLE,
    FIG_PANEL,
    ANNOT_SIZE,
    condition_label,
    format_ax,
    set_publication_style,
)


def plot_patient_score_summary(
    df: pd.DataFrame,
    outcome: str,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Dot plot: each patient on y-axis, mean score on x-axis.

    Colored by condition. Error bars show range across
    quadrants and graders. Patients sorted by mean score.
    """
    set_publication_style()

    patient_stats = (
        df.groupby(["patient", "condition"])[outcome]
        .agg(mean="mean", min="min", max="max")
        .reset_index()
        .sort_values("mean", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(7, max(6, len(patient_stats) * 0.28)))

    y_positions = range(len(patient_stats))
    for i, (_, row) in enumerate(patient_stats.iterrows()):
        color = CONDITION_COLORS[int(row["condition"])]
        ax.errorbar(
            row["mean"],
            i,
            xerr=[[row["mean"] - row["min"]], [row["max"] - row["mean"]]],
            fmt="o",
            markersize=6,
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
            alpha=0.8,
        )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([f"Pt {int(pid)}" for pid in patient_stats["patient"]])
    ax.set_xticks(range(5))
    ax.set_xlim(-0.3, 4.3)
    format_ax(
        ax,
        title=f"Patient {outcome.title()} Scores (Mean and Range)",
        xlabel=f"{outcome.title()} Score",
    )

    handles = [
        mpatches.Patch(color=CONDITION_COLORS[c], label=condition_label(c))
        for c in [0, 1]
    ]
    ax.legend(handles=handles, loc="lower right")
    return fig, ax


def plot_quadrant_radar(
    df: pd.DataFrame,
    patient_id: int,
    grader: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Radar chart of scores across 8 quadrants for a single patient.

    Parameters
    ----------
    df : pd.DataFrame
    patient_id : int
    grader : str, optional
        If None, averages across graders.
    """
    set_publication_style()

    work = df[df["patient"] == patient_id].copy()
    if grader is not None:
        work = work[work["grader"] == grader]

    if len(work) == 0:
        raise ValueError(f"No data for patient {patient_id}" + (f" grader {grader}" if grader else ""))

    outcomes = ["depth", "extent"]
    quad_order = sorted(work["quadrant"].unique())
    n_quads = len(quad_order)
    angles = np.linspace(0, 2 * np.pi, n_quads, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"polar": True})

    colors = ["#7EC8C8", "#F4A582"]
    for ax, outcome, color in zip(axes, outcomes, colors):
        means = work.groupby("quadrant")[outcome].mean().reindex(quad_order)
        values = means.tolist()
        values += values[:1]

        ax.plot(angles, values, "o-", color=color, linewidth=2, markersize=6)
        ax.fill(angles, values, alpha=0.2, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([q.upper() for q in quad_order])
        ax.set_ylim(0, 4)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(["1", "2", "3", "4"], fontsize=ANNOT_SIZE - 1)
        ax.set_title(outcome.title(), fontsize=11, fontweight="bold", pad=15)

        # Annotate each vertex
        for angle, val in zip(angles[:-1], means.values):
            ax.text(
                angle, val + 0.3, f"{val:.1f}",
                ha="center", fontsize=ANNOT_SIZE, color="#333333",
            )

    grader_label = f" ({grader})" if grader else " (Avg)"
    cond = df[df["patient"] == patient_id]["condition"].iloc[0]
    fig.suptitle(
        f"Patient {patient_id} — {condition_label(cond)}{grader_label}",
        fontsize=12, fontweight="bold",
    )
    return fig, axes


def plot_top_bottom_patients(
    df: pd.DataFrame,
    outcome: str,
    n: int = 5,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Panel showing the n highest and n lowest scoring patients
    with their quadrant breakdown as stacked bars.
    """
    set_publication_style()

    patient_means = (
        df.groupby(["patient", "condition"])[outcome]
        .mean()
        .reset_index()
        .rename(columns={outcome: "mean_score"})
        .sort_values("mean_score", ascending=False)
    )

    top = patient_means.head(n)
    bottom = patient_means.tail(n).sort_values("mean_score", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, group, title_label in zip(axes, [top, bottom], [f"Top {n}", f"Bottom {n}"]):
        _plot_patient_quadrant_bars(ax, df, group, outcome)
        format_ax(ax, title=f"{title_label} Patients — {outcome.title()}", xlabel=f"{outcome.title()} Score")
        ax.set_xlim(0, 4)

    axes[0].set_ylabel("Patient")
    fig.suptitle(
        f"Highest and Lowest {outcome.title()} Scores by Patient",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Shared legend
    handles = [
        mpatches.Patch(color=CONDITION_COLORS[0], label=condition_label(0)),
        mpatches.Patch(color=CONDITION_COLORS[1], label=condition_label(1)),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=ANNOT_SIZE)
    return fig, axes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _plot_patient_quadrant_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    patient_group: pd.DataFrame,
    outcome: str,
) -> None:
    """Draw horizontal bars for a set of patients with mean and range."""
    patients = patient_group["patient"].tolist()
    conditions = patient_group.set_index("patient")["condition"].to_dict()

    y_pos = list(range(len(patients)))
    for i, pid in enumerate(patients):
        subset = df[df["patient"] == pid]
        mean_val = subset[outcome].mean()
        min_val = subset[outcome].min()
        max_val = subset[outcome].max()
        color = CONDITION_COLORS[int(conditions[pid])]

        ax.barh(i, mean_val, height=0.6, color=color, alpha=0.7, edgecolor="#333333", linewidth=0.5)
        ax.errorbar(
            mean_val, i,
            xerr=[[mean_val - min_val], [max_val - mean_val]],
            fmt="none", ecolor="#333333", elinewidth=1, capsize=3,
        )
        ax.text(mean_val + 0.1, i, f"{mean_val:.2f}", va="center", fontsize=ANNOT_SIZE, color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Pt {int(pid)}" for pid in patients])
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
