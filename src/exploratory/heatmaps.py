"""
Patient x quadrant heatmaps.

Reveal spatial scoring patterns across quadrants and
systematic patient-level variation.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from src.exploratory.style import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    SCORE_COLOR_LIST,
    FIG_SINGLE,
    FIG_WIDE,
    ANNOT_SIZE,
    condition_label,
    format_ax,
    set_publication_style,
)


def plot_patient_quadrant_heatmap(
    df: pd.DataFrame,
    outcome: str,
    grader: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Heatmap: patients (y) x quadrants (x), color = score.

    Rows sorted by mean score descending. If grader is None,
    averages across both graders.

    Parameters
    ----------
    df : pd.DataFrame
    outcome : str
    grader : str, optional
        Filter to a specific grader, or None to average.
    """
    set_publication_style()
    work = df.copy()

    if grader is not None:
        work = work[work["grader"] == grader]

    pivot = (
        work.groupby(["patient", "quadrant"])[outcome]
        .mean()
        .reset_index()
        .pivot(index="patient", columns="quadrant", values=outcome)
    )

    # Sort by row mean descending
    pivot = pivot.assign(_mean=pivot.mean(axis=1)).sort_values("_mean", ascending=False).drop(columns="_mean")

    # Ensure quadrant columns are in order
    quad_order = sorted(pivot.columns)
    pivot = pivot[quad_order]

    cmap = mcolors.LinearSegmentedColormap.from_list("score_cmap", SCORE_COLOR_LIST, N=256)
    fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) * 0.3)))

    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=0, vmax=4)

    # Cell annotations
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            text_color = "white" if val >= 3 else "#333333"
            ax.text(
                j, i, f"{val:.1f}",
                ha="center", va="center",
                fontsize=ANNOT_SIZE, color=text_color,
            )

    ax.set_xticks(range(len(quad_order)))
    ax.set_xticklabels([q.upper() for q in quad_order])
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([f"Pt {pid}" for pid in pivot.index])

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, label=f"{outcome.title()} Score")
    cbar.set_ticks(range(5))

    grader_label = f" ({grader})" if grader else " (Avg)"
    format_ax(ax, title=f"{outcome.title()} Scores by Patient and Quadrant{grader_label}", xlabel="Quadrant")
    ax.grid(False)
    return fig, ax


def plot_grader_difference_heatmap(
    df: pd.DataFrame,
    outcome: str,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Heatmap of |Grader 1 - Grader 2| per image, highlighting disagreement.
    """
    set_publication_style()
    graders = sorted(df["grader"].unique())
    if len(graders) != 2:
        raise ValueError(f"Expected 2 graders, found {len(graders)}")

    g1 = df[df["grader"] == graders[0]].set_index(["patient", "quadrant"])[outcome]
    g2 = df[df["grader"] == graders[1]].set_index(["patient", "quadrant"])[outcome]
    diff = (g1 - g2).abs().reset_index()
    diff.columns = ["patient", "quadrant", "abs_diff"]

    pivot = diff.pivot(index="patient", columns="quadrant", values="abs_diff")
    pivot = pivot.assign(_mean=pivot.mean(axis=1)).sort_values("_mean", ascending=False).drop(columns="_mean")

    quad_order = sorted(pivot.columns)
    pivot = pivot[quad_order]

    # Diverging palette: white (0) to deep rose (max diff)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "diff_cmap", ["#FFFFFF", "#FDCDAC", "#E8998D", "#C46A6A"], N=256
    )
    fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) * 0.3)))

    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=0, vmax=4)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            text_color = "white" if val >= 3 else "#333333"
            ax.text(
                j, i, f"{val:.0f}",
                ha="center", va="center",
                fontsize=ANNOT_SIZE, color=text_color,
            )

    ax.set_xticks(range(len(quad_order)))
    ax.set_xticklabels([q.upper() for q in quad_order])
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([f"Pt {pid}" for pid in pivot.index])

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, label="|Grader 1 - Grader 2|")
    cbar.set_ticks(range(5))

    format_ax(ax, title=f"Grader Disagreement — {outcome.title()}", xlabel="Quadrant")
    ax.grid(False)
    return fig, ax


def plot_condition_heatmap_pair(
    df: pd.DataFrame,
    outcome: str,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Side-by-side heatmaps: Condition 0 patients (left) and Condition 1 (right).
    """
    set_publication_style()
    cmap = mcolors.LinearSegmentedColormap.from_list("score_cmap", SCORE_COLOR_LIST, N=256)

    conditions = [0, 1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    for ax, cond in zip(axes, conditions):
        subset = df[df["condition"] == cond]
        pivot = (
            subset.groupby(["patient", "quadrant"])[outcome]
            .mean()
            .reset_index()
            .pivot(index="patient", columns="quadrant", values=outcome)
        )
        pivot = pivot.assign(_mean=pivot.mean(axis=1)).sort_values("_mean", ascending=False).drop(columns="_mean")
        quad_order = sorted(pivot.columns)
        pivot = pivot[quad_order]

        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=0, vmax=4)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                text_color = "white" if val >= 3 else "#333333"
                ax.text(
                    j, i, f"{val:.1f}",
                    ha="center", va="center",
                    fontsize=ANNOT_SIZE - 1, color=text_color,
                )

        ax.set_xticks(range(len(quad_order)))
        ax.set_xticklabels([q.upper() for q in quad_order])
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels([f"Pt {pid}" for pid in pivot.index])
        format_ax(ax, title=condition_label(cond), xlabel="Quadrant")
        ax.grid(False)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label=f"{outcome.title()} Score").set_ticks(range(5))
    fig.suptitle(
        f"{outcome.title()} Scores by Condition — Patient × Quadrant",
        fontsize=14, fontweight="bold", y=1.02,
    )
    return fig, axes
