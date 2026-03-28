"""
Score frequency and proportion plots.

Visualize how depth and extent scores (0–4) are distributed
overall and by grouping variables.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exploratory.style import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    SCORE_COLOR_LIST,
    SCORE_COLORS,
    FIG_SINGLE,
    FIG_WIDE,
    ANNOT_SIZE,
    annotate_bars_pct,
    condition_label,
    format_ax,
    set_publication_style,
)


def plot_score_distribution(
    df: pd.DataFrame,
    outcome: str,
    by: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Grouped bar chart of score frequencies (0–4).

    Parameters
    ----------
    df : pd.DataFrame
        Raw data from load_raw_data().
    outcome : str
        'depth' or 'extent'.
    by : str, optional
        Grouping column (e.g. 'condition'). If None, plots overall distribution.

    Returns
    -------
    (fig, ax)
    """
    set_publication_style()
    scores = list(range(5))

    if by is None:
        counts = df[outcome].value_counts().reindex(scores, fill_value=0)
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        bars = ax.bar(scores, counts.values, color=SCORE_COLOR_LIST, edgecolor="white")
        annotate_bars_pct(ax, bars, total=len(df))
        format_ax(ax, title=f"{outcome.title()} Score Distribution", xlabel="Score", ylabel="Count")
        ax.set_xticks(scores)
        return fig, ax

    groups = sorted(df[by].unique())
    n_groups = len(groups)
    width = 0.8 / n_groups
    x = np.arange(len(scores))

    color_map = _get_color_map(by, groups)

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    for i, grp in enumerate(groups):
        subset = df[df[by] == grp]
        counts = subset[outcome].value_counts().reindex(scores, fill_value=0)
        offset = (i - (n_groups - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            counts.values,
            width=width * 0.9,
            label=_group_label(by, grp),
            color=color_map[grp],
            edgecolor="white",
        )
        annotate_bars_pct(ax, bars, total=len(subset), fontsize=ANNOT_SIZE - 1)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in scores])
    format_ax(
        ax,
        title=f"{outcome.title()} Score Distribution by {by.title()}",
        xlabel="Score",
        ylabel="Count",
    )
    ax.legend(loc="upper right")
    return fig, ax


def plot_score_proportions(
    df: pd.DataFrame,
    outcome: str,
    by: str,
) -> tuple[plt.Figure, plt.Axes]:
    """
    100% stacked horizontal bar chart showing score composition per group.

    Parameters
    ----------
    df : pd.DataFrame
    outcome : str
    by : str

    Returns
    -------
    (fig, ax)
    """
    set_publication_style()
    scores = list(range(5))
    groups = sorted(df[by].unique())

    proportions = {}
    for grp in groups:
        subset = df[df[by] == grp]
        counts = subset[outcome].value_counts().reindex(scores, fill_value=0)
        proportions[grp] = counts / len(subset)

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    y_pos = np.arange(len(groups))

    lefts = np.zeros(len(groups))
    for score in scores:
        widths = [proportions[grp][score] for grp in groups]
        bars = ax.barh(
            y_pos,
            widths,
            left=lefts,
            height=0.6,
            label=f"Score {score}",
            color=SCORE_COLORS[score],
            edgecolor="white",
            linewidth=0.5,
        )
        # Annotate cells with percentage if >= 5%
        for j, w in enumerate(widths):
            if w >= 0.05:
                ax.text(
                    lefts[j] + w / 2,
                    y_pos[j],
                    f"{w:.0%}",
                    ha="center",
                    va="center",
                    fontsize=ANNOT_SIZE,
                    color="#333333",
                )
        lefts = lefts + np.array(widths)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([_group_label(by, g) for g in groups])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    format_ax(ax, title=f"{outcome.title()} Score Composition by {by.title()}", xlabel="Proportion")
    ax.legend(loc="lower right", ncol=5)
    ax.grid(False)
    return fig, ax


def plot_combined_distribution(
    df: pd.DataFrame,
) -> tuple[plt.Figure, np.ndarray]:
    """
    2×1 panel: depth distribution on top, extent on bottom, split by condition.

    Returns
    -------
    (fig, axes) where axes is an array of 2 Axes
    """
    set_publication_style()
    fig, axes = plt.subplots(2, 1, figsize=(8, 7))

    for ax, outcome in zip(axes, ["depth", "extent"]):
        scores = list(range(5))
        width = 0.35
        x = np.arange(len(scores))

        for i, cond in enumerate([0, 1]):
            subset = df[df["condition"] == cond]
            counts = subset[outcome].value_counts().reindex(scores, fill_value=0)
            offset = -width / 2 + i * width
            bars = ax.bar(
                x + offset,
                counts.values,
                width=width * 0.9,
                label=condition_label(cond),
                color=CONDITION_COLORS[cond],
                edgecolor="white",
            )
            annotate_bars_pct(ax, bars, total=len(subset), fontsize=ANNOT_SIZE - 1)

        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in scores])
        format_ax(ax, title=f"{outcome.title()} Score Distribution by Condition", xlabel="Score", ylabel="Count")
        ax.legend(loc="upper right")

    return fig, axes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_color_map(by: str, groups: list) -> dict:
    """Return an appropriate color mapping for the grouping variable."""
    if by == "condition":
        return CONDITION_COLORS
    from src.exploratory.style import GRADER_COLORS, PROCEDURE_COLORS
    if by == "grader":
        return GRADER_COLORS
    if by == "procedure_code":
        return PROCEDURE_COLORS
    # Fallback: generate from a pastel colormap
    cmap = plt.cm.Pastel1
    return {g: cmap(i / max(len(groups) - 1, 1)) for i, g in enumerate(groups)}


def _group_label(by: str, val) -> str:
    """Return a display label for a group value."""
    if by == "condition":
        return condition_label(val)
    return str(val)
