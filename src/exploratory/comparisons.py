"""
Side-by-side group comparison plots.

Violin, box, and raincloud plots comparing distributions across
condition, grader, and procedure code. No statistical tests.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.exploratory.style import (
    CONDITION_COLORS,
    CONDITION_COLOR_LIST,
    CONDITION_LABELS,
    GRADER_COLORS,
    PROCEDURE_COLORS,
    SCORE_COLOR_LIST,
    FIG_SINGLE,
    FIG_PANEL,
    FIG_WIDE,
    ANNOT_SIZE,
    annotate_median,
    condition_label,
    format_ax,
    set_publication_style,
)


def plot_violin_by_condition(
    df: pd.DataFrame,
    outcome: str,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Split violin plots for Condition 0 vs 1 with embedded box plots.

    Includes jittered individual points and annotated medians.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    conditions = [0, 1]
    data_groups = [df[df["condition"] == c][outcome].values for c in conditions]

    positions = [1, 2]
    parts = ax.violinplot(
        data_groups,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(CONDITION_COLORS[conditions[i]])
        body.set_edgecolor("#666666")
        body.set_alpha(0.7)

    # Embedded box plots
    bp = ax.boxplot(
        data_groups,
        positions=positions,
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        zorder=3,
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("white")
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.2)
    for element in ["whiskers", "caps"]:
        for line in bp[element]:
            line.set_color("#333333")
            line.set_linewidth(1)
    for line in bp["medians"]:
        line.set_color("#333333")
        line.set_linewidth(2)

    # Jittered strip
    rng = np.random.default_rng(42)
    for i, cond in enumerate(conditions):
        vals = data_groups[i]
        jitter = rng.uniform(-0.06, 0.06, size=len(vals))
        ax.scatter(
            positions[i] + jitter,
            vals,
            s=8,
            color=CONDITION_COLORS[cond],
            alpha=0.4,
            edgecolors="none",
            zorder=2,
        )

    # Median annotations
    for i, cond in enumerate(conditions):
        med = np.median(data_groups[i])
        annotate_median(ax, positions[i] + 0.2, med)

    ax.set_xticks(positions)
    ax.set_xticklabels([condition_label(c) for c in conditions])
    ax.set_yticks(range(5))
    format_ax(ax, title=f"{outcome.title()} by Condition", ylabel=f"{outcome.title()} Score")

    handles = [
        mpatches.Patch(color=CONDITION_COLORS[c], label=condition_label(c))
        for c in conditions
    ]
    ax.legend(handles=handles, loc="upper right")
    return fig, ax


def plot_box_by_procedure(
    df: pd.DataFrame,
    outcome: str,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Grouped box plots by procedure code, colored by condition.
    """
    set_publication_style()
    procedures = sorted(df["procedure_code"].unique())
    conditions = [0, 1]

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    width = 0.35
    all_positions = []
    all_data = []
    all_colors = []

    for pi, proc in enumerate(procedures):
        center = pi * 2
        for ci, cond in enumerate(conditions):
            subset = df[(df["procedure_code"] == proc) & (df["condition"] == cond)]
            pos = center + (ci - 0.5) * width * 1.5
            all_positions.append(pos)
            all_data.append(subset[outcome].values)
            all_colors.append(CONDITION_COLORS[cond])

    bp = ax.boxplot(
        all_data,
        positions=all_positions,
        widths=width,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(all_colors[i])
        patch.set_edgecolor("#333333")
        patch.set_alpha(0.8)
    for element in ["whiskers", "caps"]:
        for line in bp[element]:
            line.set_color("#333333")
    for line in bp["medians"]:
        line.set_color("#333333")
        line.set_linewidth(2)

    # Annotate medians
    for i, data in enumerate(all_data):
        if len(data) > 0:
            med = np.median(data)
            ax.text(
                all_positions[i],
                med + 0.15,
                f"{med:.1f}",
                ha="center",
                va="bottom",
                fontsize=ANNOT_SIZE,
                fontweight="bold",
                color="#333333",
            )

    ax.set_xticks([pi * 2 for pi in range(len(procedures))])
    ax.set_xticklabels([f"Procedure {p}" for p in procedures])
    ax.set_yticks(range(5))
    format_ax(
        ax,
        title=f"{outcome.title()} by Procedure Code and Condition",
        ylabel=f"{outcome.title()} Score",
    )

    handles = [
        mpatches.Patch(color=CONDITION_COLORS[c], label=condition_label(c))
        for c in conditions
    ]
    ax.legend(handles=handles, loc="upper right")
    return fig, ax


def plot_box_by_grader(
    df: pd.DataFrame,
    outcome: str,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Box plots comparing Grader 1 vs Grader 2 scoring patterns.
    """
    set_publication_style()
    graders = sorted(df["grader"].unique())

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    data_groups = [df[df["grader"] == g][outcome].values for g in graders]

    bp = ax.boxplot(
        data_groups,
        positions=list(range(1, len(graders) + 1)),
        widths=0.5,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(GRADER_COLORS[graders[i]])
        patch.set_edgecolor("#333333")
        patch.set_alpha(0.8)
    for element in ["whiskers", "caps"]:
        for line in bp[element]:
            line.set_color("#333333")
    for line in bp["medians"]:
        line.set_color("#333333")
        line.set_linewidth(2)

    # Jittered strip
    rng = np.random.default_rng(42)
    for i, g in enumerate(graders):
        vals = data_groups[i]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            (i + 1) + jitter,
            vals,
            s=6,
            color=GRADER_COLORS[g],
            alpha=0.3,
            edgecolors="none",
            zorder=2,
        )

    # Median annotations
    for i, g in enumerate(graders):
        med = np.median(data_groups[i])
        annotate_median(ax, i + 1.3, med)

    ax.set_xticks(range(1, len(graders) + 1))
    ax.set_xticklabels(graders)
    ax.set_yticks(range(5))
    format_ax(ax, title=f"{outcome.title()} by Grader", ylabel=f"{outcome.title()} Score")
    return fig, ax


def plot_raincloud(
    df: pd.DataFrame,
    outcome: str,
    by: str,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Raincloud plot: half-violin + jittered strip + box plot.

    Provides the most complete single-panel view of a distribution.
    """
    set_publication_style()
    groups = sorted(df[by].unique())
    color_map = _get_color_map(by, groups)

    fig, ax = plt.subplots(figsize=(7, max(3, len(groups) * 1.2)))

    rng = np.random.default_rng(42)
    for i, grp in enumerate(groups):
        vals = df[df[by] == grp][outcome].values
        y_center = i

        # Half-violin (upper half)
        parts = ax.violinplot(
            vals,
            positions=[y_center],
            vert=False,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in parts["bodies"]:
            # Clip to upper half
            m = np.mean(body.get_paths()[0].vertices[:, 1])
            body.get_paths()[0].vertices[:, 1] = np.clip(
                body.get_paths()[0].vertices[:, 1], m, None
            )
            body.set_facecolor(color_map[grp])
            body.set_edgecolor("#666666")
            body.set_alpha(0.6)

        # Jittered strip (lower half)
        jitter = rng.uniform(-0.15, -0.02, size=len(vals))
        ax.scatter(
            vals,
            y_center + jitter,
            s=8,
            color=color_map[grp],
            alpha=0.5,
            edgecolors="none",
            zorder=2,
        )

        # Box plot (narrow, centered below)
        bp = ax.boxplot(
            vals,
            positions=[y_center - 0.22],
            widths=0.08,
            vert=False,
            patch_artist=True,
            showfliers=False,
            zorder=3,
        )
        bp["boxes"][0].set_facecolor("white")
        bp["boxes"][0].set_edgecolor("#333333")
        bp["medians"][0].set_color("#333333")
        bp["medians"][0].set_linewidth(2)
        for element in ["whiskers", "caps"]:
            for line in bp[element]:
                line.set_color("#333333")

    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([_group_label(by, g) for g in groups])
    ax.set_xticks(range(5))
    format_ax(
        ax,
        title=f"{outcome.title()} Distribution by {by.replace('_', ' ').title()}",
        xlabel=f"{outcome.title()} Score",
    )
    ax.invert_yaxis()
    return fig, ax


def plot_condition_comparison_panel(
    df: pd.DataFrame,
) -> tuple[plt.Figure, np.ndarray]:
    """
    2×2 panel: depth violin, extent violin, depth proportions, extent proportions.
    """
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=FIG_PANEL)

    # Top row: violins
    for col_idx, outcome in enumerate(["depth", "extent"]):
        ax = axes[0, col_idx]
        conditions = [0, 1]
        data_groups = [df[df["condition"] == c][outcome].values for c in conditions]

        parts = ax.violinplot(data_groups, positions=[1, 2], showmeans=False, showmedians=False, showextrema=False)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(CONDITION_COLORS[conditions[i]])
            body.set_edgecolor("#666666")
            body.set_alpha(0.7)

        bp = ax.boxplot(data_groups, positions=[1, 2], widths=0.15, patch_artist=True, showfliers=False, zorder=3)
        for patch in bp["boxes"]:
            patch.set_facecolor("white")
            patch.set_edgecolor("#333333")
        for line in bp["medians"]:
            line.set_color("#333333")
            line.set_linewidth(2)
        for element in ["whiskers", "caps"]:
            for line in bp[element]:
                line.set_color("#333333")

        ax.set_xticks([1, 2])
        ax.set_xticklabels([condition_label(c) for c in conditions])
        ax.set_yticks(range(5))
        format_ax(ax, title=f"{outcome.title()} by Condition", ylabel="Score")

    # Bottom row: stacked proportions
    for col_idx, outcome in enumerate(["depth", "extent"]):
        ax = axes[1, col_idx]
        scores = list(range(5))
        conditions = [0, 1]
        x = np.arange(len(conditions))
        bottoms = np.zeros(len(conditions))

        for score in scores:
            heights = []
            for cond in conditions:
                subset = df[df["condition"] == cond]
                count = (subset[outcome] == score).sum()
                heights.append(count / len(subset))

            ax.bar(
                x,
                heights,
                bottom=bottoms,
                width=0.6,
                label=f"Score {score}",
                color=SCORE_COLOR_LIST[score],
                edgecolor="white",
                linewidth=0.5,
            )
            # Annotate if proportion >= 5%
            for j, h in enumerate(heights):
                if h >= 0.05:
                    ax.text(
                        x[j],
                        bottoms[j] + h / 2,
                        f"{h:.0%}",
                        ha="center",
                        va="center",
                        fontsize=ANNOT_SIZE - 1,
                        color="#333333",
                    )
            bottoms = bottoms + np.array(heights)

        ax.set_xticks(x)
        ax.set_xticklabels([condition_label(c) for c in conditions])
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        format_ax(ax, title=f"{outcome.title()} Score Composition", ylabel="Proportion")
        ax.grid(False)
        if col_idx == 1:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=ANNOT_SIZE)

    fig.suptitle("Condition Comparison Overview", fontsize=14, fontweight="bold", y=1.02)
    return fig, axes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_color_map(by: str, groups: list) -> dict:
    if by == "condition":
        return CONDITION_COLORS
    if by == "grader":
        return GRADER_COLORS
    if by == "procedure_code":
        return PROCEDURE_COLORS
    cmap = plt.cm.Pastel1
    return {g: cmap(i / max(len(groups) - 1, 1)) for i, g in enumerate(groups)}


def _group_label(by: str, val) -> str:
    if by == "condition":
        return condition_label(val)
    return str(val)
