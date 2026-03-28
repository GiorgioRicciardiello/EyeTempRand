"""
Grader agreement visualizations.

Visual exploration of inter-rater scoring patterns.
No statistical tests — ICC and kappa live in src/analysis/icc.py.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.exploratory.style import (
    GRADER_COLORS,
    SCORE_COLOR_LIST,
    FIG_SINGLE,
    ANNOT_SIZE,
    format_ax,
    set_publication_style,
)


def plot_grader_scatter(
    df: pd.DataFrame,
    outcome: str,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Scatter of Grader 1 vs Grader 2 scores with jitter.

    Point size proportional to count at each coordinate.
    Diagonal reference line marks perfect agreement.
    Marginal histograms on top and right.
    """
    set_publication_style()

    graders = sorted(df["grader"].unique())
    if len(graders) != 2:
        raise ValueError(f"Expected 2 graders, found {len(graders)}")

    wide = _pivot_graders(df, outcome, graders)

    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        wspace=0.05,
        hspace=0.05,
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Count at each (g1, g2) pair
    counts = wide.groupby([graders[0], graders[1]]).size().reset_index(name="n")

    # Bubble scatter
    rng = np.random.default_rng(42)
    for _, row in counts.iterrows():
        g1_val, g2_val, n = row[graders[0]], row[graders[1]], row["n"]
        ax_main.scatter(
            g1_val + rng.uniform(-0.1, 0.1),
            g2_val + rng.uniform(-0.1, 0.1),
            s=n * 15,
            color="#7EC8C8",
            alpha=0.6,
            edgecolors="#333333",
            linewidth=0.5,
            zorder=3,
        )
        ax_main.text(
            g1_val + 0.18,
            g2_val + 0.18,
            str(n),
            fontsize=ANNOT_SIZE,
            color="#333333",
            ha="left",
        )

    # Diagonal
    ax_main.plot([-0.5, 4.5], [-0.5, 4.5], "--", color="#999999", linewidth=1, zorder=1)
    ax_main.set_xlim(-0.5, 4.5)
    ax_main.set_ylim(-0.5, 4.5)
    ax_main.set_xticks(range(5))
    ax_main.set_yticks(range(5))
    format_ax(ax_main, xlabel=graders[0], ylabel=graders[1])

    # Marginal histograms
    bins = np.arange(-0.5, 5.5, 1)
    ax_top.hist(wide[graders[0]], bins=bins, color=GRADER_COLORS[graders[0]], edgecolor="white", alpha=0.7)
    ax_top.tick_params(labelbottom=False)
    ax_top.set_ylabel("Count")
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    ax_right.hist(
        wide[graders[1]], bins=bins, orientation="horizontal",
        color=GRADER_COLORS[graders[1]], edgecolor="white", alpha=0.7,
    )
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel("Count")
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    fig.suptitle(
        f"Grader Agreement — {outcome.title()}",
        fontsize=12, fontweight="bold",
    )
    return fig, ax_main


def plot_agreement_confusion(
    df: pd.DataFrame,
    outcome: str,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Confusion-matrix-style heatmap of Grader 1 x Grader 2 scores.

    Cells on the diagonal represent perfect agreement.
    Off-diagonal cells are shaded warmer to highlight disagreement.
    """
    set_publication_style()
    graders = sorted(df["grader"].unique())
    wide = _pivot_graders(df, outcome, graders)

    scores = list(range(5))
    matrix = np.zeros((5, 5), dtype=int)
    for _, row in wide.iterrows():
        g1 = int(row[graders[0]])
        g2 = int(row[graders[1]])
        matrix[g1, g2] += 1

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="equal")

    for i in range(5):
        for j in range(5):
            val = matrix[i, j]
            total = matrix.sum()
            pct = 100 * val / total if total > 0 else 0
            text_color = "white" if val > matrix.max() * 0.6 else "#333333"
            ax.text(
                j, i,
                f"{val}\n({pct:.1f}%)",
                ha="center", va="center",
                fontsize=ANNOT_SIZE, color=text_color,
            )

    ax.set_xticks(scores)
    ax.set_yticks(scores)
    format_ax(ax, title=f"Grader Agreement Matrix — {outcome.title()}", xlabel=graders[1], ylabel=graders[0])
    fig.colorbar(im, ax=ax, shrink=0.7, label="Count")
    ax.grid(False)

    # Highlight diagonal
    for i in range(5):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#333333", linewidth=2))

    return fig, ax


def plot_grader_bland_altman(
    df: pd.DataFrame,
    outcome: str,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Bland-Altman plot: x = mean of graders, y = difference (G1 - G2).

    Horizontal lines for mean difference and +/- 1.96 SD limits.
    """
    set_publication_style()
    graders = sorted(df["grader"].unique())
    wide = _pivot_graders(df, outcome, graders)

    mean_score = (wide[graders[0]] + wide[graders[1]]) / 2
    diff = wide[graders[0]] - wide[graders[1]]

    mean_diff = diff.mean()
    sd_diff = diff.std()
    upper = mean_diff + 1.96 * sd_diff
    lower = mean_diff - 1.96 * sd_diff

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    # Jitter for overlapping ordinal values
    rng = np.random.default_rng(42)
    jitter_x = rng.uniform(-0.08, 0.08, size=len(mean_score))
    jitter_y = rng.uniform(-0.08, 0.08, size=len(diff))

    ax.scatter(
        mean_score + jitter_x,
        diff + jitter_y,
        s=20, color="#7EC8C8", alpha=0.5, edgecolors="#333333", linewidth=0.3,
    )

    # Reference lines
    ax.axhline(mean_diff, color="#333333", linewidth=1.5, linestyle="-", label=f"Mean diff = {mean_diff:.2f}")
    ax.axhline(upper, color="#E8998D", linewidth=1, linestyle="--", label=f"+1.96 SD = {upper:.2f}")
    ax.axhline(lower, color="#E8998D", linewidth=1, linestyle="--", label=f"-1.96 SD = {lower:.2f}")
    ax.axhline(0, color="#CCCCCC", linewidth=0.5, linestyle=":")

    # Annotations on the right margin
    x_max = ax.get_xlim()[1]
    for val, label in [(mean_diff, f"{mean_diff:.2f}"), (upper, f"{upper:.2f}"), (lower, f"{lower:.2f}")]:
        ax.text(x_max + 0.05, val, label, va="center", fontsize=ANNOT_SIZE, color="#333333")

    format_ax(
        ax,
        title=f"Bland-Altman Plot — {outcome.title()}",
        xlabel=f"Mean of {graders[0]} and {graders[1]}",
        ylabel=f"Difference ({graders[0]} - {graders[1]})",
    )
    ax.legend(loc="lower left", fontsize=ANNOT_SIZE)
    return fig, ax


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pivot_graders(df: pd.DataFrame, outcome: str, graders: list[str]) -> pd.DataFrame:
    """Pivot data so each row has one column per grader's score."""
    wide = df.pivot_table(
        index=["patient", "quadrant"],
        columns="grader",
        values=outcome,
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide
