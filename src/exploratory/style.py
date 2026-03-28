"""
Publication style configuration, palettes, and shared helpers.

Every plotting module imports from here to ensure visual consistency.
All condition labels respect blinding — never reveal treatment identity.
"""

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.container import BarContainer

from src.config import get_path

# ---------------------------------------------------------------------------
# Blinding-safe labels
# ---------------------------------------------------------------------------
CONDITION_LABELS: dict[int, str] = {0: "Condition 0", 1: "Condition 1"}
GRADER_LABELS: dict[str, str] = {
    "Grader 1": "Grader 1",
    "Grader 2": "Grader 2",
}

# ---------------------------------------------------------------------------
# Palettes (pastel, colorblind-safe, print-friendly)
# ---------------------------------------------------------------------------
CONDITION_COLORS: dict[int, str] = {0: "#7EC8C8", 1: "#F4A582"}

SCORE_COLORS: dict[int, str] = {
    0: "#E0E0E0",
    1: "#A8D5E2",
    2: "#7EC8C8",
    3: "#F9D576",
    4: "#E8998D",
}

GRADER_COLORS: dict[str, str] = {
    "Grader 1": "#8DA0CB",
    "Grader 2": "#A6D854",
}

PROCEDURE_COLORS: dict[int, str] = {
    1: "#B3B3E6",
    2: "#FDCDAC",
    3: "#B3E2CD",
}

# Ordered lists for matplotlib convenience
CONDITION_COLOR_LIST: list[str] = [CONDITION_COLORS[0], CONDITION_COLORS[1]]
SCORE_COLOR_LIST: list[str] = [SCORE_COLORS[i] for i in range(5)]

# ---------------------------------------------------------------------------
# Typography and layout constants
# ---------------------------------------------------------------------------
FONT_FAMILY = "DejaVu Sans"
TITLE_SIZE = 12
LABEL_SIZE = 10
TICK_SIZE = 9
ANNOT_SIZE = 8

FIG_SINGLE = (6, 4)
FIG_PANEL = (10, 8)
FIG_WIDE = (10, 4)

SAVE_DPI = 300
SCREEN_DPI = 100

# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------

def set_publication_style() -> None:
    """Configure matplotlib rcParams for journal-quality figures."""
    mpl.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "Arial", "Helvetica"],
        "font.size": TICK_SIZE,
        # Axes
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold",
        "axes.labelsize": LABEL_SIZE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        # Grid
        "grid.alpha": 0.3,
        "grid.color": "#CCCCCC",
        "grid.linewidth": 0.5,
        # Ticks
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Legend
        "legend.fontsize": ANNOT_SIZE,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        # Figure
        "figure.dpi": SCREEN_DPI,
        "savefig.dpi": SAVE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "figure.constrained_layout.use": True,
        # Lines
        "lines.linewidth": 1.5,
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_figure(
    fig: plt.Figure,
    name: str,
    path_key: str = "output_exploratory",
) -> Path:
    """
    Save figure to the directory specified by a config path key.

    Parameters
    ----------
    fig : matplotlib Figure
    name : str
        Base filename without extension (e.g. 'score_distribution_depth').
    path_key : str
        Key under `paths:` in config.yaml. Defaults to 'output_exploratory'.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    fig_dir = get_path(path_key, mkdir=True)

    png_path = fig_dir / f"{name}.png"
    pdf_path = fig_dir / f"{name}.pdf"

    fig.savefig(png_path)
    fig.savefig(pdf_path)
    return png_path


def save_dataframe(
    df: pd.DataFrame,
    name: str,
    path_key: str = "output_exploratory",
) -> Path:
    """
    Save a DataFrame as CSV alongside its companion figure.

    Parameters
    ----------
    df : pd.DataFrame
        Summary data underlying a figure.
    name : str
        Base filename without extension (matches the figure name).
    path_key : str
        Key under `paths:` in config.yaml. Defaults to 'output_exploratory'.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    out_dir = get_path(path_key, mkdir=True)
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def annotate_bars(
    ax: plt.Axes,
    bars: BarContainer | Sequence,
    fmt: str = "{:.0f}",
    offset: float = 0.5,
    fontsize: int = ANNOT_SIZE,
) -> None:
    """Add value labels above each bar."""
    for bar in bars:
        height = bar.get_height()
        if height == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#333333",
        )


def annotate_bars_pct(
    ax: plt.Axes,
    bars: BarContainer | Sequence,
    total: int,
    offset: float = 0.5,
    fontsize: int = ANNOT_SIZE,
) -> None:
    """Add count (percentage) labels above each bar."""
    for bar in bars:
        height = bar.get_height()
        if height == 0:
            continue
        pct = 100 * height / total if total > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{int(height)} ({pct:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#333333",
        )


def annotate_median(
    ax: plt.Axes,
    x: float,
    median_val: float,
    color: str = "#333333",
) -> None:
    """Add a median annotation at a given x position."""
    ax.text(
        x,
        median_val,
        f" {median_val:.1f}",
        ha="left",
        va="center",
        fontsize=ANNOT_SIZE,
        fontweight="bold",
        color=color,
    )


def condition_label(val: int) -> str:
    """Return the blinding-safe label for a condition value."""
    return CONDITION_LABELS.get(val, f"Condition {val}")


def format_ax(
    ax: plt.Axes,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    """Apply consistent title and axis labels."""
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
