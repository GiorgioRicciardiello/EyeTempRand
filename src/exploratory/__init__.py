"""
Exploratory visualization library for EyeTempRand.

Publication-grade figures for distribution exploration and group comparisons.
No statistical inference — purely descriptive visuals.
"""

from src.exploratory.style import (
    set_publication_style,
    save_figure,
    save_dataframe,
    CONDITION_COLORS,
    CONDITION_LABELS,
    SCORE_COLORS,
    GRADER_COLORS,
    PROCEDURE_COLORS,
)
from src.exploratory import summaries
from src.exploratory.distributions import (
    plot_score_distribution,
    plot_score_proportions,
    plot_combined_distribution,
)
from src.exploratory.comparisons import (
    plot_violin_by_condition,
    plot_box_by_procedure,
    plot_box_by_grader,
    plot_raincloud,
    plot_condition_comparison_panel,
)
from src.exploratory.heatmaps import (
    plot_patient_quadrant_heatmap,
    plot_grader_difference_heatmap,
    plot_condition_heatmap_pair,
)
from src.exploratory.agreement import (
    plot_grader_scatter,
    plot_agreement_confusion,
    plot_grader_bland_altman,
)
from src.exploratory.patient_profiles import (
    plot_patient_score_summary,
    plot_quadrant_radar,
    plot_top_bottom_patients,
)

__all__ = [
    "set_publication_style",
    "save_figure",
    "save_dataframe",
    "summaries",
    "CONDITION_COLORS",
    "CONDITION_LABELS",
    "SCORE_COLORS",
    "GRADER_COLORS",
    "PROCEDURE_COLORS",
    "plot_score_distribution",
    "plot_score_proportions",
    "plot_combined_distribution",
    "plot_violin_by_condition",
    "plot_box_by_procedure",
    "plot_box_by_grader",
    "plot_raincloud",
    "plot_condition_comparison_panel",
    "plot_patient_quadrant_heatmap",
    "plot_grader_difference_heatmap",
    "plot_condition_heatmap_pair",
    "plot_grader_scatter",
    "plot_agreement_confusion",
    "plot_grader_bland_altman",
    "plot_patient_score_summary",
    "plot_quadrant_radar",
    "plot_top_bottom_patients",
]
