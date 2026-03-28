"""
Entry point for exploratory analysis.

Generates all publication-grade exploratory figures and companion
CSV DataFrames, saved to the directory defined in config.yaml
(paths.output_exploratory).

Usage:
    python -m src.exploratory.run
    python -m src.exploratory.run --outcomes depth
    python -m src.exploratory.run --skip heatmaps agreement
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import get_path, load_config
from src.data.loader import load_raw_data
from src.exploratory.style import save_dataframe, save_figure, set_publication_style
from src.exploratory import (
    distributions,
    comparisons,
    heatmaps,
    agreement,
    patient_profiles,
)
from src.exploratory import summaries


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exploratory analysis figures and data tables."
    )
    parser.add_argument(
        "--outcomes",
        nargs="+",
        default=None,
        help="Outcomes to plot (default: all from config).",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["distributions", "comparisons", "heatmaps", "agreement", "profiles"],
        help="Module groups to skip.",
    )
    return parser.parse_args(argv)


def _save(fig, df_data, name):
    """Save figure (PNG+PDF) and companion CSV, then close figure."""
    save_figure(fig, name)
    save_dataframe(df_data, name)
    plt.close(fig)


def run(argv: list[str] | None = None) -> None:
    """Generate all exploratory figures and companion CSVs."""
    args = _parse_args(argv)
    cfg = load_config()
    outcomes = args.outcomes or cfg["data"]["outcome_columns"]
    skip = set(args.skip)

    set_publication_style()
    df = load_raw_data()
    out_dir = get_path("output_exploratory", mkdir=True)

    print(f"Data loaded: {df.shape[0]} rows, {df['patient'].nunique()} patients")
    print(f"Outcomes: {outcomes}")
    print(f"Output directory: {out_dir}")
    print()

    generated = 0

    # --- Distributions ---
    if "distributions" not in skip:
        print("== Distributions ==")
        for outcome in outcomes:
            name = f"dist_{outcome}_by_condition"
            fig, _ = distributions.plot_score_distribution(df, outcome, by="condition")
            data = summaries.score_frequency(df, outcome, by="condition")
            _save(fig, data, name)
            print(f"  {name}")
            generated += 1

            name = f"prop_{outcome}_by_condition"
            fig, _ = distributions.plot_score_proportions(df, outcome, by="condition")
            _save(fig, data, name)  # same underlying data
            print(f"  {name}")
            generated += 1

        name = "combined_distribution"
        fig, _ = distributions.plot_combined_distribution(df)
        data = summaries.combined_distribution(df)
        _save(fig, data, name)
        print(f"  {name}")
        generated += 1

    # --- Comparisons ---
    if "comparisons" not in skip:
        print("== Comparisons ==")
        for outcome in outcomes:
            name = f"violin_{outcome}_condition"
            fig, _ = comparisons.plot_violin_by_condition(df, outcome)
            data = summaries.summary_by_group(df, outcome, by="condition")
            _save(fig, data, name)
            print(f"  {name}")
            generated += 1

            name = f"box_{outcome}_procedure"
            fig, _ = comparisons.plot_box_by_procedure(df, outcome)
            data = summaries.summary_by_procedure_condition(df, outcome)
            _save(fig, data, name)
            print(f"  {name}")
            generated += 1

            name = f"box_{outcome}_grader"
            fig, _ = comparisons.plot_box_by_grader(df, outcome)
            data = summaries.summary_by_group(df, outcome, by="grader")
            _save(fig, data, name)
            print(f"  {name}")
            generated += 1

            name = f"raincloud_{outcome}_condition"
            fig, _ = comparisons.plot_raincloud(df, outcome, by="condition")
            data = summaries.summary_by_group(df, outcome, by="condition")
            _save(fig, data, name)
            print(f"  {name}")
            generated += 1

        name = "condition_comparison_panel"
        fig, _ = comparisons.plot_condition_comparison_panel(df)
        data = summaries.condition_comparison(df)
        _save(fig, data, name)
        print(f"  {name}")
        generated += 1

    # --- Heatmaps ---
    if "heatmaps" not in skip:
        print("== Heatmaps ==")
        for outcome in outcomes:
            name = f"heatmap_{outcome}_patient_quadrant"
            fig, _ = heatmaps.plot_patient_quadrant_heatmap(df, outcome)
            data = summaries.patient_quadrant_scores(df, outcome)
            _save(fig, data, name)
            print(f"  {name}")
            generated += 1

            name = f"heatmap_grader_diff_{outcome}"
            fig, _ = heatmaps.plot_grader_difference_heatmap(df, outcome)
            data = summaries.grader_difference(df, outcome)
            _save(fig, data, name)
            print(f"  {name}")
            generated += 1

            name = f"heatmap_condition_pair_{outcome}"
            fig, _ = heatmaps.plot_condition_heatmap_pair(df, outcome)
            data = summaries.condition_quadrant_scores(df, outcome)
            _save(fig, data, name)
            print(f"  {name}")
            generated += 1

    # --- Agreement ---
    if "agreement" not in skip:
        print("== Agreement ==")
        for outcome in outcomes:
            name = f"grader_scatter_{outcome}"
            fig, _ = agreement.plot_grader_scatter(df, outcome)
            data = summaries.grader_pair_scores(df, outcome)
            _save(fig, data, name)
            print(f"  {name}")
            generated += 1

            name = f"agreement_confusion_{outcome}"
            fig, _ = agreement.plot_agreement_confusion(df, outcome)
            data = summaries.agreement_matrix(df, outcome)
            _save(fig, data, name)
            print(f"  {name}")
            generated += 1

            name = f"bland_altman_{outcome}"
            fig, _ = agreement.plot_grader_bland_altman(df, outcome)
            data = summaries.bland_altman_summary(df, outcome)
            _save(fig, data, name)
            print(f"  {name}")
            generated += 1

    # --- Patient Profiles ---
    if "profiles" not in skip:
        print("== Patient Profiles ==")
        for outcome in outcomes:
            name = f"patient_summary_{outcome}"
            fig, _ = patient_profiles.plot_patient_score_summary(df, outcome)
            data = summaries.patient_score_summary(df, outcome)
            _save(fig, data, name)
            print(f"  {name}")
            generated += 1

            name = f"top_bottom_{outcome}"
            fig, _ = patient_profiles.plot_top_bottom_patients(df, outcome, n=5)
            data = summaries.top_bottom_patients(df, outcome, n=5)
            _save(fig, data, name)
            print(f"  {name}")
            generated += 1

        sample_patients = sorted(df["patient"].unique())[:4]
        for pid in sample_patients:
            name = f"radar_patient_{pid}"
            fig, _ = patient_profiles.plot_quadrant_radar(df, patient_id=pid)
            data = summaries.quadrant_radar_data(df, pid)
            _save(fig, data, name)
            print(f"  {name}")
            generated += 1

    print()
    print(f"Done — {generated} figure+CSV pairs saved to {out_dir}")


if __name__ == "__main__":
    run()
