"""
Entry point for the full statistical analysis pipeline.

Orchestrates all 9 phases and produces the master summary table.

Usage:
    python -m src.analysis.run_analysis
    python -m src.analysis.run_analysis --skip ordinal
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.config import get_path, load_config
from src.data.loader import load_raw_data, add_binary_outcomes
from src.analysis import icc as icc_module
from src.analysis import models, diagnostics, effect_size, power, composite, sensitivity


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run full statistical analysis.")
    parser.add_argument(
        "--skip", nargs="*", default=[],
        choices=["icc", "lmm", "ordinal", "gee_ordinal", "conditional_logistic",
                 "binary", "composite", "diagnostics", "effect_sizes", "power", "sensitivity"],
    )
    return parser.parse_args(argv)


def run(argv=None):
    args = _parse_args(argv)
    skip = set(args.skip)
    cfg = load_config()
    alpha = cfg["analysis"]["alpha"]
    outcomes = cfg["data"]["outcome_columns"]

    base_dir = get_path("output_analysis", mkdir=True)

    df = load_raw_data()
    df = add_binary_outcomes(df, threshold=cfg["analysis"]["binary_threshold"])
    df = add_binary_outcomes(
        df.rename(columns={"depth_binary": "_d1", "extent_binary": "_e1"}),
        threshold=cfg["analysis"]["clinically_significant_threshold"],
    )
    # Restore names — threshold 1
    df = df.rename(columns={"_d1": "depth_binary_1", "_e1": "extent_binary_1"})
    df = df.rename(columns={"depth_binary": "depth_binary_2", "extent_binary": "extent_binary_2"})

    # Build composite
    df = composite.build_composite(df)
    all_outcomes = outcomes + ["composite"]

    master_rows = []

    print(f"Data: {len(df)} rows, {df['patient'].nunique()} patients")
    print(f"Conditions: {df.groupby('condition')['patient'].nunique().to_dict()}")
    print(f"Alpha: {alpha}")
    print()

    # ================================================================
    # PHASE 0: ICC
    # ================================================================
    if "icc" not in skip:
        print("== Phase 0: Inter-Rater Reliability ==")
        phase_dir = base_dir / "phase0_icc"
        phase_dir.mkdir(parents=True, exist_ok=True)

        icc_results = icc_module.run_icc_analysis(df)
        for outcome in outcomes:
            icc_table = icc_results["icc"][outcome]
            icc_table.to_csv(phase_dir / f"icc_{outcome}.csv", index=False)
            print(f"  {outcome}: {icc_results['recommendation'][outcome]}")

        # Grader bias test
        bias_rows = []
        graders = sorted(df["grader"].unique())
        for outcome in outcomes:
            wide = df.pivot_table(
                index=["patient", "quadrant"], columns="grader",
                values=outcome, aggfunc="first",
            ).reset_index()
            diff = wide[graders[0]] - wide[graders[1]]
            t_stat, t_p = _safe_ttest(diff)
            w_stat, w_p = _safe_wilcoxon(diff)
            bias_rows.append({
                "outcome": outcome,
                "mean_diff_g1_minus_g2": diff.mean(),
                "sd_diff": diff.std(),
                "paired_t_stat": t_stat, "paired_t_p": t_p,
                "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
                "systematic_bias": t_p < alpha,
            })
        bias_df = pd.DataFrame(bias_rows)
        bias_df.to_csv(phase_dir / "grader_bias_test.csv", index=False)
        print(f"  Grader bias: {bias_df[['outcome','mean_diff_g1_minus_g2','paired_t_p']].to_string(index=False)}")
        print()

    # ================================================================
    # PHASE 1: LMM
    # ================================================================
    if "lmm" not in skip:
        print("== Phase 1: Linear Mixed Models ==")
        phase_dir = base_dir / "phase1_lmm"
        phase_dir.mkdir(parents=True, exist_ok=True)

        variance_rows = []
        for outcome in all_outcomes:
            result = models.fit_lmm(df, outcome, include_grader=True)
            fe = models.summarize_lmm(result, model_name=f"lmm_{outcome}")
            fe.to_csv(phase_dir / f"lmm_{outcome}_results.csv", index=False)

            vc = models.extract_variance_components(result, model_name=f"lmm_{outcome}")
            variance_rows.append(vc)

            cond = fe[fe["term"].str.contains("condition")]
            if len(cond) > 0:
                beta = cond["estimate"].values[0]
                p = cond["pvalue"].values[0]
                ci_l = cond["ci_lower"].values[0]
                ci_u = cond["ci_upper"].values[0]
                se = cond["se"].values[0]
                sig = "YES" if p < alpha else "no"
                d = effect_size.cohens_d_from_model(beta, np.sqrt(result.scale))
                print(f"  {outcome}: beta={beta:.4f}, 95%CI=[{ci_l:.4f}, {ci_u:.4f}], "
                      f"p={p:.4f} ({sig}), d={d:.3f}")
                master_rows.append(_master_row(
                    phase="1_lmm", outcome=outcome, model_type="LMM",
                    estimate=beta, se=se, ci_lower=ci_l, ci_upper=ci_u,
                    p_value=p, effect_size_type="cohens_d", effect_size_val=d,
                    n_patients=df["patient"].nunique(), n_obs=len(df),
                ))

        pd.concat(variance_rows).to_csv(phase_dir / "lmm_variance_components.csv", index=False)
        print()

    # ================================================================
    # PHASE 2: CLMM (ordinal)
    # ================================================================
    if "ordinal" not in skip:
        print("== Phase 2: Ordinal Mixed Models (CLMM) ==")
        phase_dir = base_dir / "phase2_ordinal"
        phase_dir.mkdir(parents=True, exist_ok=True)

        for outcome in outcomes:
            clmm_result = models.fit_clmm(df, outcome, include_grader=True)
            if clmm_result.get("model_type") == "clmm":
                clmm_result["coefficients"].to_csv(phase_dir / f"clmm_{outcome}_results.csv", index=False)
                or_val = clmm_result["condition_or"]
                p = clmm_result["condition_p"]
                sig = "YES" if p < alpha else "no"
                print(f"  {outcome}: OR={or_val:.3f}, p={p:.4f} ({sig})")
                master_rows.append(_master_row(
                    phase="2_ordinal", outcome=outcome, model_type="CLMM",
                    estimate=clmm_result["condition_log_or"],
                    se=np.nan, ci_lower=clmm_result["condition_or_ci_lower"],
                    ci_upper=clmm_result["condition_or_ci_upper"],
                    p_value=p, effect_size_type="proportional_OR",
                    effect_size_val=or_val,
                    n_patients=df["patient"].nunique(), n_obs=len(df),
                ))
            else:
                print(f"  {outcome}: {clmm_result.get('warning', 'fallback to LMM')}")
                pd.DataFrame([{"note": clmm_result.get("warning", "")}]).to_csv(
                    phase_dir / f"clmm_{outcome}_fallback.csv", index=False
                )
        print()

    # ================================================================
    # PHASE 2.5: GEE for Ordinal Outcomes (paired structure)
    # ================================================================
    if "gee_ordinal" not in skip:
        print("== Phase 2.5: GEE for Ordinal Outcomes (Paired Structure) ==")
        phase_dir = base_dir / "phase2p5_gee_ordinal"
        phase_dir.mkdir(parents=True, exist_ok=True)

        print("  NOTE: GEE with exchangeable correlation respects within-patient pairing")
        print("  and provides population-averaged estimates robust to non-normality.")
        print()

        for outcome in outcomes:
            try:
                gee_ord_result = models.fit_gee_ordinal(
                    df, outcome, include_grader=True, corr_struct="exchangeable"
                )
                gee_ord_fe = models.summarize_gee_ordinal(gee_ord_result, model_name=f"gee_ord_{outcome}")
                gee_ord_fe.to_csv(phase_dir / f"gee_ordinal_{outcome}_results.csv", index=False)

                cond = gee_ord_fe[gee_ord_fe["term"].str.contains("condition")]
                if len(cond) > 0:
                    beta = cond["estimate"].values[0]
                    se = cond["se"].values[0]
                    p = cond["pvalue"].values[0]
                    ci_l = cond["ci_lower"].values[0]
                    ci_u = cond["ci_upper"].values[0]
                    sig = "YES" if p < alpha else "no"
                    d = effect_size.cohens_d_from_model(beta, np.sqrt(gee_ord_result.scale))
                    print(f"  {outcome}: beta={beta:.4f}, 95%CI=[{ci_l:.4f}, {ci_u:.4f}], "
                          f"p={p:.4f} ({sig}), d={d:.3f}")
                    master_rows.append(_master_row(
                        phase="2p5_gee_ordinal", outcome=outcome, model_type="GEE-Ordinal",
                        estimate=beta, se=se, ci_lower=ci_l, ci_upper=ci_u,
                        p_value=p, effect_size_type="cohens_d", effect_size_val=d,
                        n_patients=df["patient"].nunique(), n_obs=len(df),
                        notes="Population-averaged; exchangeable within-patient correlation",
                    ))
            except Exception as e:
                print(f"  {outcome}: GEE ordinal failed — {e}")
        print()

    # ================================================================
    # PHASE 3: Conditional Logistic Regression (paired binary outcomes)
    # ================================================================
    if "conditional_logistic" not in skip:
        print("== Phase 3: Conditional Logistic Regression (Paired Binary) ==")
        phase_dir = base_dir / "phase3_conditional_logistic"
        phase_dir.mkdir(parents=True, exist_ok=True)

        print("  NOTE: Conditional logistic is the epidemiological gold standard for")
        print("  matched/paired case-control and cohort studies. It eliminates patient-level")
        print("  nuisance parameters entirely via conditional likelihood.")
        print("  Requires eye-level condition assignment; currently falling back to GLMM.")
        print()

        for outcome in outcomes:
            for threshold, label in [(1, "any"), (2, "significant")]:
                bin_col = f"{outcome}_binary_{threshold}"
                if bin_col not in df.columns:
                    continue

                try:
                    clogit_result = models.fit_conditional_logistic(df, bin_col)
                    if clogit_result.get("model_type") == "conditional_logistic":
                        clogit_result["coefficients"].to_csv(
                            phase_dir / f"clogit_{outcome}_{label}_coefficients.csv", index=False
                        )
                        or_val = clogit_result["condition_or"]
                        p = clogit_result["condition_p"]
                        sig = "YES" if p < alpha else "no"
                        print(f"  {outcome} {label}: OR={or_val:.3f} (95%CI [{clogit_result['condition_or_ci_lower']:.3f}, "
                              f"{clogit_result['condition_or_ci_upper']:.3f}]), p={p:.4f} ({sig})")
                        master_rows.append(_master_row(
                            phase="3_conditional_logistic", outcome=f"{outcome}_{label}",
                            model_type="Conditional Logistic",
                            estimate=clogit_result["condition_log_or"],
                            se=clogit_result["condition_se"],
                            ci_lower=np.log(clogit_result["condition_or_ci_lower"]),
                            ci_upper=np.log(clogit_result["condition_or_ci_upper"]),
                            p_value=p, effect_size_type="OR",
                            effect_size_val=or_val,
                            n_patients=df["patient"].nunique(), n_obs=len(df),
                            notes="Matched pairs (epidemiological standard); eliminates patient-level confounding",
                        ))
                    else:
                        print(f"  {outcome} {label}: {clogit_result.get('warning', 'fallback to GLMM')}")
                except Exception as e:
                    print(f"  {outcome} {label}: Conditional logistic failed — {e}")
        print()

    # ================================================================
    # PHASE 4: Binary outcomes (GEE - updated label)
    # ================================================================
    if "binary" not in skip:
        print("== Phase 4: Binary Outcome Models (GEE) ==")
        phase_dir = base_dir / "phase4_binary"
        phase_dir.mkdir(parents=True, exist_ok=True)

        for outcome in outcomes:
            for threshold, label in [(1, "any"), (2, "significant")]:
                bin_col = f"{outcome}_binary_{threshold}"
                if bin_col not in df.columns:
                    continue

                # GEE
                try:
                    gee_result = models.fit_gee(df, bin_col, include_grader=True)
                    gee_fe = models.summarize_gee(gee_result, model_name=f"gee_{outcome}_{label}")
                    gee_fe.to_csv(phase_dir / f"gee_{outcome}_{label}.csv", index=False)

                    cond = gee_fe[gee_fe["term"].str.contains("condition")]
                    if len(cond) > 0:
                        or_val = cond["or"].values[0]
                        p = cond["pvalue"].values[0]
                        sig = "YES" if p < alpha else "no"
                        print(f"  {outcome} {label}: OR={or_val:.3f}, p={p:.4f} ({sig})")

                        # Risk difference
                        p0 = df[df["condition"] == 0][bin_col].mean()
                        p1 = df[df["condition"] == 1][bin_col].mean()
                        rd = effect_size.risk_difference(p1, p0)

                        master_rows.append(_master_row(
                            phase="4_binary", outcome=f"{outcome}_{label}",
                            model_type="GEE", estimate=cond["estimate"].values[0],
                            se=cond["se"].values[0],
                            ci_lower=cond["ci_lower"].values[0],
                            ci_upper=cond["ci_upper"].values[0],
                            p_value=p, effect_size_type="OR",
                            effect_size_val=or_val,
                            n_patients=df["patient"].nunique(), n_obs=len(df),
                            notes=f"RD={rd['risk_difference']:.3f}, {rd['nnt_label']}={rd['nnt_or_nnh']:.1f}",
                        ))
                except Exception as e:
                    print(f"  {outcome} {label}: GEE failed — {e}")
        print()

    # ================================================================
    # PHASE 5: Composite
    # ================================================================
    if "composite" not in skip:
        print("== Phase 5: Composite Outcome ==")
        phase_dir = base_dir / "phase5_composite"
        phase_dir.mkdir(parents=True, exist_ok=True)

        # Validation
        comp_valid = composite.validate_composite(df)
        comp_valid.to_csv(phase_dir / "composite_validation.csv", index=False)
        print(f"  Correlation depth-extent: rho={comp_valid['spearman_rho'].values[0]:.3f}")
        print(f"  Cronbach alpha: {comp_valid['cronbach_alpha'].values[0]:.3f}")
        print(f"  {comp_valid['interpretation'].values[0]}")

        # LMM on composite already done in Phase 1 if not skipped
        print()

    # ================================================================
    # PHASE 6: Diagnostics
    # ================================================================
    if "diagnostics" not in skip:
        print("== Phase 6: Model Diagnostics ==")
        phase_dir = base_dir / "phase6_diagnostics"
        phase_dir.mkdir(parents=True, exist_ok=True)

        for outcome in all_outcomes:
            result = models.fit_lmm(df, outcome, include_grader=True)
            diag = diagnostics.run_lmm_diagnostics(result, f"lmm_{outcome}", phase_dir)
            norm_ok = diag["normality_ok"].values[0]
            re_ok = diag["re_normality_ok"].values[0]
            print(f"  {outcome}: residuals normal={norm_ok}, RE normal={re_ok}, "
                  f"outliers={diag['n_outliers_2sd'].values[0]}")

        # Collinearity check
        balance = diagnostics.collinearity_check(df, ["condition", "procedure_code"])
        balance.to_csv(phase_dir / "condition_procedure_crosstab.csv", index=False)

        # Influence analysis
        for outcome in outcomes:
            diagnostics.influence_on_condition(df, outcome, f"lmm_{outcome}", phase_dir)
            print(f"  {outcome}: leave-one-out influence analysis complete")
        print()

    # ================================================================
    # PHASE 7: Effect sizes
    # ================================================================
    if "effect_sizes" not in skip:
        print("== Phase 7: Effect Sizes ==")
        phase_dir = base_dir / "phase7_effect_sizes"
        phase_dir.mkdir(parents=True, exist_ok=True)

        es_parts = []
        for outcome in all_outcomes:
            es = effect_size.compute_all_effect_sizes(df, outcome)
            es_parts.append(es)
            d = es["cohens_d"].values[0]
            cliff = es["cliffs_delta"].values[0]
            print(f"  {outcome}: Cohen's d={d:.3f} ({es['cohens_d_interpretation'].values[0]}), "
                  f"Cliff's delta={cliff:.3f} ({es['cliffs_interpretation'].values[0]})")

        all_es = pd.concat(es_parts, ignore_index=True)
        all_es.to_csv(phase_dir / "effect_sizes_all.csv", index=False)
        print()

    # ================================================================
    # PHASE 8: Power
    # ================================================================
    if "power" not in skip:
        print("== Phase 8: Power Analysis ==")
        phase_dir = base_dir / "phase8_power"
        phase_dir.mkdir(parents=True, exist_ok=True)

        power_df = power.run_power_analysis(df, all_outcomes, phase_dir)
        for _, row in power_df.iterrows():
            print(f"  {row['outcome']}: observed d={row['observed_d']:.3f}, "
                  f"power={row['power_patient_level']:.1%}, "
                  f"MDE(80%)={row['mde_d_80pct']:.3f}, "
                  f"n needed={row['n_per_group_for_80pct']}")
        print()

    # ================================================================
    # PHASE 9: Sensitivity
    # ================================================================
    if "sensitivity" not in skip:
        print("== Phase 9: Sensitivity Analyses ==")
        phase_dir = base_dir / "phase9_sensitivity"
        phase_dir.mkdir(parents=True, exist_ok=True)

        # Random structure comparison
        rs_parts = [sensitivity.compare_random_structures(df, o) for o in outcomes]
        pd.concat(rs_parts).to_csv(phase_dir / "sensitivity_random_structure.csv", index=False)
        print("  Random structure comparison complete")

        # Interaction
        int_parts = [sensitivity.test_interaction(df, o) for o in outcomes]
        pd.concat(int_parts).to_csv(phase_dir / "sensitivity_interaction.csv", index=False)
        print("  Interaction tests complete")

        # Grader sensitivity
        gs_parts = [sensitivity.grader_sensitivity(df, o) for o in outcomes]
        pd.concat(gs_parts).to_csv(phase_dir / "sensitivity_grader.csv", index=False)
        print("  Grader sensitivity complete")

        # Non-parametric
        np_parts = [sensitivity.nonparametric_tests(df, o) for o in all_outcomes]
        pd.concat(np_parts).to_csv(phase_dir / "sensitivity_nonparametric.csv", index=False)
        print("  Non-parametric tests complete")

        # Balance assessment
        balance = sensitivity.assess_balance(df)
        balance.to_csv(phase_dir / "bias_assessment_balance.csv", index=False)
        print("  Balance assessment complete")

        # Multiple comparisons
        primary_p = []
        primary_labels = []
        for row in master_rows:
            if row["phase"] == "1_lmm" and row["outcome"] in outcomes:
                primary_p.append(row["p_value"])
                primary_labels.append(f"lmm_{row['outcome']}")
        if primary_p:
            mc = sensitivity.adjust_pvalues(primary_p, primary_labels)
            mc.to_csv(phase_dir / "sensitivity_multiple_comparisons.csv", index=False)
            print(f"  Multiple comparisons: {len(primary_p)} primary tests corrected")
        print()

    # ================================================================
    # MASTER SUMMARY TABLE
    # ================================================================
    print("== Writing Master Summary Table ==")
    master_df = pd.DataFrame(master_rows)
    if len(master_df) > 0:
        master_df["significant_raw"] = master_df["p_value"] < alpha
        master_df["effect_size_interpretation"] = master_df["effect_size"].apply(
            lambda d: effect_size.interpret_cohens_d(d) if not np.isnan(d) else "N/A"
        )
    master_df.to_csv(base_dir / "summary_table.csv", index=False)
    print(f"  {len(master_df)} rows written to summary_table.csv")
    print()

    # Print final summary
    if len(master_df) > 0:
        print("=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        for _, row in master_df.iterrows():
            sig_marker = " ***" if row.get("significant_raw") else ""
            print(f"  [{row['phase']}] {row['outcome']} ({row['model_type']}): "
                  f"est={row['estimate']:.4f}, p={row['p_value']:.4f}{sig_marker}, "
                  f"effect={row['effect_size']:.3f}")
        print()

    print(f"All outputs saved to {base_dir}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _master_row(
    phase, outcome, model_type, estimate, se, ci_lower, ci_upper,
    p_value, effect_size_type, effect_size_val,
    n_patients, n_obs, notes="",
):
    return {
        "phase": phase,
        "outcome": outcome,
        "model_type": model_type,
        "estimate": estimate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "effect_size_type": effect_size_type,
        "effect_size": effect_size_val,
        "n_patients": n_patients,
        "n_observations": n_obs,
        "notes": notes,
    }


def _safe_ttest(diff):
    from scipy import stats as st
    try:
        return st.ttest_1samp(diff.dropna(), 0)
    except Exception:
        return np.nan, np.nan


def _safe_wilcoxon(diff):
    from scipy import stats as st
    try:
        d = diff.dropna()
        d = d[d != 0]
        if len(d) < 2:
            return np.nan, np.nan
        return st.wilcoxon(d)
    except Exception:
        return np.nan, np.nan


if __name__ == "__main__":
    run()
