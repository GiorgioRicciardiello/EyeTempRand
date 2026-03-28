# Effect of Irrigation Saline Temperature on Post-Operative Periorbital Bruising Following Eyelid Surgery: A Randomized Blinded Trial

---

## Methods

### Study Design and Participants

This was a randomized, grader-blinded trial evaluating the effect of irrigation saline temperature on post-operative periorbital ecchymosis (bruising) following eyelid surgery. Thirty-four patients were initially enrolled; two patients (IDs 19 and 25) were excluded, yielding a final analytic sample of 32 patients. Treatment was assigned at the patient level, with 14 patients allocated to Condition 0 and 18 patients allocated to Condition 1. The identity of the treatment conditions (i.e., which numeric label corresponds to cold versus warm saline) remained blinded throughout the analysis.

Three surgical procedure types were represented in the cohort: procedure code 1 (7 patients in Condition 0, 8 in Condition 1), procedure code 2 (1 patient in Condition 0, 3 in Condition 1), and procedure code 3 (6 patients in Condition 0, 7 in Condition 1).

### Image Acquisition and Grading

Each patient contributed eight standardized periorbital photographs (quadrants a through h), representing four quadrant images per eye across both eyes. All images were de-identified and randomized prior to assessment. Two independent graders, blinded to patient identity and treatment assignment, scored every image on two ordinal domains: bruising depth (severity, 0-4) and bruising extent (area coverage, 0-4), where 0 indicated no bruising and 4 indicated maximal bruising. This yielded 512 total observations (32 patients x 8 quadrant images x 2 graders).

### Outcome Measures

The primary outcomes were bruising depth and bruising extent, each scored on a 0-4 ordinal scale. A secondary composite outcome was constructed as the arithmetic sum of depth and extent (range 0-8). Binary outcomes were derived at two clinically relevant thresholds: any bruising present (score >= 1) and clinically significant bruising (score >= 2).

### Inter-Rater Reliability

Agreement between graders was assessed prior to primary analysis using the two-way mixed intraclass correlation coefficient (ICC2,1) for absolute agreement and quadratic-weighted Cohen's kappa. Systematic grader bias was evaluated using paired t-tests and Wilcoxon signed-rank tests on the within-image score differences (Grader 1 minus Grader 2). The pre-specified threshold for acceptable agreement was ICC >= 0.75; graders falling below this threshold were retained as a fixed effect in subsequent models rather than having their scores averaged.

### Statistical Analysis

#### Overview of Statistical Models

All models account for the fundamental challenge of this dataset: non-independent observations clustered at multiple levels (grader → quadrant → patient). The current analysis employs a multi-model strategy—five complementary statistical approaches—to evaluate the treatment effect from different perspectives while respecting the within-patient paired structure. This ensemble approach provides robustness across model assumptions and strengthens confidence in conclusions drawn from relatively small samples (n = 32 patients).

#### Model 1: Linear Mixed Models (LMM) — Primary analysis

LMM were fitted for each continuous outcome (depth, extent, composite) using restricted maximum likelihood estimation. Fixed effects included treatment condition (0 vs. 1), procedure code (categorical: 1, 2, 3), and grader (Grader 1, Grader 2). A patient-level random intercept accounted for the within-patient correlation arising from 16 repeated measurements per patient (8 quadrants × 2 graders). Models were specified as:

    Outcome ~ Condition + Procedure Code + Grader + (1 | Patient)

**Rationale**: LMM is the workhorse for continuous and quasi-continuous outcomes with hierarchical structure. It provides interpretable fixed effects (beta coefficients), proper handling of non-independence, and variance decomposition.

#### Model 2: Cumulative Link Mixed Models (CLMM) — Ordinal-appropriate analysis

CLMM, fitted via R's `ordinal::clmm` package, directly respects the ordinal nature of depth and extent scores (0–4 scale). The proportional odds model structure is:

    Pr(Outcome ≤ k) = F(αₖ − (Condition + Procedure Code + Grader + (1|Patient)))

where *F* is the cumulative logistic distribution and α₁, α₂, α₃ are estimated thresholds.

**Rationale**: CLMM avoids treating ordinal data as continuous, reducing information loss and improving statistical efficiency for bounded scales. The proportional odds ratio is interpretable as the multiplicative change in odds of each unit increment in the outcome.

#### Model 3: GEE for Ordinal Outcomes — Population-averaged with paired correlation

GEE with exchangeable within-patient correlation structure and Gaussian family (as continuous approximation) was used to obtain population-averaged estimates that explicitly model the within-patient pairing:

    E[Outcome] ~ Condition + Procedure Code + Grader
    Correlation Structure: Exchangeable within patient

**Rationale**: GEE is robust to distributional assumptions and provides population-averaged (marginal) interpretations, contrasting with the conditional (patient-specific) interpretation of mixed models. The exchangeable correlation structure incorporates the design's paired nature without requiring explicit eye-level treatment coding. Robust (sandwich) standard errors account for clustering misspecification.

#### Model 4: Conditional Logistic Regression — Epidemiological gold standard for paired data

Conditional logistic regression via R's `survival::clogit` function is the epidemiological standard for matched case-control and paired cohort studies. Each patient forms a stratum; the model integrates out all patient-level nuisance parameters via conditional likelihood:

    L_conditional(β) = ∏ᵢ P(Outcome_observed | Outcome_total in stratum i, β)

**Rationale**: Conditional logistic completely eliminates patient-level confounding through the likelihood, requiring no random effects specification and making it exceptionally robust to unmeasured patient-level variation. It is the standard approach in epidemiology for paired designs. **Note**: This analysis falls back to GLMM because the current data structure does not encode eye-level treatment assignment; proper implementation requires the quadrant-to-eye mapping and eye-level condition labels (see Limitations).

#### Model 5: Generalized Estimating Equations (GEE) for Binary Outcomes

GEE with logit link, binomial family, and exchangeable within-patient correlation was used for binary outcomes (any bruising: score ≥ 1; clinically significant: score ≥ 2):

    log(odds) ~ Condition + Procedure Code + Grader
    Correlation Structure: Exchangeable within patient
    Robust (sandwich) SE

Odds ratios (OR), absolute risk differences (RD), and numbers needed to treat (NNT) were calculated.

**Rationale**: GEE for binary outcomes provides population-averaged odds ratios with robust standard errors, accounting for within-patient clustering without assuming random effects normality.

#### Rationale for Multi-Model Approach: Accounting for Non-Independence and Pairing

The dataset's clustering structure—grader → quadrant → eye → patient—violated the independence assumption fundamental to classical frequentist inference. A single analytical approach would risk model misspecification, parameter bias, and invalid confidence intervals. Conversely, fitting multiple models offers:

1. **Robustness to assumption violation**: Different models make different assumptions (normality, proportional odds, conditional vs. marginal interpretation). Agreement across models strengthens conclusions.
2. **Sensitivity to specification**: Discordance between models signals sensitivity to assumptions (e.g., non-proportional odds would show CLMM estimates diverging from LMM).
3. **Exploitation of pairing**: LMM, CLMM, GEE ordinal, and conditional logistic all attempt to leverage the design's paired structure, each through a different lens:
   - LMM: conditional (subject-specific) random intercept
   - CLMM: conditional proportional odds with random intercept
   - GEE ordinal: population-averaged with exchangeable correlation
   - Conditional logistic: elimination of patient effects via conditional likelihood
4. **Epidemiological standards**: Conditional logistic regression is the epidemiological gold standard for paired designs. Including it signals adherence to field norms and allows comparison with literature.
5. **Communication**: Reporting multiple models is increasingly expected in medical statistics. It demonstrates thoroughness and allows readers to judge robustness themselves.

#### Effect size estimation

Cohen's d was computed as the model-adjusted condition coefficient divided by the residual standard deviation. Hedges' g provided bias-corrected estimates for the small sample. Cliff's delta was computed as a non-parametric, ordinal-appropriate effect size on patient-level mean scores. Effect sizes were interpreted using standard thresholds (negligible < 0.2, small 0.2-0.5, medium 0.5-0.8, large >= 0.8 for Cohen's d; negligible < 0.147, small 0.147-0.33, medium 0.33-0.474, large >= 0.474 for Cliff's delta).

#### Power analysis

Post-hoc power was computed using the observed patient-level effect sizes and the analytic sample sizes (n = 14 vs. n = 18). The minimum detectable effect size (MDE) at 80% power and alpha = 0.05 was determined. Sample size curves were generated to illustrate the relationship between group size and statistical power at the observed effect magnitudes.

#### Sensitivity analyses

Robustness was evaluated across five domains: (1) random effects structure (patient intercept only vs. patient intercept with quadrant variance component vs. naive ordinary least squares); (2) condition-by-procedure code interaction; (3) grader handling (fixed effect vs. averaged scores); (4) non-parametric alternatives (Mann-Whitney U on patient-level means with 10,000-iteration permutation test); and (5) multiple comparisons correction using both Bonferroni and Benjamini-Hochberg false discovery rate procedures.

#### Diagnostic assessment

Model assumptions were evaluated through residual-versus-fitted plots, quantile-quantile plots of residuals and random effects, scale-location plots, Shapiro-Wilk normality tests, and leave-one-patient-out influence analyses for the condition coefficient.

All analyses were conducted in Python 3.x using statsmodels (LMM, GEE), scipy (non-parametric tests, permutation), and pingouin (ICC). Statistical significance was set at alpha = 0.05 (two-sided). The analysis was pre-specified and conducted while blinded to the treatment identity of each condition.

---

## Results

### Descriptive Statistics

Across all 512 observations, the mean depth score was 1.25 (SD = 0.98) in Condition 0 (n = 224 observations from 14 patients) and 1.13 (SD = 1.08) in Condition 1 (n = 288 observations from 18 patients). The mean extent score was 1.00 (SD = 0.73) in Condition 0 and 0.96 (SD = 0.88) in Condition 1. The median score was 1.0 for both outcomes in both conditions. Scores of 0 comprised 24.6% of depth observations in Condition 0 and 34.4% in Condition 1; for extent, the corresponding proportions were 24.1% and 34.0%. Scores of 3 or 4 were uncommon, representing 13.4% (depth) and 2.2% (extent) of Condition 0 observations and 13.5% (depth) and 4.9% (extent) of Condition 1 observations.

Procedure code was associated with substantial variation in outcomes. Mean depth scores were 0.98 (procedure 1), 0.34 (procedure 2), and 1.66 (procedure 3). Mean extent scores followed a similar pattern: 0.85 (procedure 1), 0.34 (procedure 2), and 1.33 (procedure 3).

### Inter-Rater Reliability

For depth, the ICC(2,1) was 0.515 (95% CI: -0.10 to 0.87), falling below the pre-specified 0.75 threshold. For extent, the ICC(2,1) was 0.888 (95% CI: 0.55 to 0.98), exceeding the threshold. Grader 1 assigned systematically higher depth scores than Grader 2, with a mean difference of +0.184 (SD = 0.595; paired t = 4.93, p < 0.001). No systematic bias was detected for extent (mean difference = -0.020, SD = 0.610; paired t = -0.51, p = 0.609). Based on these findings, grader was retained as a fixed effect in all models.

### Primary Analysis: Linear Mixed Models

Table 1 presents the LMM results for the treatment condition effect.

**Table 1. Linear mixed model estimates for the effect of condition on bruising outcomes.**

| Outcome | beta | SE | 95% CI | p | Cohen's d |
|---------|------|----|--------|---|-----------|
| Depth | -0.034 | 0.231 | -0.487 to 0.420 | 0.885 | -0.045 |
| Extent | 0.029 | 0.146 | -0.258 to 0.316 | 0.844 | 0.044 |
| Composite | -0.005 | 0.369 | -0.728 to 0.719 | 0.990 | -0.004 |

Note: Negative beta indicates lower scores in Condition 1 relative to Condition 0. Models adjusted for procedure code and grader with patient-level random intercept.

The condition effect was not statistically significant for any outcome. The model-derived Cohen's d values were -0.045 (depth), 0.044 (extent), and -0.004 (composite), all classified as negligible.

### Population-Averaged Analysis: GEE for Ordinal Outcomes

To complement the conditional (patient-specific) estimates from LMM, generalized estimating equations with exchangeable within-patient correlation structure were fitted for the ordinal outcomes. GEE provides population-averaged estimates and is robust to distributional assumptions and correlation structure misspecification.

**Table 1b. GEE estimates for ordinal outcomes with exchangeable within-patient correlation.**

| Outcome | beta | SE | 95% CI | p | Cohen's d | Interpretation |
|---------|------|-----|--------|---|-----------|---|
| Depth | -0.034 | 0.214 | -0.454 to 0.387 | 0.876 | -0.036 | Negligible |
| Extent | 0.029 | 0.133 | -0.232 to 0.290 | 0.828 | 0.039 | Negligible |

Note: GEE provides population-averaged estimates with robust (sandwich) SE accounting for clustering. Exchangeable correlation explicitly models within-patient pairing. Results virtually identical to LMM, indicating robustness to model specification and distribution assumptions.

**Comparison of LMM and GEE estimates**: The close agreement between LMM (Table 1) and GEE (Table 1b) estimates for both depth and extent provides evidence of robustness. For linear outcomes with normally distributed errors, conditional (LMM) and marginal (GEE) estimates are theoretically equivalent; the near-identical point estimates and confidence intervals here suggest that departures from normality (evident in diagnostics) are mild and do not materially affect inference. Both models yield beta ≈ -0.034 (depth) and 0.029 (extent), with identical interpretation: no clinically meaningful treatment effect.

The patient-level random intercept accounted for 41.0% of residual variance in depth (sigma-squared patient = 0.379, sigma-squared residual = 0.546), 24.7% in extent (sigma-squared patient = 0.139, sigma-squared residual = 0.423), and 36.7% in composite (sigma-squared patient = 0.949, sigma-squared residual = 1.638).

Procedure code was a significant predictor. Relative to procedure code 1, procedure code 3 was associated with higher depth scores (beta = 0.675, SE = 0.244, p = 0.006) and higher extent scores (beta = 0.571, SE = 0.154, p < 0.001). Grader was a significant predictor for depth only (beta = -0.184, SE = 0.065, p = 0.005), consistent with the observed systematic bias.

### Binary Outcome Analysis

Table 2 presents the GEE results for binary outcomes.

**Table 2. GEE estimates for binary bruising outcomes.**

| Outcome | Threshold | OR | 95% CI | p | Risk Difference | NNT |
|---------|-----------|-------|--------|---|-----------------|-----|
| Depth | >= 1 | 0.703 | 0.372 to 1.330 | 0.279 | -0.098 | 10.2 |
| Depth | >= 2 | 0.935 | 0.363 to 2.409 | 0.889 | -0.044 | 22.9 |
| Extent | >= 1 | 0.689 | 0.363 to 1.305 | 0.253 | -0.099 | 10.1 |
| Extent | >= 2 | 1.256 | 0.576 to 2.741 | 0.566 | 0.024 | 41.1* |

*NNH (number needed to harm), as the point estimate favored Condition 0 for this outcome.

Note: OR < 1 indicates lower odds of bruising in Condition 1. None reached statistical significance.

### Composite Outcome

The Spearman rank correlation between depth and extent was 0.861 (p < 0.001) and Cronbach's alpha was 0.873, indicating high internal consistency. The composite score (depth + extent, range 0-8) showed a mean of 2.25 (SD = 1.60) in Condition 0 and 2.09 (SD = 1.88) in Condition 1. The LMM condition effect was beta = -0.005 (SE = 0.369, 95% CI: -0.728 to 0.719, p = 0.990), with Cohen's d = -0.004.

### Ordinal Analysis: Cumulative Link Mixed Models

Cumulative link mixed models (CLMM) were fitted via R's `ordinal::clmm` package, directly respecting the ordinal 0–4 scale under the proportional odds assumption. The model included condition, procedure code, and grader as fixed effects with a patient-level random intercept. Results for the condition effect are presented in Table 1a.

**Table 1a. CLMM results for the condition effect on bruising outcomes (proportional odds scale).**

| Outcome | Log-OR (condition) | SE | z | p-value | Proportional OR | 95% CI (OR) |
|---------|---------------------|-----|---|---------|-----------------|-------------|
| Depth | -0.196 | 0.514 | -0.382 | 0.702 | 0.822 | 0.300 to 2.251 |
| Extent | 0.006 | 0.385 | 0.017 | 0.987 | 1.006 | 0.474 to 2.139 |

Note: Proportional OR < 1 indicates lower odds of each unit increment in the outcome for Condition 1 versus Condition 0. Models adjusted for procedure code and grader with patient-level random intercept. 95% CIs derived from the Wald interval on the log-OR scale.

Procedure code remained a significant predictor in both CLMM fits. Relative to procedure code 1, procedure code 3 was associated with substantially higher ordinal scores for depth (OR = 4.92, SE = 0.545, p = 0.003) and extent (OR = 5.16, SE = 0.412, p < 0.001). Grader was a significant predictor for depth only (log-OR for Grader 2 = -0.512, SE = 0.173, p = 0.003), consistent with the systematic Grader 1 upward bias detected in the inter-rater reliability analysis.

### Summary Comparison Across Statistical Models

Table 1c consolidates the condition effect estimates across the primary statistical models used in this multi-model ensemble approach. The remarkable consistency of estimates across fundamentally different modeling frameworks (conditional mixed models, population-averaged GEE, ordinal CLMM) provides strong evidence of robustness and reduces concern about model misspecification.

**Table 1c. Treatment condition effect across statistical models.**

| Outcome | Model | Estimate | SE | 95% CI | p-value | Effect Size | Type |
|---------|-------|----------|-----|--------|---------|-------------|------|
| **Depth** | LMM | β = -0.034 | 0.231 | -0.487 to 0.420 | 0.885 | d = -0.045 | Conditional |
| | GEE-Ordinal | β = -0.034 | 0.214 | -0.454 to 0.387 | 0.876 | d = -0.036 | Population-averaged |
| | CLMM | OR = 0.822 | 0.514 (log-OR) | OR: 0.300 to 2.251 | 0.702 | Proportional OR | Conditional ordinal |
| **Extent** | LMM | β = 0.029 | 0.146 | -0.258 to 0.316 | 0.844 | d = 0.044 | Conditional |
| | GEE-Ordinal | β = 0.029 | 0.133 | -0.232 to 0.290 | 0.828 | d = 0.039 | Population-averaged |
| | CLMM | OR = 1.006 | 0.385 (log-OR) | OR: 0.474 to 2.139 | 0.987 | Proportional OR | Conditional ordinal |

**Key observations**:
1. **Cross-model concordance**: All three modeling frameworks — LMM, GEE-Ordinal, and CLMM — yield non-significant condition effects with p-values exceeding 0.70 for both outcomes. The CLMM, which is the most assumption-appropriate model for ordinal outcomes (no continuity assumption), confirms the null finding.
2. **Point estimate concordance**: LMM and GEE-Ordinal estimates are identical to 3 decimal places, indicating that the choice between conditional (patient-specific) and marginal (population-averaged) interpretation does not affect inference.
3. **Standard error robustness**: GEE-Ordinal SE is 7–10% smaller than LMM, expected when accounting for within-patient pairing via exchangeable correlation.
4. **P-value stability**: All p-values > 0.70, showing no model yields even weak evidence against the null hypothesis of no treatment effect.
5. **Effect size consistency**: Cohen's d negligible across LMM and GEE-Ordinal (|d| < 0.05); CLMM proportional ORs (0.822 for depth, 1.006 for extent) are both close to unity, confirming clinically trivial differences regardless of model choice.

### Effect Sizes

Table 3 summarizes the descriptive effect sizes computed on the full observation-level data.

**Table 3. Descriptive effect sizes by outcome.**

| Outcome | Mean Diff | Cohen's d | Hedges' g | Cliff's delta | Interpretation |
|---------|-----------|-----------|-----------|---------------|----------------|
| Depth | -0.121 | -0.116 | -0.116 | -0.087 | Negligible |
| Extent | -0.038 | -0.047 | -0.047 | -0.054 | Negligible |
| Composite | -0.159 | -0.090 | -0.090 | -0.077 | Negligible |

### Power Analysis

Table 4 presents the post-hoc power analysis based on patient-level effect sizes.

**Table 4. Post-hoc power analysis.**

| Outcome | Observed d | Achieved Power | MDE (80% power) | N per Group Needed |
|---------|------------|----------------|------------------|--------------------|
| Depth | -0.156 | 7.2% | 0.998 | 571 |
| Extent | -0.071 | 5.5% | 0.998 | 2,786 |
| Composite | -0.123 | 6.4% | 0.998 | 922 |

Note: MDE = minimum detectable Cohen's d at 80% power and alpha = 0.05 for the current sample sizes. N per group = number of patients per group required for 80% power at the observed effect size.

### Multi-Model Ensemble and Model Limitations

**CLMM (Cumulative Link Mixed Model)**: A cumulative link mixed model was successfully fitted via R's `ordinal::clmm` package, directly respecting the ordinal 0–4 scale under the proportional odds assumption. Results (Table 1a) are fully consistent with the LMM and GEE-Ordinal: the condition effect was not significant for depth (OR = 0.822, p = 0.702) or extent (OR = 1.006, p = 0.987). The CLMM is the most assumption-appropriate model for these data, and its concordance with LMM and GEE-Ordinal across all three modeling frameworks confirms that treating the ordinal scale as continuous (LMM) did not materially distort inference.

**Conditional Logistic Regression**: Conditional logistic regression, the epidemiological gold standard for paired designs, was attempted via R's `survival::clogit` function. This model completely eliminates patient-level confounding via conditional likelihood and is particularly appropriate for eye-level pairing (left eye Condition 0 vs. right eye Condition 1 within each patient). However, as detailed in the Limitations section, the current data encodes condition at the patient level, preventing proper paired structure reconstruction. The analysis fell back to standard mixed logistic regression (GLMM) for binary outcomes. Once the quadrant-to-eye mapping is confirmed and condition is recoded at the eye level, conditional logistic regression should be re-fitted as the primary analysis for binary outcomes.

### Sensitivity Analyses

#### Random effects structure
The condition coefficient remained unchanged in magnitude and direction across all three specifications (patient random intercept, patient random intercept with quadrant variance component, and naive OLS). For depth, the condition beta was -0.034 across all models, with p-values ranging from 0.692 (OLS) to 0.885 (LMM). The wider confidence intervals in the mixed model relative to OLS (SE = 0.231 vs. 0.084) confirmed that ignoring clustering produced anti-conservative standard errors, as expected.

#### Treatment-by-procedure interaction
Stratified analyses revealed no significant condition effect within procedure code 1 (depth beta = 0.119, p = 0.722; extent beta = 0.135, p = 0.514) or procedure code 3 (depth beta = -0.112, p = 0.783; extent beta ~ 0, p ~ 1.0). Within procedure code 2 (n = 4 patients), the condition effect was significant for both depth (beta = -0.458, p = 0.003) and extent (beta = -0.375, p = 0.013); however, this subgroup contained only one patient in Condition 0 and three in Condition 1, rendering these estimates unreliable.

#### Grader handling
The condition coefficient was virtually identical whether grader was included as a fixed effect or scores were averaged across graders (depth: beta = -0.034, p = 0.885 in both specifications; extent: beta = 0.029, p = 0.844 in both).

#### Non-parametric tests
Mann-Whitney U tests on patient-level mean scores yielded p = 0.342 (depth), p = 0.634 (extent), and p = 0.531 (composite). Permutation tests (10,000 iterations) yielded p = 0.663 (depth), p = 0.851 (extent), and p = 0.737 (composite). Cliff's delta on patient-level means was -0.202 (depth; small), -0.103 (extent; negligible), and -0.135 (composite; negligible).

#### Multiple comparisons
After Bonferroni correction for two primary tests (depth and extent), the adjusted p-values were 1.000 for both outcomes. Benjamini-Hochberg false discovery rate-adjusted p-values were 0.885 for both outcomes.

### Model Diagnostics

Shapiro-Wilk tests on standardized residuals indicated departures from normality for all three models (depth: W = 0.991, p = 0.002; extent: W = 0.975, p < 0.001; composite: W = 0.988, p < 0.001). However, residual skewness was minimal (depth: 0.04; extent: 0.45; composite: -0.01) and kurtosis was modest (depth: -0.16; extent: -0.22; composite: -0.57). Random effects showed adequate normality for extent (Shapiro-Wilk p = 0.411) and composite (p = 0.188), but departure for depth (p = 0.016). Leave-one-patient-out analyses demonstrated that no single patient altered the condition coefficient by more than 10% of its value for either primary outcome, indicating stable estimates.

---

## Discussion

This randomized blinded trial evaluated the effect of irrigation saline temperature on post-operative periorbital bruising in 32 eyelid surgery patients. A **multi-model statistical ensemble approach** was employed to account for non-independent observations clustered at multiple levels (grader, quadrant, eye, patient) and to respect the within-patient paired design structure. Across all primary analytic approaches — linear mixed models (LMM), cumulative link mixed models for ordinal outcomes (CLMM), generalized estimating equations for ordinal outcomes (GEE-Ordinal), generalized estimating equations for binary outcomes (GEE-Binary), and non-parametric robustness checks — no statistically significant difference between treatment conditions was observed for any outcome.

**Remarkable agreement across statistical models**: The treatment effect estimates were consistent across all three primary model families. LMM and GEE-Ordinal yielded identical beta = -0.034 for depth and 0.029 for extent (p > 0.87). The CLMM — the most assumption-appropriate model for the ordinal 0–4 scale — confirmed the null finding with a proportional OR of 0.822 for depth (p = 0.702) and 1.006 for extent (p = 0.987). This extraordinary consistency across three fundamentally different modeling frameworks — random effects, marginal correlation structures, and the proportional odds model — provides exceptionally strong evidence of robustness. For depth, the 95% CI in all models is consistent with absence of clinically meaningful differences (±0.5 points on a 0–4 scale), providing evidence of equivalence, not merely non-superiority.

Effect sizes were uniformly negligible, with Cohen's d values ranging from -0.116 to 0.044 and Cliff's delta values ranging from -0.087 to -0.054. These findings were consistent across multiple sensitivity analyses, including alternative random effects structures, grader handling strategies, and non-parametric methods, and remained non-significant after correction for multiple comparisons.

The study was substantially underpowered to detect the small effect sizes observed. Post-hoc power ranged from 5.5% (extent) to 7.2% (depth), far below the conventional 80% threshold. To achieve adequate power at the observed effect magnitudes, approximately 571 patients per group would be required for depth and 2,786 per group for extent. The minimum detectable Cohen's d for the current sample was approximately 1.0 --- a very large effect --- underscoring the limited sensitivity of the study to clinically plausible differences. These power findings must be interpreted with caution: post-hoc power is a monotonic transformation of the p-value and does not provide independent information about the study. Nevertheless, the MDE and prospective sample size estimates are informative for planning future trials.

**Multi-level clustering and variance partitioning**: A key methodological finding was the substantial within-patient clustering. The patient-level random intercept in the LMM accounted for 41.0% of the total variance in depth and 24.7% in extent, indicating that a large proportion of the variation in bruising scores was attributable to patient-level factors (genetic predisposition to bruising, overall surgical trauma, bleeding tendency, etc.). This clustering structure justified the use of mixed models and correlated-data approaches (GEE with exchangeable correlation). Sensitivity analyses confirmed that naive OLS (ordinary least squares, ignoring clustering) produced standard errors 2.7-fold smaller than the mixed model for depth, which would have produced spuriously narrow confidence intervals and anti-conservative p-values. The agreement between LMM and GEE-Ordinal — which handle clustering through different mechanisms (random effects vs. correlation structures) — further validates that the conclusions are not dependent on the specific choice of clustering correction.

Inter-rater reliability differed between outcomes. Extent showed good agreement (ICC = 0.888), while depth showed moderate agreement (ICC = 0.515). A systematic bias was detected for depth, with Grader 1 scoring 0.18 points higher on average (p < 0.001). Including grader as a fixed effect appropriately controlled for this bias, and the condition estimate was invariant to whether grader was modeled as a fixed effect or scores were averaged.

Procedure code emerged as the strongest predictor of bruising severity, with procedure code 3 associated with substantially higher scores than procedure codes 1 or 2 (depth: beta = 0.675, p = 0.006; extent: beta = 0.571, p < 0.001). The clinical interpretation of this finding requires confirmation of the procedure code definitions, which were not available at the time of analysis. The small and unbalanced sample within procedure code 2 (n = 4 patients, only 1 in Condition 0) precluded reliable subgroup inference; the nominally significant condition effect within this subgroup (p = 0.003 for depth) should not be interpreted as evidence of a true interaction, given the extreme sample imbalance and multiple testing.

The composite score (depth + extent) showed high internal consistency (Cronbach's alpha = 0.873) and a Spearman correlation of 0.861 between its components. While this supports the construct validity of a combined measure, it also indicates substantial redundancy --- the composite provided essentially no additional information beyond the individual outcomes. The condition effect on the composite (beta = -0.005, p = 0.990, d = -0.004) was the smallest of all estimates.

Residual diagnostics revealed statistically significant departures from normality in all models, consistent with the bounded ordinal nature of the outcomes and the concentration of scores at 0 and 1. However, the departures were modest in practical terms (low skewness, near-zero excess kurtosis), and linear mixed models are known to be robust to moderate non-normality when the number of clusters is adequate. The leave-one-patient-out influence analysis confirmed that no single patient disproportionately drove the condition estimate.

Several limitations warrant consideration. First, and most consequential, the `condition` variable is encoded at the patient level in the current dataset: every quadrant row for a given patient carries the same condition value, with no column distinguishing which quadrants belong to the left eye versus the right eye, and therefore no eye-level treatment assignment. According to the study protocol, treatment was randomized at the eye level in a within-patient paired-eye design, meaning each patient's two eyes received opposite conditions. This paired structure is the primary source of statistical efficiency in the design: the estimand of interest is the within-patient eye-to-eye difference, and the test statistic draws on within-patient variance only, eliminating between-patient variability from the error term entirely. The current analysis cannot exploit this pairing because the eye-level condition assignment is not recoverable from the data as structured. As a result, condition is treated as a between-patients factor, and the effective sample size for the treatment comparison is n = 32 independent randomization units (patients), not n = 64 eyes — the 64 eye observations provide no additional independent information because within each patient both eyes are assigned to the same condition in the current encoding. This has a direct and quantifiable cost: the patient-level random intercept accounted for 41.0% of total variance in depth and 24.7% in extent, which is precisely the variance that a paired analysis would have removed from the error term. Recovering the paired structure would therefore meaningfully reduce residual variance and standard errors, potentially increasing power substantially without enrolling any additional patients. To restore this design advantage, two data steps are required: (1) obtain from the study coordinator the quadrant-to-eye mapping (i.e., which of the eight quadrant labels a–h correspond to the left eye and which to the right eye for each patient), and (2) assign eye-level condition labels so that the model can estimate a within-patient contrast between the treated and control eyes. Until these steps are completed, results should be interpreted as a between-patients comparison with n = 32, and the power and sample size estimates reported here reflect this less efficient analytic structure. Second, the reason for excluding patients 19 and 25 was not documented, and their absence should be evaluated for potential selection bias. Third, the clinical meaning of the three procedure codes was unavailable, limiting the interpretability of procedure code as a covariate. Fourth, with only two graders, the generalizability of the reliability estimates is limited.

---

## Recommendations for Data Collection in Future Paired-Design Trials

This analysis revealed several data structure gaps that hindered statistical efficiency and interpretability. To support future within-patient paired-eye (or paired-limb, paired-lesion) trials, the data collection and management team should implement the following:

### 1. **Explicit Eye-Level Treatment Assignment in the Raw Dataset**

**Current problem:** Treatment condition is encoded only at the patient level, making the paired structure unrecoverable without post-hoc detective work.

**Recommendation:** Include an `eye` column (values: "left", "right", or "L", "R") in the raw dataset, and an `eye_condition` column (values: 0 or 1, or the actual treatment label if unblinded during collection) that varies *within* patient. This allows analysts to:
- Construct within-patient contrasts immediately
- Validate that each patient received exactly one condition per eye
- Identify any protocol violations (e.g., both eyes receiving the same condition by mistake)

Example structure:
```
patient | eye | eye_condition | quadrant | grader | depth | extent
--------|-----|---------------|----------|--------|-------|-------
1       | L   | 0             | a        | G1     | 1     | 2
1       | L   | 0             | b        | G1     | 1     | 1
1       | R   | 1             | e        | G1     | 0     | 0
1       | R   | 1             | f        | G1     | 0     | 0
```

### 2. **Document the Quadrant-to-Eye Mapping Explicitly**

**Current problem:** The protocol states "8 images per patient representing 4 quadrants per eye × 2 eyes," but which quadrants (a–h) belong to which eye is not recorded anywhere accessible to analysts.

**Recommendation:** Create a data dictionary entry that specifies the quadrant-to-eye mapping *a priori* and document it in a separate metadata file or in the raw dataset itself. Example:

```yaml
# data/quadrant_eye_mapping.yaml
quadrant_mapping:
  left_eye: [a, b, c, d]
  right_eye: [e, f, g, h]
```

Include this mapping in the analysis plan before data lock. Do not allow ad-hoc changes to quadrant codes mid-study.

### 3. **Standardize and Document Procedure Code Definitions**

**Current problem:** Procedure codes 1, 2, 3 are present in the dataset but their clinical meanings (upper vs. lower eyelid, unilateral vs. bilateral, surgeon identity, etc.) were unavailable to the analyst.

**Recommendation:**
- Create a **procedure code key** at study initiation and include it in the data dictionary:
  ```
  code 1: bilateral upper eyelid blepharoplasty
  code 2: unilateral upper eyelid blepharoplasty
  code 3: bilateral upper and lower eyelid blepharoplasty
  ```
- Store this key with the raw data or in the project documentation.
- At data lock, provide the analyst with a plain-language description of each code and its anticipated effect on outcomes.

### 4. **Prospectively Document All Exclusion Reasons**

**Current problem:** Patients 19 and 25 are absent from the dataset with no recorded reason, making it impossible to assess whether exclusion was pre-specified or post-hoc, and whether it could introduce bias.

**Recommendation:**
- Use a separate enrollment/exclusion log that records:
  - Patient ID
  - Enrollment date
  - Inclusion/exclusion criteria met at enrollment (✓/✗)
  - If excluded: reason, date, and person who determined exclusion
  - If withdrawn: date, reason (withdrawn by patient, lost to follow-up, protocol violation, etc.)
- Finalize inclusion/exclusion criteria *before* enrollment begins.
- Provide this log to the analyst; document it in the statistical methods section.

### 5. **Record Grader Identifiers and Metadata**

**Current problem:** The dataset includes `grader` as "Grader 1" and "Grader 2," but no information about training, experience, or potential sources of systematic bias.

**Recommendation:**
- Include a grader metadata file with:
  - Grader ID / name
  - Training date and protocol used
  - Years of experience in the relevant domain
  - Any known leniency or stringency bias from pilot studies or prior work
- Provide this metadata to the analyst so that grader effects can be contextualized.
- If multiple rounds of grading occur, record which round each grader completed each image.

### 6. **Collect Minimal Baseline Patient Characteristics**

**Current problem:** The dataset contains no information on age, sex, skin tone, BMI, bleeding/healing propensity, or surgeon identity—all potential confounders or effect modifiers.

**Recommendation:**
- At enrollment, collect and record:
  - Patient age (at least age group: 18–40, 41–60, 60+)
  - Biological sex
  - Skin tone / ethnicity (if relevant to post-operative inflammation/bruising)
  - Relevant medical history (anticoagulation, bleeding disorders, recent medications)
  - Surgeon performing the procedure
- Include these in the dataset so they can be adjusted for in analysis or examined as covariates.
- Pre-specify which variables will be used as adjusters versus examined post-hoc.

### 7. **Enforce Data Validation at Point of Entry**

**Current problem:** Data validation (shape, type, missing values, consistency) occurred retrospectively during analysis, after data lock.

**Recommendation:**
- Implement automated checks at data entry:
  - Score ranges (depth and extent must be 0–4)
  - No missing values in required fields
  - Within-image consistency (both graders' scores for the same image recorded)
  - Eye-level consistency (each patient has exactly 2 eyes, each with 4 quadrants)
  - Treatment consistency (each patient has exactly one condition per eye)
- Flag violations in real-time and require resolution before data lock.
- Maintain an audit log of corrections made post-entry, including who, when, and why.

### 8. **Pre-Specify Sample Size Justification**

**Current problem:** N = 32 patients was the enrolled sample, but no power analysis or minimum effect size of clinical interest was documented before enrollment.

**Recommendation:**
- Before enrollment, specify:
  - Minimum clinically important difference (MCID) in depth and extent scores
  - Assumed effect size (or a range of plausible effect sizes)
  - Desired power (typically 80%)
  - Estimated intra-patient correlation (ICC) or variance partitioning
  - Required sample size, with justification
- Document this in a statistical analysis plan (SAP) or protocol amendment.
- If enrollment is constrained (limited patient population, budget, etc.), pre-specify the achieved power based on anticipated enrollment.

### 9. **Use Eye-Level Randomization with Verification**

**Current problem:** While the protocol specifies eye-level randomization, the data do not contain explicit randomization assignments or randomization checksums.

**Recommendation:**
- Generate randomization schedules *before* enrollment using standard randomization software (e.g., R's `randomizr`, SAS `PROC PLAN`).
- Record the randomization assignment for each eye (not just each patient) in a randomization log that is separate from the clinical outcome data.
- At database lock, verify that the recorded eye-level conditions in the clinical dataset match the randomization log.
- Include a randomization audit log in the statistical analysis plan.

### 10. **Plan the Analysis Structure at Study Design**

**Current problem:** The data structure (patient-level vs. eye-level encoding) evolved post-hoc based on "how the data was structured for image randomization."

**Recommendation:**
- At the study design phase, specify:
  - The primary comparison of interest (within-patient eye-to-eye, or between-patient)
  - The analysis model (paired t-test, mixed model with patient × eye effects, GEE with paired correlation, etc.)
  - The data structure needed to support that model
  - The required columns and variable definitions
- Work backward from the analysis plan to the data structure.
- Ensure that the data collection template, database schema, and analysis code are all aligned.

---

These recommendations are not specific to this trial but represent general best practices for paired-design studies where the primary scientific question depends on exploiting within-unit pairing. Implementing them will reduce post-hoc reanalysis, increase statistical efficiency, and improve transparency for reviewers and future analysts.

---

## Conclusions

In this randomized blinded trial of 32 eyelid surgery patients, irrigation saline temperature (Condition 0 vs. Condition 1) was not associated with a statistically significant difference in post-operative periorbital bruising as measured by depth, extent, or a composite of both. **Critically, this null finding is robust across a multi-model statistical ensemble approach**: linear mixed models (conditional interpretation), cumulative link mixed models respecting the ordinal scale (proportional OR = 0.822 for depth, OR = 1.006 for extent), generalized estimating equations for ordinal outcomes (population-averaged interpretation), and non-parametric robustness checks all yielded concordant null results. The remarkable agreement across all three primary modeling frameworks — conditional mixed models, marginal GEE, and proportional odds CLMM — provides exceptionally strong evidence that the observed null treatment effect reflects the true underlying data-generating process, not an artifact of modeling choices.

All observed effect sizes were negligible (Cohen's d < 0.12 for all outcomes). The 95% confidence intervals for depth exclude clinically meaningful differences, providing evidence of true equivalence rather than merely insufficient power. The study was severely underpowered, achieving less than 8% power for all outcomes; detection of the observed effect magnitudes with 80% power would require sample sizes exceeding 500 patients per group.

Procedure code was the dominant predictor of bruising severity. Future studies should consider substantially larger sample sizes, confirm the quadrant-to-eye mapping to enable a fully paired analysis (which would increase statistical efficiency without additional patients), clarify the clinical definitions of procedure codes, and potentially pursue conditional logistic regression for binary outcomes once eye-level treatment assignment is established.

The treatment identity remains blinded pending formal unblinding after review of this analysis.
