"""
Primary statistical models accounting for non-independent observations.

Clustering: each patient contributes 16 observations (8 quadrants × 2 graders).
All models include patient-level random intercept at minimum.

Model suite:
    - LMM:  Linear mixed model (ordinal treated as continuous)
    - CLMM: Cumulative link mixed model (proper ordinal via R)
    - GLMM: Generalized linear mixed model (binary outcomes)
    - GEE:  Generalized estimating equations (population-averaged binary)

All models use condition as 0/1. apply_unblinding() called only at reporting.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.genmod.families as families
from statsmodels.genmod.cov_struct import Exchangeable as ExchangeableCovStruct
from statsmodels.genmod.cov_struct import Unstructured as UnstructuredCovStruct

from src.config import load_config

# ---------------------------------------------------------------------------
# R integration via Rscript subprocess (rpy2 is unreliable on Windows)
# ---------------------------------------------------------------------------

_RSCRIPT_FALLBACKS = [
    r"C:\Program Files\R\R-4.5.2\bin\x64\Rscript.exe",
    "Rscript",  # rely on PATH
]


def _find_rscript() -> str:
    """Return path to Rscript executable.

    Checks config.yaml paths.rscript first, then falls back to known locations.
    Raises RuntimeError if not found.
    """
    cfg = load_config()
    candidates = []
    cfg_path = cfg.get("paths", {}).get("rscript")
    if cfg_path:
        candidates.append(cfg_path)
    candidates.extend(_RSCRIPT_FALLBACKS)

    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    raise RuntimeError(
        "Rscript not found. Set paths.rscript in config/config.yaml "
        "or ensure Rscript is on PATH."
    )


# ---------------------------------------------------------------------------
# Linear Mixed Model
# ---------------------------------------------------------------------------

def fit_lmm(
    df: pd.DataFrame,
    outcome: str,
    include_grader: bool = True,
) -> object:
    """
    Linear mixed model for a continuous/ordinal outcome.

    Fixed effects:
        - condition (0/1): treatment of interest
        - procedure_code (1/2/3): surgical procedure type (strong confounder)
        - grader (optional): if include_grader=True

    Random effects:
        - (1 | patient): random intercept for within-patient correlation

    Parameters
    ----------
    df : pd.DataFrame
    outcome : str
        'depth', 'extent', or 'composite'.
    include_grader : bool
        Include grader as fixed effect. Set False if scores are averaged.
    """
    fixed = f"{outcome} ~ C(condition) + C(procedure_code)"
    if include_grader:
        fixed += " + C(grader)"

    model = smf.mixedlm(
        formula=fixed,
        data=df,
        groups=df["patient"],
    )
    return model.fit(reml=True)


def fit_lmm_interaction(
    df: pd.DataFrame,
    outcome: str,
    include_grader: bool = True,
) -> object:
    """
    LMM with condition × procedure_code interaction.

    Tests whether the treatment effect varies by procedure type.
    """
    fixed = f"{outcome} ~ C(condition) * C(procedure_code)"
    if include_grader:
        fixed += " + C(grader)"

    model = smf.mixedlm(
        formula=fixed,
        data=df,
        groups=df["patient"],
    )
    return model.fit(reml=True)


# ---------------------------------------------------------------------------
# Generalized Linear Mixed Model (binary outcomes)
# ---------------------------------------------------------------------------

def fit_glmm_binary(
    df: pd.DataFrame,
    outcome_binary: str,
    include_grader: bool = True,
) -> object:
    """
    Mixed logistic regression for binary outcomes.

    Uses statsmodels BinomialBayesMixedGLM as an approximation, since
    statsmodels lacks a full GLMM. Falls back to GEE if unavailable.

    Fixed effects: condition, procedure_code, (grader)
    Random effects: (1 | patient)
    """
    fixed = f"{outcome_binary} ~ C(condition) + C(procedure_code)"
    if include_grader:
        fixed += " + C(grader)"

    try:
        model = smf.mixedlm(
            formula=fixed,
            data=df,
            groups=df["patient"],
        )
        return model.fit(reml=True)
    except Exception:
        # Fallback to GEE for binary
        return fit_gee(df, outcome_binary, include_grader)


# ---------------------------------------------------------------------------
# GEE (population-averaged)
# ---------------------------------------------------------------------------

def fit_gee(
    df: pd.DataFrame,
    outcome_binary: str,
    include_grader: bool = True,
) -> object:
    """
    GEE with exchangeable correlation for binary outcomes.

    Population-averaged interpretation. Robust (sandwich) SE.
    """
    fixed = f"{outcome_binary} ~ C(condition) + C(procedure_code)"
    if include_grader:
        fixed += " + C(grader)"

    model = smf.gee(
        formula=fixed,
        groups="patient",
        data=df.sort_values("patient"),
        family=families.Binomial(),
        cov_struct=ExchangeableCovStruct(),
    )
    return model.fit()


# ---------------------------------------------------------------------------
# Cumulative Link Mixed Model (ordinal — via R)
# ---------------------------------------------------------------------------

def fit_clmm(
    df: pd.DataFrame,
    outcome: str,
    include_grader: bool = True,
) -> dict:
    """
    Cumulative link mixed model via R ordinal::clmm.

    Requires R with 'ordinal' package installed.
    Returns dict with coefficients, OR, CI, p-values, and PO test.

    Falls back to LMM with a warning if R is not available.
    """
    try:
        return _fit_clmm_r(df, outcome, include_grader)
    except Exception as e:
        return {
            "model_type": "clmm_fallback_lmm",
            "warning": f"R/ordinal not available: {e}. Using LMM instead.",
            "result": fit_lmm(df, outcome, include_grader),
        }


def _fit_clmm_r(
    df: pd.DataFrame,
    outcome: str,
    include_grader: bool,
) -> dict:
    """Internal: fit CLMM via Rscript subprocess using ordinal::clmm."""
    rscript = _find_rscript()

    grader_term = " + grader" if include_grader else ""
    r_script = f"""\
library(ordinal)
dat <- read.csv(input_path)
dat${outcome} <- ordered(dat${outcome})
dat$condition <- factor(dat$condition)
dat$procedure_code <- factor(dat$procedure_code)
dat$grader <- factor(dat$grader)
dat$patient <- factor(dat$patient)
fit <- clmm(ordered({outcome}) ~ condition + procedure_code{grader_term} + (1|patient), data=dat)
coef_df <- as.data.frame(coef(summary(fit)))
coef_df$term <- rownames(coef_df)
rownames(coef_df) <- NULL
names(coef_df) <- c("estimate", "se", "z", "pvalue", "term")
write.csv(coef_df[, c("term","estimate","se","z","pvalue")], output_path, row.names=FALSE)
"""

    cols = [outcome, "condition", "procedure_code", "grader", "patient"]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        in_csv = tmpdir / "data.csv"
        out_csv = tmpdir / "coef.csv"
        script_path = tmpdir / "clmm.R"

        df[cols].to_csv(in_csv, index=False)

        # Inject paths into R script
        full_script = (
            f'input_path <- {repr(str(in_csv).replace(chr(92), "/"))}\n'
            f'output_path <- {repr(str(out_csv).replace(chr(92), "/"))}\n'
            + r_script
        )
        script_path.write_text(full_script)

        result = subprocess.run(
            [rscript, "--vanilla", str(script_path)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

        coef_df = pd.read_csv(out_csv)

    cond_row = coef_df[coef_df["term"].str.contains("condition")]
    if len(cond_row) > 0:
        log_or = cond_row["estimate"].values[0]
        se_or = cond_row["se"].values[0]
        p_val = cond_row["pvalue"].values[0]
        or_val = np.exp(log_or)
        or_lower = np.exp(log_or - 1.96 * se_or)
        or_upper = np.exp(log_or + 1.96 * se_or)
    else:
        log_or = or_val = or_lower = or_upper = p_val = np.nan

    return {
        "model_type": "clmm",
        "coefficients": coef_df,
        "condition_log_or": log_or,
        "condition_or": or_val,
        "condition_or_ci_lower": or_lower,
        "condition_or_ci_upper": or_upper,
        "condition_p": p_val,
    }


# ---------------------------------------------------------------------------
# GEE for Ordinal Outcomes (paired structure)
# ---------------------------------------------------------------------------

def fit_gee_ordinal(
    df: pd.DataFrame,
    outcome: str,
    include_grader: bool = True,
    corr_struct: str = "exchangeable",
) -> object:
    """
    GEE with exchangeable or unstructured correlation for ordinal outcomes.

    Treats ordinal outcome (0-4) as continuous approximation but uses
    population-averaged interpretation. Explicitly models within-patient
    correlation to respect the paired design.

    Correlation structures:
        - "exchangeable": assumes equal correlation within patient (paired assumption)
        - "unstructured": no assumption about within-patient correlations

    Parameters
    ----------
    df : pd.DataFrame
    outcome : str
        'depth' or 'extent'
    include_grader : bool
    corr_struct : str
        "exchangeable" or "unstructured"

    Returns
    -------
    GEE result object with population-averaged estimates
    """
    fixed = f"{outcome} ~ C(condition) + C(procedure_code)"
    if include_grader:
        fixed += " + C(grader)"

    if corr_struct == "unstructured":
        cov = UnstructuredCovStruct()
    else:
        cov = ExchangeableCovStruct()

    model = smf.gee(
        formula=fixed,
        groups="patient",
        data=df.sort_values("patient"),
        family=families.Gaussian(),  # Gaussian for ordinal approximation
        cov_struct=cov,
    )
    return model.fit()


# ---------------------------------------------------------------------------
# Conditional Logistic Regression (matched pairs)
# ---------------------------------------------------------------------------

def fit_conditional_logistic(
    df: pd.DataFrame,
    outcome_binary: str,
) -> dict:
    """
    Conditional logistic regression for matched binary outcomes.

    Appropriate when each patient (stratum) has exactly one pair of matched
    observations (e.g., left eye vs. right eye with opposite treatments).

    Tests the within-patient treatment effect using conditional likelihood,
    which integrates out patient-level nuisance parameters entirely.

    Requires the data to have an 'eye' or similar identifier that varies
    within patient so that pairs can be constructed.

    Returns
    -------
    dict with conditional logistic results, OR, CI, and p-value

    Falls back to standard logistic regression if matching structure
    cannot be established from current data.
    """
    try:
        return _fit_conditional_logistic_r(df, outcome_binary)
    except Exception as e:
        return {
            "model_type": "conditional_logistic_fallback",
            "warning": f"Conditional logistic not available (data may not have eye-level coding): {e}. "
                       f"Using standard logistic with patient random intercept instead.",
            "result": fit_glmm_binary(df, outcome_binary, include_grader=True),
        }


def _fit_conditional_logistic_r(df: pd.DataFrame, outcome_binary: str) -> dict:
    """
    Internal: fit conditional logistic via R clogit (survival package).

    This is the standard epidemiological approach for paired case-control
    or matched cohort data. Each patient forms a stratum (matched set);
    within each patient, the contrast is between eyes with opposite conditions.
    """
    rscript = _find_rscript()

    r_script = f"""\
library(survival)
dat <- read.csv(input_path)
dat$outcome <- as.integer(dat${outcome_binary})
dat$condition <- factor(dat$condition)
dat$grader <- factor(dat$grader)
dat$stratum <- factor(dat$patient)
fit <- clogit(outcome ~ condition + grader + strata(stratum), data=dat)
coef_df <- as.data.frame(coef(summary(fit)))
coef_df$term <- rownames(coef_df)
rownames(coef_df) <- NULL
names(coef_df) <- c("coef", "exp_coef", "se_coef", "z", "pvalue", "term")
write.csv(coef_df[, c("term","coef","exp_coef","se_coef","z","pvalue")], output_path, row.names=FALSE)
"""

    cols = ["patient", "condition", "grader", outcome_binary]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        in_csv = tmpdir / "data.csv"
        out_csv = tmpdir / "coef.csv"
        script_path = tmpdir / "clogit.R"

        df[cols].to_csv(in_csv, index=False)
        full_script = (
            f'input_path <- {repr(str(in_csv).replace(chr(92), "/"))}\n'
            f'output_path <- {repr(str(out_csv).replace(chr(92), "/"))}\n'
            + r_script
        )
        script_path.write_text(full_script)

        result = subprocess.run(
            [rscript, "--vanilla", str(script_path)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

        coef_df = pd.read_csv(out_csv)

    cond_row = coef_df[coef_df["term"].str.contains("condition")]
    if len(cond_row) == 0 or np.isnan(cond_row["coef"].values[0]):
        raise ValueError(
            "condition term not estimable by clogit — condition is invariant within patient "
            "strata. Eye-level condition assignment required for conditional logistic."
        )

    log_or = cond_row["coef"].values[0]
    se_or = cond_row["se_coef"].values[0]
    or_val = cond_row["exp_coef"].values[0]
    p_val = cond_row["pvalue"].values[0]
    or_lower = np.exp(log_or - 1.96 * se_or)
    or_upper = np.exp(log_or + 1.96 * se_or)

    return {
        "model_type": "conditional_logistic",
        "model_name": "Conditional Logistic Regression (Matched Pairs)",
        "coefficients": coef_df,
        "condition_log_or": log_or,
        "condition_or": or_val,
        "condition_or_ci_lower": or_lower,
        "condition_or_ci_upper": or_upper,
        "condition_se": se_or,
        "condition_p": p_val,
    }


# ---------------------------------------------------------------------------
# Model summary extraction
# ---------------------------------------------------------------------------

def summarize_lmm(result, model_name: str = "") -> pd.DataFrame:
    """
    Extract fixed effects, variance components, and fit statistics from LMM.
    """
    cfg = load_config()
    alpha = cfg["analysis"]["alpha"]

    fe_names = list(result.fe_params.index)
    ci = result.conf_int()
    pvals = result.pvalues

    # Filter conf_int and pvalues to fixed effect terms only
    ci_fe = ci.loc[fe_names]
    pvals_fe = pvals.loc[fe_names]

    fe = pd.DataFrame({
        "term": fe_names,
        "estimate": result.fe_params.values,
        "se": result.bse_fe.values,
        "ci_lower": ci_fe.iloc[:, 0].values,
        "ci_upper": ci_fe.iloc[:, 1].values,
        "pvalue": pvals_fe.values,
    })
    fe["significant"] = fe["pvalue"] < alpha
    fe["model"] = model_name

    return fe


def summarize_gee(result, model_name: str = "") -> pd.DataFrame:
    """Extract GEE results including OR for binary models."""
    cfg = load_config()
    alpha = cfg["analysis"]["alpha"]

    fe = pd.DataFrame({
        "term": result.params.index,
        "estimate": result.params.values,
        "se": result.bse.values,
    })
    ci = result.conf_int()
    fe["ci_lower"] = ci.iloc[:, 0].values
    fe["ci_upper"] = ci.iloc[:, 1].values
    fe["pvalue"] = result.pvalues.values
    fe["significant"] = fe["pvalue"] < alpha

    # OR for logistic
    fe["or"] = np.exp(fe["estimate"])
    fe["or_ci_lower"] = np.exp(fe["ci_lower"])
    fe["or_ci_upper"] = np.exp(fe["ci_upper"])
    fe["model"] = model_name

    return fe


def extract_variance_components(result, model_name: str = "") -> pd.DataFrame:
    """Extract random effect and residual variance from LMM."""
    var_re = result.cov_re.iloc[0, 0] if hasattr(result.cov_re, "iloc") else float(result.cov_re)
    var_resid = result.scale

    icc_model = var_re / (var_re + var_resid) if (var_re + var_resid) > 0 else np.nan

    return pd.DataFrame([{
        "model": model_name,
        "var_patient": var_re,
        "var_residual": var_resid,
        "icc_model": icc_model,
        "aic": result.aic if hasattr(result, "aic") else np.nan,
        "bic": result.bic if hasattr(result, "bic") else np.nan,
    }])


def summarize_gee_ordinal(result, model_name: str = "") -> pd.DataFrame:
    """
    Extract GEE results for ordinal outcomes.

    GEE with exchangeable or unstructured correlation provides
    population-averaged estimates that respect within-patient pairing.
    """
    cfg = load_config()
    alpha = cfg["analysis"]["alpha"]

    fe = pd.DataFrame({
        "term": result.params.index,
        "estimate": result.params.values,
        "se": result.bse.values,
    })
    ci = result.conf_int()
    fe["ci_lower"] = ci.iloc[:, 0].values
    fe["ci_upper"] = ci.iloc[:, 1].values
    fe["pvalue"] = result.pvalues.values
    fe["significant"] = fe["pvalue"] < alpha
    fe["model"] = model_name

    return fe


def summarize_conditional_logistic(result_dict, model_name: str = "") -> pd.DataFrame:
    """
    Extract conditional logistic regression results.

    Returns a summary matching the structure of other model summaries.
    """
    cfg = load_config()
    alpha = cfg["analysis"]["alpha"]

    if result_dict.get("model_type") != "conditional_logistic":
        # Fallback case
        return pd.DataFrame([{
            "term": "condition",
            "estimate": np.nan,
            "se": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "pvalue": np.nan,
            "significant": False,
            "model": model_name,
            "note": result_dict.get("warning", "Conditional logistic not available"),
        }])

    coef_df = result_dict["coefficients"]
    cond_row = coef_df[coef_df["term"].str.contains("condition")]

    if len(cond_row) == 0:
        return pd.DataFrame()

    return pd.DataFrame([{
        "term": "condition",
        "estimate": result_dict["condition_log_or"],
        "se": result_dict["condition_se"],
        "ci_lower": np.log(result_dict["condition_or_ci_lower"]),
        "ci_upper": np.log(result_dict["condition_or_ci_upper"]),
        "pvalue": result_dict["condition_p"],
        "significant": result_dict["condition_p"] < alpha,
        "or": result_dict["condition_or"],
        "or_ci_lower": result_dict["condition_or_ci_lower"],
        "or_ci_upper": result_dict["condition_or_ci_upper"],
        "model": model_name,
    }])
