"""
Composite outcome construction and validation.

Combines depth and extent into a single score for secondary analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.config import load_config


def build_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add composite score = depth + extent (range 0–8).

    Returns a new DataFrame with added columns:
    - composite: sum of depth + extent
    - composite_binary_1: composite >= 1
    - composite_binary_3: composite >= 3 (clinically significant)
    """
    cfg = load_config()
    result = df.copy()
    result["composite"] = result["depth"] + result["extent"]

    result["composite_binary_1"] = (result["composite"] >= 1).astype(int)
    result["composite_binary_3"] = (result["composite"] >= 3).astype(int)
    return result


def validate_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assess whether combining depth and extent is justified.

    Returns a DataFrame with:
    - Spearman correlation between depth and extent
    - Cronbach's alpha for the two-item scale
    - Interpretation and warnings
    """
    rho, p_rho = stats.spearmanr(df["depth"], df["extent"])

    # Cronbach's alpha for 2 items
    var_depth = df["depth"].var()
    var_extent = df["extent"].var()
    var_total = (df["depth"] + df["extent"]).var()
    k = 2
    alpha = (k / (k - 1)) * (1 - (var_depth + var_extent) / var_total)

    if rho > 0.85:
        warning = "High redundancy — depth and extent may measure the same construct"
    elif rho < 0.3:
        warning = "Low correlation — composite may obscure differential treatment effects"
    else:
        warning = "Acceptable correlation — composite is justified"

    return pd.DataFrame([{
        "spearman_rho": rho,
        "spearman_p": p_rho,
        "cronbach_alpha": alpha,
        "var_depth": var_depth,
        "var_extent": var_extent,
        "var_composite": var_total,
        "interpretation": warning,
    }])
