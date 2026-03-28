"""
Raw data loader and validator.

All scripts and notebooks must import data through this module.
Never read irrigation_data.xlsx directly outside of this file.
"""

import pandas as pd
from pathlib import Path
from src.config import load_config


def load_raw_data() -> pd.DataFrame:
    """
    Load and validate the raw irrigation dataset.

    Returns
    -------
    pd.DataFrame
        Validated raw data with dtypes enforced.

    Raises
    ------
    FileNotFoundError
        If the raw data file does not exist.
    ValueError
        If the dataset does not match expected shape or values.
    """
    cfg = load_config()
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / cfg["paths"]["raw_data"]

    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at: {path}")

    df = pd.read_excel(path, sheet_name="Sheet1")
    _validate(df, cfg)
    df = _enforce_dtypes(df)
    return df


def _validate(df: pd.DataFrame, cfg: dict) -> None:
    expected_rows = cfg["data"]["expected_rows"]
    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, got {len(df)}."
        )

    missing = df.isnull().sum()
    if missing.any():
        raise ValueError(f"Unexpected missing values:\n{missing[missing > 0]}")

    for col in cfg["data"]["outcome_columns"]:
        out_of_range = ~df[col].between(
            cfg["data"]["outcome_min"], cfg["data"]["outcome_max"]
        )
        if out_of_range.any():
            raise ValueError(
                f"Column '{col}' has values outside "
                f"[{cfg['data']['outcome_min']}, {cfg['data']['outcome_max']}]: "
                f"{df.loc[out_of_range, col].unique().tolist()}"
            )

    invalid_conditions = ~df["condition"].isin(cfg["data"]["condition_values"])
    if invalid_conditions.any():
        raise ValueError(
            f"Unexpected condition values: "
            f"{df.loc[invalid_conditions, 'condition'].unique().tolist()}"
        )

    actual_patients = set(df["patient"].unique())
    excluded = set(cfg["data"]["excluded_patients"])
    unexpected = actual_patients & excluded
    if unexpected:
        raise ValueError(
            f"Excluded patients {unexpected} are present in the data."
        )


def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["patient"] = df["patient"].astype(int)
    df["procedure_code"] = df["procedure_code"].astype(int)
    df["condition"] = df["condition"].astype(int)
    df["depth"] = df["depth"].astype(int)
    df["extent"] = df["extent"].astype(int)
    df["quadrant"] = df["quadrant"].astype(str).str.strip().str.lower()
    df["grader"] = df["grader"].astype(str).str.strip()
    return df


def add_binary_outcomes(df: pd.DataFrame, threshold: int = 1) -> pd.DataFrame:
    """
    Add binary outcome columns: 1 if score >= threshold, else 0.

    Parameters
    ----------
    df : pd.DataFrame
    threshold : int
        Minimum score to classify as bruising present (default 1).

    Returns
    -------
    pd.DataFrame
        New DataFrame with added columns `depth_binary` and `extent_binary`.
    """
    result = df.copy()
    result["depth_binary"] = (result["depth"] >= threshold).astype(int)
    result["extent_binary"] = (result["extent"] >= threshold).astype(int)
    return result
