"""
Project configuration loader.

Loads config/config.yaml for general settings.
The blinding config (config/blinding.yaml) is loaded ONLY via apply_unblinding(),
which must not be called until all model specifications are locked.
"""

from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    """Load general project config. Safe to call at any time."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_path(key: str, mkdir: bool = False) -> Path:
    """
    Resolve a path from config.yaml relative to PROJECT_ROOT.

    Parameters
    ----------
    key : str
        Key under `paths:` in config.yaml (e.g. 'output_exploratory').
    mkdir : bool
        If True, create the directory (and parents) if it doesn't exist.

    Returns
    -------
    Path
        Absolute path.

    Raises
    ------
    KeyError
        If the key is not found under `paths:` in config.yaml.
    """
    cfg = load_config()
    try:
        relative = cfg["paths"][key]
    except KeyError:
        raise KeyError(f"Path key '{key}' not found in config.yaml paths section")
    resolved = PROJECT_ROOT / relative
    if mkdir:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def apply_unblinding(df, condition_col: str = "condition") -> "pd.DataFrame":
    """
    Relabel the condition column using the blinding config.

    ONLY call this function after all model specifications are locked and
    the study coordinator has approved unblinding.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the condition column with values 0/1.
    condition_col : str
        Name of the condition column to relabel.

    Returns
    -------
    pd.DataFrame
        New DataFrame with an added `condition_label` column. The original
        numeric `condition` column is preserved unchanged.

    Raises
    ------
    FileNotFoundError
        If config/blinding.yaml does not exist.
    ValueError
        If unblinding.unblinded is False in the config, or labels are not set.
    """
    import pandas as pd

    blinding_path = PROJECT_ROOT / "config" / "blinding.yaml"
    if not blinding_path.exists():
        raise FileNotFoundError(
            "config/blinding.yaml not found. "
            "This file is gitignored and must be created locally."
        )

    with open(blinding_path, "r") as f:
        blinding = yaml.safe_load(f).get("blinding", {})

    if not blinding.get("unblinded", False):
        raise ValueError(
            "blinding.yaml has unblinded=false. "
            "Set unblinded=true only after all model specs are locked and "
            "unblinding is approved."
        )

    label_map = blinding.get("condition_labels", {})
    if any(v is None for v in label_map.values()):
        raise ValueError(
            "One or more condition labels are null in blinding.yaml. "
            "Fill in the true treatment labels before calling apply_unblinding()."
        )

    result = df.copy()
    result["condition_label"] = result[condition_col].map(
        {int(k): v for k, v in label_map.items()}
    )
    return result
