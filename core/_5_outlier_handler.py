# core/_5_outlier_handler.py

from typing import Dict, Tuple
import numpy as np
import pandas as pd


# ----------------------------
# Bound fitting
# ----------------------------
def fit_outlier_bounds(
    df: pd.DataFrame,
    method: str = "iqr",
    z_thresh: float = 3.0,
    iqr_multiplier: float = 1.5
) -> Dict[str, Tuple[float, float]]:
    """
    Fit outlier bounds per numeric column.
    """
    bounds: Dict[str, Tuple[float, float]] = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        if method == "zscore":
            mean = series.mean()
            std = series.std()
            lower = mean - z_thresh * std
            upper = mean + z_thresh * std

        elif method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr

        else:
            raise ValueError("method must be 'iqr' or 'zscore'")

        bounds[col] = (lower, upper)

    return bounds


# ----------------------------
# Removal strategy
# ----------------------------
def remove_outliers(
    df: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]],
    max_row_loss_ratio: float = 0.2
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Remove rows containing outliers.
    Enforces maximum row loss constraint.
    """
    df_clean = df.copy()
    removal_stats: Dict[str, int] = {}

    initial_rows = len(df_clean)

    for col, (lower, upper) in bounds.items():
        before = len(df_clean)
        df_clean = df_clean[
            (df_clean[col] >= lower) & (df_clean[col] <= upper)
        ]
        removed = before - len(df_clean)
        removal_stats[col] = removed

        # Safety guard
        if (initial_rows - len(df_clean)) / initial_rows > max_row_loss_ratio:
            raise RuntimeError(
                f"Outlier removal exceeded max_row_loss_ratio ({max_row_loss_ratio})"
            )

    return df_clean, removal_stats


# ----------------------------
# Capping strategy
# ----------------------------
def cap_outliers(
    df: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]]
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Cap outliers instead of removing rows.
    """
    df_capped = df.copy()
    cap_stats: Dict[str, int] = {}

    for col, (lower, upper) in bounds.items():
        series = df_capped[col]
        mask = (series < lower) | (series > upper)
        cap_stats[col] = int(mask.sum())
        df_capped[col] = series.clip(lower, upper)

    return df_capped, cap_stats


# ----------------------------
# Adaptive handler (AI-facing)
# ----------------------------
def adaptive_outlier_handling(
    df: pd.DataFrame,
    method: str = "iqr",
    strategy: str = "auto",
    row_threshold: int = 1000,
    max_row_loss_ratio: float = 0.2
) -> Tuple[pd.DataFrame, Dict]:
    """
    Adaptive outlier handling with safety constraints.
    Returns cleaned DataFrame and execution metadata.
    """
    bounds = fit_outlier_bounds(df, method=method)
    metadata = {
        "method": method,
        "strategy": strategy,
        "row_count": len(df),
        "affected_columns": list(bounds.keys())
    }

    if strategy == "auto":
        strategy = "cap" if len(df) <= row_threshold else "remove"
        metadata["auto_selected"] = strategy

    if strategy == "cap":
        df_out, stats = cap_outliers(df, bounds)
        metadata["cap_stats"] = stats

    elif strategy == "remove":
        df_out, stats = remove_outliers(
            df, bounds, max_row_loss_ratio=max_row_loss_ratio
        )
        metadata["removal_stats"] = stats

    else:
        raise ValueError("strategy must be 'cap', 'remove', or 'auto'")

    return df_out, metadata
