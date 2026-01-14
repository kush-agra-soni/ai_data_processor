# core/_12_validator.py

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------
# NaN validation
# ----------------------------
def check_final_nans(df: pd.DataFrame) -> Dict:
    """
    Check for remaining NaN values.
    Returns structured report.
    """
    nan_counts = df.isna().sum()
    nan_columns = nan_counts[nan_counts > 0]

    return {
        "has_nans": not nan_columns.empty,
        "nan_columns": nan_columns.to_dict()
    }


# ----------------------------
# Type validation
# ----------------------------
def check_invalid_types(
    df: pd.DataFrame,
    allowed_dtypes: Tuple[str, ...] = ("number", "bool", "datetime64[ns]")
) -> Dict:
    """
    Detect columns that are not numeric / boolean / datetime.
    """
    invalid_columns = []

    for col in df.columns:
        dtype = str(df[col].dtype)
        if not any(allowed in dtype for allowed in allowed_dtypes):
            invalid_columns.append(col)

    return {
        "invalid_type_columns": invalid_columns,
        "count": len(invalid_columns)
    }


# ----------------------------
# Outlier recheck
# ----------------------------
def recheck_outliers(
    df: pd.DataFrame,
    method: str = "zscore",
    threshold: float = 3.0
) -> Dict:
    """
    Recheck for remaining outliers.
    Returns count per column.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    result: Dict = {}

    if numeric_df.empty:
        return {"outliers": {}}

    if method == "zscore":
        z_scores = np.abs(stats.zscore(numeric_df, nan_policy="omit"))
        z_scores = pd.DataFrame(z_scores, columns=numeric_df.columns)
        result = (z_scores > threshold).sum().to_dict()

    elif method == "iqr":
        q1 = numeric_df.quantile(0.25)
        q3 = numeric_df.quantile(0.75)
        iqr = q3 - q1
        outliers = (
            (numeric_df < (q1 - 1.5 * iqr)) |
            (numeric_df > (q3 + 1.5 * iqr))
        )
        result = outliers.sum().to_dict()

    else:
        raise ValueError("method must be 'zscore' or 'iqr'")

    # Keep only columns with detected outliers
    result = {k: v for k, v in result.items() if v > 0}

    return {
        "method": method,
        "outlier_columns": result
    }


# ----------------------------
# Final validation wrapper
# ----------------------------
def validate_dataset(df: pd.DataFrame) -> Dict:
    """
    Run full validation suite.
    This is the canonical validator entry point.
    """
    report = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "nan_check": check_final_nans(df),
        "type_check": check_invalid_types(df),
        "outlier_check": recheck_outliers(df)
    }

    report["is_valid"] = (
        not report["nan_check"]["has_nans"]
        and report["type_check"]["count"] == 0
        and len(report["outlier_check"]["outlier_columns"]) == 0
    )

    return report
