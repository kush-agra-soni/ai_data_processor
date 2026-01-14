# core/_10_scaler.py

from typing import Tuple, Dict

import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)


# ----------------------------
# Fit scaler
# ----------------------------
def fit_scaler(
    X: pd.DataFrame,
    scaler_type: str = "standard"
) -> Tuple[object, Dict]:
    """
    Fit a scaler on feature matrix X.

    Returns:
        scaler
        metadata
    """
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(
            "scaler_type must be one of ['standard', 'minmax', 'robust']"
        )

    scaler.fit(X)

    metadata = {
        "scaler_type": scaler_type,
        "n_features": X.shape[1],
        "feature_names": list(X.columns)
    }

    return scaler, metadata


# ----------------------------
# Transform with scaler
# ----------------------------
def transform_scaler(
    X: pd.DataFrame,
    scaler
) -> pd.DataFrame:
    """
    Transform feature matrix using a fitted scaler.
    """
    X_scaled = scaler.transform(X)

    return pd.DataFrame(
        X_scaled,
        columns=X.columns,
        index=X.index
    )
