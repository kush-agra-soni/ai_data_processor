# core/_9_encoder.py

from typing import Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder
)


# ----------------------------
# Label Encoding
# ----------------------------
def fit_label_encoders(
    df: pd.DataFrame,
    columns: List[str] | None = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Fit LabelEncoders on specified categorical columns.
    """
    df = df.copy()
    encoders: Dict[str, LabelEncoder] = {}

    if columns is None:
        columns = df.select_dtypes(include=["object", "string"]).columns.tolist()

    for col in columns:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


def transform_label_encoders(
    df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder]
) -> pd.DataFrame:
    """
    Apply fitted LabelEncoders.
    """
    df = df.copy()

    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = le.transform(df[col])

    return df


# ----------------------------
# One-Hot Encoding
# ----------------------------
def fit_onehot_encoder(
    df: pd.DataFrame,
    columns: List[str] | None = None
) -> Tuple[OneHotEncoder, List[str]]:
    """
    Fit OneHotEncoder on specified columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=["object", "string"]).columns.tolist()

    ohe = OneHotEncoder(
        sparse=False,
        handle_unknown="ignore"
    )
    ohe.fit(df[columns])

    return ohe, columns


def transform_onehot_encoder(
    df: pd.DataFrame,
    ohe: OneHotEncoder,
    columns: List[str]
) -> pd.DataFrame:
    """
    Apply OneHotEncoder and return expanded DataFrame.
    """
    df = df.copy()

    encoded = ohe.transform(df[columns])
    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(columns),
        index=df.index
    )

    df = df.drop(columns=columns)
    df = pd.concat([df, encoded_df], axis=1)

    return df


# ----------------------------
# Ordinal Encoding
# ----------------------------
def fit_ordinal_encoder(
    df: pd.DataFrame,
    columns_with_order: Dict[str, List[str]]
) -> Tuple[OrdinalEncoder, List[str]]:
    """
    Fit OrdinalEncoder with explicit category order.
    """
    cols = list(columns_with_order.keys())
    categories = [columns_with_order[col] for col in cols]

    oe = OrdinalEncoder(categories=categories)
    oe.fit(df[cols])

    return oe, cols


def transform_ordinal_encoder(
    df: pd.DataFrame,
    oe: OrdinalEncoder,
    columns: List[str]
) -> pd.DataFrame:
    """
    Apply OrdinalEncoder.
    """
    df = df.copy()
    df[columns] = oe.transform(df[columns])
    return df


# ----------------------------
# Frequency Encoding
# ----------------------------
def frequency_encoder(
    df: pd.DataFrame,
    columns: List[str] | None = None
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Replace categories with their frequency proportion.
    Returns transformed DataFrame and encoding maps.
    """
    df = df.copy()
    metadata: Dict[str, Dict] = {}

    if columns is None:
        columns = df.select_dtypes(include=["object", "string"]).columns.tolist()

    for col in columns:
        freq_map = df[col].value_counts(normalize=True)
        df[col] = df[col].map(freq_map)
        metadata[col] = freq_map.to_dict()

    return df, metadata


# ----------------------------
# AI-friendly single entry
# ----------------------------
def encode_all(
    df: pd.DataFrame,
    strategy: str = "label"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical features using a single strategy.

    strategy:
        - 'label'
        - 'onehot'
        - 'frequency'
    """
    metadata: Dict = {"strategy": strategy}

    if strategy == "label":
        df, encoders = fit_label_encoders(df)
        metadata["encoders"] = list(encoders.keys())
        return df, metadata

    if strategy == "onehot":
        ohe, cols = fit_onehot_encoder(df)
        df = transform_onehot_encoder(df, ohe, cols)
        metadata["encoded_columns"] = cols
        return df, metadata

    if strategy == "frequency":
        df, freq_maps = frequency_encoder(df)
        metadata["encoded_columns"] = list(freq_maps.keys())
        return df, metadata

    raise ValueError(
        "strategy must be one of ['label', 'onehot', 'frequency']"
    )
