# core/_7_dtype_handler.py

from typing import Dict, Tuple
import re

import pandas as pd
from dateutil.parser import parse


# ----------------------------
# Proactive type detection
# ----------------------------
def detect_types_proactively(
    df: pd.DataFrame,
    sample_size: int = 10
) -> Dict[str, str]:
    """
    Predict column types from a small sample of values.
    Returns: {column_name: predicted_type}
    """
    predicted_types: Dict[str, str] = {}
    time_pattern = re.compile(r"^\d{2}:\d{2}:\d{2}$")

    for col in df.columns:
        sample = (
            df[col]
            .dropna()
            .astype(str)
            .head(sample_size)
            .str.strip()
        )

        if sample.empty:
            predicted_types[col] = "unknown"
            continue

        # Boolean
        if sample.str.lower().isin(
            ["true", "false", "yes", "no", "1", "0"]
        ).all():
            predicted_types[col] = "bool"
            continue

        # Time-only
        if sample.str.match(time_pattern).all():
            predicted_types[col] = "time"
            continue

        # Date
        date_hits = 0
        for val in sample:
            try:
                parse(val, fuzzy=False, dayfirst=True)
                date_hits += 1
            except Exception:
                pass

        if date_hits >= len(sample) * 0.8:
            predicted_types[col] = "date"
            continue

        # Numeric
        numeric = pd.to_numeric(sample, errors="coerce")
        if numeric.notna().all():
            if (numeric % 1 == 0).all():
                predicted_types[col] = "int"
            else:
                predicted_types[col] = "float"
            continue

        predicted_types[col] = "str"

    return predicted_types


# ----------------------------
# Enforce explicit types
# ----------------------------
def enforce_column_types(
    df: pd.DataFrame,
    type_map: Dict[str, object]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Force column types based on an explicit mapping.
    Returns transformed DataFrame and metadata.
    """
    df = df.copy()
    metadata: Dict = {"enforced": {}, "failed": {}}

    for col, expected_type in type_map.items():
        if col not in df.columns:
            metadata["failed"][col] = "column_not_found"
            continue

        try:
            df[col] = df[col].astype(expected_type)
            metadata["enforced"][col] = str(expected_type)
        except Exception as e:
            metadata["failed"][col] = str(e)

    return df, metadata


# ----------------------------
# Auto conversion
# ----------------------------
def convert_types_as_needed(
    df: pd.DataFrame,
    min_success_ratio: float = 0.8
) -> Tuple[pd.DataFrame, Dict]:
    """
    Automatically convert object/string columns to more specific types.
    Returns transformed DataFrame and metadata.
    """
    df = df.copy()
    metadata: Dict = {}

    for col in df.columns:
        if not pd.api.types.is_object_dtype(df[col]):
            continue

        series = df[col]

        # Try numeric
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() >= len(series) * min_success_ratio:
            df[col] = numeric
            metadata[col] = "numeric"
            continue

        # Try datetime
        datetime = pd.to_datetime(series, errors="coerce")
        if datetime.notna().sum() >= len(series) * min_success_ratio:
            df[col] = datetime
            metadata[col] = "datetime"
            continue

        # Try boolean
        lowered = series.astype(str).str.lower().str.strip()
        if lowered.isin(
            ["true", "false", "yes", "no", "1", "0"]
        ).all():
            df[col] = lowered.map(
                lambda x: x in ["true", "yes", "1"]
            )
            metadata[col] = "boolean"
            continue

    return df, metadata
