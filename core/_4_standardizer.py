# core/_4_standardizer.py

from typing import Dict
import pandas as pd
import numpy as np


class Standardizer:
    """
    Deterministic value-level standardization.
    No prints, no logging, no side effects.
    """

    def __init__(self, unit_map: Dict[str, list] | None = None):
        self.unit_map = unit_map or {
            "kg": [r"\bkilograms?\b", r"\bkgs?\b", r"\bkg\.?\b"],
            "usd": [r"\$|usd|us dollars?"]
        }

    # ----------------------------
    # Numeric standardization
    # ----------------------------
    def standardize_numerical_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts numeric values from mixed columns and converts them
        to int / float when safe.
        """
        df = df.copy()

        for col in df.columns:
            series = df[col]

            if series.dtype == "object":
                # Try extracting numeric values
                extracted = series.astype(str).str.extract(
                    r"(-?\d+\.\d+|-?\d+)", expand=False
                )

                if extracted.notna().sum() >= len(series) * 0.7:
                    numeric = pd.to_numeric(extracted, errors="coerce")

                    # Decide int vs float
                    if (numeric.dropna() % 1 == 0).all():
                        df[col] = numeric.astype("Int64")
                    else:
                        df[col] = numeric.astype("float")

            elif pd.api.types.is_numeric_dtype(series):
                df[col] = pd.to_numeric(series, errors="coerce")

        return df

    # ----------------------------
    # String normalization
    # ----------------------------
    def standardize_string_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lowercase, trim, and normalize whitespace in string columns.
        """
        df = df.copy()

        string_cols = df.select_dtypes(include=["object", "string"]).columns

        for col in string_cols:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", " ", regex=True)
            )

        return df

    # ----------------------------
    # Date standardization
    # ----------------------------
    def standardize_date_format(
        self,
        df: pd.DataFrame,
        output_format: str = "%Y-%m-%d",
        min_valid_ratio: float = 0.5
    ) -> pd.DataFrame:
        """
        Converts datetime-like columns to a consistent string format.
        """
        df = df.copy()

        for col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                continue

            parsed = pd.to_datetime(df[col], errors="coerce")

            if parsed.notna().sum() >= len(df) * min_valid_ratio:
                df[col] = parsed.dt.strftime(output_format)

        return df

    # ----------------------------
    # Unit normalization
    # ----------------------------
    def standardize_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes unit representations in string columns.
        """
        df = df.copy()

        string_cols = df.select_dtypes(include=["object", "string"]).columns

        for col in string_cols:
            for standard_unit, patterns in self.unit_map.items():
                df[col] = df[col].str.replace(
                    "|".join(patterns),
                    standard_unit,
                    regex=True
                )

        return df

    # ----------------------------
    # Boolean normalization
    # ----------------------------
    def standardize_boolean_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes boolean-like values into True / False.
        """
        df = df.copy()

        for col in df.columns:
            series = df[col]

            if series.dtype == "object":
                lowered = series.astype(str).str.lower().str.strip()

                if lowered.isin(
                    ["true", "false", "yes", "no", "1", "0"]
                ).all():
                    df[col] = lowered.map(
                        lambda x: x in ["true", "yes", "1"]
                    )

        return df

    # ----------------------------
    # Canonical entry point
    # ----------------------------
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Canonical standardization entry point.
        """
        df = self.standardize_numerical_format(df)
        df = self.standardize_string_format(df)
        df = self.standardize_date_format(df)
        df = self.standardize_units(df)
        df = self.standardize_boolean_format(df)

        return df
