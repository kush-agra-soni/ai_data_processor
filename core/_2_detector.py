import re
import pandas as pd
from typing import Dict, Tuple
from dateutil.parser import parse

class DataDetector:
    """
    Detects and normalizes primitive data types in a DataFrame.
    This class is deterministic and import-safe (no side effects).
    """

    def __init__(self):
        self.time_pattern = re.compile(r"^\d{2}:\d{2}:\d{2}$")  # HH:MM:SS

    # ----------------------------
    # Time-only column detection
    # ----------------------------
    def detect_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect columns that strictly match HH:MM:SS and convert them
        to time-only string representation.
        """
        df = df.copy()

        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue

            str_values = series.astype(str).str.strip()

            if str_values.str.match(self.time_pattern).all():
                df[col] = (
                    pd.to_datetime(df[col], format="%H:%M:%S", errors="coerce")
                    .dt.time
                    .astype("string")
                )

        return df

    # ----------------------------
    # Single-column type detection
    # ----------------------------
    def detect_column_type(self, df: pd.DataFrame, column: str) -> str:
        """
        Detect and coerce the most appropriate data type for a column.
        """
        series = df[column].dropna()
        if series.empty:
            return "unknown"

        str_values = series.astype(str).str.strip()

        # Boolean detection
        if str_values.str.lower().isin(
            ["true", "false", "yes", "no", "1", "0"]
        ).all():
            df[column] = str_values.str.lower().map(
                lambda x: x in ["true", "yes", "1"]
            )
            return "bool"

        # Date detection
        if str_values.apply(self._is_date).all():
            df[column] = pd.to_datetime(df[column], errors="coerce")
            return "date"

        # Numeric detection
        numeric_values = pd.to_numeric(str_values, errors="coerce")
        if numeric_values.notna().all():
            if (numeric_values % 1 == 0).all():
                df[column] = numeric_values.astype("Int64")
                return "int"
            df[column] = numeric_values.astype("float")
            return "float"

        # Fallback: string
        df[column] = df[column].astype("string")
        return "str"

    # ----------------------------
    # Full DataFrame detection
    # ----------------------------
    def detect_column_types(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Detect and coerce column types for the entire DataFrame.
        """
        df = df.copy()
        detected_types: Dict[str, str] = {}

        for col in df.columns:
            detected_types[col] = self.detect_column_type(df, col)

        return df, detected_types

    # ----------------------------
    # Public entry point
    # ----------------------------
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Canonical entry point used by pipeline / AI executor.
        """
        df = self.detect_time_columns(df)
        df, detected_types = self.detect_column_types(df)
        return df, detected_types

    # ----------------------------
    # Helpers
    # ----------------------------
    @staticmethod
    def _is_date(value: str) -> bool:
        try:
            parse(value, fuzzy=False, dayfirst=True)
            return True
        except Exception:
            return False
