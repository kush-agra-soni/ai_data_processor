# ai/profiler.py

from typing import Dict, Any, List
import hashlib

import pandas as pd
import numpy as np

from core._7_dtype_handler import detect_types_proactively


# ==========================================================
# Profiler
# ==========================================================
class DataProfiler:
    """
    Converts a raw DataFrame into an LLM-safe, compact, structured profile.
    This is the ONLY data Gemini is allowed to see.
    """

    def __init__(
        self,
        sample_rows: int = 10,
        top_k_values: int = 5,
        pii_keywords: List[str] | None = None
    ):
        self.sample_rows = sample_rows
        self.top_k_values = top_k_values
        self.pii_keywords = pii_keywords or [
            "email", "mail",
            "phone", "mobile",
            "name", "first_name", "last_name",
            "address", "street", "city", "zip",
            "aadhaar", "pan", "ssn", "passport"
        ]

    # ----------------------------
    # Public entry
    # ----------------------------
    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a full profiling report.
        """
        profile = {
            "dataset_fingerprint": self._fingerprint(df),
            "shape": {
                "rows": df.shape[0],
                "columns": df.shape[1],
            },
            "columns": self._profile_columns(df),
            "numeric_summary": self._numeric_summary(df),
            "categorical_summary": self._categorical_summary(df),
            "predicted_types": detect_types_proactively(df),
            "sample_rows": self._sample_rows(df),
            "suspected_pii_columns": self._detect_pii_columns(df),
        }

        return profile

    # ======================================================
    # Internals
    # ======================================================
    def _profile_columns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        columns_profile = []

        for col in df.columns:
            series = df[col]

            col_profile = {
                "name": col,
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
                "null_ratio": float(series.isna().mean()),
                "unique_count": int(series.nunique(dropna=True)),
            }

            if pd.api.types.is_numeric_dtype(series):
                col_profile.update({
                    "min": self._safe_float(series.min()),
                    "max": self._safe_float(series.max()),
                    "mean": self._safe_float(series.mean()),
                    "std": self._safe_float(series.std()),
                })

            else:
                value_counts = (
                    series.astype(str)
                    .value_counts(dropna=True)
                    .head(self.top_k_values)
                )
                col_profile["top_values"] = value_counts.to_dict()

            columns_profile.append(col_profile)

        return columns_profile

    # ----------------------------
    def _numeric_summary(self, df: pd.DataFrame) -> Dict[str, Dict]:
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return {}

        desc = numeric_df.describe().to_dict()
        return {
            col: {
                k: self._safe_float(v)
                for k, v in stats.items()
            }
            for col, stats in desc.items()
        }

    # ----------------------------
    def _categorical_summary(self, df: pd.DataFrame) -> Dict[str, Dict]:
        cat_df = df.select_dtypes(exclude=[np.number])

        summary = {}
        for col in cat_df.columns:
            vc = cat_df[col].astype(str).value_counts(dropna=True)
            summary[col] = {
                "unique": int(vc.shape[0]),
                "top_values": vc.head(self.top_k_values).to_dict(),
            }

        return summary

    # ----------------------------
    def _sample_rows(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Return masked sample rows for LLM context.
        """
        sample = df.head(self.sample_rows).copy()

        for col in sample.columns:
            if pd.api.types.is_numeric_dtype(sample[col]):
                continue
            sample[col] = sample[col].astype(str).str.slice(0, 32)

        return sample.to_dict(orient="records")

    # ----------------------------
    def _detect_pii_columns(self, df: pd.DataFrame) -> List[str]:
        suspected = []

        for col in df.columns:
            name = col.lower()
            if any(keyword in name for keyword in self.pii_keywords):
                suspected.append(col)
                continue

            # pattern-based detection (email / phone)
            series = df[col].dropna().astype(str)
            if series.empty:
                continue

            email_hits = series.str.contains(r"@.+\.", regex=True).mean()
            phone_hits = series.str.contains(r"\d{8,}", regex=True).mean()

            if email_hits > 0.6 or phone_hits > 0.6:
                suspected.append(col)

        return list(set(suspected))

    # ----------------------------
    def _fingerprint(self, df: pd.DataFrame) -> str:
        """
        Create a stable fingerprint for caching LLM plans.
        """
        payload = (
            str(df.shape)
            + "|".join(df.columns)
            + "|".join(str(t) for t in df.dtypes)
        )
        return hashlib.md5(payload.encode()).hexdigest()

    # ----------------------------
    @staticmethod
    def _safe_float(value) -> float | None:
        try:
            if pd.isna(value):
                return None
            return float(value)
        except Exception:
            return None


# ==========================================================
# Factory
# ==========================================================
def get_profiler() -> DataProfiler:
    """
    Factory method for profiler.
    """
    return DataProfiler()
