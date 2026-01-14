# core/pipeline.py

from typing import Dict, Any, Tuple

import pandas as pd

# ----------------------------
# Core imports (deterministic)
# ----------------------------
from core._2_detector import DataDetector
from core._3_cleaner import Cleaner
from core._4_standardizer import Standardizer
from core._5_outlier_handler import adaptive_outlier_handling
from core._6_filler import impute_missing_values, extract_dates
from core._7_dtype_handler import (
    detect_types_proactively,
    enforce_column_types,
    convert_types_as_needed,
)
from core._8_feature_selector import (
    fit_feature_selector,
    drop_selected_features,
    auto_feature_importance,
)
from core._9_encoder import encode_all
from core._10_scaler import fit_scaler, transform_scaler
from core._11_resampler import fit_resampler, apply_resampling
from core._12_validator import validate_dataset


# ==========================================================
# Pipeline Registry
# ==========================================================
class Pipeline:
    """
    Canonical execution engine.
    AI (or manual config) can only call steps defined here.
    """

    def __init__(self):
        self.detector = DataDetector()
        self.cleaner = Cleaner()
        self.standardizer = Standardizer()

    # ----------------------------
    # Stage 1: Detection
    # ----------------------------
    def detect(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        df, detected_types = self.detector.run(df)
        proactive = detect_types_proactively(df)
        return df, {
            "detected_types": detected_types,
            "proactive_prediction": proactive,
        }

    # ----------------------------
    # Stage 2: Cleaning
    # ----------------------------
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.cleaner.run(df)

    # ----------------------------
    # Stage 3: Standardization
    # ----------------------------
    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.standardizer.run(df)

    # ----------------------------
    # Stage 4: Outlier handling
    # ----------------------------
    def handle_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        strategy: str = "auto",
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        return adaptive_outlier_handling(
            df,
            method=method,
            strategy=strategy,
            **kwargs
        )

    # ----------------------------
    # Stage 5: Missing values
    # ----------------------------
    def impute(
        self,
        df: pd.DataFrame,
        strategy: str = "auto",
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        return impute_missing_values(
            df,
            strategy=strategy,
            **kwargs
        )

    # ----------------------------
    # Stage 6: Date feature extraction
    # ----------------------------
    def extract_dates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        return extract_dates(df)

    # ----------------------------
    # Stage 7: Type enforcement
    # ----------------------------
    def enforce_types(
        self,
        df: pd.DataFrame,
        type_map: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict]:
        return enforce_column_types(df, type_map)

    def auto_convert_types(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        return convert_types_as_needed(df)

    # ----------------------------
    # Stage 8: Encoding
    # ----------------------------
    def encode(
        self,
        df: pd.DataFrame,
        strategy: str = "label"
    ) -> Tuple[pd.DataFrame, Dict]:
        return encode_all(df, strategy=strategy)

    # ----------------------------
    # Stage 9: Feature selection
    # ----------------------------
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "rfe",
        k: int | str = "all"
    ) -> Tuple[pd.DataFrame, Dict]:
        selected, meta = fit_feature_selector(
            X, y, method=method, k=k
        )
        X = drop_selected_features(X, selected)
        return X, meta

    def feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = "random_forest"
    ) -> Tuple[pd.DataFrame, Dict]:
        return auto_feature_importance(X, y, model_type=model_type)

    # ----------------------------
    # Stage 10: Scaling
    # ----------------------------
    def scale(
        self,
        X: pd.DataFrame,
        scaler_type: str = "standard"
    ) -> Tuple[pd.DataFrame, Dict]:
        scaler, meta = fit_scaler(X, scaler_type=scaler_type)
        X_scaled = transform_scaler(X, scaler)
        return X_scaled, meta

    # ----------------------------
    # Stage 11: Resampling
    # ----------------------------
    def resample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        strategy: str = "smote",
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        resampler = fit_resampler(strategy=strategy)
        return apply_resampling(X, y, resampler, **kwargs)

    # ----------------------------
    # Stage 12: Validation
    # ----------------------------
    def validate(self, df: pd.DataFrame) -> Dict:
        return validate_dataset(df)


# ==========================================================
# Public factory (used by AI executor)
# ==========================================================
def get_pipeline() -> Pipeline:
    """
    Factory method.
    Ensures a single, clean execution surface.
    """
    return Pipeline()
