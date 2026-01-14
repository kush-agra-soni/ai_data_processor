# core/_6_filler.py

from typing import Dict, Tuple
import warnings
import re

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Internal regression imputer
# ----------------------------
def _regression_impute(
    df: pd.DataFrame,
    target_col: str,
    model,
    categorical: bool = False
) -> pd.DataFrame:
    """
    Impute missing values in target_col using regression.
    """
    df = df.copy()

    predictors = [
        c for c in df.columns
        if c != target_col and df[c].notna().all()
    ]

    if not predictors:
        raise ValueError(
            f"No complete predictor columns available for '{target_col}'"
        )

    complete = df[df[target_col].notna()]
    missing = df[df[target_col].isna()]

    if missing.empty:
        return df

    X_train = complete[predictors]
    y_train = complete[target_col]
    X_test = missing[predictors]

    pipeline = make_pipeline(StandardScaler(), model)

    if categorical:
        mapping = {v: i for i, v in enumerate(y_train.dropna().unique())}
        inv_mapping = {i: v for v, i in mapping.items()}
        y_train_num = y_train.map(mapping)

        pipeline.fit(X_train, y_train_num)
        preds = pipeline.predict(X_test)
        preds = [inv_mapping.get(int(round(x)), np.nan) for x in preds]

    else:
        try:
            pipeline.fit(X_train, np.log1p(y_train))
            preds = np.expm1(pipeline.predict(X_test))
        except Exception:
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

    df.loc[df[target_col].isna(), target_col] = preds
    return df


# ----------------------------
# Main imputation function
# ----------------------------
def impute_missing_values(
    df: pd.DataFrame,
    strategy: str = "mean",
    n_neighbors: int = 3,
    max_impute_ratio: float = 0.3
) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values with strict safety constraints.

    Returns:
        df_imputed
        metadata (for AI logs)
    """
    df = df.copy()
    metadata: Dict = {"strategy": strategy, "columns": {}}

    missing_ratio = df.isna().mean().max()
    if missing_ratio > max_impute_ratio:
        raise RuntimeError(
            f"Missing value ratio {missing_ratio:.2f} exceeds "
            f"max_impute_ratio {max_impute_ratio}"
        )

    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(exclude=["number"]).columns

    # ---- Simple strategies
    if strategy in {"mean", "median", "mode"}:
        for col in df.columns:
            if df[col].isna().any():
                if col in numeric_cols:
                    value = (
                        df[col].mean()
                        if strategy == "mean"
                        else df[col].median()
                        if strategy == "median"
                        else df[col].mode(dropna=True)[0]
                    )
                else:
                    value = df[col].mode(dropna=True)[0]

                df[col] = df[col].fillna(value)
                metadata["columns"][col] = strategy

    # ---- KNN
    elif strategy == "knn":
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        metadata["columns"] = {col: "knn" for col in numeric_cols}

    # ---- Linear regression
    elif strategy == "linreg":
        for col in numeric_cols:
            if df[col].isna().any():
                df = _regression_impute(
                    df, col, model=LinearRegression()
                )
                metadata["columns"][col] = "linreg"

    # ---- Logistic regression
    elif strategy == "logreg":
        for col in categorical_cols:
            if df[col].isna().any():
                df = _regression_impute(
                    df, col, model=LogisticRegression(max_iter=200),
                    categorical=True
                )
                metadata["columns"][col] = "logreg"

    # ---- Auto strategy
    elif strategy == "auto":
        for col in numeric_cols:
            if df[col].isna().any():
                try:
                    df = _regression_impute(
                        df, col, model=LinearRegression()
                    )
                    metadata["columns"][col] = "linreg"
                except Exception:
                    imputer = KNNImputer(n_neighbors=n_neighbors)
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    metadata["columns"][col] = "knn"

        for col in categorical_cols:
            if df[col].isna().any():
                try:
                    df = _regression_impute(
                        df, col,
                        model=LogisticRegression(max_iter=200),
                        categorical=True
                    )
                    metadata["columns"][col] = "logreg"
                except Exception:
                    warnings.warn(
                        f"Auto imputation failed for categorical column '{col}'"
                    )

    else:
        raise ValueError(
            "strategy must be one of "
            "['mean', 'median', 'mode', 'knn', 'linreg', 'logreg', 'auto']"
        )

    return df, metadata


# ----------------------------
# Date feature extraction
# ----------------------------
def extract_dates(
    df: pd.DataFrame,
    date_format: str | None = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Extract day / month / year features from date-like string columns.
    """
    df = df.copy()
    metadata: Dict = {"extracted_columns": []}
    time_only_pattern = re.compile(r"^\d{2}:\d{2}:\d{2}$")

    for col in df.select_dtypes(include=["object", "string"]).columns:
        sample = df[col].dropna().astype(str).head(5)

        if sample.empty or sample.str.match(time_only_pattern).all():
            continue

        parsed = pd.to_datetime(df[col], errors="coerce", format=date_format)

        if parsed.notna().sum() >= len(df) * 0.5:
            df[f"{col}_day"] = parsed.dt.day
            df[f"{col}_month"] = parsed.dt.month
            df[f"{col}_year"] = parsed.dt.year
            metadata["extracted_columns"].append(col)

    return df, metadata
