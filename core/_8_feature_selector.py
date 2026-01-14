# core/_8_feature_selector.py

from typing import List, Tuple, Dict

import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ----------------------------
# Feature selection
# ----------------------------
def fit_feature_selector(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "rfe",
    estimator=None,
    k: int | str = "all"
) -> Tuple[List[str], Dict]:
    """
    Select important features based on a supervised method.

    Returns:
        selected_features
        metadata
    """
    if estimator is None:
        estimator = LogisticRegression(max_iter=500)

    metadata: Dict = {
        "method": method,
        "k": k,
        "estimator": estimator.__class__.__name__
    }

    if method == "rfe":
        selector = RFE(estimator, n_features_to_select=k)
        selector.fit(X, y)
        mask = selector.support_

    elif method == "selectkbest":
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)
        mask = selector.get_support()

    else:
        raise ValueError("method must be 'rfe' or 'selectkbest'")

    selected_features = X.columns[mask].tolist()
    metadata["selected_features"] = selected_features

    return selected_features, metadata


# ----------------------------
# Drop features
# ----------------------------
def drop_selected_features(
    X: pd.DataFrame,
    selected_features: List[str]
) -> pd.DataFrame:
    """
    Drops selected features from feature matrix.
    """
    return X.drop(columns=selected_features, errors="ignore")


# ----------------------------
# Feature importance ranking
# ----------------------------
def auto_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "random_forest",
    top_n: int = 10
) -> Tuple[pd.DataFrame, Dict]:
    """
    Rank features by importance using tree-based models.

    Returns:
        feature_importance_df
        metadata
    """
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError("Currently supported model_type: 'random_forest'")

    model.fit(X, y)

    importances = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    metadata = {
        "model_type": model_type,
        "top_n": top_n,
        "top_features": feature_importance_df.head(top_n)["feature"].tolist()
    }

    return feature_importance_df.head(top_n), metadata
