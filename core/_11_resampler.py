# core/_11_resampler.py

from typing import Tuple, Dict

import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# ----------------------------
# Initialize resampler
# ----------------------------
def fit_resampler(
    strategy: str = "smote",
    random_state: int = 42,
    **kwargs
):
    """
    Initialize a resampler.

    strategy:
        - 'smote'
        - 'random_over'
        - 'random_under'
    """
    if strategy == "smote":
        resampler = SMOTE(random_state=random_state, **kwargs)
    elif strategy == "random_over":
        resampler = RandomOverSampler(random_state=random_state, **kwargs)
    elif strategy == "random_under":
        resampler = RandomUnderSampler(random_state=random_state, **kwargs)
    else:
        raise ValueError(
            "strategy must be one of "
            "['smote', 'random_over', 'random_under']"
        )

    return resampler


# ----------------------------
# Apply resampling safely
# ----------------------------
def apply_resampling(
    X: pd.DataFrame,
    y: pd.Series,
    resampler,
    max_size_multiplier: float = 3.0
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Apply resampling with strict size constraints.

    Returns:
        X_resampled
        y_resampled
        metadata
    """
    original_size = len(y)

    X_res, y_res = resampler.fit_resample(X, y)

    # Safety guard: prevent dataset explosion
    if len(y_res) > original_size * max_size_multiplier:
        raise RuntimeError(
            f"Resampling exceeded max_size_multiplier "
            f"({max_size_multiplier}). "
            f"Original: {original_size}, New: {len(y_res)}"
        )

    metadata: Dict = {
        "strategy": resampler.__class__.__name__,
        "original_size": original_size,
        "resampled_size": len(y_res),
        "class_distribution_before": y.value_counts().to_dict(),
        "class_distribution_after": y_res.value_counts().to_dict(),
    }

    return X_res, y_res, metadata
