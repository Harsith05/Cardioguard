"""Helpers so the DataFrame-based pipeline also works where sklearn passes plain arrays."""
import numpy as np
import pandas as pd

from common.features import INPUT_FEATURES


def to_frame(X) -> pd.DataFrame:
    """Array -> DataFrame with the model's input column names (picklable, unlike a lambda)."""
    if isinstance(X, pd.DataFrame):
        return X[INPUT_FEATURES]
    return pd.DataFrame(np.asarray(X), columns=INPUT_FEATURES)
