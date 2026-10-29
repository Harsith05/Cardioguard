"""Shared feature definitions and engineering.

Imported by BOTH training and the API, so the model always sees features
built the same way (no training/serving skew).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Raw inputs the API accepts (age already converted to years)
NUMERIC_INPUTS = ["age_years", "height", "weight", "ap_hi", "ap_lo"]
CATEGORICAL_INPUTS = ["gender", "cholesterol", "gluc"]
BINARY_INPUTS = ["smoke", "alco", "active"]
INPUT_FEATURES = NUMERIC_INPUTS + CATEGORICAL_INPUTS + BINARY_INPUTS
TARGET = "cardio"

# Columns created by FeatureEngineer
ENGINEERED = ["bmi", "pulse_pressure", "map"]
MODEL_NUMERIC = NUMERIC_INPUTS + ENGINEERED

# Valid ranges, used by data cleaning AND API validation
VALID_RANGES = {
    "age_years": (18, 100),
    "height": (120, 220),      # cm
    "weight": (30, 200),       # kg
    "ap_hi": (80, 250),        # systolic mmHg
    "ap_lo": (40, 160),        # diastolic mmHg
}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds BMI, pulse pressure and mean arterial pressure."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        h_m = X["height"] / 100.0
        X["bmi"] = X["weight"] / (h_m ** 2)
        X["pulse_pressure"] = X["ap_hi"] - X["ap_lo"]
        X["map"] = (X["ap_hi"] + 2 * X["ap_lo"]) / 3.0
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(INPUT_FEATURES + ENGINEERED)
