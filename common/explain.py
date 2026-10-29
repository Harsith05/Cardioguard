"""SHAP explanations for the CardioGuard pipeline.

The saved model is a Pipeline: FeatureEngineer -> ColumnTransformer -> classifier.
SHAP runs on the classifier, using the transformed features. Then:
  * one-hot columns are summed back to their original feature
    (cholesterol_1/2/3 -> cholesterol), and
  * related measurements are summed into readable groups
    (ap_hi + ap_lo + MAP + pulse pressure -> "Blood pressure").
SHAP values are additive, so both sums are exact.

Works with LightGBM / XGBoost / Random Forest (TreeExplainer) and
Logistic Regression (LinearExplainer).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import shap

from common.features import CATEGORICAL_INPUTS, ENGINEERED, INPUT_FEATURES

# ---- display metadata -------------------------------------------------------
LABELS = {
    "age_years": "Age", "gender": "Sex", "height": "Height", "weight": "Weight",
    "ap_hi": "Systolic BP", "ap_lo": "Diastolic BP", "cholesterol": "Cholesterol",
    "gluc": "Glucose", "smoke": "Smoking", "alco": "Alcohol", "active": "Physical activity",
    "bmi": "BMI", "pulse_pressure": "Pulse pressure", "map": "Mean arterial pressure",
}
LEVELS = {1: "normal", 2: "above normal", 3: "well above normal"}

GROUPS = {  # readable group -> member features
    "blood_pressure": ["ap_hi", "ap_lo", "map", "pulse_pressure"],
    "body_size": ["height", "weight", "bmi"],
    "age": ["age_years"], "sex": ["gender"], "cholesterol": ["cholesterol"],
    "glucose": ["gluc"], "smoking": ["smoke"], "alcohol": ["alco"], "activity": ["active"],
}
GROUP_LABELS = {
    "blood_pressure": "Blood pressure", "body_size": "Body mass index", "age": "Age",
    "sex": "Sex", "cholesterol": "Cholesterol", "glucose": "Glucose",
    "smoking": "Smoking", "alcohol": "Alcohol", "activity": "Physical activity",
}
EXPLAINED_FEATURES = INPUT_FEATURES + ENGINEERED


def display_value(feature: str, row: dict) -> str:
    v = row[feature]
    if feature == "age_years":
        return f"{v:g} years"
    if feature == "gender":
        return "Female" if int(v) == 1 else "Male"
    if feature in ("height",):
        return f"{v:g} cm"
    if feature == "weight":
        return f"{v:g} kg"
    if feature in ("ap_hi", "ap_lo", "pulse_pressure", "map"):
        return f"{v:.0f} mmHg"
    if feature == "bmi":
        return f"{v:.1f}"
    if feature in ("cholesterol", "gluc"):
        return LEVELS.get(int(v), str(v)).capitalize()
    if feature == "active":
        return "Active" if int(v) == 1 else "Not active"
    if feature in ("smoke", "alco"):
        return "Yes" if int(v) == 1 else "No"
    return str(v)


def group_value(group: str, row: dict) -> str:
    if group == "blood_pressure":
        return f"{row['ap_hi']:.0f}/{row['ap_lo']:.0f} mmHg"
    if group == "body_size":
        return f"BMI {row['bmi']:.1f}"
    return display_value(GROUPS[group][0], row)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class PipelineExplainer:
    def __init__(self, pipeline, background: pd.DataFrame, background_size: int = 200):
        self.fe = pipeline.named_steps["features"]
        self.prep = pipeline.named_steps["prep"]
        self.model = pipeline.named_steps["model"]
        self.columns = list(self.prep.get_feature_names_out())
        self.col_to_feature = [self._source_feature(c) for c in self.columns]

        name = type(self.model).__name__.lower()
        if "logistic" in name:
            bg = self._transform(background.sample(min(background_size, len(background)), random_state=0))
            self.explainer = shap.LinearExplainer(self.model, bg)
            self.output_space = "log_odds"
        else:
            self.explainer = shap.TreeExplainer(self.model)
            # sklearn forests explain probabilities; boosted trees explain log-odds
            self.output_space = "probability" if "forest" in name else "log_odds"

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _source_feature(col: str) -> str:
        raw = col.split("__", 1)[1]
        for cat in CATEGORICAL_INPUTS:
            if raw.startswith(cat + "_"):
                return cat
        return raw

    def _transform(self, X: pd.DataFrame):
        Xt = self.prep.transform(self.fe.transform(X[INPUT_FEATURES]))
        return Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)

    def _raw_shap(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        exp = self.explainer(self._transform(X))
        values, base = np.asarray(exp.values), np.asarray(exp.base_values)
        if values.ndim == 3:            # (n, cols, classes) -> positive class
            values, base = values[..., 1], base[..., 1] if base.ndim == 2 else base
        return values, float(np.ravel(base)[0])

    def _per_feature(self, values: np.ndarray) -> np.ndarray:
        """(n, transformed cols) -> (n, EXPLAINED_FEATURES) by summing one-hot columns."""
        out = np.zeros((values.shape[0], len(EXPLAINED_FEATURES)))
        for j, feat in enumerate(self.col_to_feature):
            out[:, EXPLAINED_FEATURES.index(feat)] += values[:, j]
        return out

    def to_probability(self, x: float) -> float:
        return _sigmoid(x) if self.output_space == "log_odds" else x

    # -- public API -----------------------------------------------------------
    def explain(self, rows: list[dict]) -> list[dict]:
        X = pd.DataFrame(rows)[INPUT_FEATURES]
        values, base = self._raw_shap(X)
        per_feat = self._per_feature(values)
        engineered = self.fe.transform(X)
        results = []
        for i in range(len(X)):
            row = engineered.iloc[i].to_dict()
            feats = {f: float(per_feat[i, k]) for k, f in enumerate(EXPLAINED_FEATURES)}
            features = sorted(
                [{"feature": f, "label": LABELS[f], "value": display_value(f, row),
                  "shap": round(s, 4)} for f, s in feats.items()],
                key=lambda d: -abs(d["shap"]))
            groups = sorted(
                [{"group": g, "label": GROUP_LABELS[g], "value": group_value(g, row),
                  "shap": round(sum(feats[m] for m in members), 4),
                  "members": members} for g, members in GROUPS.items()],
                key=lambda d: -abs(d["shap"]))
            total = base + per_feat[i].sum()
            results.append({
                "output_space": self.output_space,
                "base_value": round(base, 4),
                "base_probability": round(self.to_probability(base), 4),
                "output_value": round(float(total), 4),
                "groups": groups,
                "features": features,
            })
        return results

    def global_importance(self, X: pd.DataFrame) -> dict:
        """Mean |SHAP| over a sample: what the model relies on overall."""
        values, _ = self._raw_shap(X)
        per_feat = self._per_feature(values)
        feat_imp = np.abs(per_feat).mean(axis=0)
        grp_vals = np.stack([per_feat[:, [EXPLAINED_FEATURES.index(m) for m in members]].sum(axis=1)
                             for members in GROUPS.values()], axis=1)
        grp_imp = np.abs(grp_vals).mean(axis=0)
        return {
            "output_space": self.output_space,
            "sample_size": int(len(X)),
            "groups": sorted([{"group": g, "label": GROUP_LABELS[g], "mean_abs_shap": round(float(v), 4)}
                              for g, v in zip(GROUPS, grp_imp)], key=lambda d: -d["mean_abs_shap"]),
            "features": sorted([{"feature": f, "label": LABELS[f], "mean_abs_shap": round(float(v), 4)}
                                for f, v in zip(EXPLAINED_FEATURES, feat_imp)],
                               key=lambda d: -d["mean_abs_shap"]),
        }
