"""Phase 1 - Step 2: train baseline models, pick the best, save model + reference data.

Usage:
    python -m training.train
    python -m training.train --mlflow-uri http://localhost:5000   # log to an MLflow server

Selection rule: highest validation ROC-AUC. The test set is touched only once,
for the chosen model.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from common.features import (BINARY_INPUTS, CATEGORICAL_INPUTS, INPUT_FEATURES,
                             MODEL_NUMERIC, TARGET, FeatureEngineer)

SEED = 42
DATA = Path("data/processed")
MODELS = Path("models")
REFERENCE = Path("data/reference")


def preprocessor(scale: bool) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, MODEL_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_INPUTS),
        ("bin", "passthrough", BINARY_INPUTS),
    ])


# Medical knowledge built into the model (monotonic constraints): predicted risk may only go UP
# when these go up, everything else equal. Cholesterol and glucose are kept as ordered numbers
# (1 < 2 < 3) instead of one-hot columns so the rule can apply to them.
MONOTONE_INCREASING = {"age_years", "ap_hi", "ap_lo", "map", "cholesterol", "gluc"}
ORDINAL_INPUTS = ["cholesterol", "gluc"]
# pulse pressure (systolic - diastolic) is left out: it FALLS when diastolic BP rises, so with it
# the model could still lower the risk for a higher blood pressure.
MONOTONE_NUMERIC = [c for c in MODEL_NUMERIC if c != "pulse_pressure"]
MONOTONE_COLUMNS = MONOTONE_NUMERIC + ORDINAL_INPUTS + ["gender_1", "gender_2"] + BINARY_INPUTS


def monotone_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", "passthrough", MONOTONE_NUMERIC),
        ("ord", "passthrough", ORDINAL_INPUTS),
        ("cat", OneHotEncoder(categories=[[1, 2]], handle_unknown="ignore"), ["gender"]),
        ("bin", "passthrough", BINARY_INPUTS),
    ])


def monotone_candidates() -> dict[str, Pipeline]:
    """LightGBM and XGBoost that cannot learn medically impossible rules
    (e.g. "higher blood pressure = lower risk"). Costs ~0.001 validation AUC."""
    cons = [1 if c in MONOTONE_INCREASING else 0 for c in MONOTONE_COLUMNS]

    def pipe(model):
        return Pipeline([("features", FeatureEngineer()), ("prep", monotone_preprocessor()), ("model", model)])
    return {
        "xgboost": pipe(XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=-1, random_state=SEED, monotone_constraints=tuple(cons))),
        "lightgbm": pipe(LGBMClassifier(
            n_estimators=400, num_leaves=31, learning_rate=0.03, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=SEED, verbose=-1, monotone_constraints=cons,
            monotone_constraints_method="advanced")),
    }


def candidates(monotone: bool = False) -> dict[str, Pipeline]:
    if monotone:
        return monotone_candidates()

    def pipe(model, scale=False):
        return Pipeline([("features", FeatureEngineer()),
                         ("prep", preprocessor(scale)),
                         ("model", model)])
    return {
        "logistic_regression": pipe(LogisticRegression(max_iter=2000, C=1.0), scale=True),
        "random_forest": pipe(RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=20, n_jobs=-1, random_state=SEED)),
        "xgboost": pipe(XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=SEED)),
        "lightgbm": pipe(LGBMClassifier(
            n_estimators=400, num_leaves=31, learning_rate=0.03, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)),
    }


def metrics(y, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "brier": brier_score_loss(y, proba),
    }


def try_mlflow(uri: str | None):
    try:
        import mlflow
        mlflow.set_tracking_uri(uri or f"file:{Path('mlruns').resolve()}")
        mlflow.set_experiment("cardioguard-baseline")
        return mlflow
    except Exception as e:  # MLflow is optional in Phase 1
        print(f"[warn] MLflow disabled: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlflow-uri", default=None)
    ap.add_argument("--monotone", action="store_true",
                    help="only LightGBM/XGBoost with medical monotonic constraints (risk rises with BP, age, ...)")
    ap.add_argument("--include-production", action="store_true",
                    help="also train on human diagnoses and pseudo-labels from production (Phase 5)")
    ap.add_argument("--pseudo-weight", type=float, default=None,
                    help="sample weight of pseudo-labels (default: semisup/config.yaml)")
    args = ap.parse_args()

    train = pd.read_parquet(DATA / "train.parquet")
    val = pd.read_parquet(DATA / "val.parquet")
    test = pd.read_parquet(DATA / "test.parquet")
    weights, composition = None, {"training": len(train)}
    if args.include_production:
        import os

        from sqlalchemy import create_engine

        from semisup.pseudo_label import load_config
        from semisup.training_set import build
        pw = args.pseudo_weight if args.pseudo_weight is not None else load_config()["pseudo_label_weight"]
        engine = create_engine(os.getenv("DATABASE_URL", "postgresql+psycopg2://cardio:cardio@localhost:5432/cardio"))
        train = build(engine, train, pseudo_weight=pw)
        from semisup.training_set import WEIGHT_COL
        weights = train[WEIGHT_COL].to_numpy()
        composition = train["source"].value_counts().to_dict()
        print(f"Training rows by source: {composition} (pseudo-label weight {pw})")
        print("[warn] Simulated traffic draws patients from the TEST set, so with --include-production\n"
              "       the test metrics are optimistic. Judge this model by its validation metrics\n"
              "       (or: python -m semisup.compare_retrain).")
    Xtr, ytr = train[INPUT_FEATURES], train[TARGET]
    Xva, yva = val[INPUT_FEATURES], val[TARGET]
    Xte, yte = test[INPUT_FEATURES], test[TARGET]

    mlflow = try_mlflow(args.mlflow_uri)
    results, fitted = {}, {}

    for name, pipe in candidates(args.monotone).items():
        t0 = time.perf_counter()
        pipe.fit(Xtr, ytr, **({"model__sample_weight": weights} if weights is not None else {}))
        fit_s = time.perf_counter() - t0
        m = metrics(yva, pipe.predict_proba(Xva)[:, 1])
        m["fit_seconds"] = fit_s
        results[name], fitted[name] = m, pipe
        print(f"{name:20s} val AUC={m['roc_auc']:.4f}  recall={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  brier={m['brier']:.4f}  ({fit_s:.1f}s)")
        if mlflow:
            with mlflow.start_run(run_name=name):
                mlflow.log_params({"model": name, **{
                    k: v for k, v in pipe.named_steps["model"].get_params().items()
                    if isinstance(v, (int, float, str, bool)) and v is not None}})
                mlflow.log_metrics({f"val_{k}": v for k, v in m.items()})

    best = max(results, key=lambda k: results[k]["roc_auc"])
    model = fitted[best]
    test_m = metrics(yte, model.predict_proba(Xte)[:, 1])
    version = datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")
    print(f"\nBest: {best}  test AUC={test_m['roc_auc']:.4f}  recall={test_m['recall']:.4f}")

    # --- save model ---
    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / "model.joblib")
    import platform

    import lightgbm
    import sklearn
    import xgboost
    meta = {
        "model_version": version,
        "model_name": best,
        # a pickled model should be loaded with the same library versions
        "environment": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                        "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": INPUT_FEATURES,
        "threshold": 0.5,
        "train_rows": len(train),
        "train_composition": composition,      # training / human / pseudo rows
        # simulated production patients come from the test split -> test metrics may be optimistic
        "test_may_overlap_production": bool(args.include_production),
        "validation": {k: round(v, 4) for k, v in results[best].items()},
        "test": {k: round(v, 4) for k, v in test_m.items()},
        "all_candidates_val": {n: {k: round(v, 4) for k, v in r.items()} for n, r in results.items()},
    }
    (MODELS / "model_meta.json").write_text(json.dumps(meta, indent=2))

    # --- reference data for drift monitoring (Phase 2) ---
    REFERENCE.mkdir(parents=True, exist_ok=True)
    ref = train[INPUT_FEATURES + [TARGET]].copy()
    ref["prediction_proba"] = model.predict_proba(Xtr)[:, 1]
    ref["prediction"] = (ref["prediction_proba"] >= 0.5).astype(int)
    ref["model_version"] = version
    ref.to_parquet(REFERENCE / "reference.parquet", index=False)

    if mlflow:
        with mlflow.start_run(run_name=f"{best}-selected"):
            mlflow.set_tag("selected", "true")
            mlflow.log_param("model", best)
            mlflow.log_param("model_version", version)
            mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
            mlflow.log_artifact(str(MODELS / "model_meta.json"))
            mlflow.log_artifact(str(MODELS / "model.joblib"))

    print(f"Saved models/model.joblib ({version}) and data/reference/reference.parquet")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
