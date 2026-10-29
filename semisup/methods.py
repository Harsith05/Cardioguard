"""Semi-supervised learning methods (Phase 5).

Shared by the production pseudo-label job and the label-fraction experiment.

  self_training()    scikit-learn SelfTrainingClassifier around the same LightGBM pipeline
                     as the production model. Each round it trains on the labelled rows,
                     labels the unlabelled rows it is >= 90% sure about, and retrains.
  label_spreading()  scikit-learn LabelSpreading on a k-nearest-neighbour graph of
                     scaled features: labels "flow" from labelled patients to similar ones.
  select()           the agreement filter + guardrails: keep a pseudo-label only if both
                     methods agree and both are confident, keep the reference class ratio,
                     and cap how many are added.

Analogy: two students mark a pile of practice questions. A mark is kept only when both
are confident and give the same answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier

from common.features import (BINARY_INPUTS, CATEGORICAL_INPUTS, INPUT_FEATURES,
                             MODEL_NUMERIC, FeatureEngineer)
from common.frames import to_frame

UNLABELLED = -1


def base_model(seed: int = 42) -> Pipeline:
    """Same LightGBM pipeline as training/train.py, but it also accepts plain arrays."""
    prep = ColumnTransformer([
        ("num", "passthrough", MODEL_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_INPUTS),
        ("bin", "passthrough", BINARY_INPUTS),
    ])
    model = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.03, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    return Pipeline([("frame", FunctionTransformer(to_frame)), ("features", FeatureEngineer()),
                     ("prep", prep), ("model", model)])


def spreading_features() -> Pipeline:
    """Scaled features for the k-NN graph: distances only make sense on comparable scales."""
    prep = ColumnTransformer([
        ("num", StandardScaler(), MODEL_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_INPUTS),
        ("bin", "passthrough", BINARY_INPUTS),
    ])
    return Pipeline([("frame", FunctionTransformer(to_frame)), ("features", FeatureEngineer()), ("prep", prep)])


def self_training(X_l: pd.DataFrame, y_l: np.ndarray, X_u: pd.DataFrame, threshold: float = 0.9,
                  max_iter: int = 10, seed: int = 42):
    """Returns (P(disease) for X_u from the final self-trained model, fitted classifier)."""
    X = pd.concat([X_l[INPUT_FEATURES], X_u[INPUT_FEATURES]], ignore_index=True)
    y = np.concatenate([np.asarray(y_l, int), np.full(len(X_u), UNLABELLED)])
    st = SelfTrainingClassifier(base_model(seed), threshold=threshold, criterion="threshold",
                                max_iter=max_iter)
    st.fit(X.to_numpy(float), y)
    return st.predict_proba(X_u[INPUT_FEATURES].to_numpy(float))[:, 1], st


def label_spreading(X_l: pd.DataFrame, y_l: np.ndarray, X_u: pd.DataFrame, n_neighbors: int = 10,
                    alpha: float = 0.2, max_iter: int = 50):
    """Returns P(disease) for X_u from the label distribution after spreading."""
    X = pd.concat([X_l[INPUT_FEATURES], X_u[INPUT_FEATURES]], ignore_index=True)
    y = np.concatenate([np.asarray(y_l, int), np.full(len(X_u), UNLABELLED)])
    Z = spreading_features().fit_transform(X)
    ls = LabelSpreading(kernel="knn", n_neighbors=n_neighbors, alpha=alpha, max_iter=max_iter)
    ls.fit(Z, y)
    dist = ls.label_distributions_[len(X_l):]
    classes = list(ls.classes_)
    return dist[:, classes.index(1)]


def select(st_p: np.ndarray, ls_p: np.ndarray, st_threshold: float = 0.9, ls_threshold: float = 0.7,
           positive_rate: float = 0.5, max_accept: int | None = None, require_agreement: bool = True,
           use_label_spreading: bool = True) -> dict:
    """Agreement filter + class balance + cap. Returns accepted indices, labels and stats."""
    st_p, ls_p = np.asarray(st_p, float), np.asarray(ls_p, float)
    st_label, ls_label = (st_p >= 0.5).astype(int), (ls_p >= 0.5).astype(int)
    st_conf, ls_conf = np.maximum(st_p, 1 - st_p), np.maximum(ls_p, 1 - ls_p)

    ok = st_conf >= st_threshold
    if use_label_spreading:
        ok &= ls_conf >= ls_threshold
        if require_agreement:
            ok &= st_label == ls_label
    conf = (st_conf + ls_conf) / 2 if use_label_spreading else st_conf

    pos = np.flatnonzero(ok & (st_label == 1))
    neg = np.flatnonzero(ok & (st_label == 0))
    pos = pos[np.argsort(-conf[pos])]       # most confident first
    neg = neg[np.argsort(-conf[neg])]
    # Class balance: largest total that keeps the reference ratio with the candidates available
    limits = [len(pos) / positive_rate if positive_rate > 0 else np.inf,
              len(neg) / (1 - positive_rate) if positive_rate < 1 else np.inf]
    if max_accept is not None:
        limits.append(max_accept)
    total = int(min(limits))
    n_pos = int(round(total * positive_rate))
    idx = np.concatenate([pos[:n_pos], neg[:total - n_pos]]).astype(int)
    return {
        "index": idx,
        "label": st_label[idx],
        "confidence": conf[idx],
        "n_unlabelled": int(len(st_p)),
        "n_confident_self_training": int((st_conf >= st_threshold).sum()),
        "n_confident_spreading": int((ls_conf >= ls_threshold).sum()),
        "n_agree_confident": int(ok.sum()),
        "n_accepted": int(len(idx)),
        "disagreement_rate": float((st_label != ls_label).mean()) if len(st_p) else 0.0,
        "mean_confidence": float(st_conf.mean()) if len(st_p) else 0.0,
    }
