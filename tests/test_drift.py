"""Phase 2 tests: drift detection on simulated windows (no database needed)."""
import joblib
import numpy as np
import pandas as pd
import pytest

from common.features import INPUT_FEATURES
from monitoring.drift import analyse, auc_with_ci, load_config, psi
from simulate.traffic import apply_scenario

REF_AUC = 0.8024


@pytest.fixture(scope="module")
def setup():
    cfg = load_config()
    model = joblib.load("models/model.joblib")
    ref = pd.read_parquet("data/reference/reference.parquet").sample(8000, random_state=0)
    test = pd.read_parquet("data/processed/test.parquet")
    return cfg, model, ref, test


def window(setup, scenario, seed=0, n=1000, label_rate=0.3):
    cfg, model, ref, test = setup
    rng = np.random.default_rng(seed)
    df = apply_scenario(test.sample(n, random_state=seed), scenario, rng)
    df["prediction_proba"] = model.predict_proba(df[INPUT_FEATURES])[:, 1]
    df["label"] = np.where(rng.random(len(df)) < label_rate, df["cardio"], np.nan)
    df["label_source"] = np.where(df["label"].notna(), "human", None)
    return df


def check(result, name):
    return next(c for c in result["checks"] if c["check"] == name)


def test_normal_traffic_is_ok(setup):
    cfg, _, ref, _ = setup
    r = analyse(ref, window(setup, "normal", seed=3), cfg, reference_auc=REF_AUC)
    assert r["status"] == "ok", r["reasons"]


def test_bp_shift_is_critical_and_names_the_feature(setup):
    cfg, _, ref, _ = setup
    r = analyse(ref, window(setup, "bp_shift"), cfg, reference_auc=REF_AUC)
    assert r["status"] == "critical"
    assert check(r, "psi_key_features")["detail"].endswith("ap_hi")
    drifted = {f["feature"] for f in r["features"] if f["detected"]}
    assert {"ap_hi", "ap_lo"} <= drifted


def test_cholesterol_shift_caught_by_categorical_psi(setup):
    cfg, _, ref, _ = setup
    r = analyse(ref, window(setup, "cholesterol"), cfg, reference_auc=REF_AUC)
    psi_chol = next(f["psi"] for f in r["features"] if f["feature"] == "cholesterol")
    assert psi_chol >= cfg["psi"]["critical"]


def test_concept_drift_only_visible_in_accuracy(setup):
    cfg, _, ref, _ = setup
    r = analyse(ref, window(setup, "concept"), cfg, reference_auc=REF_AUC)
    assert check(r, "feature_drift_share")["status"] == "ok"      # inputs look normal
    assert check(r, "auc_drop")["status"] == "critical"           # but accuracy collapsed


def test_prediction_drift_needs_two_windows_for_critical(setup):
    cfg, _, ref, _ = setup
    w = window(setup, "bp_shift")
    first = analyse(ref, w, cfg, previous_prediction_drift=[False])
    second = analyse(ref, w, cfg, previous_prediction_drift=[True])
    assert check(first, "prediction_drift")["status"] == "warning"
    assert check(second, "prediction_drift")["status"] == "critical"


def test_small_window_is_skipped(setup):
    cfg, _, ref, _ = setup
    r = analyse(ref, window(setup, "normal", n=50), cfg)
    assert r["status"] == "insufficient_data"


def test_auc_noise_does_not_alarm(setup):
    """A low AUC from few labels must not count unless it is significant."""
    y = np.array([0, 1] * 60)
    p = np.clip(y * 0.3 + np.random.default_rng(0).random(len(y)) * 0.7, 0, 1)
    auc, lo, hi = auc_with_ci(y, p, 300, 0.99)
    assert lo <= auc <= hi


def test_psi_basics():
    a = pd.Series(np.random.default_rng(0).normal(0, 1, 5000))
    assert psi(a, a) < 0.01
    assert psi(a, a + 1) > 0.25
    c = pd.Series([1] * 80 + [2] * 15 + [3] * 5)
    assert psi(c, c, categorical=True) < 0.01
    assert psi(c, pd.Series([1] * 30 + [2] * 35 + [3] * 35), categorical=True) > 0.25
