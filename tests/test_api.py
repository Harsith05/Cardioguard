import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db import db_logger, predictions
from app.main import app

GOOD = {"age_years": 52, "gender": 2, "height": 172, "weight": 84, "ap_hi": 140,
        "ap_lo": 90, "cholesterol": 2, "gluc": 1, "smoke": 0, "alco": 0, "active": 1}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok"
    assert r["logging"]["sql"] is True


def test_predict_and_log(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["prediction"] in (0, 1)
    with db_logger.engine.connect() as conn:
        row = conn.execute(select(predictions).where(
            predictions.c.request_id == body["request_id"])).mappings().one()
    assert row["ap_hi"] == 140 and row["label"] is None


def test_higher_risk_profile_scores_higher(client):
    low = {**GOOD, "age_years": 30, "ap_hi": 115, "ap_lo": 75, "cholesterol": 1, "weight": 65}
    high = {**GOOD, "age_years": 62, "ap_hi": 170, "ap_lo": 100, "cholesterol": 3, "weight": 100}
    p_low = client.post("/predict", json=low).json()["probability"]
    p_high = client.post("/predict", json=high).json()["probability"]
    assert p_high > p_low


@pytest.mark.parametrize("patch", [
    {"ap_hi": 16020}, {"ap_lo": -70}, {"height": 55}, {"cholesterol": 4},
    {"ap_hi": 80, "ap_lo": 90},  # diastolic above systolic
])
def test_rejects_invalid_input(client, patch):
    assert client.post("/predict", json={**GOOD, **patch}).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"patients": [GOOD] * 25})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 25


def test_feedback_label(client):
    rid = client.post("/predict", json=GOOD).json()["request_id"]
    assert client.post(f"/feedback/{rid}", json={"label": 1}).status_code == 200
    assert client.post("/feedback/does-not-exist", json={"label": 1}).status_code == 404
    with db_logger.engine.connect() as conn:
        row = conn.execute(select(predictions).where(
            predictions.c.request_id == rid)).mappings().one()
    assert row["label"] == 1 and row["label_source"] == "human"


def test_latency_under_100ms(client):
    client.post("/predict", json=GOOD)  # warm-up
    t = []
    for _ in range(50):
        t0 = time.perf_counter()
        client.post("/predict", json=GOOD)
        t.append((time.perf_counter() - t0) * 1000)
    t.sort()
    assert t[int(0.95 * len(t))] < 100, f"p95 {t[int(0.95 * len(t))]:.1f} ms"


def test_explain_adds_up_to_prediction(client):
    import math
    r = client.post("/predict/explain", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    e = body["explanation"]
    assert {g["group"] for g in e["groups"]} >= {"blood_pressure", "age", "cholesterol"}
    total = e["base_value"] + sum(f["shap"] for f in e["features"])
    p = 1 / (1 + math.exp(-total)) if e["output_space"] == "log_odds" else total
    assert abs(p - body["probability"]) < 0.01           # SHAP is additive
    # groups are exact sums of their member features
    feats = {f["feature"]: f["shap"] for f in e["features"]}
    bp = next(g for g in e["groups"] if g["group"] == "blood_pressure")
    assert abs(bp["shap"] - sum(feats[m] for m in bp["members"])) < 1e-3


def test_explain_direction_makes_sense(client):
    high_bp = {**GOOD, "ap_hi": 180, "ap_lo": 110}
    e = client.post("/predict/explain", json=high_bp).json()["explanation"]
    bp = next(g for g in e["groups"] if g["group"] == "blood_pressure")
    assert bp["shap"] > 0                                # high blood pressure raises risk


def test_global_importance(client):
    g = client.get("/explain/global").json()
    assert g["groups"][0]["group"] == "blood_pressure"   # strongest factor in this data
    assert len(g["features"]) == 14


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Why this result" in r.text


def test_monitoring_flow(client):
    """Predictions -> drift run -> latest/history/report endpoints."""
    import pandas as pd
    from common.features import INPUT_FEATURES
    rows = pd.read_parquet("data/processed/test.parquet").sample(300, random_state=0)[INPUT_FEATURES]
    patients = rows.astype(float).to_dict("records")
    for p in patients:
        for k in ("gender", "cholesterol", "gluc", "smoke", "alco", "active"):
            p[k] = int(p[k])
    assert client.post("/predict/batch", json={"patients": patients}).status_code == 200
    r = client.post("/monitoring/drift/run")
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    latest = client.get("/monitoring/drift/latest").json()
    assert latest["run"]["id"] == run_id
    assert latest["run"]["status"] in {"ok", "warning", "critical"}
    assert len(latest["features"]) >= 11
    assert client.get("/monitoring/drift/history").json()["runs"][0]["id"] == run_id
    assert client.get(f"/monitoring/drift/runs/{run_id}/report").status_code == 200
    assert client.get("/monitoring").status_code == 200


def test_metrics_endpoint(client):
    client.post("/predict", json=GOOD)
    text = client.get("/metrics").text
    for name in ["http_requests_total", "http_request_duration_seconds_bucket",
                 "cardioguard_predictions_total", "cardioguard_prediction_probability_bucket",
                 "cardioguard_model_inference_seconds_bucket", "cardioguard_input_value_sum",
                 "cardioguard_reference_feature_mean", "cardioguard_model_info",
                 "cardioguard_db_logging_up"]:
        assert name in text, name
    assert 'handler="/predict"' in text


def test_chaos_errors_are_counted_as_5xx(client):
    from app import chaos
    client.post("/debug/chaos", params={"error_rate": 1.0})
    try:
        assert client.post("/predict", json=GOOD).status_code == 500
    finally:
        client.post("/debug/chaos", params={"error_rate": 0})
    text = client.get("/metrics").text
    assert any(l.startswith("http_requests_total") and 'handler="/predict"' in l and 'status="5xx"' in l
               for l in text.splitlines())
    assert client.post("/predict", json=GOOD).status_code == 200
    assert chaos.state["error_rate"] == 0


def test_drift_metrics_push_payload():
    """Build the Pushgateway payload without sending it."""
    from unittest import mock
    from monitoring import metrics_push
    result = {"status": "critical", "n_rows": 1000, "n_labelled": 200, "retrain_recommended": True,
              "checks": [{"check": "feature_drift_share", "value": 0.18, "status": "ok"},
                         {"check": "psi_key_features", "value": 1.9, "status": "critical"}],
              "features": [{"feature": "ap_hi", "score": 1.17, "threshold": 0.1, "psi": 1.93}],
              "prediction_drift": {"score": 1.0}, "roc_auc": 0.73, "mean_confidence": 0.8}
    with mock.patch.object(metrics_push, "PUSHGATEWAY_URL", "gw:9091"), \
         mock.patch.object(metrics_push, "push_to_gateway") as push:
        assert metrics_push.push_drift_metrics(result, "v1")
    from prometheus_client import generate_latest
    body = generate_latest(push.call_args.kwargs["registry"]).decode()
    assert "cardioguard_drift_status 2.0" in body
    assert 'cardioguard_feature_psi{feature="ap_hi"} 1.93' in body
    assert "cardioguard_retrain_recommended 1.0" in body
    assert 'cardioguard_drift_model_info{model_version="v1"} 1.0' in body
    assert "grouping_key" not in push.call_args.kwargs     # one group: new pushes replace old ones


def test_rechecking_same_window_does_not_extend_streak():
    """Pressing 'Run drift check' twice on the same data is not two windows in a row."""
    import pandas as pd
    from monitoring.drift_job import distinct_windows
    t = pd.Timestamp("2026-10-27 10:00", tz="UTC")
    current = pd.DataFrame({"created_at": [t - pd.Timedelta(minutes=5), t]})
    previous = [  # newest first
        {"id": 3, "window_end": t},                               # same data as now
        {"id": 2, "window_end": t},                               # same data again
        {"id": 1, "window_end": t - pd.Timedelta(hours=1)},       # a real earlier window
    ]
    assert [r["id"] for r in distinct_windows(previous, current)] == [1]


def test_model_info(client):
    info = client.get("/model-info").json()
    assert info["model_version"].startswith("v")
    assert info["test"]["roc_auc"] > 0.75


def test_streak_ignores_runs_of_a_previous_model():
    """After a promotion, critical runs of the OLD champion must not count towards retraining."""
    import inspect
    from monitoring import drift_job
    src = inspect.getsource(drift_job.run)
    assert 'r.get("model_version") == version' in src
