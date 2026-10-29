"""Prometheus metrics for the prediction service (Phase 3).

Exposed at /metrics and scraped by Prometheus every 15 s.

HTTP (from prometheus-fastapi-instrumentator)
  http_requests_total{handler, method, status}          status grouped: 2xx / 4xx / 5xx
  http_request_duration_seconds{handler, method}        histogram -> p50 / p95 / p99 latency

Model and data (defined here, prefix cardioguard_)
  predictions_total{model_version, risk_level, endpoint}
  prediction_probability{model_version}                 histogram -> score distribution
  model_inference_seconds                               model time only (no HTTP / logging)
  batch_size                                            patients per request
  shap_seconds                                          time to explain one patient
  feedback_total{label}                                 confirmed diagnoses received
  input_value{feature}  (summary)                       live mean of key inputs
  reference_feature_mean{feature}                       training mean, for comparison
  model_info{model_version, model_name, role}           1 per loaded model (champion / candidate)
  model_errors_total{model_version}                     failures inside a model (canary health)
  canary_fraction, deployment_status                    Phase 6 rollout state
  db_logging_up{store}                                  1 if Postgres / MongoDB logging works
  log_failures_total{store}                             failed log writes
"""
from __future__ import annotations

import logging

from prometheus_client import Counter, Gauge, Histogram, Summary
from prometheus_fastapi_instrumentator import Instrumentator

log = logging.getLogger("cardioguard.metrics")
NS = "cardioguard"
LIVE_FEATURES = ["ap_hi", "ap_lo", "age_years", "bmi"]

PREDICTIONS = Counter(f"{NS}_predictions_total", "Predictions served",
                      ["model_version", "risk_level", "endpoint"])
PROBABILITY = Histogram(f"{NS}_prediction_probability", "Predicted probability of disease",
                        ["model_version"], buckets=[i / 10 for i in range(1, 11)])
INFERENCE = Histogram(f"{NS}_model_inference_seconds", "Model predict_proba time per request",
                      ["model_version"], buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1))
BATCH = Histogram(f"{NS}_batch_size", "Patients per prediction request",
                  buckets=(1, 2, 5, 10, 25, 50, 100, 250, 500, 1000))
SHAP = Histogram(f"{NS}_shap_seconds", "SHAP explanation time per patient",
                 buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1))
FEEDBACK = Counter(f"{NS}_feedback_total", "Confirmed diagnoses received", ["label"])
INPUT = Summary(f"{NS}_input_value", "Values of key inputs in production", ["feature"])
REFERENCE_MEAN = Gauge(f"{NS}_reference_feature_mean", "Mean of the input in the training data", ["feature"])
MODEL_INFO = Gauge(f"{NS}_model_info", "Loaded models (role = champion / candidate)",
                   ["model_version", "model_name", "role"])
MODEL_ERRORS = Counter(f"{NS}_model_errors_total", "Prediction failures inside the model", ["model_version"])
CANARY_FRACTION = Gauge(f"{NS}_canary_fraction", "Share of requests sent to the candidate model")
DEPLOYMENT_STATUS = Gauge(f"{NS}_deployment_status",
                          "0 stable, 1 canary, 2 awaiting approval, 3 approved (canary starting)")
DEPLOYMENT_LEVEL = {"stable": 0, "canary": 1, "awaiting_approval": 2, "approved": 3}
DB_UP = Gauge(f"{NS}_db_logging_up", "1 if prediction logging to this store works", ["store"])
LOG_FAILURES = Counter(f"{NS}_log_failures_total", "Failed prediction log writes", ["store"])

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5)


def setup(app) -> None:
    """Add HTTP metrics and the /metrics endpoint."""
    Instrumentator(
        should_group_status_codes=True,      # 2xx / 4xx / 5xx
        should_ignore_untemplated=True,      # ignore 404 scans of random paths
        excluded_handlers=["/metrics", "/favicon.ico"],
    ).instrument(app, latency_lowr_buckets=LATENCY_BUCKETS).expose(
        app, endpoint="/metrics", include_in_schema=False)


def record_predictions(records: list[dict], endpoint: str, inference_seconds: float,
                       observe_batch: bool = True) -> None:
    INFERENCE.labels(records[0]["model_version"]).observe(inference_seconds)
    if observe_batch:
        BATCH.observe(len(records))
    for r in records:
        p = r["probability"]
        level = "high" if p >= 0.7 else "moderate" if p >= 0.4 else "low"
        PREDICTIONS.labels(r["model_version"], level, endpoint).inc()
        PROBABILITY.labels(r["model_version"]).observe(p)
        INPUT.labels("ap_hi").observe(r["ap_hi"])
        INPUT.labels("ap_lo").observe(r["ap_lo"])
        INPUT.labels("age_years").observe(r["age_years"])
        INPUT.labels("bmi").observe(r["weight"] / (r["height"] / 100) ** 2)


def set_models(champion, candidate, canary_fraction: float, status: str) -> None:
    MODEL_INFO.clear()
    for role, lm in (("champion", champion), ("candidate", candidate)):
        if lm is not None:
            MODEL_INFO.labels(lm.version, lm.meta.get("model_name", "unknown"), role).set(1)
    CANARY_FRACTION.set(canary_fraction if candidate is not None else 0.0)
    DEPLOYMENT_STATUS.set(DEPLOYMENT_LEVEL.get(status, 0))


def set_reference_means(reference_path) -> None:
    try:
        import pandas as pd
        ref = pd.read_parquet(reference_path, columns=["ap_hi", "ap_lo", "age_years", "height", "weight"])
        ref["bmi"] = ref["weight"] / (ref["height"] / 100) ** 2
        for f in LIVE_FEATURES:
            REFERENCE_MEAN.labels(f).set(float(ref[f].mean()))
    except Exception as e:
        log.warning("Reference means unavailable: %s", e)
