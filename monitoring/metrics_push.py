"""Push drift results to the Prometheus Pushgateway (Phase 3).

The drift job is a short batch job, so Prometheus cannot scrape it. Instead each run
pushes its numbers to the Pushgateway, and Prometheus scrapes the gateway.

Metrics (job="cardioguard_drift")
  cardioguard_drift_status                 0 ok, 1 warning, 2 critical, -1 not enough data
  cardioguard_drift_check_status{check}    same scale, per check
  cardioguard_drift_share                  share of inputs flagged by Evidently
  cardioguard_drift_max_psi
  cardioguard_feature_drift_score{feature} Evidently score (Wasserstein / Jensen-Shannon)
  cardioguard_feature_drift_threshold{feature}
  cardioguard_feature_psi{feature}
  cardioguard_prediction_drift_score
  cardioguard_window_roc_auc               on human-labelled rows (absent if too few)
  cardioguard_window_mean_confidence
  cardioguard_window_rows / cardioguard_window_labelled_rows
  cardioguard_retrain_recommended          1 / 0
  cardioguard_drift_last_run_timestamp_seconds

Set PUSHGATEWAY_URL (default localhost:9091). Empty string disables pushing.
"""
from __future__ import annotations

import logging
import os
import time

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

log = logging.getLogger("cardioguard.metrics_push")
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "localhost:9091")
LEVEL = {"ok": 0, "warning": 1, "critical": 2, "insufficient_data": -1}


def push_drift_metrics(result: dict, model_version: str) -> bool:
    if not PUSHGATEWAY_URL:
        return False
    reg = CollectorRegistry()
    g = lambda name, doc, labels=(): Gauge(f"cardioguard_{name}", doc, list(labels), registry=reg)

    g("drift_status", "Overall drift status (0 ok, 1 warning, 2 critical, -1 not enough data)") \
        .set(LEVEL.get(result["status"], -1))
    check = g("drift_check_status", "Status of each drift check", ["check"])
    for c in result.get("checks", []):
        check.labels(c["check"]).set(LEVEL.get(c["status"], -1))

    values = {c["check"]: c["value"] for c in result.get("checks", [])}
    if values.get("feature_drift_share") is not None:
        g("drift_share", "Share of inputs with drift").set(values["feature_drift_share"])
    if values.get("psi_key_features") is not None:
        g("drift_max_psi", "Highest PSI among key inputs").set(values["psi_key_features"])

    score = g("feature_drift_score", "Evidently drift score per input", ["feature"])
    thr = g("feature_drift_threshold", "Drift threshold per input", ["feature"])
    psi = g("feature_psi", "PSI per key input", ["feature"])
    for f in result.get("features", []):
        if f.get("score") is not None:
            score.labels(f["feature"]).set(f["score"])
            thr.labels(f["feature"]).set(f["threshold"])
        if f.get("psi") is not None:
            psi.labels(f["feature"]).set(f["psi"])

    pred = result.get("prediction_drift") or {}
    if pred.get("score") is not None:
        g("prediction_drift_score", "Drift score of the predicted probability").set(pred["score"])
    if result.get("roc_auc") is not None:
        g("window_roc_auc", "ROC-AUC on human-labelled rows in the window").set(result["roc_auc"])
    if result.get("mean_confidence") is not None:
        g("window_mean_confidence", "Mean confidence in the window").set(result["mean_confidence"])
    g("window_rows", "Rows in the drift window").set(result.get("n_rows", 0))
    g("window_labelled_rows", "Human-labelled rows in the drift window").set(result.get("n_labelled", 0))
    g("retrain_recommended", "1 if retraining is recommended").set(1 if result.get("retrain_recommended") else 0)
    g("drift_last_run_timestamp_seconds", "When the last drift run finished").set(time.time())
    g("drift_model_info", "Model version checked by the last drift run", ["model_version"]) \
        .labels(model_version).set(1)

    try:
        # One group for the whole job: each push REPLACES the previous one. Grouping by
        # model version would leave an old model's "critical" in place after retraining.
        push_to_gateway(PUSHGATEWAY_URL, job="cardioguard_drift", registry=reg, timeout=3)
        return True
    except Exception as e:
        log.warning("Could not push drift metrics to %s: %s", PUSHGATEWAY_URL, e)
        return False
