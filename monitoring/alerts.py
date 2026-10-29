"""Alerts and the retraining hook.

Warning / critical runs are logged, stored in MongoDB `alerts`, and optionally posted
to a webhook (Slack, Discord, Teams, ...) if ALERT_WEBHOOK_URL is set.

`trigger_retraining` starts the Phase 6 `cardio_self_healing` DAG through the Airflow 3
REST API when AIRFLOW_URL is set (Docker: http://airflow:8080); otherwise it only logs.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import httpx

log = logging.getLogger("cardioguard.alerts")

WEBHOOK = os.getenv("ALERT_WEBHOOK_URL", "")
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "").rstrip("/")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")
DAG_ID = os.getenv("RETRAIN_DAG_ID", "cardio_self_healing")


def build_alert(run_id: int, result: dict, model_version: str) -> dict:
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc),
        "severity": result["status"],
        "model_version": model_version,
        "reasons": result.get("reasons", []),
        "retrain_recommended": result.get("retrain_recommended", False),
    }


def send_alert(alert: dict, store=None):
    msg = (f"[CardioGuard] {alert['severity'].upper()} drift on model {alert['model_version']} "
           f"(run {alert['run_id']}): " + "; ".join(alert["reasons"]))
    (log.error if alert["severity"] == "critical" else log.warning)(msg)
    if store is not None:
        store.save_alert(alert)
    if WEBHOOK:
        try:
            httpx.post(WEBHOOK, json={"text": msg, "content": msg}, timeout=5)
        except Exception as e:
            log.error("Webhook failed: %s", e)


def _airflow_token(client: httpx.Client) -> str:
    """Airflow 3 uses JWT tokens: exchange user/password for one."""
    r = client.post(f"{AIRFLOW_URL}/auth/token", json={"username": AIRFLOW_USER, "password": AIRFLOW_PASSWORD})
    r.raise_for_status()
    return r.json()["access_token"]


def trigger_retraining(run_id: int, reasons: list[str]) -> bool:
    """Start the cardio_self_healing DAG. The DAG itself decides (cool-down, diagnosis...)."""
    if not AIRFLOW_URL:
        log.warning("Retraining recommended (run %s) - AIRFLOW_URL not set, run `python -m mlops.pipeline` by hand.",
                    run_id)
        return False
    try:
        with httpx.Client(timeout=10) as c:
            token = _airflow_token(c)
            r = c.post(f"{AIRFLOW_URL}/api/v2/dags/{DAG_ID}/dagRuns",
                       headers={"Authorization": f"Bearer {token}"},
                       json={"logical_date": None, "conf": {"drift_run_id": run_id, "reasons": reasons}})
            if r.status_code == 409:
                log.warning("Retraining DAG is already running: not starting another")
                return False
            r.raise_for_status()
        log.warning("Retraining DAG triggered: %s", r.json().get("dag_run_id"))
        return True
    except Exception as e:
        log.error("Could not trigger retraining: %s", e)
        return False
