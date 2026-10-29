"""Run one drift check: load data -> analyse -> save report -> alert.

Usage:
    python -m monitoring.drift_job                 # latest 1,000 predictions (thresholds.yaml)
    python -m monitoring.drift_job --window 500
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from monitoring.alerts import build_alert, send_alert, trigger_retraining
from monitoring.drift import analyse, load_config
from monitoring.store import Store

log = logging.getLogger("cardioguard.drift_job")
REPORT_DIR = Path(os.getenv("DRIFT_REPORT_DIR", "reports/drift"))
META = Path("models/model_meta.json")
REFERENCE_PARQUET = "data/reference/reference.parquet"


def _same_time(a, b) -> bool:
    if a is None or b is None:
        return False
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    if (a.tzinfo is None) != (b.tzinfo is None):  # SQLite drops the timezone
        a, b = a.tz_localize(None), b.tz_localize(None)
    return abs((a - b).total_seconds()) < 1


def distinct_windows(previous: list[dict], current: pd.DataFrame) -> list[dict]:
    """Earlier runs that looked at DIFFERENT data, newest first.

    Re-checking the same window (e.g. pressing "Run drift check" twice with no new
    traffic) must not count as another window "in a row" for the critical/retrain rules.
    """
    seen = []
    if len(current):
        seen.append(current["created_at"].max())
    out = []
    for r in previous:
        if any(_same_time(r.get("window_end"), s) for s in seen):
            continue
        seen.append(r.get("window_end"))
        out.append(r)
    return out


def run(window: int | None = None, store: Store | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    store = store or Store()
    # Phase 6: monitor the LIVE champion (it changes after a promotion), not the file on disk
    from mlops import deployment
    version = deployment.champion_version(store.engine)
    reference_auc = deployment.champion_auc(store.engine)

    reference = store.load_reference(version, cfg["reference_sample"], REFERENCE_PARQUET)
    current = store.load_window(window or cfg["window_size"], model_version=version)

    # only earlier runs of THIS model count towards "N windows in a row" (a new champion starts fresh)
    previous = distinct_windows([r for r in store.previous_runs(20) if r.get("model_version") == version], current)
    prev_pred = [bool(r["prediction_drift_detected"]) for r in previous]

    if current.empty:
        result = {"n_rows": 0, "status": "insufficient_data", "reasons": ["No predictions logged yet."],
                  "checks": [], "features": []}
    else:
        result = analyse(reference, current, cfg, reference_auc=reference_auc,
                         previous_prediction_drift=prev_pred)

    # retraining: critical for N runs in a row (this one included)
    k = cfg["retraining"]["consecutive_critical"]
    recent = [result["status"]] + [r["status"] for r in previous]
    result["retrain_recommended"] = len(recent) >= k and all(s == "critical" for s in recent[:k])

    report_path = None
    snapshot = result.pop("snapshot", None)
    run_id = store.save_run(result, version, current, None)
    if snapshot is not None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = str(REPORT_DIR / f"run_{run_id}.html")
        snapshot.save_html(report_path)
        with store.engine.begin() as conn:
            from monitoring.store import drift_runs
            conn.execute(drift_runs.update().where(drift_runs.c.id == run_id).values(report_path=report_path))

    if result["status"] in ("warning", "critical"):
        send_alert(build_alert(run_id, result, version), store)
    if result["retrain_recommended"]:
        trigger_retraining(run_id, result["reasons"])

    from monitoring.metrics_push import push_drift_metrics
    push_drift_metrics(result, version)

    result.update(run_id=run_id, report_path=report_path, model_version=version)
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=None)
    args = ap.parse_args()
    r = run(args.window)
    print(f"\nRun {r['run_id']}: {r['status'].upper()}  ({r['n_rows']} rows, {r.get('n_labelled', 0)} labelled)")
    for c in r.get("checks", []):
        print(f"  {c['status']:17s} {c['check']:20s} {c['detail']}")
    if r.get("retrain_recommended"):
        print("  -> retraining recommended")
    if r.get("report_path"):
        print(f"Report: {r['report_path']}")


if __name__ == "__main__":
    main()
