"""Generate the Grafana dashboard JSON (Phase 3).

    python infra/grafana/build_dashboard.py      -> infra/grafana/dashboards/cardioguard.json

Writing panels in Python keeps the ~1,500-line JSON consistent and easy to change.
Rows:
  1. Service health   API up, requests/s, error rate, p95 latency, logging
  2. Traffic          requests by endpoint, predictions by risk level
  3. Reliability      error rate and latency over time (alert limits drawn as lines)
  4. Model health     score distribution, mean risk, model/SHAP time, version, labels
  5. Live inputs      live mean of key inputs vs their training mean
  6. Drift checks     Evidently results (Pushgateway) and drift history (Postgres)
  7. Semi-supervised  pseudo-label runs: status, acceptance, disagreement, history
  8. Self-healing     rollout state, canary share, pipeline outcome, per-model traffic, events
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "dashboards" / "cardioguard.json"
PROM = {"type": "prometheus", "uid": "prometheus"}
PG = {"type": "grafana-postgresql-datasource", "uid": "cardio-postgres"}
API = 'job="cardioguard-api"'
PRED = f'{API}, handler=~"/predict.*"'

GREEN, AMBER, RED, GREY, TEAL = "#2E7D5B", "#B07A12", "#B23A3A", "#8A97A3", "#0F5E66"

_panels: list[dict] = []
_id = 0
_y = 0


def _next_id() -> int:
    global _id
    _id += 1
    return _id


def row(title: str):
    global _y
    _panels.append({"type": "row", "title": title, "collapsed": False, "id": _next_id(),
                    "gridPos": {"h": 1, "w": 24, "x": 0, "y": _y}, "panels": []})
    _y += 1


def place(panel: dict, x: int, w: int, h: int, new_line: bool = False) -> dict:
    global _y
    panel["gridPos"] = {"h": h, "w": w, "x": x, "y": _y}
    panel["id"] = _next_id()
    _panels.append(panel)
    if new_line:
        _y += h
    return panel


def prom(expr: str, legend: str = "", ref: str = "A", instant: bool = False, fmt: str | None = None) -> dict:
    t = {"datasource": PROM, "expr": expr, "legendFormat": legend or "__auto", "refId": ref,
         "range": not instant, "instant": instant}
    if fmt:
        t["format"] = fmt
    return t


def sql(query: str, fmt: str = "table", ref: str = "A") -> dict:
    return {"datasource": PG, "editorMode": "code", "format": fmt, "rawQuery": True,
            "rawSql": query, "refId": ref}


def steps(*pairs) -> dict:
    return {"mode": "absolute", "steps": [{"color": c, "value": v} for c, v in pairs]}


def value_map(mapping: dict) -> list[dict]:
    return [{"type": "value", "options": {str(k): {"text": t, "color": c, "index": i}
                                          for i, (k, (t, c)) in enumerate(mapping.items())}}]


def timeseries(title, targets, unit="short", desc="", stack=False, limit=None, ds=PROM,
               overrides=None, min_=None, max_=None) -> dict:
    custom = {"drawStyle": "line", "lineWidth": 2, "fillOpacity": 12 if stack else 6,
              "showPoints": "never", "spanNulls": True, "gradientMode": "none",
              "stacking": {"mode": "normal" if stack else "none", "group": "A"},
              "thresholdsStyle": {"mode": "dashed" if limit is not None else "off"}}
    thr = steps((GREEN, None), (RED, limit)) if limit is not None else steps((GREEN, None))
    defaults = {"unit": unit, "custom": custom, "color": {"mode": "palette-classic"}, "thresholds": thr}
    if min_ is not None:
        defaults["min"] = min_
    if max_ is not None:
        defaults["max"] = max_
    return {"type": "timeseries", "title": title, "description": desc, "datasource": ds,
            "targets": targets,
            "fieldConfig": {"defaults": defaults, "overrides": overrides or []},
            "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                        "tooltip": {"mode": "multi", "sort": "desc"}}}


def stat(title, target, unit="short", thresholds=None, mappings=None, desc="", decimals=None,
         color_mode="value", fields="", show_values=False) -> dict:
    defaults = {"unit": unit, "color": {"mode": "thresholds"},
                "thresholds": thresholds or steps((TEAL, None)), "mappings": mappings or []}
    if decimals is not None:
        defaults["decimals"] = decimals
    return {"type": "stat", "title": title, "description": desc, "datasource": target["datasource"],
            "targets": [target],
            "fieldConfig": {"defaults": defaults, "overrides": []},
            "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": fields, "values": show_values},
                        "colorMode": color_mode, "graphMode": "none", "textMode": "value",
                        "justifyMode": "center", "orientation": "auto", "wideLayout": True,
                        "showPercentChange": False}}


def bargauge(title, target, unit="short", thresholds=None, desc="", orientation="horizontal",
             min_=0, max_=None, display="basic") -> dict:
    defaults = {"unit": unit, "min": min_, "color": {"mode": "thresholds"},
                "thresholds": thresholds or steps((TEAL, None))}
    if max_ is not None:
        defaults["max"] = max_
    return {"type": "bargauge", "title": title, "description": desc, "datasource": PROM,
            "targets": [target],
            "fieldConfig": {"defaults": defaults, "overrides": []},
            "options": {"orientation": orientation, "displayMode": display, "valueMode": "color",
                        "namePlacement": "auto", "showUnfilled": True, "sizing": "auto",
                        "minVizHeight": 16, "maxVizHeight": 300,
                        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}}


def color_by_name(name: str, color: str) -> dict:
    return {"matcher": {"id": "byName", "options": name},
            "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}}]}


def dashed(name: str) -> dict:
    return {"matcher": {"id": "byName", "options": name},
            "properties": [{"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [10, 10]}},
                           {"id": "color", "value": {"mode": "fixed", "fixedColor": GREY}}]}


# ------------------------------------------------------------------ 1. Service health
row("Service health")
place(stat("Prediction API", prom(f"up{{{API}}}", instant=True),
           mappings=value_map({1: ("Up", GREEN), 0: ("Down", RED)}),
           thresholds=steps((RED, None), (GREEN, 1)), color_mode="background",
           desc="Can Prometheus reach /metrics? Alert: PredictionAPIDown after 1 minute."), 0, 4, 4)
place(stat("Requests / second", prom(f"sum(rate(http_requests_total{{{API}}}[1m]))", instant=True),
           unit="reqps", decimals=1, desc="All endpoints, last minute."), 4, 5, 4)
place(stat("Prediction error rate (5 min)", prom("cardioguard:prediction_error_ratio:rate5m", instant=True),
           unit="percentunit", decimals=1, thresholds=steps((GREEN, None), (AMBER, 0.01), (RED, 0.05)),
           desc="Share of /predict* requests returning 5xx. Alert above 5% for 5 minutes."), 9, 5, 4)
place(stat("p95 prediction latency", prom("cardioguard:prediction_latency_seconds:p95_5m", instant=True),
           unit="s", decimals=0, thresholds=steps((GREEN, None), (AMBER, 0.25), (RED, 0.5)),
           desc="95% of prediction requests finish faster than this. Alert above 500 ms for 5 minutes."), 14, 5, 4)
place(stat("Saving predictions", prom('cardioguard_db_logging_up{store="postgres"}', instant=True),
           mappings=value_map({1: ("Yes", GREEN), 0: ("No", RED)}),
           thresholds=steps((RED, None), (GREEN, 1)), color_mode="background",
           desc="Is the API writing predictions to PostgreSQL? Drift checks need this."), 19, 5, 4, new_line=True)

# ------------------------------------------------------------------ 2. Traffic
row("Traffic")
place(timeseries("Requests per second by endpoint",
                 [prom("cardioguard:http_requests:rate1m", "{{handler}}")], unit="reqps",
                 desc="From the recording rule cardioguard:http_requests:rate1m."), 0, 12, 8)
place(timeseries("Predictions per minute by risk level",
                 [prom("sum by (risk_level) (rate(cardioguard_predictions_total[5m])) * 60", "{{risk_level}}")],
                 unit="short", stack=True,
                 overrides=[color_by_name("low", GREEN), color_by_name("moderate", AMBER), color_by_name("high", RED)],
                 desc="Low < 40%, moderate 40-70%, high > 70% predicted risk. A shift in the mix is an early sign of drift."),
      12, 12, 8, new_line=True)

# ------------------------------------------------------------------ 3. Reliability
row("Reliability")
place(timeseries("Prediction error rate",
                 [prom("cardioguard:prediction_error_ratio:rate5m", "Server errors (5xx)", "A"),
                  prom(f'sum(rate(http_requests_total{{{PRED}, status="4xx"}}[5m])) '
                       f'/ sum(rate(http_requests_total{{{PRED}}}[5m]))', "Rejected input (4xx)", "B")],
                 unit="percentunit", limit=0.05, min_=0,
                 overrides=[color_by_name("Server errors (5xx)", RED), color_by_name("Rejected input (4xx)", AMBER)],
                 desc="Dashed line: 5% alert limit (applies to 5xx). 4xx are invalid inputs rejected by validation."),
      0, 12, 8)
place(timeseries("Prediction latency",
                 [prom(f"histogram_quantile({q}, sum by (le) (rate(http_request_duration_seconds_bucket{{{PRED}}}[5m])))",
                       f"p{int(q * 100)}", ref) for q, ref in [(0.5, "A"), (0.95, "B"), (0.99, "C")]],
                 unit="s", limit=0.5, min_=0,
                 desc="Whole request time, including validation. Dashed line: 500 ms alert limit for p95."),
      12, 12, 8, new_line=True)

# ------------------------------------------------------------------ 4. Model health
row("Model health")
dist = bargauge("Predicted risk distribution (last 15 min)",
                prom('sum by (le) (increase(cardioguard_prediction_probability_bucket{le!="+Inf"}[15m]))',
                     "≤ {{le}}", instant=True, fmt="heatmap"),
                unit="short", orientation="vertical", display="gradient",
                thresholds=steps((GREEN, None)),
                desc="How many patients got each predicted risk. Training data is roughly flat with peaks at both ends.")
place(dist, 0, 8, 8)
place(timeseries("Mean predicted risk",
                 [prom("sum(rate(cardioguard_prediction_probability_sum[5m])) / sum(rate(cardioguard_prediction_probability_count[5m]))",
                       "Mean predicted risk")],
                 unit="percentunit", min_=0, max_=1,
                 desc="About 50% in the training data (the dataset is balanced)."), 8, 8, 8)
place(timeseries("Model and SHAP time (p95)",
                 [prom("histogram_quantile(0.95, sum by (le) (rate(cardioguard_model_inference_seconds_bucket[5m])))", "Model", "A"),
                  prom("histogram_quantile(0.95, sum by (le) (rate(cardioguard_shap_seconds_bucket[5m])))", "SHAP explanation", "B")],
                 unit="s", min_=0,
                 desc="Time inside the model only, without HTTP or logging. The gap to the request latency is overhead."),
      16, 8, 8, new_line=True)
place(stat("Model version", prom('max by (model_version) (cardioguard_model_info{role="champion"})', instant=True, fmt="table"),
           fields="/^model_version$/", show_values=True, desc="Champion loaded by the API (Phase 6: from MLflow)."), 0, 6, 4)
place(stat("Confirmed diagnoses (1 h)", prom("sum(increase(cardioguard_feedback_total[1h])) or vector(0)", instant=True),
           decimals=0, desc="Ground truth received via /feedback. The accuracy check needs about 100 per drift window."), 6, 6, 4)
place(stat("Invalid inputs rejected (1 h)",
           prom(f'sum(increase(http_requests_total{{{PRED}, status="4xx"}}[1h])) or vector(0)', instant=True),
           decimals=0, thresholds=steps((TEAL, None), (AMBER, 50)),
           desc="Requests with impossible values (e.g. diastolic above systolic)."), 12, 6, 4)
place(stat("Prediction log failures (1 h)",
           prom("sum(increase(cardioguard_log_failures_total[1h])) or vector(0)", instant=True),
           decimals=0, thresholds=steps((GREEN, None), (RED, 1)),
           desc="Writes to PostgreSQL or MongoDB that failed."), 18, 6, 4, new_line=True)

# ------------------------------------------------------------------ 5. Live inputs
row("Live inputs vs training data")
for i, (feat, label, unit) in enumerate([("ap_hi", "Systolic BP", "mmHg"), ("ap_lo", "Diastolic BP", "mmHg"),
                                         ("age_years", "Age", "years"), ("bmi", "BMI", "")]):
    place(timeseries(f"{label}: live mean vs training",
                     [prom(f'sum(rate(cardioguard_input_value_sum{{feature="{feat}"}}[5m])) '
                           f'/ sum(rate(cardioguard_input_value_count{{feature="{feat}"}}[5m]))', "Live (5 min)", "A"),
                      prom(f'max(cardioguard_reference_feature_mean{{feature="{feat}"}})', "Training", "B")],
                     unit="none", overrides=[dashed("Training"), color_by_name("Live (5 min)", TEAL)],
                     desc=f"Real-time view of {label.lower()}{' (' + unit + ')' if unit else ''}. "
                          "Shows a shift within minutes; the drift check confirms it statistically."),
          i * 6, 6, 7, new_line=(i == 3))

# ------------------------------------------------------------------ 6. Drift checks
row("Drift checks (Evidently)")
place(stat("Latest drift status", prom("max(cardioguard_drift_status)", instant=True),
           mappings=value_map({-1: ("Not enough data", GREY), 0: ("OK", GREEN), 1: ("Warning", AMBER), 2: ("Critical", RED)}),
           thresholds=steps((GREY, None), (GREEN, 0), (AMBER, 1), (RED, 2)), color_mode="background",
           desc="Pushed by the drift job. Alerts: DataDriftWarning, DataDriftCritical."), 0, 6, 4)
place(stat("Retraining", prom("max(cardioguard_retrain_recommended)", instant=True),
           mappings=value_map({0: ("Not needed", GREEN), 1: ("Recommended", RED)}),
           thresholds=steps((GREEN, None), (RED, 1)), color_mode="background",
           desc="Critical drift in consecutive windows. Phase 6 retrains automatically."), 6, 6, 4)
place(stat("Highest PSI", prom("max(cardioguard_drift_max_psi)", instant=True), decimals=2,
           thresholds=steps((GREEN, None), (AMBER, 0.1), (RED, 0.25)),
           desc="0.10 moderate shift, 0.25 major shift."), 12, 4, 4)
place(stat("ROC-AUC (confirmed cases)", prom("max(cardioguard_window_roc_auc)", instant=True), decimals=3,
           thresholds=steps((RED, None), (AMBER, 0.752), (GREEN, 0.772)),
           desc="On human-labelled rows in the last drift window. Training: 0.802."), 16, 4, 4)
place(stat("Last drift check", prom("time() - max(cardioguard_drift_last_run_timestamp_seconds)", instant=True),
           unit="s", decimals=0, thresholds=steps((GREEN, None), (AMBER, 3600), (RED, 7200)),
           desc="Time since the drift job last ran. Alert DriftCheckStale after 2 hours."), 20, 4, 4, new_line=True)
place(bargauge("Drift per input (score ÷ threshold)",
               prom("max by (feature) (cardioguard_feature_drift_score / cardioguard_feature_drift_threshold)",
                    "{{feature}}", instant=True),
               unit="none", thresholds=steps((TEAL, None), (RED, 1)), display="basic",
               desc="1 or more = drifted (red). Evidently: Wasserstein for numeric inputs, Jensen-Shannon for categories."),
      0, 12, 9)
place(timeseries("Drift history",
                 [sql("SELECT run_at AS time, max_psi AS \"Highest PSI\", prediction_drift_score AS \"Prediction drift\", "
                      "drift_share AS \"Share of inputs drifted\" FROM drift_runs "
                      "WHERE $__timeFilter(run_at) AND status <> 'insufficient_data' ORDER BY 1", fmt="time_series")],
                 ds=PG, unit="none", min_=0,
                 desc="One point per drift run, from the drift_runs table in PostgreSQL."), 12, 12, 9, new_line=True)
recent = {
    "type": "table", "title": "Recent drift checks", "datasource": PG,
    "description": "Latest 20 runs from PostgreSQL. Open the monitoring page for the full Evidently report.",
    "targets": [sql("SELECT run_at AS \"Checked\", status AS \"Status\", n_rows AS \"Patients\", "
                    "n_labelled AS \"Confirmed\", round(drift_share::numeric, 2) AS \"Inputs drifted\", "
                    "round(max_psi::numeric, 2) AS \"Highest PSI\", round(roc_auc::numeric, 3) AS \"ROC-AUC\", "
                    "CASE WHEN retrain_recommended THEN 'Yes' ELSE '' END AS \"Retrain\", "
                    "replace(reasons, E'\\n', ' | ') AS \"Reasons\" "
                    "FROM drift_runs ORDER BY run_at DESC LIMIT 20")],
    "fieldConfig": {"defaults": {"custom": {"align": "auto", "cellOptions": {"type": "auto"}}},
                    "overrides": [{"matcher": {"id": "byName", "options": "Status"},
                                   "properties": [{"id": "custom.cellOptions", "value": {"type": "color-background"}},
                                                  {"id": "mappings", "value": value_map({
                                                      "ok": ("OK", GREEN), "warning": ("Warning", AMBER),
                                                      "critical": ("Critical", RED),
                                                      "insufficient_data": ("Not enough data", GREY)})}]},
                                  {"matcher": {"id": "byName", "options": "Reasons"},
                                   "properties": [{"id": "custom.width", "value": 520}]}]},
    "options": {"showHeader": True, "cellHeight": "sm", "footer": {"show": False}},
}
place(recent, 0, 24, 9, new_line=True)

# ------------------------------------------------------------------ 7. Semi-supervised learning (Phase 5)
row("Semi-supervised learning (pseudo-labels)")
place(stat("Last pseudo-label run", prom("max(cardioguard_pseudo_label_status)", instant=True),
           mappings=value_map({-1: ("Not enough data", GREY), 0: ("Labels added", GREEN), 1: ("Skipped: drift", AMBER)}),
           thresholds=steps((GREY, None), (GREEN, 0), (AMBER, 1)), color_mode="background",
           desc="Runs after each drift check. Skipped while drift is critical: model confidence is unreliable then."),
      0, 6, 4)
place(stat("Accepted last run", prom("max(cardioguard_pseudo_label_accepted)", instant=True), decimals=0,
           desc="Pseudo-labels added by the last run (after agreement filter, class balance and caps)."), 6, 4, 4)
place(stat("Acceptance rate", prom("max(cardioguard_pseudo_label_acceptance_rate)", instant=True),
           unit="percentunit", decimals=1, thresholds=steps((RED, None), (AMBER, 0.05), (GREEN, 0.10)),
           desc="Accepted / unconfirmed patients. About 13% on normal traffic: the model is well "
                "calibrated, so it is rarely 90% sure. Alert LowPseudoLabelAcceptance below 5%."), 10, 5, 4)
place(stat("Methods disagree", prom("max(cardioguard_pseudo_label_disagreement_rate)", instant=True),
           unit="percentunit", decimals=1, thresholds=steps((GREEN, None), (AMBER, 0.35), (RED, 0.5)),
           desc="Share of unconfirmed patients where self-training and label spreading give different labels."),
      15, 5, 4)
place(stat("Pseudo-labels stored", prom("max(cardioguard_pseudo_label_total)", instant=True), decimals=0,
           desc="Capped at 1x the real labels so machine guesses never outnumber real diagnoses."),
      20, 4, 4, new_line=True)
place(timeseries("Pseudo-labelling over time",
                 [sql("SELECT run_at AS time, acceptance_rate AS \"Acceptance rate\", "
                      "disagreement_rate AS \"Methods disagree\", mean_confidence AS \"Mean confidence\" "
                      "FROM pseudo_label_runs WHERE $__timeFilter(run_at) AND status = 'ok' ORDER BY 1",
                      fmt="time_series")],
                 ds=PG, unit="percentunit", min_=0, max_=1,
                 overrides=[color_by_name("Acceptance rate", GREEN), color_by_name("Methods disagree", AMBER),
                            color_by_name("Mean confidence", TEAL)],
                 desc="From the pseudo_label_runs table. Watch for acceptance falling while disagreement rises."),
      0, 24, 8, new_line=True)

# ------------------------------------------------------------------ 8. Self-healing (Phase 6)
OUTCOMES = {0: ("Promoted", GREEN), 1: ("Nothing to do", GREY), 2: ("Baseline updated", GREEN), 3: ("Cool-down", GREY),
            4: ("Busy", GREY), 5: ("Stopped: bad data", RED), 6: ("Rejected by checks", AMBER),
            7: ("Rejected by person", AMBER), 8: ("No decision (timeout)", AMBER), 9: ("Rolled back", RED),
            10: ("Failed", RED)}
row("Self-healing (MLflow + Airflow)")
place(stat("Rollout", prom("max(cardioguard_deployment_status)", instant=True),
           mappings=value_map({0: ("Stable", GREEN), 1: ("Canary", TEAL), 2: ("Waiting for approval", AMBER),
                               3: ("Approved", TEAL)}),
           thresholds=steps((GREEN, None), (TEAL, 1), (AMBER, 2), (TEAL, 3)), color_mode="background",
           desc="From mlops/deployment.py. Approve or reject on the monitoring page."), 0, 5, 4)
place(stat("Traffic on candidate", prom("max(cardioguard_canary_fraction)", instant=True), unit="percentunit",
           decimals=0, desc="Share of requests the API sends to the candidate during a canary release."), 5, 4, 4)
cand = stat("Candidate", prom('max by (model_version) (cardioguard_model_info{role="candidate"})', instant=True,
                               fmt="table"), fields="/^model_version$/", show_values=True,
            desc="Model on trial. Empty when no rollout is running.")
cand["fieldConfig"]["defaults"]["noValue"] = "None"
place(cand, 9, 5, 4)
place(stat("Last pipeline run", prom("max(cardioguard_self_healing_outcome)", instant=True),
           mappings=value_map(OUTCOMES), color_mode="background",
           thresholds=steps((GREY, None)),
           desc="Outcome of the last cardio_self_healing run (Airflow or python -m mlops.pipeline)."), 14, 6, 4)
place(stat("Since last run", prom("time() - max(cardioguard_self_healing_last_run_timestamp_seconds)", instant=True),
           unit="s", decimals=0, desc="Time since the self-healing pipeline last finished."), 20, 4, 4, new_line=True)
place(timeseries("Requests per model version",
                 [prom("sum by (model_version) (rate(cardioguard_model_inference_seconds_count[5m]))", "{{model_version}}")],
                 unit="reqps", min_=0,
                 desc="During a canary, about 10% of requests go to the candidate. After promotion it takes all."),
      0, 12, 8)
place(timeseries("Model time p95 per version",
                 [prom("histogram_quantile(0.95, sum by (le, model_version) "
                       "(rate(cardioguard_model_inference_seconds_bucket[5m])))", "{{model_version}}")],
                 unit="s", min_=0,
                 desc="The canary check rolls back if the candidate is much slower than the champion."),
      12, 12, 8, new_line=True)
events = {
    "type": "table", "title": "Self-healing events", "datasource": PG,
    "description": "Latest 25 entries of deployment_events: training, checks, approvals, canary, promotion, rollback.",
    "targets": [sql("SELECT at AS \"When\", event AS \"Event\", model_version AS \"Model\", "
                    "registry_version AS \"MLflow v\", actor AS \"By\", left(detail::text, 300) AS \"Detail\" "
                    "FROM deployment_events ORDER BY at DESC LIMIT 25")],
    "fieldConfig": {"defaults": {"custom": {"align": "auto", "cellOptions": {"type": "auto"}}},
                    "overrides": [{"matcher": {"id": "byName", "options": "Event"},
                                   "properties": [{"id": "custom.cellOptions", "value": {"type": "color-text"}},
                                                  {"id": "mappings", "value": [
                                                      {"type": "regex", "options": {"pattern": ".*(promoted|passed_gates|approved).*",
                                                                                    "result": {"color": GREEN, "index": 0}}},
                                                      {"type": "regex", "options": {"pattern": ".*(rolled_back|rejected|failed|stopped|timeout).*",
                                                                                    "result": {"color": RED, "index": 1}}}]}]},
                                  {"matcher": {"id": "byName", "options": "Detail"},
                                   "properties": [{"id": "custom.width", "value": 520}]}]},
    "options": {"showHeader": True, "cellHeight": "sm", "footer": {"show": False}},
}
place(events, 0, 24, 9, new_line=True)

dashboard = {
    "uid": "cardioguard-main",
    "title": "CardioGuard - Model & Service Monitoring",
    "description": "Live service metrics (Prometheus), drift results (Evidently via Pushgateway + PostgreSQL) and self-healing (Phase 6).",
    "tags": ["cardioguard", "mlops"],
    "timezone": "browser",
    "editable": True,
    "graphTooltip": 1,
    "refresh": "15s",
    "time": {"from": "now-1h", "to": "now"},
    "schemaVersion": 39,
    "version": 1,
    "templating": {"list": []},
    "annotations": {"list": [
        {"builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"}, "enable": True, "hide": True,
         "iconColor": "rgba(0, 211, 255, 1)", "name": "Annotations & Alerts", "type": "dashboard"},
        {"name": "Drift alerts", "datasource": PG, "enable": True, "iconColor": RED,
         "target": sql("SELECT run_at AS time, 'Drift ' || status AS text, status AS tags FROM drift_runs "
                       "WHERE $__timeFilter(run_at) AND status IN ('warning', 'critical') ORDER BY 1")},
        {"name": "Model changes", "datasource": PG, "enable": True, "iconColor": TEAL,
         "target": sql("SELECT at AS time, event || ' ' || coalesce(model_version, '') AS text, event AS tags "
                       "FROM deployment_events WHERE $__timeFilter(at) "
                       "AND event IN ('canary_started', 'promoted', 'rolled_back') ORDER BY 1")},
    ]},
    "panels": _panels,
}

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dashboard, indent=2, ensure_ascii=False))
    n = sum(p["type"] != "row" for p in _panels)
    print(f"Wrote {OUT} ({n} panels in {sum(p['type'] == 'row' for p in _panels)} rows)")
