[← Prediction Service](01-prediction-service.md) · [All docs](README.md) · [Observability →](03-observability.md)

# Phase 2: Data drift monitoring (Evidently AI)

Every drift check compares the **latest 1,000 production predictions** with the
**training data** and gives each of six checks a status: OK, warning or critical.
The worst status becomes the run's status.

| Check | How | Warning | Critical |
|---|---|---|---|
| Inputs that drifted | Evidently per feature: Wasserstein (numeric), Jensen–Shannon (categorical), threshold 0.1 | ≥ 30% of inputs | ≥ 50% |
| Shift in key measurements | PSI on systolic/diastolic BP, age, BMI, cholesterol, glucose | ≥ 0.10 | ≥ 0.25 |
| Change in predictions | Evidently Wasserstein on the predicted probability | drifted once | 2 windows in a row |
| Model confidence | mean max(p, 1−p) vs training | drops 5 points | drops 10 points |
| Accuracy on confirmed diagnoses | ROC-AUC on human labels, **99% bootstrap CI** | drop ≥ 0.03 *and significant* | drop ≥ 0.05 *and significant* |
| Invalid inputs | missing / out of range / diastolic ≥ systolic | ≥ 2% of rows | ≥ 5% |

**Retraining is recommended** when two runs in a row are critical. Until Phase 6 this only
logs; once Airflow runs, set `AIRFLOW_API_URL` and it starts the `cardio_self_healing` DAG.

All thresholds live in `monitoring/thresholds.yaml`.

## Run it

```bash
# API must be running (uvicorn app.main:app --port 8000)

# 1. Simulate traffic (patients from the unseen test set, 20% get a confirmed diagnosis)
python -m simulate.traffic --scenario normal --n 1000
python -m simulate.traffic --scenario bp_shift --n 1000

# 2. Run one drift check
python -m monitoring.drift_job

# 3. Or check every 60 minutes
python -m monitoring.scheduler --every 60
```

Monitoring page: **http://localhost:8000/monitoring**. It shows the latest status, each
check, drift per input, history, and a link to the full Evidently HTML report.
There is also a **Run drift check** button.

| Method | Path | Purpose |
|---|---|---|
| GET | `/monitoring/drift/latest` | Latest run + per-feature scores |
| GET | `/monitoring/drift/history?limit=30` | Recent runs |
| GET | `/monitoring/drift/runs/{id}/report` | Evidently HTML report |
| POST | `/monitoring/drift/run` | Run a check now |

## Simulation scenarios

| Scenario | What changes | Expected |
|---|---|---|
| `normal` | nothing | OK |
| `bp_shift` | BP +20/+10 mmHg (mis-calibrated device; true diagnosis unchanged) | data + accuracy drift |
| `aging` | everyone 8 years older | data drift, accuracy holds |
| `cholesterol` | far more high-cholesterol readings, unrelated to diagnosis | data + prediction drift |
| `concept` | inputs unchanged, diagnosis flips for under-50s | accuracy drift only |

## Results (9 consecutive runs, 1,000 patients each, ~20% labelled)

| Run | Scenario | Status | Max PSI | Prediction drift | ROC-AUC | Retrain |
|---|---|---|---|---|---|---|
| 1 | normal | OK | 0.00 | 0.014 | 0.726 (not significant) | – |
| 2 | normal | OK | 0.01 | 0.028 | 0.846 | – |
| 3 | bp_shift | Critical | 1.69 | 0.978 | 0.766 | – |
| 4 | bp_shift | Critical | 1.86 | 1.023 | 0.690 | Yes |
| 5 | normal | OK | 0.01 | 0.039 | 0.828 | – |
| 6 | aging | Critical | 2.11 | 0.305 | 0.816 | – |
| 7 | cholesterol | Critical | 0.85 | 0.352 | 0.771 | Yes |
| 8 | concept | Critical | 0.01 | 0.026 | 0.416 | Yes |
| 9 | concept | Critical | 0.02 | 0.056 | 0.550 | Yes |

Findings worth reporting:

1. **No false alarms on normal traffic, every drift type caught.** Concept drift (run 8)
   is invisible to every input check and only shows in accuracy, which is why labels matter.
2. **Data drift does not always hurt the model.** Aging (run 6) shifts age strongly
   (PSI 2.11) but ROC-AUC holds at 0.816.
3. **A raw AUC threshold on ~200 labels is mostly noise.** Run 1 is normal traffic with
   AUC 0.726. Over 300 simulated normal windows, a plain "AUC dropped ≥ 0.03" rule raised a
   false alarm in **18%** of them. Requiring the 99% bootstrap interval to lie below the
   training AUC cut that to **0%**, while concept drift was still caught in **100%** of windows.

## Where results are stored

| Where | What |
|---|---|
| Postgres `drift_runs` | one row per run: status, drift share, max PSI, prediction drift, AUC, retrain flag, checks (JSON) |
| Postgres `drift_feature_scores` | one row per input per run (ready for Grafana in Phase 3) |
| MongoDB `drift_reports` / `alerts` | full JSON of each run / warning and critical alerts |
| `reports/drift/run_<id>.html` | Evidently HTML report |

Set `ALERT_WEBHOOK_URL` to also post alerts to Slack, Discord or Teams.

[← Prediction Service](01-prediction-service.md) · [All docs](README.md) · [Observability →](03-observability.md)
