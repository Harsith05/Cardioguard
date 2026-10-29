[← Drift Monitoring](02-drift-monitoring.md) · [All docs](README.md) · [Containerization →](04-containerization.md)

# Phase 3: Operational monitoring (Prometheus + Grafana)

```
API /metrics ──(scrape every 15 s)──▶ Prometheus ──▶ Grafana dashboard
drift_job ──push──▶ Pushgateway ──(scrape)──▶ Prometheus     Postgres drift tables ──▶ Grafana
                                               └─▶ alert rules (rules.yml)
```

## Start it

```bash
docker compose up -d                      # Postgres, MongoDB, Pushgateway, Prometheus, Grafana
uvicorn app.main:app --port 8000          # API on your computer (Prometheus reaches it via host.docker.internal)
python -m simulate.load_test --rps 15 --duration 300
python -m simulate.traffic --scenario normal --n 1000 && python -m monitoring.drift_job
```

| Tool | URL |
|---|---|
| Grafana dashboard | http://localhost:3000 (opens on the CardioGuard dashboard; admin / admin to edit) |
| Prometheus | http://localhost:9090 (Status → Targets should show `cardioguard-api` UP; Alerts tab) |
| Raw metrics | http://localhost:8000/metrics |

## Metrics

| Metric | Type | Used for |
|---|---|---|
| `http_requests_total{handler,status}` | counter | traffic, error rate (2xx / 4xx / 5xx) |
| `http_request_duration_seconds` | histogram | p50 / p95 / p99 latency |
| `cardioguard_predictions_total{risk_level,endpoint}` | counter | predictions by risk level |
| `cardioguard_prediction_probability` | histogram | score distribution, mean risk |
| `cardioguard_model_inference_seconds`, `cardioguard_shap_seconds` | histogram | model and SHAP time |
| `cardioguard_input_value{feature}` + `cardioguard_reference_feature_mean` | summary + gauge | live mean of BP, age, BMI vs training |
| `cardioguard_feedback_total`, `cardioguard_db_logging_up`, `cardioguard_log_failures_total` | | labels, logging health |
| `cardioguard_drift_*`, `cardioguard_feature_drift_score`, `cardioguard_feature_psi` | gauges (Pushgateway) | Evidently results per drift run |

## Dashboard (28 panels, 6 rows)

| Row | Panels |
|---|---|
| Service health | API up, requests/s, error rate, p95 latency, saving predictions |
| Traffic | requests/s by endpoint, predictions/min by risk level |
| Reliability | error rate (5xx and 4xx) and latency p50/p95/p99, with alert limits as dashed lines |
| Model health | risk distribution, mean predicted risk, model/SHAP time, model version, diagnoses received, invalid inputs, log failures |
| Live inputs vs training | systolic BP, diastolic BP, age, BMI: live 5-minute mean against the training mean |
| Drift checks | latest status, retraining, highest PSI, ROC-AUC, time since last check, drift per input, drift history and recent runs (Postgres); drift alerts drawn as annotations |

The JSON is generated: edit `infra/grafana/build_dashboard.py`, run it, restart Grafana.
`python infra/grafana/check_dashboard.py` runs every panel query against live Prometheus
and Postgres and lists any panel that would show "No data".

## Alerts (`infra/prometheus/rules.yml`)

| Alert | Condition | Severity |
|---|---|---|
| PredictionAPIDown | API not scrapeable for 1 min | critical |
| HighPredictionErrorRate | > 5% of /predict requests return 5xx for 5 min | critical |
| HighPredictionLatency | p95 > 500 ms for 5 min | warning |
| PredictionLoggingDown / PredictionLogWritesFailing | predictions not reaching Postgres | warning |
| DataDriftWarning / DataDriftCritical | latest drift run status | warning / critical |
| RetrainingRecommended | critical drift in consecutive windows | critical |
| DriftCheckStale | no drift run for 2 hours | warning |

Unit tests for the rules: `promtool test rules infra/prometheus/rules_test.yml`
(in Docker: `docker compose exec prometheus promtool test rules /etc/prometheus/rules_test.yml`).

## Testing the alerts with fault injection

```bash
CHAOS_ENABLED=1 uvicorn app.main:app --port 8000        # PowerShell: $env:CHAOS_ENABLED="1"
curl -X POST "localhost:8000/debug/chaos?error_rate=0.2&latency_ms=600"
python -m simulate.load_test --rps 15 --duration 420     # alerts fire after their 5-minute "for"
curl -X POST "localhost:8000/debug/chaos?error_rate=0&latency_ms=0"
```

## Verified results

| Test | Result |
|---|---|
| Normal load, 15 req/s for 60 s | 850 ok, 50 rejected (the 5% invalid inputs), p95 29 ms |
| All 37 dashboard queries (Prometheus + Postgres) | all return data after one normal + one drifted window |
| 20% injected failures + 600 ms delay | HighPredictionErrorRate fired (12.3%), HighPredictionLatency fired (730 ms p95) |
| Two blood-pressure-shift windows | DataDriftCritical (PSI 1.86) and RetrainingRecommended fired; live systolic mean 135 vs 127 training |
| API stopped | PredictionAPIDown fired |
| promtool rule unit tests | 6 test groups pass, including exact alert messages |

The live alert run used `for: 30s` instead of 5 minutes to save time; the 5-minute behaviour
is covered by the promtool unit tests.

[← Drift Monitoring](02-drift-monitoring.md) · [All docs](README.md) · [Containerization →](04-containerization.md)
