[← README](../README.md) · [All docs](README.md)

# Configuration

## `.env` (copy from `.env.example`)

Docker Compose and the Python code both read it. Variables set in the terminal take priority.

> Put comments on their **own line** when a value is empty. With `ADMIN_TOKEN=   # comment`,
> Docker Compose can take the comment as the value.

| Variable | Default | Purpose |
|---|---|---|
| `POSTGRES_USER` / `POSTGRES_PASSWORD` | cardio / cardio | Database login |
| `POSTGRES_PORT` | 5432 | Port on your computer (use 5433 if a local Postgres already uses 5432, and change `DATABASE_URL` to match) |
| `DATABASE_URL` | `postgresql+psycopg2://cardio:cardio@localhost:5432/cardio` | For Python on your computer; containers use `postgres:5432` |
| `MONGO_URI` / `MONGO_DB` | `mongodb://localhost:27017` / cardioguard | Raw JSON prediction log (optional) |
| `GRAFANA_PASSWORD` | admin | Grafana admin login |
| `PUSHGATEWAY_URL` | localhost:9091 | Where the drift job pushes results |
| `ALERT_WEBHOOK_URL` | empty | Slack / Discord / Teams webhook for alerts |
| `CHAOS_ENABLED` | 0 | 1 = allow `/debug/chaos` fault injection (testing only) |
| `PSEUDO_LABEL_ENABLED` | 1 | Pseudo-label after each scheduled drift check |
| `DRIFT_EVERY_MINUTES` | 60 | Drift check interval |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | Registry; empty = serve `models/model.joblib` |
| `PROMETHEUS_URL` | `http://localhost:9090` | Canary check reads model errors here |
| `AIRFLOW_URL` | empty | For a drift job running on your computer (containers use `http://airflow:8080`) |
| `DEPLOYMENT_POLL_SECONDS` | 15 | How fast every API replica follows a canary or promotion |
| `MLFLOW_PORT` / `AIRFLOW_PORT` / `API_PORT` | 5000 / 8080 / 8000 | Ports on your computer |
| `AIRFLOW_PASSWORD` | admin | Airflow login (user `admin`) |
| `AIRFLOW_SECRET` | dev value | JWT / session secret shared by the Airflow services: **change it** |
| `ADMIN_TOKEN` | empty | If set, approve / reject / rollback need the header `X-Admin-Token` |

## Drift thresholds (`monitoring/thresholds.yaml`)

| Section | Key settings |
|---|---|
| `window_size`, `min_rows` | Latest 1,000 predictions per check; skip below 200 |
| `feature_drift`, `psi` | Evidently thresholds, share of drifted inputs, PSI 0.10 / 0.25 |
| `prediction_drift`, `confidence` | Output distribution, mean confidence drop (5 / 10 points) |
| `performance` | AUC drop 0.03 / 0.05, counted only if significant (99% bootstrap CI), needs 100 labels |
| `calibration` | Observed sick share vs mean predicted risk: 5 / 10 points (99% significance) |
| `data_quality` | Invalid rows 2% / 5% |
| `retraining.consecutive_critical` | 2 critical checks in a row → retraining recommended |

## Pseudo-labelling (`semisup/config.yaml`)

Self-training ≥ 0.9 confidence; label spreading ≥ 0.7; both must agree. Class balance as in the
training data, at most 2,000 per run, total ≤ 1× real labels. Skipped while drift is critical.
Weight 0.5 when retraining.

## Self-healing (`mlops/config.yaml`)

Any top-level key can be overridden per run, through the Airflow trigger form (`config`) or
`python -m mlops.pipeline --config-json '{...}'`.

| Section | Key settings |
|---|---|
| `trigger` | `cooldown_hours: 24` after a promotion or rollback |
| `diagnosis` | Which checks count as input drift / accuracy loss / bad data |
| `training` | Candidate models, `monotone: true`, recency weights, concept strategies (×10, ×50, new pattern only), hold-out share |
| `gates.default` | Max drops vs champion: AUC 0.01 (old and recent), recall 0.02, Brier 0.01; fairness gap 0.05 |
| `gates.retrain_concept` | Recent AUC +0.02 **or** Brier −0.01; old-population drop > 0.05 → human approval, > 0.35 → reject |
| `approval` | Timeout 24 h |
| `canary` | 10% for 30 min, ≥ 50 predictions, error rate ≤ 1%, p95 ≤ 300 ms, risk shift within 0.10 of the expected one |
| `clinical` | Golden patients (required risk ranges) and "worse risk factor" pairs: edit with a clinician |

[← README](../README.md) · [All docs](README.md)
