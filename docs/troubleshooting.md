[← README](../README.md) · [All docs](README.md)

# Troubleshooting

These problems were all met while building and running this project.

## Setup

| Symptom | Cause | Fix |
|---|---|---|
| `password authentication failed` / connection refused on 5432 | A Windows PostgreSQL already uses port 5432 | `POSTGRES_PORT=5433` in `.env`, **and** port 5433 in `DATABASE_URL` |
| `No module named 'mlops'` (or another new module) in a container | The image wasn't rebuilt after a code change. `up -d` reuses old images. | `docker compose build api airflow-init`, then `docker compose up -d`. Check that `docker/app.Dockerfile` copies the folder. |
| `train.py: error: unrecognized arguments: --monotone` | Same: the trainer runs inside the old app image | `docker compose build api` |
| The new base model is not served after retraining | The API serves the MLflow `@champion`; `bootstrap` registers `models/model.joblib` only when MLflow is empty | Fresh start: `docker compose down -v`, then `up -d` (**deletes all data**). Or retrain through the pipeline. |
| `image "cardioguard-app:latest" already exists` during build | Several services built the same image at once | Only `api` (and `airflow-init`) have a `build` section; use `docker compose build api` |
| Bind mounts fail after deleting `models/` | Docker keeps the mount of the deleted folder | Recreate the folder and run `docker compose up -d --force-recreate` |

## Self-healing

| Symptom | Cause | Fix |
|---|---|---|
| Run ends as **"Run skipped: cool-down"** | A promotion or rollback happened in the last 24 h | Trigger with **force ✔**, or lower `trigger.cooldown_hours` for demos |
| Run ends as **"nothing to do"** | The latest drift check had no confirmed diagnoses (`--label-rate 0`) or no drift | Send labelled drift traffic and run a drift check first, or set `diagnosis` in the trigger form |
| Canary rolled back: "only N canary predictions (need 50)" | No traffic during the canary | Run the traffic loop as soon as `start_canary` begins |
| Approve button asks for an admin token | `ADMIN_TOKEN` is set, often by accident (comment on the same line as an empty value) | `docker compose exec api printenv ADMIN_TOKEN`; put comments on their own line, then `docker compose up -d api` |
| "Approval timed out" | Nobody approved within `approval_minutes` | Trigger again; the model is kept in MLflow |
| Browser fills an email into the "Reason" box | Autofill | Fixed (`autocomplete="off"`); rebuild the API image |
| MLflow returns 403 inside Docker | MLflow 3 rejects unknown `Host` headers | Service names are listed in `--allowed-hosts` (already in compose) |
| API starts slowly when MLflow is down | The MLflow client retried for minutes | Retries are capped (`MLFLOW_HTTP_REQUEST_MAX_RETRIES=2`); the API falls back to the file model |

## Airflow

| Symptom | Fix |
|---|---|
| Login: what user? | `admin` / the value of `AIRFLOW_PASSWORD` (default `admin`) |
| A run is labelled `manual__…` although nobody clicked | Runs started through the REST API (by the drift job) are labelled "manual" by Airflow 3 |
| Task log shows `No module named 'pendulum'` | Harmless: Airflow probes the ML venv; the image installs pendulum to silence it |

## Results look strange

| Observation | Explanation |
|---|---|
| A retrained model predicts the opposite for old-style patients | Concept drift: the same inputs now have different outcomes, and a model gives one answer per input. A person must approve models that forget the old pattern. See [docs/06-self-healing.md](06-self-healing.md). |
| Pseudo-labelling "Skipped: drift" | By design: model confidence is unreliable while drift is critical |
| Test-set metrics look too good after `--include-production` | The traffic simulator samples from the test set. Judge such models on the **validation** set. |

[← README](../README.md) · [All docs](README.md)
