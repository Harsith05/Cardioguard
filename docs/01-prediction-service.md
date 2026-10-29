[← README](../README.md) · [All docs](README.md) · [Drift Monitoring →](02-drift-monitoring.md)

# Phase 1: Prediction service, explainability and logging

## Results (Kaggle cardio_train, 70,000 patients)

**Cleaning** removed 2,066 rows (2.95%): impossible blood pressure (e.g. ap_hi = 16020,
negative values), diastolic ≥ systolic, height outside 120–220 cm, weight outside 30–200 kg,
and 674 exact duplicates. 67,934 rows remain, split 70/15/15 (stratified).

| Model | Val ROC-AUC | Val recall | Val F1 | Val Brier |
|---|---|---|---|---|
| Logistic Regression | 0.7926 | 0.6703 | 0.7114 | 0.1866 |
| Random Forest | 0.8010 | 0.6713 | 0.7156 | 0.1808 |
| XGBoost | 0.8016 | 0.6949 | 0.7250 | 0.1802 |
| **LightGBM (selected)** | **0.8018** | **0.6967** | **0.7254** | **0.1802** |

Selected model on the untouched test set: **ROC-AUC 0.8024, recall 0.6977**.
An AUC of about 0.80 is the usual ceiling for this dataset, since it holds only 11 basic features.

## Quick start

```bash
pip install -r requirements.txt

# 1. Clean + split data (put cardio_train.xlsx or .csv in data/raw/)
python -m training.data_prep --input data/raw/cardio_train.xlsx

# 2. Train, compare, save model + reference data (logs runs to ./mlruns)
python -m training.train

# 3. Start the databases
docker compose up -d

# 4. Copy reference data into Postgres (used by Phase 2)
python -m training.load_reference

# 5. Run the API
uvicorn app.main:app --host 0.0.0.0 --port 8000
#    Prediction form:  http://localhost:8000/
#    API docs:         http://localhost:8000/docs

# Tests (use SQLite, no Docker needed)
pytest -q
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | One patient → probability, class, risk label, model version |
| POST | `/predict/batch` | Up to 1,000 patients |
| POST | `/feedback/{request_id}` | Attach a confirmed diagnosis later (ground truth) |
| GET | `/health` | Model loaded? Postgres/Mongo logging on? |
| GET | `/model-info` | Version, metrics, features |

Example input (age in **years**; gender 1 = female, 2 = male as in the dataset):

```json
{"age_years": 52, "gender": 2, "height": 172, "weight": 84, "ap_hi": 140, "ap_lo": 90,
 "cholesterol": 2, "gluc": 1, "smoke": 0, "alco": 0, "active": 1}
```

Invalid inputs (out-of-range values, diastolic ≥ systolic) are rejected with HTTP 422,
using the same rules as data cleaning.

## Explainable AI (SHAP)

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict/explain` | Prediction + SHAP contribution of every input for this patient |
| GET | `/explain/global` | Mean \|SHAP\| over 1,000 training patients (overall importance) |

How it works (`common/explain.py`):

- `TreeExplainer` runs on the classifier (LightGBM / XGBoost / Random Forest; Logistic
  Regression uses `LinearExplainer`), using the transformed features.
- One-hot columns are summed back to their source feature (cholesterol_1/2/3 → cholesterol).
- Related measurements are summed into readable groups: **Blood pressure** = systolic +
  diastolic + mean arterial pressure + pulse pressure; **Body mass index** = height + weight + BMI.
  These features are correlated, so SHAP splits their credit; grouping shows the combined effect.
- SHAP values are additive, so `base value + sum of contributions = model output` exactly
  (checked in the tests and in the report script, error ≈ 1e-15).

The prediction page shows a "Why this result" chart: red bars raise risk, green bars lower it.
It has a toggle between grouped factors and individual measurements, plus the model's overall
importance ranking.

Figures for the paper:

```bash
python -m training.explain_report --n 2000     # -> reports/shap/*.png
```

Produces a beeswarm summary, global importance (per feature and per clinical factor),
and waterfall plots for one high-risk and one low-risk test patient.

**Finding worth discussing in the paper:** in this dataset, smoking and alcohol get small
*negative* SHAP values (they slightly lower predicted risk). This is a known artefact of the
Kaggle data (self-reported, and smokers in it have a lower disease rate), not a medical
effect. SHAP makes the bias visible, which is one of the arguments for explainability.

## What gets stored

| Where | What | Used in |
|---|---|---|
| Postgres `predictions` | features, probability, class, latency, `label` (NULL until known), `label_source` (human / pseudo) | Phase 2 drift, Phase 5 SSL, Phase 6 retraining |
| Postgres `reference_data` | training rows + model's predictions on them | Phase 2 drift baseline |
| MongoDB `raw_requests` | original JSON payload | audit, replay |
| `models/model.joblib` + `model_meta.json` | full sklearn pipeline (feature engineering + preprocessing + model) | serving |

Logging runs in a background task after the response is sent, and a database outage
never breaks a prediction.

## Layout

See [Project structure](../README.md#project-structure).

[← README](../README.md) · [All docs](README.md) · [Drift Monitoring →](02-drift-monitoring.md)
