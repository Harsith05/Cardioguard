[← Semi Supervised](05-semi-supervised.md) · [All docs](README.md) · [Demo walkthrough →](demo.md)

# Phase 6: Self-healing (MLflow + Airflow, canary, rollback)

When the drift job says "retrain", the system now fixes itself, but **carefully**. It first
works out *what kind* of drift it is, because each kind needs a different fix. A new model
must then pass checks, sometimes a person, and a live trial before it replaces the old one.

Analogy: a hospital replacing a drug. Lab tests (gates) → the ethics board signs off when in
doubt (human approval) → a small trial group (canary) → full use, or stop the trial (rollback).

## The pipeline (`mlops/pipeline.py`, run by Airflow as `dags/cardio_self_healing.py`)

```
trigger ─▶ diagnose ─┬─ stop_bad_data ───────▶ stop + alert (fix the data, not the model)
                     ├─ no_action ───────────▶ stop
                     ├─ refresh_reference ───▶ new drift baseline, no retraining
                     └─ retrain_concept / retrain_needs_approval
                            ▼
      validate ▶ pseudo-label ▶ train LightGBM + XGBoost ▶ promotion gates ──fail──▶ reject
                                                                │ pass
                 (needs approval) ▶ wait for a person ──reject / 24 h timeout──▶ stop
                                                                ▼
                               canary: 10% of patients, 30 min ▶ verify ──bad──▶ roll back
                                                                │ good
                                                   promote ▶ refresh drift baseline
```

### Step 1: which kind of drift? (`diagnose`)

| Invalid inputs | Inputs drifted | Accuracy on confirmed cases | Diagnosis | What happens |
|---|---|---|---|---|
| critical | any | any | `stop_bad_data` | stop + alert; retraining would learn the errors |
| ok | no | dropped | `retrain_concept` | retrain automatically, recent cases weighted ×10 |
| ok | yes | dropped / unknown | `retrain_needs_approval` | retrain, but **a person must approve**: it may be a mis-calibrated device, not new patients |
| ok | yes | held | `refresh_reference` | patients changed but the model is fine: update the drift baseline only |
| ok | no | held | `no_action` | nothing |

### Step 1b: how to retrain under concept drift (strategy selection)

"New-pattern cases" are the confirmed diagnoses since the drift began, meaning since the window start of the current streak of critical drift checks. The newest 30% of them are held out to judge the candidates. The pipeline then trains **3 strategies × 2 model types** and keeps the best on those held-out cases:

| Strategy | Training data | When it wins |
|---|---|---|
| `weighted_x10` | all data, new cases ×10 | mild change, few new labels |
| `weighted_x50` | all data, new cases ×50 | clear change |
| `new_pattern_only` | only new-pattern cases (≥ 300) | the old meaning no longer holds |

Offline test (simulator: diagnosis flipped for patients under 50, LightGBM):

| New-pattern labels | ×10 (old method) | ×50 | new pattern only | Champion |
|---|---|---|---|---|
| 1,300 | 0.571 | 0.648 | **0.739** | 0.512 |
| 3,000 | 0.642 | 0.748 | **0.766** | 0.512 |

(AUC on held-out new-pattern patients.) The old fixed ×10 weighting recovered only a fraction of
what the labels allow.

**Forgetting needs a person.** A model that fits the new pattern well usually does worse on the
original validation set, which is the old truth. Under `retrain_concept`, losing more than
0.05 AUC there no longer rejects the model. Instead it is **sent for human approval**, with
the reason ("forgets the old pattern by −0.27 AUC: approve only if the old pattern really no
longer holds"). Losing more than 0.35 is still rejected outright.

### Step 1c: medical knowledge (so statistics can't promote nonsense)

Statistical gates only ask "does the model score well on recent labels?". With the flipped-label
stress test, the best-scoring model said a 45-year-old smoker with BP 160/100 had **12%** risk.
It scored well, but it was medically absurd. Two safeguards now stop this:

| Safeguard | What it does | Cost |
|---|---|---|
| **Monotonic constraints** (`training.monotone: true`) | Retrained LightGBM/XGBoost may only **raise** the risk when age, systolic/diastolic BP, mean arterial pressure, cholesterol or glucose rise. Cholesterol and glucose are kept as ordered numbers (1 < 2 < 3), and pulse pressure is dropped, so "higher BP → lower risk" is impossible by construction. | Validation AUC 0.8018 → **0.801** |
| **Clinical sanity gate** (`mlops/clinical.py`, cases in `config.yaml → clinical`) | 6 textbook patients with a required risk range, plus 12 "same patient, one risk factor worse" pairs (risk may not fall more than 0.02). Any failure → **rejected**, however good the statistics. | none |

Your current champion (trained before this change) fails 2 of the 18 checks: its risk *falls*
when glucose goes from normal to high. That's a quirk of the dataset that the constrained model
can't learn. To retrain the base model with the constraints:
`python -m training.train --monotone`.

### Step 1d: realistic concept drift and calibration

The new `guideline` scenario (`simulate.traffic --scenario guideline`) models a real change: a
new guideline diagnoses disease earlier, so 60% of undiagnosed patients with BP ≥ 130/80 are now
diagnosed. The **ranking** barely changes (champion AUC still 0.806), but the risk **level** is
wrong: 73% of confirmed patients are sick, while the model predicts 50%. So:

- a new drift check, **`calibration`** (observed sick share vs mean predicted risk, critical above a 10-point gap, significant at 99%), detects it, where the AUC check doesn't;
- the concept gate accepts **AUC +0.02 *or* Brier −0.01**, because the fix is calibration, not ranking;
- the canary compares the live risk shift with the shift **expected from confirmed cases**, so a correctly higher risk isn't rolled back.

### Result: before vs after (`python -m mlops.experiment` → `reports/self_healing/summary.md`)

| Scenario | New-pattern labels | Setting | Model chosen | New-world AUC (champion) | Brier (champion) | Old-world AUC | Clinical failures | Outcome |
|---|---|---|---|---|---|---|---|---|
| Flipped labels (absurd) | 1,300 | before | xgboost / new only | 0.742 (0.515) | 0.208 (0.308) | 0.542 | **9** | needs approval: only a person can stop it |
| | 1,300 | **after** | lightgbm / ×10 | 0.553 (0.515) | 0.277 (0.308) | **0.792** | **0** | canary: adapts only within medical bounds |
| | 3,000 | before | xgboost / new only | 0.777 (0.515) | 0.189 (0.308) | 0.535 | **10** | needs approval |
| | 3,000 | **after** | lightgbm / ×10 | 0.583 (0.515) | 0.253 (0.308) | **0.778** | **0** | canary |
| Guideline (realistic) | 1,300 | before | xgboost / ×10 | 0.776 (0.760) | 0.199 (0.223) | 0.799 | **3** | canary: unsafe model auto-promoted |
| | 1,300 | **after** | xgboost / ×10 | 0.778 (0.760) | 0.199 (0.223) | 0.796 | **0** | canary |
| | 3,000 | before | xgboost / ×10 | 0.786 (0.760) | 0.183 (0.223) | 0.795 | **3** | canary: unsafe |
| | 3,000 | **after** | xgboost / ×50 | 0.788 (0.760) | **0.163** (0.223) | 0.777 | **0** | canary |

- **Absurd drift:** the constrained pipeline refuses to learn the impossible rule. It keeps the old knowledge and adapts only the plausible part (a higher overall risk).
- **Realistic drift:** the same improvement as before (Brier −27%), with no medically wrong answers.
- **Detection:** the guideline drift is invisible to the AUC check (0.806) and caught by calibration (73% vs 50%).

### Step 2: promotion gates (candidate vs current champion)

| Gate | Normal limit | `retrain_concept` limit |
|---|---|---|
| AUC drop on the validation set (old population) | ≤ 0.01 | ≤ 0.35; above 0.05 → **a person must approve** |
| AUC on the newest 30% of confirmed cases | drop ≤ 0.01 | **gain ≥ 0.02, or Brier drop ≥ 0.01** |
| Recall drop (missed patients) | ≤ 0.02 | ≤ 0.05 |
| Brier score increase (calibration) | ≤ 0.01 | ≤ 0.03 |
| Recall gap women vs men (fairness) | ≤ 0.05 | ≤ 0.05 |
| **Clinical sanity** (6 textbook patients + 12 "worse risk factor" pairs) | 0 failures | 0 failures |

The newest 30% of confirmed cases are **held out** from training, so the "recent" gate is honest.

### Step 3: canary release (`app/model_loader.py`)

- The `model_deployment` row in Postgres is the single switch.
- Every API container re-reads it every `DEPLOYMENT_POLL_SECONDS` and swaps models **in memory, without a restart**.
- During a canary, each **patient** (also inside batch requests) goes to the candidate with a 10% chance.
- If the candidate throws an error, that request falls back to the champion, and the error is counted.
- After 30 minutes the pipeline compares the two models on live traffic:

| Check | Roll back if |
|---|---|
| Evidence | fewer than 50 canary predictions |
| Errors | > 1% of candidate requests |
| Speed | p95 model time > 300 ms, or > 2× the champion's (only above 20 ms: below that, batch size dominates) |
| Behaviour | the live risk shift vs the champion differs by > 0.10 from the shift expected on confirmed recent cases |

Safety nets:
- **Roll back now** button on /monitoring (the pipeline notices and stops);
- if any Airflow task crashes, `cleanup_after_failure` ends a canary this run started;
- `max_active_runs=1`;
- a 24 h cool-down after a promotion or rollback.

### Model registry (MLflow)

Every candidate becomes a version of the registered model `cardioguard`:
- alias **`champion`**: the model serving traffic;
- alias **`candidate`**: the model on trial;
- tags: `status` (`unverified` → `champion` / `retired` / `rolled_back`), plus the diagnosis, gate results and the Airflow run id;
- runs log the metrics, the training-data composition (training / human / pseudo rows) and `model_meta.json`.

Models are saved with **skops** instead of pickle. Loading refuses any Python type that isn't listed as trusted, and the registry only trusts `common.`, `sklearn.`, `lightgbm.`, `xgboost.`, `numpy.`, `scipy.` and `pandas.` types.

### Why Airflow runs the ML code in a separate Python

The DAG uses `@task.external_python`. Airflow's own Python schedules the tasks, and the steps
run in `/opt/cardio`, a venv with the **same pinned libraries as the API image**. So
Airflow's dependencies never clash with scikit-learn/LightGBM, and a model trained by Airflow
loads in the API with identical versions.

## Start it

```bash
docker compose up -d --build        # now also: db-init, mlflow, airflow-init, airflow, airflow-scheduler, airflow-dag-processor
```

| Service | URL | Login |
|---|---|---|
| Monitoring + Self-healing panel | http://localhost:8000/monitoring | – |
| MLflow (models, versions, aliases) | http://localhost:5000 | – |
| Airflow (DAG `cardio_self_healing`) | http://localhost:8080 | admin / admin (`AIRFLOW_PASSWORD`) |

On first start:
- `db-init` creates the `mlflow` and `airflow` databases. This also works on your existing Postgres volume.
- `bootstrap` registers `models/model.joblib` as the first champion.

## Use it

| How | Command |
|---|---|
| Automatically | the drift scheduler starts the DAG when the drift check recommends retraining |
| By hand in Airflow | open `cardio_self_healing` → Trigger. Parameters: `force`, `diagnosis`, `canary_minutes`, `approval_minutes`, `config` |
| Without Airflow (same steps) | `docker compose exec api python -m mlops.pipeline --force --canary-minutes 2` |
| Approve / reject / roll back | buttons on /monitoring, or `POST /deployment/approve`, `/reject`, `/rollback` |

To try it quickly, push concept drift and shorten the canary:

```bash
python -m simulate.traffic --scenario concept --n 2000 --label-rate 0.3
# then trigger the DAG in Airflow with {"force": true, "canary_minutes": 3}
python -m simulate.traffic --scenario normal --n 2000      # traffic for the canary to judge
```

If `ADMIN_TOKEN` is set in `.env`, approve/reject/rollback need the header `X-Admin-Token`.
The page asks for it.

## Monitoring

- **/monitoring:** a "Self-healing" panel showing the status, the traffic split (champion vs candidate), the champion and candidate versions, Approve / Reject / Roll back, and an event log.
- **Grafana:** a "Self-healing (MLflow + Airflow)" row with rollout status, % on the candidate, the candidate version, the last pipeline outcome, requests and p95 model time **per model version**, and an events table. Promotions and rollbacks also appear as markers on every time chart.
- **Alerts:**
  - `RetrainingAwaitingApproval` (30 min);
  - `CanaryRunningTooLong` (2 h);
  - `ModelRolledBack` (for 1 h after a rollback);
  - `SelfHealingBlocked` (critical: bad data, failed gates or a crashed task, so the drifted model is still serving).

  Unit tests are in `rules_test.yml`.

## Results (each path run end to end)

| Scenario | Diagnosis | What happened | Correct? |
|---|---|---|---|
| Concept drift (same inputs, new disease relation): AUC fell to 0.557 | `retrain_concept` | Candidate: recent AUC 0.548 → **0.580** (+0.030), old population −0.002. All 5 gates passed. Canary of 131 patients was as fast and as stable as the champion. **Promoted, hot-swapped without a restart.** | ✓ (repair is partial: the recent AUC is still low, see below) |
| BP device fault (+15 mmHg on every reading) | `retrain_needs_approval` | Candidate **rejected by gates** (AUC drop on recent cases 0.011 > 0.01) | ✓ retraining can't fix a broken device |
| Same, gates relaxed, approved by a person | – | Canary → **rolled back automatically**: "only 29 canary predictions (need 50)" | ✓ no evidence = no promotion |
| Person rejects | – | `rejected_by_person`; champion untouched; candidate never served | ✓ |
| Older patients (age shift), accuracy held (AUC 0.809) | `refresh_reference` | New baseline from the newest 2,000 patients; the next drift check on the same population came back **OK** | ✓ no needless retraining |
| Airflow in Docker, triggered by the drift scheduler with no drift | – | `check_and_diagnose` → `stop` (no action) | ✓ |
| Airflow in Docker, forced retrain | `retrain_concept` | 47,553 training + 1,883 human + 1,249 pseudo-labels; canary 120 vs 1,200 patients → **promoted**; API switched to the new champion within seconds | ✓ |
| Airflow in Docker, approval in the browser | `retrain_needs_approval` | Waiting for approval → *Approve* (name + reason logged) → canary at 10% → *Roll back now* → the DAG noticed and ended `rolled_back` | ✓ |
| Concept drift, **strategy selection** (Docker, 1,235 new-pattern labels) | `retrain_concept` | 6 candidates tried. ×10: recent AUC 0.63–0.64. ×50: 0.73–0.74. **XGBoost new-pattern-only: 0.774** (champion 0.600, **+0.174**). It forgets the old pattern (−0.27), so it was **held for approval**, approved, then went to the canary. | ✓ |
| A task crashes during the canary | – | `cleanup_after_failure` rolled the canary back; the run is marked `failed` | ✓ never stuck at 10% |

**Honest limit (for the paper):** with the original fixed ×10 weighting, retraining only partly
repaired strong concept drift (+0.03 AUC). Trying several strategies and selecting on held-out
new-pattern cases raised the repair to +0.17. The price is forgetting the old pattern, which is
why a person must confirm the old pattern is really gone. Self-healing restores *what the labels
allow*. It cannot make up for missing ground truth.

## Bugs found by testing (fixed)

1. **A crash left a canary running.** The 10% of traffic stayed on an unverified model. Every failure path now rolls back.
2. **The retraining streak counted the old model's runs.** After a promotion, the old champion's critical drift runs triggered the next retraining at once. The streak now counts only the current model's runs.
3. **The refreshed baseline mixed old and new patients.** Using all traffic watered down the new population, so drift came back the next hour. It now uses the newest 2 drift windows only.
4. **Wrong dtypes broke training.** Concatenating an empty pseudo-label table turned the features into `object` dtype, and the encoder crashed (`isnan`). Features are cast to float before training.
5. **Config overrides wiped settings.** Overriding one gate replaced the whole gate block (`KeyError`). Overrides now deep-merge.
6. **The canary could start on the wrong path.** In the DAG, "none failed, at least one succeeded" counted the shared config task as a success, so `start_canary` ran on the stop path. It now depends only on the gate/approval results.
7. **The pseudo-labelling data was missing in Airflow.** Only visible on a run where drift wasn't critical. `data/reference` is now mounted into the Airflow services.
8. **MLflow rejected Docker requests (403).** MLflow 3 blocks unknown `Host` headers, so `--allowed-hosts` lists the service names.
9. **skops refused to load the model.** It found untrusted types. The registry now trusts an explicit list of packages, and refuses to register anything outside it.
10. **The canary saw 0 patients.** Routing was decided per *request*, and the traffic came in batches of 80, so 29 batches at 10% had a 5% chance of all going to the champion, which is what happened. The API now routes each patient independently.
11. **The API hung at startup when MLflow was down.** The MLflow client retried for minutes. Retries are now capped (`MLFLOW_HTTP_REQUEST_MAX_RETRIES=2`), and the API falls back to the file model quickly.
12. **Retraining couldn't learn strong concept drift.** A fixed ×10 weight was too timid. Strategy selection is described above.
13. **Statistics promoted a medically absurd model.** With flipped labels, the best-scoring model gave a 45-year-old hypertensive smoker 12% risk. Added monotonic constraints and a clinical sanity gate.
14. **Realistic concept drift was invisible.** A guideline change keeps the AUC (0.806) but shifts the risk level (73% vs 50%). Added a calibration drift check, a calibration-aware gate and an expected-shift canary check.

[← Semi Supervised](05-semi-supervised.md) · [All docs](README.md) · [Demo walkthrough →](demo.md)
