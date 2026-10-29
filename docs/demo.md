[← README](../README.md) · [All docs](README.md)

# Demo walkthrough (about 20 minutes)

This walkthrough shows the whole system on a fresh stack, from normal traffic through realistic
drift to automatic retraining, a canary release and promotion. The commands work on Windows
PowerShell and on bash (replace the PowerShell `for` loop with a bash loop).

Keep **two terminals** open in the project folder: one for commands, one for traffic that must
keep running during the canary.

## 0. Faster settings for a demo

Production defaults are a 30-minute canary and a 24-hour cool-down. For a live demo, in
`mlops/config.yaml` set:

```yaml
trigger:
  cooldown_hours: 0
canary:
  wait_minutes: 3
```

Then rebuild Airflow:

```bash
docker compose build airflow-init && docker compose up -d
```

Set the values back afterwards.

## 1. Check the stack

| Open | Expect |
|---|---|
| http://localhost:8000/health | `"status":"ok"`, deployment `stable` |
| http://localhost:8000/monitoring | Self-healing: **Stable**, champion = MLflow v1 |
| http://localhost:5000 → Models | `cardioguard` v1 **@champion** |
| http://localhost:8080 | DAG `cardio_self_healing` (admin / admin) |

Confirm the champion is medically sound:

```bash
docker compose exec api python -c "from mlops import clinical, pipeline as p; m, meta = p.load_champion(p.engine()); print(meta['model_version'], clinical.check(m, p.load_config()['clinical'])['failures'] or 'clinical checks: all passed')"
```

## 2. Prediction and explanation

Open http://localhost:8000. Enter a patient (e.g. age 55, BP 150/95, cholesterol "well above
normal") and read the risk and the **"Why this result"** chart: blood pressure pushes the risk up.

## 3. Normal traffic: baseline

```bash
python -m simulate.traffic --scenario normal --n 2000 --label-rate 0.3
```

On /monitoring, click **Run drift check**. Expect **OK** for every check, including
"Predicted vs actual risk" (about 50% vs 50%). Then click **Run pseudo-labelling**: about 10–15%
of unconfirmed patients get a pseudo-label.

## 4. Realistic concept drift: a new guideline

A new guideline diagnoses disease earlier at BP ≥ 130/80. The inputs don't change, but the
outcomes do.

```bash
python -m simulate.traffic --scenario guideline --n 2500 --label-rate 0.5 --seed 21
```
Click **Run drift check**. Expect **Critical**: accuracy (AUC) still looks OK, but **"Predicted
vs actual risk"** says about 70% are sick while the model predicts about 48%.

```bash
python -m simulate.traffic --scenario guideline --n 2500 --label-rate 0.5 --seed 22
```
Click **Run drift check** again. Expect **Critical** and **"Retraining recommended"**. The drift
job now starts the Airflow DAG by itself.

## 5. Watch it heal

In Airflow, open `cardio_self_healing` → the new run → **Graph**:

| Task | What it does |
|---|---|
| `check_and_diagnose` | `retrain_concept`: the inputs are normal but the risk level is wrong |
| `train_candidates` | 3 strategies × LightGBM/XGBoost, all with medical constraints |
| `promotion_gates` | Statistics, fairness and 18 clinical sanity checks |
| `start_canary` | 10% of patients go to the new model |

**As soon as `start_canary` starts**, run canary traffic in the second terminal:

```powershell
for ($i=1; $i -le 80; $i++) { python -m simulate.traffic --scenario guideline --n 80 --label-rate 0 --seed $i; Start-Sleep 3 }
```

```bash
for i in $(seq 1 80); do python -m simulate.traffic --scenario guideline --n 80 --label-rate 0 --seed $i; sleep 3; done
```

After `wait_minutes`: **Promoted**. The API switches models without a restart, MLflow v2
becomes **@champion**, and Grafana marks the promotion on every chart.

If the panel says **Waiting for approval**, the new model forgets a lot of the old pattern, so a
person must confirm. Enter your name and a reason, then click **Approve canary release**.

## 6. Prove it improved

```bash
python -m simulate.traffic --scenario guideline --n 1500 --label-rate 0.5 --seed 77
```

Run a drift check. "Predicted vs actual risk" should shrink from about 22 points to about
7 points, and the AUC should rise to about 0.82.

## 7. Other paths worth showing

| Scenario | Command | Expected outcome |
|---|---|---|
| Device fault (BP +20/+10) | `--scenario bp_shift` | Diagnosis `retrain_needs_approval`: a person decides; often rejected by the gates |
| Older patients, model still accurate | `--scenario aging` | Only the drift baseline is refreshed; no retraining |
| Medically impossible labels | `--scenario concept` | The constrained models can't learn it; at most a small, plausible adjustment |
| Emergency brake | **Roll back now** during a canary | 100% back on the champion within seconds; the DAG ends `rolled_back` |
| Manual run | Airflow → **Trigger** (`force`, `diagnosis`, `canary_minutes`) | Any path on demand |

## Where to read the numbers

| Place | Shows |
|---|---|
| MLflow run `candidate-v…` | `holdout_*` (old patients), `recent_*` (new-pattern patients), `clinical_failures`, strategy |
| Airflow `train_candidates` → XCom | All 6 candidates side by side, candidate vs champion |
| Airflow `promotion_gates` → XCom | Every gate: value, limit, passed |
| /monitoring → Self-healing | Event log of the whole run |
| Grafana → Self-healing row | Rollout status, traffic per model version, events |

[← README](../README.md) · [All docs](README.md)
