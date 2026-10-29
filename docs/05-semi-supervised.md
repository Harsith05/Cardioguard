[← Containerization](04-containerization.md) · [All docs](README.md) · [Self Healing →](06-self-healing.md)

# Phase 5: Semi-supervised learning (pseudo-labels)

Most production patients never get a confirmed diagnosis. Phase 5 turns confident model
predictions on those patients into **pseudo-labels** for retraining. Guardrails stop the
model from teaching itself its own mistakes.

Analogy: two students mark a pile of practice questions. A mark is kept only when **both are
confident and give the same answer**.

## How a pseudo-label is accepted (`semisup/`)

| Step | Rule |
|---|---|
| 1. Drift guardrail | skip the run if the latest drift check is **critical** (model confidence is unreliable then) |
| 2. Candidates | unconfirmed predictions from the last 7 days, valid inputs only, **not yet scored by this model version** |
| 3. Self-training | scikit-learn `SelfTrainingClassifier` around the production LightGBM pipeline; needs P ≥ 0.9 or ≤ 0.1 |
| 4. Label spreading | scikit-learn `LabelSpreading` (k-NN graph, k = 10, scaled features); needs confidence ≥ 0.7 |
| 5. Agreement filter | both confident **and** same label |
| 6. Class balance | accepted in the training data's disease ratio (~50/50) |
| 7. Caps | ≤ 2,000 per run; pseudo-labels in total ≤ 1× real labels |

Pseudo-labels live in their **own table** (`pseudo_labels`), never in `predictions.label`:
- a human diagnosis can never be overwritten;
- the drift job's accuracy check uses human labels only;
- when a doctor later confirms a patient, the human label wins.

Other safeguards:
- **One run at a time**, enforced with a PostgreSQL advisory lock. A second run returns `busy`.
- **Duplicate-safe inserts**.

Retraining weights each label source differently:

| Source | Weight |
|---|---|
| Training data | 1.0 |
| Human diagnoses | 1.0 |
| Pseudo-labels | 0.5 |

## Run it

```bash
# runs automatically after each drift check (drift-scheduler, every DRIFT_EVERY_MINUTES)
docker compose exec api python -m semisup.pseudo_label          # or the button on /monitoring

docker compose --profile tools run --rm trainer python -m semisup.compare_retrain   # safe to use them?
docker compose --profile tools run --rm trainer python -m training.train --include-production
docker compose --profile tools run --rm trainer python -m semisup.experiment         # paper experiment (~10-20 min)
```

Monitoring:
- **/monitoring page:** a "Pseudo-labels for unconfirmed patients" section showing how patients are filtered step by step.
- **Grafana:** a "Semi-supervised learning" row with status, accepted, acceptance rate, disagreement, total, and history.
- **Alert `LowPseudoLabelAcceptance`:** fires when the acceptance rate drops below 5%. Normal traffic gives about 13%.

## Result 1: label-fraction experiment (paper)

`python -m semisup.experiment` hides most training labels, then compares methods on the
held-out test set (5 seeds, mean ± sd). Full table: `reports/ssl/summary.md`; figure:
`reports/ssl/label_fraction.png`.

| Labels | Supervised only | Self-training | Label spreading | **Agreement filter** | All labels |
|---|---|---|---|---|---|
| 1% (475) | 0.7450 ± 0.013 | 0.7656 | 0.7343 | **0.7690 ± 0.011** (+0.024) | 0.8024 |
| 5% | 0.7641 ± 0.005 | 0.7829 | 0.7762 | **0.7849 ± 0.003** (+0.021) | 0.8024 |
| 10% | 0.7808 ± 0.002 | 0.7932 | 0.7882 | **0.7938 ± 0.001** (+0.013) | 0.8024 |
| 20% | 0.7921 ± 0.002 | 0.7962 | 0.7949 | 0.7957 ± 0.001 (+0.004) | 0.8024 |

- **The agreement filter is best or tied at every label budget.** It recovers 42%, 54%, 60% and 35% of the gap to full supervision at 1%, 5%, 10% and 20% labels.
- **It produces the most accurate pseudo-labels:** 0.747–0.844, against 0.637–0.694 for label spreading alone. Label spreading alone even **hurts** at 1% labels (−0.011).
- **Self-training alone becomes overconfident.** At 1% labels, 85% of patients pass its "≥ 90% sure" bar, but only 74% of those labels are right. This is confirmation bias, and the agreement filter limits it.
- **There's a trade-off between quantity and quality.** Capping pseudo-labels at 1× the real labels gives very accurate labels (0.87–0.89), but too few of them to help when labels are scarce. In production, real labels are plentiful (47k), so the cap there is a safety limit, not a bottleneck.

## Result 2: production run (Docker, simulated traffic)

| Measure | Value |
|---|---|
| Acceptance on normal traffic | about 13% (4 runs: 13.1%, 12.6%, 13.3%, 13.2%) |
| Where patients drop out (one run of 2,388) | 397 ≥ 90% sure (self-training) → 374 also agreed → 312 after 50/50 balance |
| Methods disagree on | about 14% |
| Run time on 2 CPU cores | about 20 s |
| Drift critical | run skipped (scheduler and button), status pushed to Prometheus |

Acceptance is much lower than in the experiment for a good reason. Here self-training starts
from 10,000 real labels, so it stays **well calibrated** and is rarely 90% sure. With 1–5% of
the labels it became overconfident.

**Retrain with vs without pseudo-labels** (`semisup.compare_retrain`). Both models were scored on data neither trained on:

| Evaluated on | AUC without | AUC with 1,249 pseudo-labels | Calibration (mean risk vs true) |
|---|---|---|---|
| Validation set, never sent as traffic (10,190) | 0.8018 | **0.8022** | 0.499 / 0.499 vs 0.497 |
| Human-confirmed production cases (2,437) | 0.8084 | **0.8093** | 0.506 / 0.506 vs 0.508 |

→ **Pseudo-labels are safe: they match or slightly beat training without them.** With 47k real
labels they don't add much, which is exactly what the experiment predicts. Their value is
when confirmed diagnoses are scarce.

**Caveat:** the traffic simulator draws patients from the test set. Once production data is
used for training (`--include-production`), test metrics are optimistic, so judge such models
on the **validation** set. `train.py` prints this warning and records `test_may_overlap_production`.

## Bugs found by testing (fixed)

1. **Training data corruption.** The sample-weight column was called `weight`, which is also the
   patient's body weight. It silently overwrote every body weight with 1.0 / 0.5, and pseudo-labels
   looked harmful (AUC −0.006, risk biased upward). Found when "training set without pseudo-labels"
   didn't reproduce the baseline. It's now `sample_weight`, with a regression test.
2. **Acceptance rate drifting down by itself.** Rejected patients were re-scored every hour, so the
   pool filled with hard cases. Each model version now scores a patient once
   (`pseudo_label_seen`); a new model version looks again.
3. **Race between the scheduler and the button.** Both scored the same patients and one run crashed
   on a duplicate. Fixed with an advisory lock and duplicate-safe inserts.
4. **Threads fighting over the CPU.** LightGBM's OpenMP threads in two processes slowed each other
   down more than 10x on 2 cores. The image now sets `OMP_WAIT_POLICY=PASSIVE`.
5. **Package name.** The plan's `ssl/` folder would have shadowed Python's built-in `ssl` module,
   which every HTTPS and database connection needs, so it's `semisup/`.

[← Containerization](04-containerization.md) · [All docs](README.md) · [Self Healing →](06-self-healing.md)
