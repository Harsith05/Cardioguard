"""Phase 5 "done when" check: does retraining WITH pseudo-labels match or beat WITHOUT?

  A  training data only                      (the current model's recipe)
  B  training data + production pseudo-labels (weight from config)

Both are scored on data neither model trained on:
  * the VALIDATION set: never sent as simulated traffic, so B never saw these patients.
    (The simulator draws patients from the test set, so the test set is NOT clean for B.)
  * human-confirmed production cases (their real labels are left out of training here).

    python -m semisup.compare_retrain
    docker compose exec api python -m semisup.compare_retrain
"""
from __future__ import annotations

import os

import pandas as pd
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score
from sqlalchemy import create_engine, select

from common.features import INPUT_FEATURES, TARGET
from semisup.methods import base_model
from semisup.pseudo_label import load_config
from semisup.training_set import WEIGHT_COL, build


def main():
    engine = create_engine(os.getenv("DATABASE_URL", "postgresql+psycopg2://cardio:cardio@localhost:5432/cardio"))
    train = pd.read_parquet("data/processed/train.parquet")
    val = pd.read_parquet("data/processed/val.parquet")
    w = load_config()["pseudo_label_weight"]
    B = build(engine, train, pseudo_weight=w, include_human=False)
    n_pseudo = int((B.source == "pseudo").sum())
    if n_pseudo == 0:
        raise SystemExit("No pseudo-labels yet. Run: python -m semisup.pseudo_label")

    from app.db import predictions as p
    with engine.connect() as conn:
        human = pd.DataFrame(conn.execute(select(*[p.c[c] for c in INPUT_FEATURES], p.c.label)
                                          .where(p.c.label_source == "human")).mappings().all())

    a = base_model().fit(train[INPUT_FEATURES], train[TARGET])
    b = base_model().fit(B[INPUT_FEATURES], B[TARGET], model__sample_weight=B[WEIGHT_COL].to_numpy())

    sets = [("Validation set (never sent as traffic)", val[INPUT_FEATURES], val[TARGET])]
    if len(human) >= 50 and human["label"].nunique() == 2:
        sets.append(("Human-confirmed production cases", human[INPUT_FEATURES], human["label"].astype(int)))

    print(f"A: training data only ({len(train):,} rows)")
    print(f"B: training data + {n_pseudo:,} pseudo-labels (weight {w})\n")
    verdict = True
    for name, X, y in sets:
        pa, pb = a.predict_proba(X)[:, 1], b.predict_proba(X)[:, 1]
        auc_a, auc_b = roc_auc_score(y, pa), roc_auc_score(y, pb)
        verdict &= auc_b >= auc_a - 0.005          # "matches" = within half an AUC point
        print(f"{name} ({len(y):,} rows)")
        print(f"  {'':10s} {'A':>8s} {'B':>8s} {'B - A':>8s}")
        for metric, fa, fb in [("ROC-AUC", auc_a, auc_b),
                               ("Recall", recall_score(y, pa >= .5), recall_score(y, pb >= .5)),
                               ("Precision", precision_score(y, pa >= .5), precision_score(y, pb >= .5)),
                               ("Brier", brier_score_loss(y, pa), brier_score_loss(y, pb))]:
            print(f"  {metric:10s} {fa:8.4f} {fb:8.4f} {fb - fa:+8.4f}")
        print(f"  mean predicted risk: A {pa.mean():.3f}, B {pb.mean():.3f} (true rate {y.mean():.3f})\n")
    print("\nResult:", "B matches or beats A - pseudo-labels are safe to use." if verdict
          else "B is worse than A - do not retrain with these pseudo-labels.")


if __name__ == "__main__":
    main()
