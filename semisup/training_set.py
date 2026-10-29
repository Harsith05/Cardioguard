"""Build the retraining set from all label sources (Phase 5; used by Phase 6).

| Source            | Weight | Notes                                              |
|-------------------|--------|----------------------------------------------------|
| training data     | 1.0    | the original labelled patients                     |
| human diagnoses   | 1.0    | from /feedback                                     |
| pseudo-labels     | 0.5    | only where no human label exists (humans win)      |

Evaluation must never use pseudo-labels: score models on the held-out test set and on
human-confirmed production cases only.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import select

from common.features import INPUT_FEATURES, TARGET

# NOT "weight": that is a patient feature (body weight in kg). An earlier version used
# "weight" here and silently overwrote every patient's body weight with 1.0 / 0.5.
WEIGHT_COL = "sample_weight"


def build(engine, reference: pd.DataFrame, pseudo_weight: float = 0.5, include_human: bool = True,
          include_pseudo: bool = True) -> pd.DataFrame:
    from app.db import predictions as p
    from semisup.store import pseudo_labels as pl

    parts = [reference[INPUT_FEATURES + [TARGET]].assign(source="training", **{WEIGHT_COL: 1.0})]
    feats = [p.c[c] for c in INPUT_FEATURES]
    with engine.connect() as conn:
        if include_human:
            h = pd.DataFrame(conn.execute(select(*feats, p.c.label).where(p.c.label_source == "human"))
                             .mappings().all(), columns=INPUT_FEATURES + ["label"])
            parts.append(h.rename(columns={"label": TARGET}).assign(source="human", **{WEIGHT_COL: 1.0}))
        if include_pseudo:
            q = (select(*feats, pl.c.label).join(pl, pl.c.request_id == p.c.request_id)
                 .where(p.c.label.is_(None)))                     # a human label always wins
            ps = pd.DataFrame(conn.execute(q).mappings().all(), columns=INPUT_FEATURES + ["label"])
            parts.append(ps.rename(columns={"label": TARGET}).assign(source="pseudo", **{WEIGHT_COL: pseudo_weight}))
    df = pd.concat(parts, ignore_index=True)
    assert df[INPUT_FEATURES].notna().all().all(), "missing feature values in the training set"
    df[INPUT_FEATURES] = df[INPUT_FEATURES].astype(float)   # empty parts would otherwise give "object" columns
    df[TARGET] = df[TARGET].astype(int)
    return df
