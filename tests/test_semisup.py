"""Phase 5 tests: pseudo-label selection, guardrails and the production job."""
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select

from common.features import INPUT_FEATURES, TARGET
from semisup.methods import select as select_labels


# ---------------------------------------------------------------- selection rules
def test_agreement_filter_keeps_only_confident_agreeing_rows():
    st = np.array([0.95, 0.95, 0.95, 0.60, 0.02, 0.03])
    ls = np.array([0.90, 0.20, 0.55, 0.95, 0.10, 0.80])   # row1 disagrees, row2 unsure, row3 st unsure
    s = select_labels(st, ls, 0.9, 0.7, positive_rate=0.5)
    accepted = set(s["index"].tolist())
    assert accepted == {0, 4}                           # row 5: st says 0, ls says 1 -> out
    assert dict(zip(s["index"], s["label"])) == {0: 1, 4: 0}


def test_class_balance_follows_reference_ratio():
    st = np.r_[np.full(80, 0.97), np.full(20, 0.02)]      # 80 confident positives, 20 negatives
    ls = st.copy()
    s = select_labels(st, ls, 0.9, 0.7, positive_rate=0.5)
    assert s["n_accepted"] == 40 and s["label"].mean() == 0.5   # limited by the 20 negatives


def test_cap_limits_accepted_rows_and_keeps_most_confident():
    st = np.r_[np.linspace(0.91, 0.99, 50), np.linspace(0.01, 0.09, 50)]
    s = select_labels(st, st, 0.9, 0.7, positive_rate=0.5, max_accept=10)
    assert s["n_accepted"] == 10
    pos = s["index"][s["label"] == 1]
    assert st[pos].min() >= 0.95                          # the 5 most confident positives


def test_nothing_confident_means_nothing_accepted():
    st = np.full(30, 0.6)
    assert select_labels(st, st, 0.9, 0.7)["n_accepted"] == 0


# ---------------------------------------------------------------- production job (SQLite)
FAST = {"lookback_hours": 168, "max_unlabelled": 5000, "min_unlabelled": 100, "reference_sample": 2000,
        "self_training": {"threshold": 0.9, "max_iter": 2},
        "label_spreading": {"n_neighbors": 10, "alpha": 0.2, "threshold": 0.7},
        "require_agreement": True, "max_accept_per_run": 2000, "max_pseudo_ratio": 1.0,
        "balance_to_reference": True, "skip_if_drift_status": ["critical"], "pseudo_label_weight": 0.5}


@pytest.fixture()
def engine(tmp_path):
    from app.db import metadata as app_meta
    from monitoring.store import metadata as drift_meta
    from semisup.store import metadata as pl_meta
    e = create_engine(f"sqlite:///{tmp_path / 'ssl.db'}")
    for m in (app_meta, drift_meta, pl_meta):
        m.create_all(e)
    return e


def add_predictions(engine, n=400, human_rate=0.2, seed=0):
    from app.db import predictions
    test = pd.read_parquet("data/processed/test.parquet").sample(n, random_state=seed)
    rng = np.random.default_rng(seed)
    now = datetime.now(timezone.utc)
    rows = []
    for _, r in test.iterrows():
        human = rng.random() < human_rate
        rows.append({**{k: float(r[k]) for k in INPUT_FEATURES}, "request_id": str(uuid.uuid4()),
                     "created_at": now, "model_version": "vtest", "probability": 0.5, "prediction": 1,
                     "label": int(r[TARGET]) if human else None, "label_source": "human" if human else None})
    with engine.begin() as conn:
        conn.execute(predictions.insert(), rows)
    return rows


def set_drift(engine, status):
    from monitoring.store import drift_runs
    with engine.begin() as conn:
        conn.execute(drift_runs.insert().values(run_at=datetime.now(timezone.utc), status=status))


def test_job_skips_with_too_little_data(engine):
    from semisup.pseudo_label import run
    add_predictions(engine, n=50)
    s = run(FAST, engine=engine, push=False)
    assert s["status"] == "insufficient_data"


def test_job_adds_balanced_pseudo_labels_once(engine):
    from semisup.pseudo_label import run
    from semisup.store import pseudo_labels
    rows = add_predictions(engine, n=400)
    unlabelled = {r["request_id"] for r in rows if r["label"] is None}
    s = run(FAST, engine=engine, push=False)
    assert s["status"] == "ok" and s["n_accepted"] > 0
    assert 0.4 <= s["positive_share"] <= 0.6                      # balanced like the training data
    with engine.connect() as conn:
        ids = {r for (r,) in conn.execute(select(pseudo_labels.c.request_id))}
    assert ids <= unlabelled                                       # never a human-labelled row
    again = run(FAST, engine=engine, push=False)                   # nothing new to label
    assert again["status"] == "insufficient_data"


def test_job_skips_when_drift_is_critical(engine):
    from semisup.pseudo_label import run
    add_predictions(engine, n=400)
    set_drift(engine, "critical")
    s = run(FAST, engine=engine, push=False)
    assert s["status"] == "skipped_drift"
    set_drift(engine, "ok")
    assert run(FAST, engine=engine, push=False)["status"] == "ok"


def test_ratio_cap_blocks_flooding(engine):
    from semisup.pseudo_label import run
    add_predictions(engine, n=400)
    tiny = {**FAST, "max_pseudo_ratio": 0.0}                       # no room for pseudo-labels
    s = run(tiny, engine=engine, push=False)
    assert s["status"] == "ok" and s["n_accepted"] == 0


def test_training_set_human_label_wins_over_pseudo(engine):
    from app.db import predictions
    from semisup.store import pseudo_labels
    from semisup.training_set import build
    rows = add_predictions(engine, n=10, human_rate=0.0)
    rid = rows[0]["request_id"]
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(pseudo_labels.insert(), [{"request_id": r["request_id"], "label": 1, "confidence": 0.9,
                                               "created_at": now} for r in rows[:3]])
        # a doctor later confirms row 0 as healthy
        conn.execute(predictions.update().where(predictions.c.request_id == rid)
                     .values(label=0, label_source="human"))
    ref = pd.read_parquet("data/processed/train.parquet").head(100)
    df = build(engine, ref, pseudo_weight=0.5)
    assert (df.source == "human").sum() == 1 and (df.source == "pseudo").sum() == 2
    assert set(df.loc[df.source == "pseudo", "sample_weight"]) == {0.5}
    # the patient's body weight must survive untouched (regression test for the "weight" clash)
    assert np.array_equal(df.loc[df.source == "training", "weight"].to_numpy(), ref["weight"].to_numpy())
    assert (df.loc[df.source != "training", "weight"] > 20).all()
    with engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(pseudo_labels)).scalar() == 3


def test_new_model_version_rescans_rejected_rows(engine, monkeypatch):
    """Rows rejected by one model version are looked at again by the next one."""
    from semisup import pseudo_label as pl
    add_predictions(engine, n=400)
    first = pl.run(FAST, engine=engine, push=False)
    assert first["status"] == "ok"
    assert pl.run(FAST, engine=engine, push=False)["status"] == "insufficient_data"   # same model
    import json
    real = json.loads(pl.META.read_text())
    monkeypatch.setattr(pl.Path, "read_text", lambda self, *a, **k: json.dumps({**real, "model_version": "vnext"})
                        if self == pl.META else pl.Path.read_bytes(self).decode())
    again = pl.run(FAST, engine=engine, push=False)                                   # new model
    assert again["status"] == "ok"
    assert again["n_unlabelled"] == first["n_unlabelled"] - first["n_accepted"]


def test_insert_ignore_skips_duplicates(engine):
    from semisup.pseudo_label import insert_ignore
    from semisup.store import pseudo_label_seen
    row = {"request_id": "r1", "model_version": "v1", "run_id": 1}
    with engine.begin() as conn:
        conn.execute(insert_ignore(pseudo_label_seen, engine), [row])
        conn.execute(insert_ignore(pseudo_label_seen, engine), [row, {**row, "request_id": "r2"}])
    with engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(pseudo_label_seen)).scalar() == 2


def test_training_set_has_numeric_features_without_pseudo_labels(engine):
    """Regression: concat with an empty part made categorical columns 'object' -> encoder crash."""
    from semisup.training_set import build
    ref = pd.read_parquet("data/processed/train.parquet").head(50)
    df = build(engine, ref, include_human=True, include_pseudo=True)     # empty DB: no human, no pseudo
    assert all(df[c].dtype.kind == "f" for c in INPUT_FEATURES)
