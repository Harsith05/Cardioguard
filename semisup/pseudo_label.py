"""Pseudo-label unlabelled production predictions (Phase 5). Becomes an Airflow task in Phase 6.

Steps
  1. Guardrail: skip if the latest drift check is critical (confidence is unreliable then).
  2. Load unlabelled predictions from the last 7 days that this model version has not
     scored yet (accepted or not), dropping rows with invalid inputs.
  3. Labelled data: a sample of the training data + every human-confirmed diagnosis.
  4. Self-training and label spreading both score the unlabelled rows.
  5. Agreement filter + class balance + caps -> accepted pseudo-labels.
  6. Save them (pseudo_labels), save the run summary (pseudo_label_runs), push metrics.

    python -m semisup.pseudo_label
    docker compose exec api python -m semisup.pseudo_label
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml
from sqlalchemy import create_engine, desc, func, select, text

from common.features import INPUT_FEATURES, TARGET
from semisup.methods import label_spreading, select as select_labels, self_training
from semisup.store import metadata, pseudo_label_runs, pseudo_label_seen, pseudo_labels

log = logging.getLogger("cardioguard.pseudo_label")
CFG_PATH = Path(__file__).with_name("config.yaml")
META = Path("models/model_meta.json")
REFERENCE_PARQUET = Path("data/reference/reference.parquet")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://cardio:cardio@localhost:5432/cardio")
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "localhost:9091")
STATUS_LEVEL = {"ok": 0, "skipped_drift": 1, "insufficient_data": -1}
LOCK_KEY = 7_005_501            # any constant; identifies this job's PostgreSQL advisory lock


@contextmanager
def single_run(engine):
    """Only one pseudo-label run at a time (scheduler + "Run" button could overlap).

    PostgreSQL advisory lock; yields False if another run holds it. SQLite (tests): no lock.
    """
    if engine.dialect.name != "postgresql":
        yield True
        return
    conn = engine.connect()
    got = False
    try:
        got = bool(conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": LOCK_KEY}).scalar())
        yield got
    finally:
        if got:
            conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": LOCK_KEY})
        conn.close()


def insert_ignore(table, engine):
    """INSERT ... ON CONFLICT DO NOTHING (a second guard against duplicates)."""
    if engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    return insert(table).on_conflict_do_nothing()


def load_config() -> dict:
    return yaml.safe_load(CFG_PATH.read_text())


def _engine(engine=None):
    engine = engine or create_engine(DATABASE_URL, pool_pre_ping=True)
    from app.db import metadata as app_metadata
    app_metadata.create_all(engine)         # predictions table (normally created by the API)
    metadata.create_all(engine)
    return engine


def latest_drift_status(engine) -> str | None:
    from monitoring.store import drift_runs
    from monitoring.store import metadata as drift_metadata
    drift_metadata.create_all(engine)
    with engine.connect() as conn:
        return conn.execute(select(drift_runs.c.status)
                            .where(drift_runs.c.status != "insufficient_data")
                            .order_by(desc(drift_runs.c.run_at)).limit(1)).scalar()


def load_unlabelled(engine, cfg, model_version: str) -> pd.DataFrame:
    """Unlabelled predictions not pseudo-labelled yet and not already scored by this model version."""
    from app.db import predictions as p
    since = datetime.now(timezone.utc) - timedelta(hours=cfg["lookback_hours"])
    labelled = select(pseudo_labels.c.request_id)
    seen = select(pseudo_label_seen.c.request_id).where(pseudo_label_seen.c.model_version == model_version)
    q = (select(p).where(p.c.label.is_(None), p.c.created_at >= since,
                         p.c.request_id.not_in(labelled), p.c.request_id.not_in(seen))
         .order_by(desc(p.c.created_at), desc(p.c.id)).limit(cfg["max_unlabelled"]))
    with engine.connect() as conn:
        df = pd.DataFrame(conn.execute(q).mappings().all())
    if df.empty:
        return df
    from monitoring.drift import quality_issues
    return df[~quality_issues(df)].reset_index(drop=True)


def load_human_labels(engine) -> pd.DataFrame:
    from app.db import predictions as p
    q = select(*[p.c[c] for c in INPUT_FEATURES], p.c.label).where(p.c.label_source == "human")
    with engine.connect() as conn:
        return pd.DataFrame(conn.execute(q).mappings().all(), columns=INPUT_FEATURES + ["label"])


def count_pseudo(engine) -> int:
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(pseudo_labels)).scalar() or 0)


def run(cfg: dict | None = None, engine=None, push: bool = True) -> dict:
    cfg = cfg or load_config()
    engine = _engine(engine)
    with single_run(engine) as got_lock:
        if not got_lock:
            return {"run_id": None, "status": "busy", "seconds": 0,
                    "reason": "Another pseudo-labelling run is in progress; try again in a minute.",
                    "total_pseudo_labels": count_pseudo(engine)}
        return _run(cfg, engine, push)


def _run(cfg: dict, engine, push: bool) -> dict:
    t0 = time.time()
    from mlops import deployment
    version = deployment.champion_version(engine)         # the live champion (Phase 6)
    summary = {"run_at": datetime.now(timezone.utc), "model_version": version}

    drift = latest_drift_status(engine)
    if drift in cfg["skip_if_drift_status"]:
        summary.update(status="skipped_drift",
                       reason=f"Latest drift check is {drift}: model confidence is unreliable, "
                              "so no pseudo-labels this run.")
        return _finish(engine, summary, t0, push)

    U = load_unlabelled(engine, cfg, version)
    if len(U) < cfg["min_unlabelled"]:
        summary.update(status="insufficient_data", n_unlabelled=len(U),
                       reason=f"{len(U)} new unlabelled predictions; need {cfg['min_unlabelled']}.")
        return _finish(engine, summary, t0, push)

    reference = pd.read_parquet(REFERENCE_PARQUET)
    ref_sample = reference.sample(min(cfg["reference_sample"], len(reference)), random_state=0)
    human = load_human_labels(engine)
    L = pd.concat([ref_sample[INPUT_FEATURES].assign(label=ref_sample[TARGET].to_numpy()),
                   human], ignore_index=True)
    y_l = L["label"].astype(int).to_numpy()

    st_cfg, ls_cfg = cfg["self_training"], cfg["label_spreading"]
    st_p, _ = self_training(L, y_l, U, st_cfg["threshold"], st_cfg["max_iter"])
    ls_p = label_spreading(L, y_l, U, ls_cfg["n_neighbors"], ls_cfg["alpha"])

    # caps: per run, and pseudo-labels overall <= ratio x real labels
    real_labels = len(reference) + len(human)
    room = int(cfg["max_pseudo_ratio"] * real_labels) - count_pseudo(engine)
    max_accept = max(0, min(cfg["max_accept_per_run"], room))
    pos_rate = float(reference[TARGET].mean()) if cfg["balance_to_reference"] else float((st_p >= 0.5).mean())
    sel = select_labels(st_p, ls_p, st_cfg["threshold"], ls_cfg["threshold"], pos_rate,
                        max_accept, cfg["require_agreement"])

    summary.update(status="ok", n_unlabelled=len(U), n_human_labels=len(human),
                   **{k: sel[k] for k in ("n_confident_self_training", "n_confident_spreading",
                                          "n_agree_confident", "n_accepted", "disagreement_rate",
                                          "mean_confidence")},
                   acceptance_rate=sel["n_accepted"] / len(U),
                   positive_share=float(sel["label"].mean()) if sel["n_accepted"] else None)
    return _finish(engine, summary, t0, push, U=U, sel=sel, st_p=st_p, ls_p=ls_p)


def _finish(engine, summary, t0, push, U=None, sel=None, st_p=None, ls_p=None) -> dict:
    summary["seconds"] = round(time.time() - t0, 1)
    with engine.begin() as conn:
        run_id = conn.execute(pseudo_label_runs.insert().values(**summary)).inserted_primary_key[0]
        if U is not None and len(U):              # remember everything this model version scored
            conn.execute(insert_ignore(pseudo_label_seen, engine), [
                {"request_id": rid, "model_version": summary["model_version"], "run_id": run_id}
                for rid in U["request_id"]])
        if sel is not None and sel["n_accepted"]:
            now = datetime.now(timezone.utc)
            conn.execute(insert_ignore(pseudo_labels, engine), [
                {"request_id": U.loc[i, "request_id"], "label": int(lab), "confidence": float(c),
                 "self_training_prob": float(st_p[i]), "label_spreading_prob": float(ls_p[i]),
                 "run_id": run_id, "model_version": summary["model_version"], "created_at": now}
                for i, lab, c in zip(sel["index"], sel["label"], sel["confidence"])])
    total = count_pseudo(engine)
    with engine.begin() as conn:
        conn.execute(pseudo_label_runs.update().where(pseudo_label_runs.c.id == run_id)
                     .values(total_pseudo_labels=total))
    summary.update(run_id=run_id, total_pseudo_labels=total)
    if push:
        push_metrics(summary)
    return summary


def push_metrics(s: dict) -> bool:
    """Pushgateway job "cardioguard_pseudo_label" (each push replaces the previous one)."""
    if not PUSHGATEWAY_URL:
        return False
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
    reg = CollectorRegistry()
    g = lambda name, doc: Gauge(f"cardioguard_pseudo_label_{name}", doc, registry=reg)
    g("status", "0 ok, 1 skipped (drift), -1 not enough data").set(STATUS_LEVEL.get(s["status"], -1))
    g("unlabelled", "Unlabelled predictions considered in the last run").set(s.get("n_unlabelled") or 0)
    g("accepted", "Pseudo-labels accepted in the last run").set(s.get("n_accepted") or 0)
    g("total", "Pseudo-labels stored in total").set(s.get("total_pseudo_labels") or 0)
    if s["status"] == "ok":
        g("acceptance_rate", "Accepted / unlabelled in the last run").set(s["acceptance_rate"])
        g("disagreement_rate", "Share where self-training and label spreading disagree").set(s["disagreement_rate"])
        g("mean_confidence", "Mean self-training confidence on unlabelled rows").set(s["mean_confidence"])
    g("last_run_timestamp_seconds", "When the job last ran").set(time.time())
    try:
        push_to_gateway(PUSHGATEWAY_URL, job="cardioguard_pseudo_label", registry=reg, timeout=3)
        return True
    except Exception as e:
        log.warning("Could not push pseudo-label metrics: %s", e)
        return False


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = run()
    print(f"\nPseudo-label run {s['run_id'] or '-'}: {s['status'].upper()}  ({s['seconds']}s)")
    if s["status"] != "ok":
        print(f"  {s['reason']}")
    else:
        print(f"  unlabelled {s['n_unlabelled']}, human labels used {s['n_human_labels']}")
        print(f"  confident: self-training {s['n_confident_self_training']}, label spreading "
              f"{s['n_confident_spreading']}, both + agree {s['n_agree_confident']}")
        print(f"  accepted {s['n_accepted']} ({s['acceptance_rate']:.1%}), disease share "
              f"{(s['positive_share'] or 0):.1%}, disagreement {s['disagreement_rate']:.1%}")
    print(f"  pseudo-labels stored in total: {s['total_pseudo_labels']}")


if __name__ == "__main__":
    main()
