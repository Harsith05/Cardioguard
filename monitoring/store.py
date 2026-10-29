"""Where drift results live.

PostgreSQL  drift_runs            one row per run (status, key numbers) -> Grafana in Phase 3
            drift_feature_scores  one row per feature per run
MongoDB     drift_reports         full JSON of each run
            alerts                warning / critical alerts
Files       reports/drift/<run_id>.html   Evidently HTML report
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer,
                        MetaData, String, Table, Text, create_engine, desc, select, text)

from common.features import INPUT_FEATURES

log = logging.getLogger("cardioguard.monitoring")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://cardio:cardio@localhost:5432/cardio")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "cardioguard")

metadata = MetaData()
drift_runs = Table(
    "drift_runs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_at", DateTime(timezone=True), nullable=False, index=True),
    Column("model_version", String(32)),
    Column("window_start", DateTime(timezone=True)),
    Column("window_end", DateTime(timezone=True)),
    Column("n_rows", Integer),
    Column("n_labelled", Integer),
    Column("status", String(20), index=True),
    Column("drift_share", Float),
    Column("max_psi", Float),
    Column("prediction_drift_score", Float),
    Column("prediction_drift_detected", Boolean),
    Column("mean_confidence", Float),
    Column("mean_probability", Float),
    Column("roc_auc", Float),
    Column("retrain_recommended", Boolean, default=False),
    Column("reasons", Text),
    Column("checks", JSON),
    Column("report_path", String(255)),
)
drift_feature_scores = Table(
    "drift_feature_scores", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Integer, ForeignKey("drift_runs.id"), index=True),
    Column("run_at", DateTime(timezone=True), index=True),
    Column("feature", String(32)),
    Column("method", String(32)),
    Column("score", Float),
    Column("threshold", Float),
    Column("detected", Boolean),
    Column("psi", Float),
)


class Store:
    def __init__(self, database_url: str = DATABASE_URL, mongo_uri: str | None = MONGO_URI):
        self.engine = create_engine(database_url, pool_pre_ping=True)
        metadata.create_all(self.engine)
        self.mongo = None
        if mongo_uri:
            try:
                from pymongo import MongoClient
                client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
                client.admin.command("ping")
                self.mongo = client[MONGO_DB]
            except Exception as e:
                log.warning("MongoDB unavailable, drift JSON only in Postgres: %s", e)

    # ---------- reading inputs
    def load_reference(self, model_version: str | None, sample: int, fallback_parquet: str) -> pd.DataFrame:
        cols = INPUT_FEATURES + ["prediction_proba"]
        try:
            q, params = "SELECT * FROM reference_data", {}
            if model_version:
                q, params = q + " WHERE model_version = :v", {"v": model_version}
            with self.engine.connect() as conn:
                ref = pd.read_sql(text(q), conn, params=params)
            if ref.empty:
                raise ValueError("reference_data is empty for this model version")
        except Exception as e:
            log.info("Using reference parquet (%s)", e)
            ref = pd.read_parquet(fallback_parquet)
        ref = ref[cols]
        return ref.sample(min(sample, len(ref)), random_state=0) if sample else ref

    def load_window(self, size: int, model_version: str | None = None) -> pd.DataFrame:
        from app.db import predictions  # same table the API writes
        q = select(predictions).order_by(desc(predictions.c.created_at), desc(predictions.c.id)).limit(size)
        if model_version:
            q = q.where(predictions.c.model_version == model_version)
        with self.engine.connect() as conn:
            df = pd.DataFrame(conn.execute(q).mappings().all())
        if df.empty:
            return df
        return (df.rename(columns={"probability": "prediction_proba"})
                  .sort_values(["created_at", "id"]).reset_index(drop=True))

    def previous_runs(self, n: int) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(drift_runs).where(drift_runs.c.status != "insufficient_data")
                                .order_by(desc(drift_runs.c.run_at)).limit(n)).mappings().all()
        return [dict(r) for r in rows]

    # ---------- writing results
    def save_run(self, result: dict, model_version: str, window: pd.DataFrame, report_path: str | None) -> int:
        now = datetime.now(timezone.utc)
        checks = result.get("checks", [])
        get = lambda name: next((c["value"] for c in checks if c["check"] == name), None)
        pred = result.get("prediction_drift") or {}
        row = {
            "run_at": now, "model_version": model_version,
            "window_start": window["created_at"].min() if len(window) else None,
            "window_end": window["created_at"].max() if len(window) else None,
            "n_rows": result["n_rows"], "n_labelled": result.get("n_labelled", 0),
            "status": result["status"], "drift_share": get("feature_drift_share"),
            "max_psi": get("psi_key_features"),
            "prediction_drift_score": pred.get("score"), "prediction_drift_detected": pred.get("detected"),
            "mean_confidence": result.get("mean_confidence"), "mean_probability": result.get("mean_probability"),
            "roc_auc": result.get("roc_auc"), "retrain_recommended": result.get("retrain_recommended", False),
            "reasons": "\n".join(result.get("reasons", [])), "checks": checks, "report_path": report_path,
        }
        with self.engine.begin() as conn:
            run_id = conn.execute(drift_runs.insert().values(**row)).inserted_primary_key[0]
            feats = [{"run_id": run_id, "run_at": now, **{k: f.get(k) for k in
                      ("feature", "method", "score", "threshold", "detected", "psi")}}
                     for f in result.get("features", [])]
            if feats:
                conn.execute(drift_feature_scores.insert(), feats)
        if self.mongo is not None:
            try:
                doc = {**row, "run_id": run_id, "features": result.get("features", [])}
                self.mongo["drift_reports"].insert_one(json.loads(json.dumps(doc, default=str)))
            except Exception as e:
                log.error("Mongo drift report failed: %s", e)
        return run_id

    def save_alert(self, alert: dict):
        if self.mongo is not None:
            try:
                self.mongo["alerts"].insert_one(dict(alert))
            except Exception as e:
                log.error("Mongo alert failed: %s", e)
