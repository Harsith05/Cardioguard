"""Prediction logging.

PostgreSQL `predictions` - one row per prediction: features, output, latency, and
                           a label column that stays NULL until ground truth (or a
                           Phase 5 pseudo-label) arrives.
MongoDB `raw_requests`    - the original JSON payload, for audit and replay.

Logging failures never break a prediction: they are caught and logged.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String, Table,
                        create_engine, update)

from app import config

log = logging.getLogger("cardioguard.db")

metadata = MetaData()
predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("request_id", String(36), unique=True, nullable=False, index=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("model_version", String(32), nullable=False, index=True),
    Column("age_years", Float), Column("gender", Integer), Column("height", Float),
    Column("weight", Float), Column("ap_hi", Float), Column("ap_lo", Float),
    Column("cholesterol", Integer), Column("gluc", Integer), Column("smoke", Integer),
    Column("alco", Integer), Column("active", Integer),
    Column("probability", Float, nullable=False),
    Column("prediction", Integer, nullable=False),
    Column("latency_ms", Float),
    Column("label", Integer, nullable=True),            # ground truth, NULL until known
    Column("label_source", String(16), nullable=True),  # 'human' | 'pseudo' (Phase 5)
    Column("label_confidence", Float, nullable=True),
    Column("labelled_at", DateTime(timezone=True), nullable=True),
)


class Logger:
    def __init__(self):
        self.engine = None
        self.mongo = None

    def connect(self):
        try:
            self.engine = create_engine(config.DATABASE_URL, pool_pre_ping=True)
            metadata.create_all(self.engine)
            log.info("SQL logging enabled")
        except Exception as e:
            self.engine = None
            log.warning("SQL logging disabled: %s", e)
        if config.MONGO_URI:
            try:
                from pymongo import MongoClient
                client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=2000)
                client.admin.command("ping")
                self.mongo = client[config.MONGO_DB]["raw_requests"]
                log.info("MongoDB logging enabled")
            except Exception as e:
                self.mongo = None
                log.warning("MongoDB logging disabled: %s", e)
        from app.metrics import DB_UP
        DB_UP.labels("postgres").set(1 if self.engine is not None else 0)
        DB_UP.labels("mongodb").set(1 if self.mongo is not None else 0)

    @property
    def status(self) -> dict:
        return {"sql": self.engine is not None, "mongo": self.mongo is not None}

    def log_predictions(self, records: list[dict], raw_payload: dict, endpoint: str):
        """Runs in a FastAPI background task, after the response is sent."""
        from app.metrics import LOG_FAILURES
        if self.engine is not None:
            try:
                with self.engine.begin() as conn:
                    conn.execute(predictions.insert(), records)
            except Exception as e:
                LOG_FAILURES.labels("postgres").inc()
                log.error("SQL log failed: %s", e)
        if self.mongo is not None:
            try:
                self.mongo.insert_one({
                    "endpoint": endpoint,
                    "received_at": datetime.now(timezone.utc),
                    "request_ids": [r["request_id"] for r in records],
                    "model_version": records[0]["model_version"],
                    "payload": raw_payload,
                })
            except Exception as e:
                LOG_FAILURES.labels("mongodb").inc()
                log.error("Mongo log failed: %s", e)

    def add_label(self, request_id: str, label: int, source: str) -> bool:
        if self.engine is None:
            raise RuntimeError("SQL logging is not available")
        with self.engine.begin() as conn:
            res = conn.execute(
                update(predictions)
                .where(predictions.c.request_id == request_id)
                .values(label=label, label_source=source, label_confidence=1.0,
                        labelled_at=datetime.now(timezone.utc)))
        return res.rowcount > 0


db_logger = Logger()
