"""Deployment state: which model serves traffic (Phase 6).

One row in `model_deployment` is the live switch. Every API container reads it every
DEPLOYMENT_POLL_SECONDS and swaps models in memory when it changes, so all replicas
follow the same decision without restarts, Docker access or Nginx edits.

  status = stable              champion only
           awaiting_approval   candidate registered, a person must approve or reject
           canary              candidate gets `canary_fraction` of requests

`deployment_events` is the audit log: every registration, approval, canary start,
promotion and rollback, with who/what caused it.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (JSON, Column, DateTime, Float, Integer, MetaData, String, Table, Text,
                        desc, select)

log = logging.getLogger("cardioguard.deployment")
META_FILE = Path("models/model_meta.json")

metadata = MetaData()
model_deployment = Table(
    "model_deployment", metadata,
    Column("id", Integer, primary_key=True),                 # always 1
    Column("status", String(24), nullable=False, default="stable"),
    Column("champion_version", String(32)),                  # our version string, e.g. v20261027104439
    Column("champion_registry_version", Integer),            # MLflow version number (NULL = file model)
    Column("champion_auc", Float),                           # champion's own held-out AUC (drift accuracy check)
    Column("candidate_version", String(32)),
    Column("candidate_registry_version", Integer),
    Column("canary_fraction", Float, default=0.0),
    Column("diagnosis", String(40)),                         # why the candidate exists
    Column("dag_run_id", String(250)),
    Column("updated_at", DateTime(timezone=True)),
    Column("note", Text),
)
deployment_events = Table(
    "deployment_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("at", DateTime(timezone=True), nullable=False, index=True),
    Column("event", String(40), nullable=False),
    Column("model_version", String(32)),
    Column("registry_version", Integer),
    Column("actor", String(80)),                             # pipeline / dag run / person / api
    Column("detail", JSON),
)


def ensure(engine) -> None:
    metadata.create_all(engine)


def _now():
    return datetime.now(timezone.utc)


def get_state(engine) -> dict | None:
    ensure(engine)
    with engine.connect() as conn:
        row = conn.execute(select(model_deployment).where(model_deployment.c.id == 1)).mappings().first()
    return dict(row) if row else None


def set_state(engine, **values) -> dict:
    ensure(engine)
    values["updated_at"] = _now()
    with engine.begin() as conn:
        exists = conn.execute(select(model_deployment.c.id).where(model_deployment.c.id == 1)).first()
        if exists:
            conn.execute(model_deployment.update().where(model_deployment.c.id == 1).values(**values))
        else:
            conn.execute(model_deployment.insert().values(id=1, **{"status": "stable", "canary_fraction": 0.0,
                                                                      **values}))
    return get_state(engine)


def log_event(engine, event: str, model_version=None, registry_version=None, actor="pipeline", **detail):
    ensure(engine)
    with engine.begin() as conn:
        conn.execute(deployment_events.insert().values(
            at=_now(), event=event, model_version=model_version, registry_version=registry_version,
            actor=actor, detail=json.loads(json.dumps(detail, default=str))))
    log.info("deployment event %s %s %s", event, model_version or "", detail or "")


def events(engine, limit: int = 30) -> list[dict]:
    ensure(engine)
    with engine.connect() as conn:
        rows = conn.execute(select(deployment_events).order_by(desc(deployment_events.c.at),
                                                               desc(deployment_events.c.id)).limit(limit)).mappings().all()
    return [dict(r) for r in rows]


def champion_auc(engine=None) -> float | None:
    """Held-out ROC-AUC of the live champion: the baseline for the drift job's accuracy check."""
    if engine is not None:
        try:
            s = get_state(engine)
            if s and s.get("champion_auc"):
                return float(s["champion_auc"])
        except Exception:
            pass
    if META_FILE.exists():
        return json.loads(META_FILE.read_text()).get("test", {}).get("roc_auc")
    return None


def champion_version(engine=None) -> str:
    """Version string of the model that should be monitored (drift, pseudo-labels)."""
    if engine is not None:
        try:
            s = get_state(engine)
            if s and s.get("champion_version"):
                return s["champion_version"]
        except Exception as e:
            log.debug("deployment state unavailable: %s", e)
    return json.loads(META_FILE.read_text())["model_version"] if META_FILE.exists() else "unknown"


# ---------------------------------------------------------------- transitions
def clear_candidate(engine, status="stable", note=None):
    return set_state(engine, status=status, candidate_version=None, candidate_registry_version=None,
                     canary_fraction=0.0, note=note)


def decide(engine, approve: bool, actor: str = "person", reason: str | None = None) -> dict:
    """A person approves or rejects a candidate that is awaiting approval."""
    s = get_state(engine)
    if not s or s["status"] != "awaiting_approval":
        raise ValueError("No model is waiting for approval")
    log_event(engine, "approved" if approve else "rejected_by_person", s["candidate_version"],
              s["candidate_registry_version"], actor=actor, reason=reason)
    if approve:
        return set_state(engine, status="approved", note=reason or "approved")
    return clear_candidate(engine, note=reason or "rejected by a person")


def manual_rollback(engine, actor: str = "person", reason: str | None = None) -> dict:
    s = get_state(engine)
    if not s or not s.get("candidate_version"):
        raise ValueError("No candidate model to roll back")
    log_event(engine, "rolled_back", s["candidate_version"], s["candidate_registry_version"], actor=actor,
              reason=reason or "manual rollback")
    return clear_candidate(engine, note=reason or "manual rollback")
