"""Phase 6 endpoints: see the rollout state, approve/reject a candidate, roll back a canary.

If ADMIN_TOKEN is set, the POST endpoints need the header  X-Admin-Token: <token>.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.db import db_logger
from app.model_loader import model_service

router = APIRouter(prefix="/deployment", tags=["deployment"])
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


class Decision(BaseModel):
    reason: str | None = None
    actor: str | None = None


def _engine():
    if db_logger.engine is None:
        raise HTTPException(503, "Database is not available")
    return db_logger.engine


def _check(token: str | None):
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise HTTPException(403, "Missing or wrong X-Admin-Token")


def _iso(d: dict) -> dict:
    return {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in d.items()}


@router.get("")
def state():
    from mlops import deployment
    e = _engine()
    s = deployment.get_state(e)
    cand = model_service.candidate
    return {"state": _iso(s) if s else None,
            "this_replica": {"champion": model_service.version,
                             "candidate": cand.version if cand else None,
                             "canary_fraction": model_service.canary_fraction},
            "events": [_iso(x) for x in deployment.events(e, 20)]}


@router.post("/approve")
def approve(body: Decision | None = None, x_admin_token: str | None = Header(None)):
    _check(x_admin_token)
    from mlops import deployment
    try:
        s = deployment.decide(_engine(), True, actor=(body and body.actor) or "person",
                              reason=body.reason if body else None)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"status": s["status"], "message": "Approved: the pipeline starts the canary release."}


@router.post("/reject")
def reject(body: Decision | None = None, x_admin_token: str | None = Header(None)):
    _check(x_admin_token)
    from mlops import deployment
    try:
        s = deployment.decide(_engine(), False, actor=(body and body.actor) or "person",
                              reason=body.reason if body else None)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"status": s["status"], "message": "Rejected: the current model stays."}


@router.post("/rollback")
def rollback(body: Decision | None = None, x_admin_token: str | None = Header(None)):
    _check(x_admin_token)
    from mlops import deployment
    try:
        s = deployment.manual_rollback(_engine(), actor=(body and body.actor) or "person",
                                       reason=body.reason if body else None)
    except ValueError as e:
        raise HTTPException(409, str(e))
    model_service.sync()
    return {"status": s["status"], "message": "Rolled back: all traffic is on the champion again."}
