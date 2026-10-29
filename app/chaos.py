"""Fault injection for testing alerts (Phase 3). OFF unless CHAOS_ENABLED=1.

With it on, prediction requests can be made to fail or slow down on purpose, so you
can check that the error-rate and latency alerts in Prometheus actually fire.

    CHAOS_ENABLED=1 uvicorn app.main:app --port 8000
    curl -X POST "localhost:8000/debug/chaos?error_rate=0.2&latency_ms=600"
    curl -X POST "localhost:8000/debug/chaos?error_rate=0&latency_ms=0"     # back to normal
"""
from __future__ import annotations

import asyncio
import os
import random

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

ENABLED = os.getenv("CHAOS_ENABLED", "0") == "1"
state = {"error_rate": 0.0, "latency_ms": 0}
router = APIRouter(prefix="/debug", tags=["debug"], include_in_schema=ENABLED)


class ChaosMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if ENABLED and request.url.path.startswith("/predict"):
            if state["latency_ms"]:
                await asyncio.sleep(state["latency_ms"] / 1000)
            if random.random() < state["error_rate"]:
                return JSONResponse({"detail": "Injected failure (chaos testing)"}, status_code=500)
        return await call_next(request)


@router.get("/chaos")
def get_chaos():
    if not ENABLED:
        raise HTTPException(404)
    return state


@router.post("/chaos")
def set_chaos(error_rate: float = Query(0.0, ge=0, le=1), latency_ms: int = Query(0, ge=0, le=10000)):
    if not ENABLED:
        raise HTTPException(404)
    state.update(error_rate=error_rate, latency_ms=latency_ms)
    return state
