"""CardioGuard prediction service.

Run:     uvicorn app.main:app --host 0.0.0.0 --port 8000
Docs:    http://localhost:8000/docs
Metrics: http://localhost:8000/metrics  (Prometheus, Phase 3)
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse

from app import config, metrics
from app.chaos import ChaosMiddleware
from app.chaos import router as chaos_router
from app.db import db_logger
from app.model_loader import model_service
from app.deployment_routes import router as deployment_router
from app.monitoring_routes import router as monitoring_router
from app.schemas import (BatchIn, BatchOut, ExplainedPredictionOut, LabelIn,
                         PatientInput, PredictionOut)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_logger.connect()
    model_service.load(db_logger.engine)    # champion (+ candidate) from the deployment state
    metrics.set_reference_means(config.REFERENCE_PATH)
    model_service.start_watcher()           # Phase 6: follow promotions / canaries / rollbacks
    yield
    model_service.stop_watcher()


app = FastAPI(title="CardioGuard Prediction Service", version="1.3.0", lifespan=lifespan)
app.include_router(monitoring_router)
app.include_router(deployment_router)
app.include_router(chaos_router)
app.add_middleware(ChaosMiddleware)   # added first, so the metrics middleware sees injected errors
metrics.setup(app)


def risk_label(p: float) -> str:
    return "high" if p >= 0.7 else "moderate" if p >= 0.4 else "low"


def _predict_group(rows: list[dict], lm):
    """Score rows with one model. If a candidate fails, its error is counted and the champion
    answers instead, so a broken canary never reaches the user."""
    t0 = time.perf_counter()
    try:
        probs = model_service.predict_with(lm, rows)
    except Exception:
        metrics.MODEL_ERRORS.labels(lm.version).inc()
        if lm is model_service.champion:
            raise
        logging.getLogger("cardioguard.api").exception("Candidate %s failed; champion answers", lm.version)
        lm = model_service.champion
        probs = model_service.predict_with(lm, rows)
    return probs, lm, time.perf_counter() - t0


def _score(patients: list[PatientInput], endpoint: str, lm=None):
    """Score patients with the champion, or the candidate for the canary share.

    The model is chosen PER PATIENT (not per request), so a batch of 80 patients sends about
    8 to the candidate during a 10% canary, instead of all-or-nothing."""
    rows = [p.model_dump() for p in patients]
    chosen = [lm] * len(rows) if lm is not None else [model_service.choose() for _ in rows]
    groups: dict[int, tuple] = {}
    for i, m in enumerate(chosen):
        groups.setdefault(id(m), (m, []))[1].append(i)

    now = datetime.now(timezone.utc)
    outs, records = [None] * len(rows), [None] * len(rows)
    total = 0.0
    for m, idx in groups.values():
        probs, used, elapsed = _predict_group([rows[i] for i in idx], m)
        total += elapsed
        latency_ms = elapsed * 1000 / len(idx)
        group_records = []
        for i, p in zip(idx, probs):
            rid = str(uuid.uuid4())
            pred = int(p >= used.threshold)
            outs[i] = PredictionOut(request_id=rid, probability=round(p, 4), prediction=pred,
                                    risk_label=risk_label(p), model_version=used.version)
            records[i] = {**rows[i], "request_id": rid, "created_at": now, "model_version": used.version,
                          "probability": p, "prediction": pred, "latency_ms": latency_ms}
            group_records.append(records[i])
        metrics.record_predictions(group_records, endpoint, elapsed, observe_batch=False)
    metrics.BATCH.observe(len(rows))
    return outs, records, total * 1000 / len(rows)


@app.post("/predict", response_model=PredictionOut)
def predict(patient: PatientInput, background: BackgroundTasks):
    outs, records, _ = _score([patient], "/predict")
    background.add_task(db_logger.log_predictions, records, patient.model_dump(), "/predict")
    return outs[0]


@app.post("/predict/explain", response_model=ExplainedPredictionOut)
def predict_explain(patient: PatientInput, background: BackgroundTasks):
    """Prediction plus SHAP contributions: which inputs pushed this patient's risk up or down."""
    lm = model_service.choose()                       # explain the model that actually answers
    if lm.explainer is None:
        lm = model_service.champion
    try:
        t0 = time.perf_counter()
        explanation = model_service.explain([patient.model_dump()], lm)[0]
        metrics.SHAP.observe(time.perf_counter() - t0)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    outs, records, _ = _score([patient], "/predict/explain", lm)
    background.add_task(db_logger.log_predictions, records, patient.model_dump(), "/predict/explain")
    return ExplainedPredictionOut(**outs[0].model_dump(), explanation=explanation)


@app.get("/explain/global")
def explain_global():
    """Mean |SHAP| over 1,000 training patients: what the model relies on overall."""
    if model_service.global_importance is None:
        raise HTTPException(503, "Explanations are not available")
    return {"model_version": model_service.version, **model_service.global_importance}


@app.post("/predict/batch", response_model=BatchOut)
def predict_batch(batch: BatchIn, background: BackgroundTasks):
    if len(batch.patients) > config.MAX_BATCH:
        raise HTTPException(413, f"Batch larger than {config.MAX_BATCH}")
    outs, records, _ = _score(batch.patients, "/predict/batch")
    background.add_task(db_logger.log_predictions, records, batch.model_dump(), "/predict/batch")
    return BatchOut(model_version=outs[0].model_version, predictions=outs)


@app.post("/feedback/{request_id}")
def add_ground_truth(request_id: str, body: LabelIn):
    """Attach a confirmed diagnosis to an earlier prediction (used by backtesting)."""
    try:
        found = db_logger.add_label(request_id, body.label, body.label_source)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    if not found:
        raise HTTPException(404, "request_id not found")
    metrics.FEEDBACK.labels(str(body.label)).inc()
    return {"request_id": request_id, "label": body.label, "status": "stored"}


UI_FILE = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def ui():
    """Prediction form for people (the API itself stays at /predict, docs at /docs)."""
    return FileResponse(UI_FILE)


@app.get("/health")
def health():
    cand = model_service.candidate
    return {"status": "ok" if model_service.ready else "model_not_loaded",
            "model_version": model_service.version, "logging": db_logger.status,
            "explanations": model_service.explainer is not None,
            "deployment": {"status": model_service.status,
                           "candidate_version": cand.version if cand else None,
                           "canary_fraction": model_service.canary_fraction}}


@app.get("/model-info")
def model_info():
    m = model_service.meta
    return {k: m.get(k) for k in
            ["model_version", "model_name", "trained_at", "threshold", "features",
             "train_rows", "validation", "test"]}
