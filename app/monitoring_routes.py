"""Phase 2 endpoints: read drift results and run a check on demand."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import desc, select

from app.db import db_logger

router = APIRouter(prefix="/monitoring", tags=["monitoring"])
PAGE = Path(__file__).parent / "static" / "monitoring.html"


def _tables():
    from monitoring.store import drift_feature_scores, drift_runs, metadata
    if db_logger.engine is None:
        raise HTTPException(503, "Database is not available")
    metadata.create_all(db_logger.engine)  # first call before any drift run
    return drift_runs, drift_feature_scores


def _run_dict(row) -> dict:
    d = dict(row)
    for k in ("run_at", "window_start", "window_end"):
        if d.get(k) is not None:
            d[k] = d[k].isoformat()
    d["reasons"] = [r for r in (d.get("reasons") or "").split("\n") if r]
    d["has_report"] = bool(d.get("report_path")) and Path(d["report_path"]).exists()
    d.pop("report_path", None)
    return d


@router.get("", include_in_schema=False)
def page():
    return FileResponse(PAGE)


@router.get("/drift/latest")
def latest():
    runs, feats = _tables()
    with db_logger.engine.connect() as conn:
        row = conn.execute(select(runs).order_by(desc(runs.c.run_at)).limit(1)).mappings().first()
        if row is None:
            return {"run": None, "features": []}
        f = conn.execute(select(feats).where(feats.c.run_id == row["id"])
                         .order_by(feats.c.feature)).mappings().all()
    return {"run": _run_dict(row),
            "features": [{k: r[k] for k in ("feature", "method", "score", "threshold", "detected", "psi")} for r in f]}


@router.get("/drift/history")
def history(limit: int = Query(30, ge=1, le=500)):
    runs, _ = _tables()
    with db_logger.engine.connect() as conn:
        rows = conn.execute(select(runs).order_by(desc(runs.c.run_at)).limit(limit)).mappings().all()
    return {"runs": [_run_dict(r) for r in rows]}


@router.get("/drift/runs/{run_id}/report", include_in_schema=False)
def report(run_id: int):
    runs, _ = _tables()
    with db_logger.engine.connect() as conn:
        path = conn.execute(select(runs.c.report_path).where(runs.c.id == run_id)).scalar()
    if not path or not Path(path).exists():
        raise HTTPException(404, "No Evidently report for this run")
    return FileResponse(path, media_type="text/html")


@router.post("/drift/run")
def run_now(window: int | None = Query(None, ge=50, le=50000)):
    """Run a drift check now (the scheduler normally does this)."""
    from monitoring.drift_job import run
    _tables()
    result = run(window)
    return {"run_id": result["run_id"], "status": result["status"],
            "retrain_recommended": result.get("retrain_recommended", False)}


# ---------------------------------------------------------------- Phase 5: pseudo-labels
def _pl_tables():
    from semisup.store import metadata, pseudo_label_runs
    if db_logger.engine is None:
        raise HTTPException(503, "Database is not available")
    metadata.create_all(db_logger.engine)
    return pseudo_label_runs


def _pl_dict(row) -> dict:
    d = dict(row)
    if d.get("run_at") is not None:
        d["run_at"] = d["run_at"].isoformat()
    return d


@router.get("/pseudo-labels/latest")
def pseudo_latest():
    runs = _pl_tables()
    with db_logger.engine.connect() as conn:
        row = conn.execute(select(runs).order_by(desc(runs.c.run_at)).limit(1)).mappings().first()
    return {"run": _pl_dict(row) if row else None}


@router.get("/pseudo-labels/history")
def pseudo_history(limit: int = Query(30, ge=1, le=500)):
    runs = _pl_tables()
    with db_logger.engine.connect() as conn:
        rows = conn.execute(select(runs).order_by(desc(runs.c.run_at)).limit(limit)).mappings().all()
    return {"runs": [_pl_dict(r) for r in rows]}


@router.post("/pseudo-labels/run")
def pseudo_run_now():
    """Run the pseudo-label job now (the scheduler normally does this after each drift check)."""
    from semisup.pseudo_label import run
    _pl_tables()
    s = run(engine=db_logger.engine)
    return {k: s.get(k) for k in ("run_id", "status", "reason", "n_unlabelled", "n_accepted",
                                  "acceptance_rate", "total_pseudo_labels")}