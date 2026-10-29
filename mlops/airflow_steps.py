"""Thin wrappers around mlops/pipeline.py for the Airflow DAG (dags/cardio_self_healing.py).

Each function is one Airflow task. They run in the CardioGuard Python environment (not
Airflow's), take and return small JSON-safe dicts (XComs), and keep the same branching as
`pipeline.run_all`, so the DAG and `python -m mlops.pipeline` always behave the same.
"""
from __future__ import annotations

import json
import logging
import os

# the pipeline uses relative paths (data/, models/): run from the project folder
os.chdir(os.getenv("CARDIO_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from mlops import deployment, pipeline as p  # noqa: E402

log = logging.getLogger("cardioguard.airflow")


def _safe(d: dict) -> dict:
    """numpy floats, timestamps... -> plain JSON types."""
    def conv(o):
        try:
            return float(o)
        except Exception:
            return str(o)
    return json.loads(json.dumps(d, default=conv))


def _done(conf, outcome, diagnosis=None, **detail):
    return _safe(p.finish(outcome, diagnosis, actor=conf["run_id"], **detail))


# ------------------------------------------------------------ 1-2. trigger + diagnosis
def check(conf: dict) -> dict:
    trig = p.check_trigger(conf)
    if not trig["proceed"]:
        return _safe({"route": "stop", "outcome": trig["outcome"], "reason": trig["reason"], "diagnosis": None})
    diag = p.diagnose(trig, conf)
    d = diag["diagnosis"]
    log.info("Diagnosis: %s - %s", d, diag["explanation"])
    route = {"stop_bad_data": "stop", "no_action": "stop", "refresh_reference": "refresh"}.get(d, "retrain")
    outcome = {"stop_bad_data": "stopped_bad_data", "no_action": "no_action"}.get(d)
    return _safe({"route": route, "outcome": outcome, "reason": diag["explanation"], "diagnosis": d, "diag": diag})


def stop(conf: dict, chk: dict) -> dict:
    return _done(conf, chk["outcome"], chk["diagnosis"], reason=chk["reason"])


def refresh(conf: dict, chk: dict) -> dict:
    r = p.refresh_reference(source="recent")
    return _done(conf, "reference_refreshed", chk["diagnosis"], reason=chk["reason"], rows=r["rows"])


# ------------------------------------------------------------ 3-6. data, training, gates
def validate(conf: dict) -> dict:
    return _safe(p.validate_data(conf))


def pseudo(conf: dict) -> dict:
    return _safe(p.pseudo_label())


def train(conf: dict, chk: dict) -> dict:
    return _safe(p.train_and_evaluate(chk["diag"], conf, conf["run_id"]))


def gates(conf: dict, tr: dict) -> dict:
    g = p.check_gates(tr, conf)
    cand = {k: tr[k] for k in ("model_version", "registry_version", "diagnosis", "candidate", "champion")}
    failed = [x["gate"] for x in g["results"] if not x["passed"]]
    # conf travels inside the result so start_canary depends only on the gate/approval path
    return _safe({"passed": g["passed"], "failed": failed, "results": g["results"], "cand": cand, "conf": conf,
                  "needs_approval": g.get("needs_approval", False), "approval_reason": g.get("approval_reason")})


def reject(conf: dict, g: dict) -> dict:
    c = g["cand"]
    return _done(conf, "rejected_by_gates", c["diagnosis"], model_version=c["model_version"],
                 registry_version=c["registry_version"], reason="failed: " + ", ".join(g["failed"]))


# ------------------------------------------------------------ 7. human approval
def request_approval(conf: dict, g: dict, chk: dict) -> dict:
    return _safe(p.request_approval(g["cand"], conf["run_id"], g.get("approval_reason") or chk["reason"]))


def wait_approval(conf: dict, g: dict) -> dict:
    return _safe(p.wait_for_approval(g["cand"], conf))


def not_approved(conf: dict, a: dict) -> dict:
    return _done(conf, a["outcome"], a["diagnosis"], model_version=a["model_version"],
                 registry_version=a["registry_version"])


# ------------------------------------------------------------ 8-9. canary, promote / roll back
def start_canary(g: dict) -> dict:
    conf = g["conf"]
    return _safe(p.start_canary(g["cand"], conf, conf["run_id"]))


def verify(conf: dict, c: dict) -> dict:
    return _safe(p.verify_canary(c, conf))


def promote(conf: dict, v: dict) -> dict:
    p.promote(v, conf["run_id"])
    p.refresh_reference(v["model_version"], v["registry_version"], source="training")
    return _done(conf, "promoted", v["diagnosis"], model_version=v["model_version"],
                 registry_version=v["registry_version"], canary=v.get("canary_stats"))


def rollback(conf: dict, v: dict) -> dict:
    p.rollback(v, v["reasons"], conf["run_id"])
    return _done(conf, "rolled_back", v["diagnosis"], model_version=v["model_version"],
                 registry_version=v["registry_version"], reason="; ".join(v["reasons"]))


# ------------------------------------------------------------ safety net
def cleanup_after_failure(conf: dict) -> dict:
    """A task crashed. If THIS run left a canary or an approval request open, undo it so the
    champion serves 100% of traffic again, then record the failure."""
    eng = p.engine()
    s = deployment.get_state(eng) or {}
    undone = None
    if s.get("status") not in (None, "stable") and s.get("dag_run_id") == conf["run_id"]:
        undone = s["status"]
        if s["status"] == "canary" and s.get("candidate_registry_version"):
            p.rollback({"model_version": s["candidate_version"], "registry_version": s["candidate_registry_version"]},
                       ["pipeline task failed"], conf["run_id"])
        else:
            deployment.clear_candidate(eng, note="pipeline task failed")
    return _done(conf, "failed", s.get("diagnosis") if undone else conf.get("diagnosis"), reason=f"a task failed; undone: {undone or 'nothing'}")
