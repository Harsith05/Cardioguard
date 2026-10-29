"""CardioGuard self-healing DAG (Phase 6, Airflow 3).

Started by the drift job when it recommends retraining (monitoring/alerts.py), or by hand
from the Airflow UI ("Trigger" with parameters). Same logic as `python -m mlops.pipeline`:

 check_and_diagnose ─▶ route ─┬─ stop                 (cool-down, busy, no action, bad data)
                              ├─ refresh_reference    (patients changed, model still fine)
                              └─ validate ▶ pseudo_label ▶ train ▶ gates ─▶ after_gates
                                                                             ├─ reject
                                                                             ├─ request_approval ▶ wait_for_approval ─▶ after_approval
                                                                             │                                           ├─ not_approved
                                                                             └───────────────┬───────────────────────────┘
                                                                                       start_canary ▶ verify_canary ─▶ after_canary
                                                                                                                       ├─ promote
                                                                                                                       └─ rollback
 cleanup_after_failure: runs only if a task crashes; ends any canary this run started.

The ML work runs in the CardioGuard Python environment (CARDIO_PYTHON), not Airflow's own,
so Airflow and scikit-learn/LightGBM never have to agree on library versions.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.sdk import Param, TriggerRule, dag, get_current_context, task

PY = os.getenv("CARDIO_PYTHON", "/opt/cardio/bin/python")
APPROVAL_TIMEOUT_H = float(os.getenv("CARDIO_APPROVAL_TIMEOUT_HOURS", "25"))
ext = dict(python=PY, expect_airflow=False, retries=0)


@dag(
    dag_id="cardio_self_healing",
    schedule=None,                         # triggered by the drift job, or by hand
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,                     # never two retrainings at the same time
    tags=["cardioguard", "self-healing"],
    params={
        "force": Param(False, type="boolean", description="Run even if the drift job did not recommend it"),
        "diagnosis": Param(None, type=["null", "string"],
                           enum=[None, "retrain_concept", "retrain_needs_approval", "refresh_reference",
                                 "no_action", "stop_bad_data"],
                           description="Skip the automatic diagnosis (testing)"),
        "canary_minutes": Param(None, type=["null", "number"], description="Override canary observation time"),
        "approval_minutes": Param(None, type=["null", "number"], description="Override approval timeout"),
        "config": Param({}, type="object", description="Override any section of mlops/config.yaml"),
    },
    doc_md=__doc__,
)
def cardio_self_healing():

    @task
    def make_conf() -> dict:
        """DAG params + run id -> the `conf` dict every pipeline step takes."""
        ctx = get_current_context()
        prm, run = dict(ctx["params"]), ctx["dag_run"]
        prm.update({k: v for k, v in (run.conf or {}).items() if k in prm})   # API-triggered runs
        cfg = dict(prm.get("config") or {})
        if prm.get("canary_minutes") is not None:
            cfg["canary"] = {**cfg.get("canary", {}), "wait_minutes": prm["canary_minutes"]}
        if prm.get("approval_minutes") is not None:
            cfg["approval"] = {**cfg.get("approval", {}), "timeout_minutes": prm["approval_minutes"]}
        return {"run_id": run.run_id, "force": bool(prm.get("force")), "diagnosis": prm.get("diagnosis"),
                "config": cfg, "drift_run_id": (run.conf or {}).get("drift_run_id")}

    # ---------------------------------------------------------- heavy steps (CardioGuard env)
    @task.external_python(task_id="check_and_diagnose", **ext)
    def check_and_diagnose(conf):
        from mlops import airflow_steps as s
        return s.check(conf)

    @task.external_python(task_id="stop", **ext)
    def stop(conf, chk):
        from mlops import airflow_steps as s
        return s.stop(conf, chk)

    @task.external_python(task_id="refresh_reference", **ext)
    def refresh_reference(conf, chk):
        from mlops import airflow_steps as s
        return s.refresh(conf, chk)

    @task.external_python(task_id="validate_data", **ext)
    def validate_data(conf):
        from mlops import airflow_steps as s
        return s.validate(conf)

    @task.external_python(task_id="pseudo_label", **ext)
    def pseudo_label(conf):
        from mlops import airflow_steps as s
        return s.pseudo(conf)

    @task.external_python(task_id="train_candidates", execution_timeout=timedelta(hours=2), **ext)
    def train(conf, chk):
        from mlops import airflow_steps as s
        return s.train(conf, chk)

    @task.external_python(task_id="promotion_gates", **ext)
    def gates(conf, tr):
        from mlops import airflow_steps as s
        return s.gates(conf, tr)

    @task.external_python(task_id="reject", **ext)
    def reject(conf, g):
        from mlops import airflow_steps as s
        return s.reject(conf, g)

    @task.external_python(task_id="request_approval", **ext)
    def request_approval(conf, g, chk):
        from mlops import airflow_steps as s
        return s.request_approval(conf, g, chk)

    @task.external_python(task_id="wait_for_approval", execution_timeout=timedelta(hours=APPROVAL_TIMEOUT_H), **ext)
    def wait_for_approval(conf, g):
        from mlops import airflow_steps as s
        return s.wait_approval(conf, g)

    @task.external_python(task_id="not_approved", **ext)
    def not_approved(conf, a):
        from mlops import airflow_steps as s
        return s.not_approved(conf, a)

    @task.external_python(task_id="start_canary", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS, **ext)
    def start_canary(g):
        # takes ONLY the gate result: with "none failed, at least one succeeded" it must not
        # count make_conf as a success, or it would run on the stop/refresh paths too
        from mlops import airflow_steps as s
        return s.start_canary(g)

    @task.external_python(task_id="verify_canary", execution_timeout=timedelta(hours=6), **ext)
    def verify_canary(conf, c):
        from mlops import airflow_steps as s
        return s.verify(conf, c)

    @task.external_python(task_id="promote", **ext)
    def promote(conf, v):
        from mlops import airflow_steps as s
        return s.promote(conf, v)

    @task.external_python(task_id="rollback", **ext)
    def rollback(conf, v):
        from mlops import airflow_steps as s
        return s.rollback(conf, v)

    @task.external_python(task_id="cleanup_after_failure", trigger_rule=TriggerRule.ONE_FAILED, **ext)
    def cleanup_after_failure(conf):
        from mlops import airflow_steps as s
        return s.cleanup_after_failure(conf)

    # ---------------------------------------------------------- decisions (tiny, Airflow env)
    @task.branch
    def route(chk: dict) -> str:
        return {"stop": "stop", "refresh": "refresh_reference"}.get(chk["route"], "validate_data")

    @task.branch
    def after_gates(g: dict, chk: dict) -> str:
        if not g["passed"]:
            return "reject"
        if chk["diagnosis"] == "retrain_needs_approval" or g.get("needs_approval"):
            return "request_approval"        # suspected device fault, or the model forgets a lot
        return "start_canary"

    @task.branch
    def after_approval(a: dict) -> str:
        return "start_canary" if a.get("approved") else "not_approved"

    @task.branch
    def after_canary(v: dict) -> str:
        return "promote" if v.get("healthy") else "rollback"

    # ---------------------------------------------------------- wiring
    conf = make_conf()
    chk = check_and_diagnose(conf)
    r = route(chk)
    v_data = validate_data(conf)
    r >> [stop(conf, chk), refresh_reference(conf, chk), v_data]

    ps = pseudo_label(conf)
    tr = train(conf, chk)
    v_data >> ps >> tr
    g = gates(conf, tr)
    ag = after_gates(g, chk)

    req = request_approval(conf, g, chk)
    wait = wait_for_approval(conf, g)
    req >> wait
    aa = after_approval(wait)
    canary = start_canary(g)
    ag >> [reject(conf, g), req, canary]
    aa >> [not_approved(conf, wait), canary]

    ver = verify_canary(conf, canary)
    ac = after_canary(ver)
    ac >> [promote(conf, ver), rollback(conf, ver)]

    [chk, v_data, ps, tr, g, req, wait, canary, ver] >> cleanup_after_failure(conf)


cardio_self_healing()
