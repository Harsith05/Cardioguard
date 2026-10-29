"""Self-healing pipeline steps (Phase 6). Airflow runs them as tasks; `run_all` runs them in one go.

    trigger ──▶ diagnose ──┬─ stop_bad_data ─────────────▶ alert, stop
                           ├─ no_action ─────────────────▶ stop
                           ├─ refresh_reference ─────────▶ new drift baseline, no retrain
                           └─ retrain_concept / retrain_needs_approval
                                  ▼
               validate ▶ pseudo-label ▶ train candidates ▶ promotion gates ──fail──▶ reject
                                                                │ pass
                              (needs approval) ▶ wait for a person ──reject/timeout──▶ stop
                                                                ▼
                                       canary (10% of traffic) ▶ verify ──bad──▶ roll back
                                                                │ good
                                                    promote ▶ refresh drift reference

Every step takes/returns small JSON-able dicts (Airflow XComs). The heavy objects live in
MLflow (models) and PostgreSQL (deployment state, events, reference data).

    python -m mlops.pipeline                       # same steps without Airflow
    python -m mlops.pipeline --force --canary-minutes 2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine, desc, select, text

from common.features import INPUT_FEATURES, TARGET
from mlops import deployment, registry

log = logging.getLogger("cardioguard.pipeline")
CFG_PATH = Path(__file__).with_name("config.yaml")
DATA = Path("data/processed")
OUTCOME_CODE = {"promoted": 0, "no_action": 1, "reference_refreshed": 2, "cooldown": 3, "busy": 4,
                "stopped_bad_data": 5, "rejected_by_gates": 6, "rejected_by_person": 7,
                "approval_timeout": 8, "rolled_back": 9, "failed": 10}


# ---------------------------------------------------------------- helpers
def load_config(overrides: dict | None = None) -> dict:
    return _merge(yaml.safe_load(CFG_PATH.read_text()), overrides or {})


def _merge(base: dict, over: dict) -> dict:
    """Deep merge: {"gates": {"default": {"max_auc_drop_recent": 0.02}}} changes only that value."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def engine():
    return create_engine(os.getenv("DATABASE_URL", "postgresql+psycopg2://cardio:cardio@localhost:5432/cardio"),
                         pool_pre_ping=True)


def _now():
    return datetime.now(timezone.utc)


def _metrics(y, p) -> dict:
    from sklearn.metrics import brier_score_loss, recall_score, roc_auc_score
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    return {"roc_auc": float(roc_auc_score(y, p)), "recall": float(recall_score(y, p >= 0.5)),
            "brier": float(brier_score_loss(y, p)), "mean_risk": float(p.mean()), "n": int(len(y))}


def _recall_gap(model, df: pd.DataFrame) -> float:
    from sklearn.metrics import recall_score
    p = model.predict_proba(df[INPUT_FEATURES])[:, 1] >= 0.5
    r = [recall_score(df[TARGET][df.gender == g], p[(df.gender == g).to_numpy()]) for g in (1, 2)]
    return float(abs(r[0] - r[1]))


def load_champion(eng):
    s = deployment.get_state(eng) or {}
    if s.get("champion_registry_version") and registry.enabled():
        return registry.load(s["champion_registry_version"])
    import joblib
    return joblib.load("models/model.joblib"), json.loads(Path("models/model_meta.json").read_text())


def push_outcome(outcome: str, diagnosis: str | None = None) -> None:
    url = os.getenv("PUSHGATEWAY_URL", "")
    if not url:
        return
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
        reg = CollectorRegistry()
        Gauge("cardioguard_self_healing_outcome", "Last pipeline outcome (see mlops/pipeline.py OUTCOME_CODE)",
              ["outcome", "diagnosis"], registry=reg).labels(outcome, diagnosis or "").set(OUTCOME_CODE.get(outcome, 10))
        Gauge("cardioguard_self_healing_last_run_timestamp_seconds", "When the pipeline last finished",
              registry=reg).set(time.time())
        push_to_gateway(url, job="cardioguard_self_healing", registry=reg, timeout=3)
    except Exception as e:
        log.warning("Could not push pipeline outcome: %s", e)


def finish(outcome: str, diagnosis: str | None = None, **detail) -> dict:
    """Record the final outcome of a run (event + metric) and return it."""
    eng = engine()
    result = {"outcome": outcome, "diagnosis": diagnosis, **detail}
    detail = dict(detail)
    deployment.log_event(eng, f"pipeline_{outcome}", detail.pop("model_version", None),
                         detail.pop("registry_version", None), actor=detail.pop("actor", "pipeline"),
                         diagnosis=diagnosis, **detail)
    push_outcome(outcome, diagnosis)
    if outcome in ("stopped_bad_data", "rejected_by_gates", "rolled_back", "approval_timeout"):
        try:
            from monitoring.alerts import send_alert
            send_alert({"run_id": "-", "severity": "warning", "model_version": "-",
                        "reasons": [f"Self-healing pipeline: {outcome} ({diagnosis}). {detail.get('reason', '')}"]})
        except Exception:
            pass
    return result


# ---------------------------------------------------------------- 1. trigger
def check_trigger(conf: dict | None = None) -> dict:
    """Should the pipeline run? Needs a retraining recommendation (or force), no rollout in
    progress, and no promotion/rollback in the cool-down period."""
    conf = conf or {}
    cfg = load_config(conf.get("config"))
    eng = engine()
    from monitoring.store import drift_runs
    from monitoring.store import metadata as drift_meta
    drift_meta.create_all(eng)
    deployment.ensure(eng)
    with eng.connect() as conn:
        latest = conn.execute(select(drift_runs).where(drift_runs.c.status != "insufficient_data")
                              .order_by(desc(drift_runs.c.run_at)).limit(1)).mappings().first()
    state = deployment.get_state(eng) or {}
    out = {"proceed": False, "drift_run_id": latest["id"] if latest else None,
           "checks": (latest or {}).get("checks") or [], "force": bool(conf.get("force")),
           "diagnosis_override": conf.get("diagnosis")}

    if state.get("status") not in (None, "stable"):
        return {**out, "reason": f"A rollout is already in progress ({state['status']})", "outcome": "busy"}
    if not out["force"]:
        cutoff = _now() - timedelta(hours=cfg["trigger"]["cooldown_hours"])
        recent = [e for e in deployment.events(eng, 50)
                  if e["event"] in ("promoted", "rolled_back") and e["at"] and
                  (e["at"] if e["at"].tzinfo else e["at"].replace(tzinfo=timezone.utc)) > cutoff]
        if recent:
            return {**out, "reason": f"Cool-down: last {recent[0]['event']} at {recent[0]['at']}", "outcome": "cooldown"}
        if latest is None:
            return {**out, "reason": "No drift check has run yet", "outcome": "no_action"}
        if cfg["trigger"]["require_retrain_recommended"] and not latest["retrain_recommended"]:
            return {**out, "reason": "Latest drift check does not recommend retraining", "outcome": "no_action"}
    return {**out, "proceed": True, "reason": "forced" if out["force"] else "retraining recommended by drift job"}


# ---------------------------------------------------------------- 2. diagnose
def diagnose(trigger: dict, conf: dict | None = None) -> dict:
    """Decide WHAT kind of drift this is, because each kind needs a different response.

    | data quality | inputs drifted | accuracy (human labels) | diagnosis                |
    |--------------|----------------|-------------------------|--------------------------|
    | critical     | any            | any                     | stop_bad_data            |
    | ok           | no             | dropped                 | retrain_concept          |
    | ok           | yes            | dropped                 | retrain_needs_approval   |
    | ok           | yes            | not enough labels       | retrain_needs_approval   |
    | ok           | yes            | held                    | refresh_reference        |
    | ok           | no             | held / unknown          | no_action                |
    """
    cfg = load_config((conf or {}).get("config"))["diagnosis"]
    status = {c["check"]: c["status"] for c in trigger.get("checks") or []}
    bad = {"warning", "critical"}
    quality_bad = status.get(cfg["data_quality_check"]) == "critical"
    inputs = any(status.get(c) in bad for c in cfg["input_checks"])
    names = cfg.get("accuracy_checks") or [cfg["accuracy_check"]]
    acc_states = [status.get(n) for n in names]
    acc_drop = any(s in bad for s in acc_states)
    acc_unknown = not acc_drop and all(s in (None, "insufficient_data") for s in acc_states)
    acc = next((s for s in acc_states if s in bad), acc_states[0])

    if trigger.get("diagnosis_override"):
        d, why = trigger["diagnosis_override"], "set by the person who started the run"
    elif quality_bad:
        d, why = "stop_bad_data", "Too many invalid inputs: fix the data source, retraining would learn errors."
    elif acc_drop and not inputs:
        d, why = "retrain_concept", "Inputs look normal but accuracy fell: the meaning of the data changed."
    elif acc_drop and inputs:
        d, why = ("retrain_needs_approval", "Inputs drifted AND accuracy fell. Could be a new population or a "
                  "measurement fault (e.g. a mis-calibrated device): a person must confirm before release.")
    elif inputs and acc_unknown:
        d, why = "retrain_needs_approval", "Inputs drifted and there are too few diagnoses to check accuracy."
    elif inputs:
        d, why = "refresh_reference", "Patients changed but the model is still accurate: update the drift baseline."
    else:
        d, why = "no_action", "No input drift and no accuracy loss."
    return {"diagnosis": d, "explanation": why, "inputs_drifted": inputs, "accuracy": acc,
            "data_quality": status.get(cfg["data_quality_check"]), "drift_run_id": trigger.get("drift_run_id")}


# ---------------------------------------------------------------- 3. validate
def human_labels(eng) -> pd.DataFrame:
    from app.db import predictions as p
    with eng.connect() as conn:
        rows = conn.execute(select(*[p.c[c] for c in INPUT_FEATURES], p.c.label, p.c.created_at)
                            .where(p.c.label_source == "human").order_by(p.c.created_at, p.c.id)).mappings().all()
    df = pd.DataFrame(rows, columns=INPUT_FEATURES + ["label", "created_at"])
    return df.rename(columns={"label": TARGET})


def validate_data(conf: dict | None = None) -> dict:
    cfg = load_config((conf or {}).get("config"))["validation"]
    from monitoring.drift import quality_issues
    train = pd.read_parquet(DATA / "train.parquet")
    missing = [c for c in INPUT_FEATURES + [TARGET] if c not in train]
    if missing:
        raise ValueError(f"Training data is missing columns: {missing}")
    human = human_labels(engine())
    invalid = float(quality_issues(human).mean()) if len(human) else 0.0
    out = {"training_rows": len(train), "human_labels": len(human), "invalid_share": round(invalid, 4)}
    if invalid > cfg["max_invalid_share"]:
        raise ValueError(f"{invalid:.1%} of labelled production rows are invalid (limit {cfg['max_invalid_share']:.0%})")
    if len(train) < cfg["min_training_rows"]:
        raise ValueError(f"Only {len(train)} training rows")
    return out


# ---------------------------------------------------------------- 4. pseudo-labels
def pseudo_label() -> dict:
    from semisup.pseudo_label import run
    s = run(engine=engine())
    return {k: s.get(k) for k in ("status", "n_accepted", "total_pseudo_labels", "reason")}


# ---------------------------------------------------------------- 5. train + evaluate
def drift_start(eng, drift_run_id: int | None = None):
    """When did the current drift begin? Window start of the oldest run in the latest streak
    of critical drift checks (same model version). None if there is no critical run."""
    from monitoring.store import drift_runs as d
    with eng.connect() as conn:
        q = select(d.c.id, d.c.status, d.c.model_version, d.c.window_start).where(d.c.status != "insufficient_data")
        if drift_run_id:
            q = q.where(d.c.id <= drift_run_id)
        runs = conn.execute(q.order_by(desc(d.c.run_at)).limit(50)).mappings().all()
    start = None
    for r in runs:
        if r["status"] != "critical" or r["model_version"] != runs[0]["model_version"]:
            break
        start = r["window_start"]
    if start is not None and start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return start


def build_training_data(eng, cfg: dict, diagnosis: str, drift_run_id: int | None = None):
    """Training rows (with a `source` column) + the newest confirmed cases kept back for evaluation.

    sources: training | human (before the drift) | human_new (since the drift) | pseudo
    Under concept drift the evaluation cases are the newest 30% of the *new-pattern* cases, so
    "recent AUC" measures exactly what the retraining is meant to fix."""
    from semisup.training_set import WEIGHT_COL, build
    from monitoring.drift import quality_issues
    t = cfg["training"]
    train = pd.read_parquet(DATA / "train.parquet")
    human = human_labels(eng)
    human = human[~quality_issues(human)].reset_index(drop=True)
    created = pd.to_datetime(human["created_at"], utc=True)

    start = drift_start(eng, drift_run_id) if diagnosis == "retrain_concept" else None
    if start is not None and (created >= start).sum() >= 100:
        is_new = (created >= start).to_numpy()
    else:                                   # unknown start: the last `human_recent_days` days
        start = None
        is_new = (created >= _now() - timedelta(days=t["human_recent_days"])).to_numpy()

    pool = human[is_new] if diagnosis == "retrain_concept" else human
    n_eval = int(len(pool) * t["holdout_recent_share"])
    eval_idx = pool.index[len(pool) - n_eval:] if n_eval else pool.index[:0]
    recent_eval = human.loc[eval_idx]
    keep = ~human.index.isin(eval_idx)
    human_train, new_train = human[keep], is_new[keep]

    recent_w = t["concept_recent_weight"] if diagnosis == "retrain_concept" else t["human_recent_weight"]
    hw = np.where(new_train, recent_w, t["human_weight"])
    src = np.where(new_train, "human_new", "human")

    pseudo = build(eng, train.head(0), pseudo_weight=t["pseudo_weight"], include_human=False)
    pseudo = pseudo[pseudo.source == "pseudo"]
    parts = [train[INPUT_FEATURES + [TARGET]].assign(source="training", **{WEIGHT_COL: 1.0}),
             human_train[INPUT_FEATURES + [TARGET]].assign(source=src, **{WEIGHT_COL: hw}),
             pseudo[INPUT_FEATURES + [TARGET, "source", WEIGHT_COL]]]
    data = pd.concat(parts, ignore_index=True)
    # concatenating with an EMPTY part (e.g. no pseudo-labels yet) turns columns into "object";
    # the one-hot encoder then fails on predict. Force numbers.
    data[INPUT_FEATURES] = data[INPUT_FEATURES].astype(float)
    data[TARGET] = data[TARGET].astype(int)
    data.attrs["drift_start"] = start.isoformat() if start is not None else None
    return data, recent_eval.astype({c: float for c in INPUT_FEATURES}), WEIGHT_COL


def strategies(data: pd.DataFrame, wcol: str, cfg: dict, diagnosis: str) -> dict:
    """Training sets to try: {name: (rows, weights)}. One ("standard") unless concept drift."""
    if diagnosis != "retrain_concept":
        return {"standard": (data, data[wcol].to_numpy())}
    t = cfg["training"]
    new = data["source"] == "human_new"
    out = {}
    for s in t["concept_strategies"]:
        if s.get("old_data", True):
            w = np.where(new, s.get("new_weight", t["concept_recent_weight"]), data[wcol].clip(upper=1.0))
            out[s["name"]] = (data, w)
        elif new.sum() >= t["min_new_pattern_rows"] and data.loc[new, TARGET].nunique() == 2:
            out[s["name"]] = (data[new], np.ones(int(new.sum())))
        else:
            log.info("Skipping strategy %s: only %d new-pattern cases (need %d)", s["name"], new.sum(),
                     t["min_new_pattern_rows"])
    return out


def train_and_evaluate(diag: dict, conf: dict | None = None, dag_run_id: str | None = None) -> dict:
    """Train each candidate type, keep the best, compare it with the champion, register it."""
    from training.train import candidates
    cfg = load_config((conf or {}).get("config"))
    eng = engine()
    d = diag["diagnosis"]
    data, recent, wcol = build_training_data(eng, cfg, d, diag.get("drift_run_id"))
    val = pd.read_parquet(DATA / "val.parquet")
    use_recent = len(recent) >= cfg["training"]["min_recent_eval"] and recent[TARGET].nunique() == 2

    def evaluate(model) -> dict:
        r = {"holdout": _metrics(val[TARGET], model.predict_proba(val[INPUT_FEATURES])[:, 1]),
             "recall_gap_sex": _recall_gap(model, val)}
        if "clinical" in cfg:
            from mlops import clinical
            r["clinical"] = clinical.check(model, cfg["clinical"])
        if use_recent:
            r["recent"] = _metrics(recent[TARGET], model.predict_proba(recent[INPUT_FEATURES])[:, 1])
        return r

    champ_model, champ_meta = load_champion(eng)
    champion = evaluate(champ_model)
    results, fitted = {}, {}
    for strategy, (rows, weights) in strategies(data, wcol, cfg, d).items():
        for model_type in cfg["training"]["candidates"]:
            name = model_type if strategy == "standard" else f"{model_type}/{strategy}"
            pipe = candidates(cfg["training"].get("monotone", False))[model_type]   # fresh, unfitted
            t0 = time.perf_counter()
            pipe.fit(rows[INPUT_FEATURES], rows[TARGET], model__sample_weight=weights)
            results[name] = {**evaluate(pipe), "fit_seconds": round(time.perf_counter() - t0, 1),
                             "train_rows": int(len(rows))}
            fitted[name] = pipe
            log.info("%s: holdout AUC %.4f%s", name, results[name]["holdout"]["roc_auc"],
                     f", recent AUC {results[name]['recent']['roc_auc']:.4f}" if use_recent else "")
    # best = clinically sane first, then highest mean of holdout and recent AUC (recent alone
    # under concept drift)
    def score(r):
        sane = not (r.get("clinical") or {}).get("failures")
        if d == "retrain_concept" and "recent" in r:
            return (sane, r["recent"]["roc_auc"])
        return (sane, np.mean([r["holdout"]["roc_auc"]] + ([r["recent"]["roc_auc"]] if "recent" in r else [])))
    best = max(results, key=lambda n: score(results[n]))
    model = fitted[best]

    import lightgbm
    import sklearn
    import xgboost
    version = _now().strftime("v%Y%m%d%H%M%S")
    composition = data["source"].value_counts().to_dict()
    strategy = best.split("/", 1)[1] if "/" in best else "standard"
    if strategy == "new_pattern_only":
        composition = {"human_new": composition.get("human_new", 0)}
    meta = {"model_version": version, "model_name": best, "threshold": 0.5, "features": INPUT_FEATURES,
            "trained_at": _now().isoformat(), "trained_by": dag_run_id or "pipeline", "diagnosis": d,
            "strategy": strategy, "drift_start": data.attrs.get("drift_start"),
            "train_rows": int(len(data)), "train_composition": composition,
            "environment": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                            "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__},
            "validation": results[best]["holdout"], "recent": results[best].get("recent"),
            "test": {"roc_auc": results[best]["holdout"]["roc_auc"]},   # baseline for the drift accuracy check
            "previous_champion": champ_meta.get("model_version")}
    reg_version = None
    if registry.enabled():
        flat = {f"holdout_{k}": v for k, v in results[best]["holdout"].items()}
        flat.update({f"recent_{k}": v for k, v in (results[best].get("recent") or {}).items()})
        if "clinical" in results[best]:
            flat["clinical_failures"] = len(results[best]["clinical"]["failures"])
        reg_version = registry.register(model, meta, metrics=flat, run_name=f"candidate-{version}",
                                        params={"model": best, "strategy": strategy, "diagnosis": d,
                                                "drift_start": data.attrs.get("drift_start"), **{f"rows_{k}": v for k, v in composition.items()}},
                                        tags={"diagnosis": d, "dag_run_id": dag_run_id or "", "status": "unverified"})
    else:                                     # no registry: keep the candidate on disk for the next step
        import joblib
        Path("models/candidates").mkdir(parents=True, exist_ok=True)
        joblib.dump(model, f"models/candidates/{version}.joblib")
    deployment.log_event(eng, "candidate_trained", version, reg_version, actor=dag_run_id or "pipeline",
                         model=best, diagnosis=d, composition=composition,
                         tried={n: {"holdout_auc": round(r["holdout"]["roc_auc"], 4),
                                    "recent_auc": round(r["recent"]["roc_auc"], 4) if "recent" in r else None,
                                    "clinical_failures": len((r.get("clinical") or {}).get("failures", []))}
                                for n, r in results.items()})
    return {"diagnosis": d, "model_version": version, "registry_version": reg_version, "model_name": best,
            "candidate": results[best], "champion": champion, "champion_version": champ_meta.get("model_version"),
            "all_candidates": {n: {"holdout_auc": r["holdout"]["roc_auc"],
                                   "recent_auc": (r.get("recent") or {}).get("roc_auc"),
                                   "clinical_failures": len((r.get("clinical") or {}).get("failures", []))}
                               for n, r in results.items()},
            "composition": composition, "recent_eval_rows": int(len(recent)) if use_recent else 0}


# ---------------------------------------------------------------- 6. promotion gates
def check_gates(tr: dict, conf: dict | None = None) -> dict:
    cfg = load_config((conf or {}).get("config"))["gates"]
    g = cfg.get(tr["diagnosis"], cfg["default"])
    c, ch = tr["candidate"], tr["champion"]
    results = []

    def gate(name, value, limit, ok, unit=""):
        results.append({"gate": name, "value": round(float(value), 4), "limit": limit, "passed": bool(ok)})

    hd = ch["holdout"]["roc_auc"] - c["holdout"]["roc_auc"]
    gate("AUC drop on validation set (old population)", hd, g["max_auc_drop_holdout"], hd <= g["max_auc_drop_holdout"])
    # under concept drift, forgetting the old pattern is expected; a lot of it needs a person's OK
    needs_approval = "approval_if_auc_drop_holdout_over" in g and hd > g["approval_if_auc_drop_holdout_over"]
    if "recent" in c and "recent" in ch:
        diff = c["recent"]["roc_auc"] - ch["recent"]["roc_auc"]
        if "min_auc_gain_recent" in g and "min_brier_drop_recent" in g:
            brier_drop = ch["recent"]["brier"] - c["recent"]["brier"]
            gate(f"Improvement on recent confirmed cases (AUC +{g['min_auc_gain_recent']} or Brier "
                 f"-{g['min_brier_drop_recent']}; Brier drop {brier_drop:+.3f})", diff, g["min_auc_gain_recent"],
                 diff >= g["min_auc_gain_recent"] or brier_drop >= g["min_brier_drop_recent"])
        elif "min_auc_gain_recent" in g:
            gate("AUC gain on recent confirmed cases", diff, g["min_auc_gain_recent"], diff >= g["min_auc_gain_recent"])
        else:
            gate("AUC drop on recent confirmed cases", -diff, g["max_auc_drop_recent"], -diff <= g["max_auc_drop_recent"])
        rd = ch["recent"]["recall"] - c["recent"]["recall"]
        bd = c["recent"]["brier"] - ch["recent"]["brier"]
    else:
        rd = ch["holdout"]["recall"] - c["holdout"]["recall"]
        bd = c["holdout"]["brier"] - ch["holdout"]["brier"]
        if "min_auc_gain_recent" in g:
            gate("AUC gain on recent confirmed cases (not enough labels)", 0, g["min_auc_gain_recent"], False)
    gate("Recall drop", rd, g["max_recall_drop"], rd <= g["max_recall_drop"])
    gate("Brier score increase (calibration)", bd, g["max_brier_increase"], bd <= g["max_brier_increase"])
    gate("Recall gap women vs men", c["recall_gap_sex"], g["max_recall_gap_sex"], c["recall_gap_sex"] <= g["max_recall_gap_sex"])
    if "clinical" in c:
        fails = c["clinical"]["failures"]
        gate("Clinical sanity (textbook patients, risk rises with risk factors)"
             + (": " + "; ".join(fails[:3]) if fails else ""), len(fails), 0, not fails)

    passed = all(r["passed"] for r in results)
    eng = engine()
    deployment.log_event(eng, "passed_gates" if passed else "rejected_by_gates", tr["model_version"],
                         tr["registry_version"], gates=results)
    if tr["registry_version"]:
        registry.tag(tr["registry_version"], "gates", "passed" if passed else "failed")
        registry.tag(tr["registry_version"], "gate_results", json.dumps(results))
    out = {"passed": passed, "results": results, "needs_approval": bool(passed and needs_approval),
           **{k: tr[k] for k in ("model_version", "registry_version", "diagnosis")}}
    if out["needs_approval"]:
        out["approval_reason"] = (f"The new model fits recent patients much better but forgets the old pattern "
                                  f"(AUC on the original validation set -{hd:.3f}). Approve only if the old "
                                  f"pattern really no longer holds.")
    return out


# ---------------------------------------------------------------- 7. human approval
def request_approval(cand: dict, dag_run_id: str | None = None, explanation: str = "") -> dict:
    eng = engine()
    deployment.set_state(eng, status="awaiting_approval", candidate_version=cand["model_version"],
                         candidate_registry_version=cand["registry_version"], canary_fraction=0.0,
                         diagnosis=cand["diagnosis"], dag_run_id=dag_run_id, note=explanation)
    deployment.log_event(eng, "approval_requested", cand["model_version"], cand["registry_version"],
                         actor=dag_run_id or "pipeline", reason=explanation)
    try:
        from monitoring.alerts import send_alert
        send_alert({"run_id": "-", "severity": "warning", "model_version": cand["model_version"],
                    "reasons": [f"A retrained model needs approval before release: {explanation} "
                                "Approve or reject on the monitoring page."]})
    except Exception:
        pass
    return {**cand, "status": "awaiting_approval"}


def wait_for_approval(cand: dict, conf: dict | None = None) -> dict:
    cfg = load_config((conf or {}).get("config"))["approval"]
    eng = engine()
    deadline = time.time() + cfg["timeout_minutes"] * 60
    while time.time() < deadline:
        s = deployment.get_state(eng) or {}
        if s.get("candidate_version") != cand["model_version"]:
            return {**cand, "approved": False, "outcome": "rejected_by_person"}
        if s.get("status") == "approved":
            return {**cand, "approved": True}
        time.sleep(cfg["poll_seconds"])
    deployment.clear_candidate(eng, note="approval timed out")
    deployment.log_event(eng, "approval_timeout", cand["model_version"], cand["registry_version"])
    return {**cand, "approved": False, "outcome": "approval_timeout"}


# ---------------------------------------------------------------- 8. canary
def start_canary(cand: dict, conf: dict | None = None, dag_run_id: str | None = None) -> dict:
    cfg = load_config((conf or {}).get("config"))["canary"]
    eng = engine()
    if cand.get("registry_version"):
        registry.set_alias("candidate", cand["registry_version"])
    started = _now()
    deployment.set_state(eng, status="canary", candidate_version=cand["model_version"],
                         candidate_registry_version=cand["registry_version"], canary_fraction=cfg["fraction"],
                         diagnosis=cand["diagnosis"], dag_run_id=dag_run_id,
                         note=f"canary {cfg['fraction']:.0%} since {started.isoformat()}")
    deployment.log_event(eng, "canary_started", cand["model_version"], cand["registry_version"],
                         actor=dag_run_id or "pipeline", fraction=cfg["fraction"])
    return {**cand, "canary_started": started.isoformat()}


def _canary_stats(eng, cand: dict, champion_version: str, since: datetime) -> dict:
    q = text("""SELECT model_version, count(*) AS n, avg(probability) AS mean_risk,
                       percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms
                FROM predictions WHERE created_at >= :since AND model_version IN (:c, :h)
                GROUP BY model_version""") if eng.dialect.name == "postgresql" else None
    if q is not None:
        with eng.connect() as conn:
            rows = {r["model_version"]: dict(r) for r in conn.execute(q, {"since": since, "c": cand["model_version"],
                                                                           "h": champion_version}).mappings()}
    else:                                     # SQLite (tests): compute in pandas
        from app.db import predictions as p
        with eng.connect() as conn:
            df = pd.DataFrame(conn.execute(select(p.c.model_version, p.c.probability, p.c.latency_ms)
                                           .where(p.c.created_at >= since)).mappings().all())
        rows = {} if df.empty else {v: {"n": len(g), "mean_risk": g.probability.mean(),
                                        "p95_ms": g.latency_ms.quantile(0.95)} for v, g in df.groupby("model_version")}
    return {"candidate": rows.get(cand["model_version"], {"n": 0}), "champion": rows.get(champion_version, {"n": 0})}


def _model_errors(model_version: str, minutes: float) -> float | None:
    url = os.getenv("PROMETHEUS_URL", "")
    if not url:
        return None
    try:
        import httpx
        q = f'sum(increase(cardioguard_model_errors_total{{model_version="{model_version}"}}[{max(1, int(minutes))}m]))'
        r = httpx.get(f"{url}/api/v1/query", params={"query": q}, timeout=5).json()["data"]["result"]
        return float(r[0]["value"][1]) if r else 0.0
    except Exception as e:
        log.warning("Prometheus not reachable for canary error check: %s", e)
        return None


def verify_canary(cand: dict, conf: dict | None = None) -> dict:
    """Watch the canary; decide healthy / unhealthy. A manual rollback ends it early."""
    cfg = load_config((conf or {}).get("config"))["canary"]
    eng = engine()
    since = datetime.fromisoformat(cand["canary_started"])
    champion_version = (deployment.get_state(eng) or {}).get("champion_version")
    deadline = time.time() + cfg["wait_minutes"] * 60
    while time.time() < deadline:
        s = deployment.get_state(eng) or {}
        if s.get("status") != "canary" or s.get("candidate_version") != cand["model_version"]:
            return {**cand, "healthy": False, "outcome": "rolled_back", "reasons": ["rolled back manually"],
                    "manual": True}
        time.sleep(cfg["poll_seconds"])

    st = _canary_stats(eng, cand, champion_version, since)
    c, h = st["candidate"], st["champion"]
    reasons = []
    if c["n"] < cfg["min_predictions"]:
        reasons.append(f"only {c['n']} canary predictions (need {cfg['min_predictions']}): cannot judge it")
    else:
        errors = _model_errors(cand["model_version"], cfg["wait_minutes"] + 1)
        if errors is not None and errors / max(c["n"], 1) > cfg["max_error_rate"]:
            reasons.append(f"error rate {errors / c['n']:.1%} > {cfg['max_error_rate']:.0%}")
        if c["p95_ms"] and c["p95_ms"] > cfg["max_latency_ms"]:
            reasons.append(f"p95 model time {c['p95_ms']:.0f} ms > {cfg['max_latency_ms']} ms")
        if h.get("n") and h.get("p95_ms") and c["p95_ms"] > cfg.get("ratio_min_ms", 0) and \
                c["p95_ms"] > cfg["max_latency_ratio"] * h["p95_ms"]:
            reasons.append(f"p95 model time {c['p95_ms']:.1f} ms is over {cfg['max_latency_ratio']}x the champion's {h['p95_ms']:.1f} ms")
        # the shift we EXPECT from the labelled recent cases (e.g. a guideline change raises risk
        # for everyone); the live shift must stay close to it
        expected = 0.0
        cr, hr = (cand.get("candidate") or {}).get("recent"), (cand.get("champion") or {}).get("recent")
        if cr and hr and "mean_risk" in cr and "mean_risk" in hr:
            expected = cr["mean_risk"] - hr["mean_risk"]
        live = c["mean_risk"] - h["mean_risk"] if h.get("n") else None
        if live is not None and abs(live - expected) > cfg["max_mean_risk_shift"]:
            reasons.append(f"mean predicted risk {c['mean_risk']:.2f} vs champion {h['mean_risk']:.2f} "
                           f"(shift {live:+.2f}, expected {expected:+.2f} from confirmed cases)")
    num = lambda x: round(float(x), 4) if isinstance(x, (int, float, np.integer, np.floating)) or \
        type(x).__name__ == "Decimal" else x
    stats = {k: {kk: num(vv) for kk, vv in v.items() if kk != "model_version"} for k, v in st.items()}
    return {**cand, "healthy": not reasons, "reasons": reasons, "canary_stats": stats}


# ---------------------------------------------------------------- 9. promote / roll back
def promote(cand: dict, dag_run_id: str | None = None) -> dict:
    eng = engine()
    if cand.get("registry_version"):
        old = registry.get("champion")
        registry.set_alias("champion", cand["registry_version"])
        registry.delete_alias("candidate")
        registry.tag(cand["registry_version"], "status", "champion")
        if old:
            registry.tag(old["version"], "status", "retired")
    auc = (cand.get("candidate") or {}).get("holdout", {}).get("roc_auc")
    deployment.set_state(eng, status="stable", champion_version=cand["model_version"],
                         champion_registry_version=cand.get("registry_version"), champion_auc=auc,
                         candidate_version=None, candidate_registry_version=None, canary_fraction=0.0,
                         note=f"promoted by {dag_run_id or 'pipeline'}")
    deployment.log_event(eng, "promoted", cand["model_version"], cand.get("registry_version"),
                         actor=dag_run_id or "pipeline", canary=cand.get("canary_stats"))
    return {**cand, "promoted": True}


def rollback(cand: dict, reasons: list[str], dag_run_id: str | None = None) -> dict:
    eng = engine()
    s = deployment.get_state(eng) or {}
    if s.get("candidate_version") == cand["model_version"]:
        deployment.clear_candidate(eng, note="rolled back: " + "; ".join(reasons))
        deployment.log_event(eng, "rolled_back", cand["model_version"], cand.get("registry_version"),
                             actor=dag_run_id or "pipeline", reasons=reasons, canary=cand.get("canary_stats"))
    if cand.get("registry_version"):
        registry.delete_alias("candidate")
        registry.tag(cand["registry_version"], "status", "rolled_back")
    return {**cand, "promoted": False}


# ---------------------------------------------------------------- 10. drift reference
def refresh_reference(model_version: str | None = None, registry_version: int | None = None,
                      source: str = "training", recent_rows: int | None = None) -> dict:
    """Write the drift baseline (reference_data) for a model version.

    source="training": the data the new champion was trained on (after a promotion).
    source="recent":   the latest production patients (population changed, model still accurate).
                       Only the newest `recent_rows` (default: 2 drift windows), so older traffic
                       from before the change does not water the new baseline down.
    """
    eng = engine()
    s = deployment.get_state(eng) or {}
    model_version = model_version or s.get("champion_version")
    registry_version = registry_version or s.get("champion_registry_version")
    model = registry.load(registry_version)[0] if (registry_version and registry.enabled()) else load_champion(eng)[0]
    if source == "recent":
        from app.db import predictions as p
        with eng.connect() as conn:
            df = pd.DataFrame(conn.execute(select(*[p.c[c] for c in INPUT_FEATURES], p.c.label)
                                           .where(p.c.model_version == model_version)
                                           .order_by(desc(p.c.created_at), desc(p.c.id))
                                           .limit(recent_rows or _recent_rows())).mappings().all())
        df = df.rename(columns={"label": TARGET})
    else:
        train = pd.read_parquet(DATA / "train.parquet")
        df = pd.concat([train[INPUT_FEATURES + [TARGET]], human_labels(eng)[INPUT_FEATURES + [TARGET]]],
                       ignore_index=True)
    if df.empty:
        raise ValueError("No rows to build a reference from")
    df["prediction_proba"] = model.predict_proba(df[INPUT_FEATURES])[:, 1]
    df["prediction"] = (df["prediction_proba"] >= 0.5).astype(int)
    df["model_version"] = model_version
    with eng.begin() as conn:
        if eng.dialect.has_table(conn, "reference_data"):
            conn.execute(text("DELETE FROM reference_data WHERE model_version = :v"), {"v": model_version})
    df.to_sql("reference_data", eng, if_exists="append", index=False, chunksize=5000)
    deployment.log_event(eng, "reference_refreshed", model_version, registry_version, rows=len(df), source=source)
    return {"model_version": model_version, "rows": int(len(df)), "source": source}


def _recent_rows() -> int:
    from monitoring.drift import load_config as drift_cfg
    return 2 * int(drift_cfg()["window_size"])


# ---------------------------------------------------------------- whole loop without Airflow
def run_all(conf: dict | None = None) -> dict:
    conf = conf or {}
    run_id = conf.get("run_id") or f"manual__{_now().strftime('%Y%m%dT%H%M%S')}"
    trig = check_trigger(conf)
    if not trig["proceed"]:
        return finish(trig["outcome"], reason=trig["reason"], actor=run_id)
    diag = diagnose(trig, conf)
    d = diag["diagnosis"]
    log.info("Diagnosis: %s - %s", d, diag["explanation"])
    if d == "stop_bad_data":
        return finish("stopped_bad_data", d, reason=diag["explanation"], actor=run_id)
    if d == "no_action":
        return finish("no_action", d, reason=diag["explanation"], actor=run_id)
    if d == "refresh_reference":
        r = refresh_reference(source="recent")
        return finish("reference_refreshed", d, reason=diag["explanation"], rows=r["rows"], actor=run_id)

    validate_data(conf)
    pseudo_label()
    tr = train_and_evaluate(diag, conf, run_id)
    gates = check_gates(tr, conf)
    if not gates["passed"]:
        failed = [g["gate"] for g in gates["results"] if not g["passed"]]
        return finish("rejected_by_gates", d, model_version=tr["model_version"],
                      registry_version=tr["registry_version"], reason="failed: " + ", ".join(failed), actor=run_id)
    cand = {**{k: tr[k] for k in ("model_version", "registry_version", "diagnosis", "candidate", "champion")}}
    if d == "retrain_needs_approval" or gates.get("needs_approval"):
        request_approval(cand, run_id, gates.get("approval_reason") or diag["explanation"])
        a = wait_for_approval(cand, conf)
        if not a["approved"]:
            return finish(a["outcome"], d, model_version=cand["model_version"],
                          registry_version=cand["registry_version"], actor=run_id)
    c = start_canary(cand, conf, run_id)
    try:
        v = verify_canary(c, conf)
        if not v["healthy"]:
            rollback(v, v["reasons"], run_id)
            return finish("rolled_back", d, model_version=cand["model_version"],
                          registry_version=cand["registry_version"], reason="; ".join(v["reasons"]), actor=run_id)
        promote(v, run_id)
    except Exception as e:                       # never leave a canary running after a crash
        log.exception("Pipeline failed during the canary: rolling back")
        rollback(c, [f"pipeline error: {e}"], run_id)
        return finish("rolled_back", d, model_version=cand["model_version"],
                      registry_version=cand["registry_version"], reason=f"pipeline error: {e}", actor=run_id)
    refresh_reference(cand["model_version"], cand["registry_version"], source="training")
    return finish("promoted", d, model_version=cand["model_version"], registry_version=cand["registry_version"],
                  canary=v.get("canary_stats"), actor=run_id)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Run the self-healing pipeline once, without Airflow")
    ap.add_argument("--force", action="store_true", help="run even if the drift job did not recommend it")
    ap.add_argument("--diagnosis", choices=["retrain_concept", "retrain_needs_approval", "refresh_reference",
                                            "no_action", "stop_bad_data"], help="skip the diagnosis step")
    ap.add_argument("--canary-minutes", type=float, help="override canary observation time")
    ap.add_argument("--approval-minutes", type=float, help="override approval timeout")
    ap.add_argument("--config-json", default="{}", help='override any config.yaml section, e.g. '
                    '\'{"gates": {"default": {"max_auc_drop_recent": 0.02}}}\'')
    a = ap.parse_args()
    conf = {"force": a.force, "diagnosis": a.diagnosis, "config": json.loads(a.config_json)}
    if a.canary_minutes is not None:
        conf["config"]["canary"] = {"wait_minutes": a.canary_minutes}
    if a.approval_minutes is not None:
        conf["config"]["approval"] = {"timeout_minutes": a.approval_minutes}
    print(json.dumps(run_all(conf), indent=2, default=str))


if __name__ == "__main__":
    main()
