"""Phase 6 tests: drift diagnosis, promotion gates, deployment state, canary routing."""
import numpy as np
import pytest
from sqlalchemy import create_engine

from mlops import deployment
from mlops.pipeline import check_gates, diagnose


def checks(**status):
    names = {"inputs": "psi_key_features", "share": "feature_drift_share", "acc": "auc_drop", "dq": "data_quality"}
    return {"checks": [{"check": names[k], "status": v} for k, v in status.items()], "drift_run_id": 1}


@pytest.mark.parametrize("st, expected", [
    (dict(dq="critical", inputs="critical", acc="critical"), "stop_bad_data"),
    (dict(dq="ok", inputs="ok", share="ok", acc="critical"), "retrain_concept"),          # concept drift
    (dict(dq="ok", inputs="critical", acc="critical"), "retrain_needs_approval"),         # device fault?
    (dict(dq="ok", inputs="critical", acc="insufficient_data"), "retrain_needs_approval"),
    (dict(dq="ok", inputs="critical", acc="ok"), "refresh_reference"),                    # older patients
    (dict(dq="ok", inputs="ok", share="ok", acc="ok"), "no_action"),
    (dict(dq="ok", share="warning", acc="ok"), "refresh_reference"),
])
def test_diagnosis_table(st, expected):
    assert diagnose(checks(**st))["diagnosis"] == expected


def test_diagnosis_override():
    t = {**checks(dq="ok", inputs="ok", acc="ok"), "diagnosis_override": "retrain_concept"}
    assert diagnose(t)["diagnosis"] == "retrain_concept"


# ---------------------------------------------------------------- gates
def result(auc_h, auc_r, recall=0.70, brier=0.18, gap=0.01):
    return {"holdout": {"roc_auc": auc_h, "recall": recall, "brier": brier},
            "recent": {"roc_auc": auc_r, "recall": recall, "brier": brier}, "recall_gap_sex": gap}


@pytest.fixture()
def no_side_effects(monkeypatch, tmp_path):
    from mlops import pipeline
    monkeypatch.setattr(pipeline, "engine", lambda: create_engine(f"sqlite:///{tmp_path / 'g.db'}"))
    return pipeline


def tr(diag, cand, champ):
    return {"diagnosis": diag, "candidate": cand, "champion": champ, "model_version": "vX", "registry_version": None}


def test_gates_pass_when_candidate_is_as_good(no_side_effects):
    assert check_gates(tr("retrain_needs_approval", result(0.800, 0.75), result(0.802, 0.75)))["passed"]


def test_gates_block_forgetting_the_old_population(no_side_effects):
    g = check_gates(tr("retrain_needs_approval", result(0.78, 0.80), result(0.80, 0.75)))
    assert not g["passed"]
    assert [x["gate"] for x in g["results"] if not x["passed"]] == ["AUC drop on validation set (old population)"]


def test_gates_block_unfair_model(no_side_effects):
    assert not check_gates(tr("retrain_needs_approval", result(0.80, 0.75, gap=0.08), result(0.80, 0.75)))["passed"]


def test_concept_gates_require_real_improvement(no_side_effects):
    champ = result(0.80, 0.55)
    assert not check_gates(tr("retrain_concept", result(0.80, 0.56), champ))["passed"]   # +0.01 only
    assert check_gates(tr("retrain_concept", result(0.77, 0.60), champ))["passed"]       # some forgetting OK


# ---------------------------------------------------------------- deployment state
@pytest.fixture()
def eng(tmp_path):
    e = create_engine(f"sqlite:///{tmp_path / 'd.db'}")
    deployment.set_state(e, status="stable", champion_version="v1", champion_registry_version=1)
    return e


def test_approval_flow(eng):
    with pytest.raises(ValueError):
        deployment.decide(eng, True)                                   # nothing waiting
    deployment.set_state(eng, status="awaiting_approval", candidate_version="v2", candidate_registry_version=2)
    assert deployment.decide(eng, True, actor="Dr. A")["status"] == "approved"
    deployment.set_state(eng, status="awaiting_approval", candidate_version="v3", candidate_registry_version=3)
    s = deployment.decide(eng, False, actor="Dr. A", reason="device fault")
    assert s["status"] == "stable" and s["candidate_version"] is None and s["champion_version"] == "v1"
    assert [e["event"] for e in deployment.events(eng)][:2] == ["rejected_by_person", "approved"]


def test_manual_rollback_clears_canary(eng):
    deployment.set_state(eng, status="canary", candidate_version="v2", candidate_registry_version=2, canary_fraction=0.1)
    s = deployment.manual_rollback(eng, reason="looks wrong")
    assert s["status"] == "stable" and s["canary_fraction"] == 0 and s["champion_version"] == "v1"


def test_champion_version_follows_state(eng):
    assert deployment.champion_version(eng) == "v1"
    deployment.set_state(eng, champion_version="v9", champion_auc=0.81)
    assert deployment.champion_version(eng) == "v9" and deployment.champion_auc(eng) == 0.81


# ---------------------------------------------------------------- canary routing in the API
class _Model:
    def __init__(self, p=0.3, fail=False):
        self.p, self.fail = p, fail

    def predict_proba(self, X):
        if self.fail:
            raise RuntimeError("broken candidate")
        return np.column_stack([1 - np.full(len(X), self.p), np.full(len(X), self.p)])


def test_router_sends_canary_share_to_candidate():
    from app.model_loader import LoadedModel, ModelService
    ms = ModelService()
    ms.champion = LoadedModel(_Model(0.2), {"model_version": "champ"})
    ms.candidate = LoadedModel(_Model(0.8), {"model_version": "cand"})
    ms.canary_fraction = 0.1
    picks = [ms.choose().version for _ in range(5000)]
    assert 0.07 < picks.count("cand") / 5000 < 0.13
    ms.canary_fraction = 0.0
    assert {ms.choose().version for _ in range(200)} == {"champ"}


def test_broken_candidate_falls_back_to_champion(monkeypatch):
    from app import main
    from app.model_loader import LoadedModel
    from app.schemas import PatientInput
    from app.model_loader import model_service as ms
    saved = (ms.champion, ms.candidate, ms.canary_fraction)
    try:
        ms.champion = LoadedModel(_Model(0.2), {"model_version": "champ"})
        bad = LoadedModel(_Model(fail=True), {"model_version": "cand"})
        p = PatientInput(age_years=50, gender=1, height=165, weight=70, ap_hi=130, ap_lo=85,
                         cholesterol=1, gluc=1, smoke=0, alco=0, active=1)
        outs, records, _ = main._score([p], "/predict", bad)
        assert outs[0].model_version == "champ" and records[0]["model_version"] == "champ"
    finally:
        ms.champion, ms.candidate, ms.canary_fraction = saved


# ---------------------------------------------------------------- config overrides + Airflow hand-off
def test_config_override_changes_only_one_value():
    from mlops.pipeline import load_config
    c = load_config({"gates": {"retrain_concept": {"min_auc_gain_recent": -1}}})
    assert c["gates"]["retrain_concept"]["min_auc_gain_recent"] == -1
    assert c["gates"]["retrain_concept"]["max_recall_drop"] == load_config()["gates"]["retrain_concept"]["max_recall_drop"]


def test_drift_job_triggers_airflow3_dag(monkeypatch):
    """Airflow 3: get a JWT from /auth/token, then POST /api/v2/dags/<id>/dagRuns with logical_date null."""
    import httpx
    from monitoring import alerts
    calls = []

    def handler(req: httpx.Request):
        calls.append(req)
        if req.url.path == "/auth/token":
            return httpx.Response(201, json={"access_token": "jwt-123"})
        return httpx.Response(200, json={"dag_run_id": "manual__x"})

    real = httpx.Client
    monkeypatch.setattr(alerts.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(alerts, "AIRFLOW_URL", "http://airflow:8080")
    assert alerts.trigger_retraining(7, ["psi critical"]) is True
    run = calls[1]
    assert run.url.path == "/api/v2/dags/cardio_self_healing/dagRuns"
    assert run.headers["authorization"] == "Bearer jwt-123"
    import json
    body = json.loads(run.content)
    assert body["logical_date"] is None and body["conf"]["drift_run_id"] == 7


def test_dag_structure():
    """The DAG parses and keeps run_all's branching (needs Airflow; skipped otherwise)."""
    pytest.importorskip("airflow.sdk")
    from pathlib import Path
    from airflow.models.dagbag import DagBag
    bag = DagBag(dag_folder=str(Path(__file__).parents[1] / "dags"), include_examples=False)
    assert not bag.import_errors, bag.import_errors
    dag = bag.dags["cardio_self_healing"]
    down = lambda t: set(dag.get_task(t).downstream_task_ids)
    assert down("route") == {"stop", "refresh_reference", "validate_data"}
    assert down("after_gates") == {"reject", "request_approval", "start_canary"}
    assert down("after_canary") == {"promote", "rollback"}
    # start_canary must not depend on make_conf (else it would run on the stop/refresh paths)
    assert "make_conf" not in dag.get_task("start_canary").upstream_task_ids
    assert dag.max_active_runs == 1


# ---------------------------------------------------------------- concept drift: strategies + escalation
def test_concept_strategies_reweight_or_drop_old_data():
    import pandas as pd
    from mlops.pipeline import load_config, strategies
    from common.features import TARGET
    cfg = load_config()
    n_new = cfg["training"]["min_new_pattern_rows"]
    data = pd.DataFrame({"source": ["training"] * 1000 + ["human_new"] * n_new + ["pseudo"] * 10,
                         TARGET: ([0, 1] * 2000)[:1010 + n_new], "w": 1.0})
    data.loc[data.source == "pseudo", "w"] = 0.5
    s = strategies(data, "w", cfg, "retrain_concept")
    assert set(s) == {"weighted_x10", "weighted_x50", "new_pattern_only"}
    rows, w = s["weighted_x50"]
    assert w[(data.source == "human_new").to_numpy()].min() == 50 and w[(data.source == "pseudo").to_numpy()].max() == 0.5
    rows, w = s["new_pattern_only"]
    assert set(rows.source) == {"human_new"} and len(rows) == n_new
    # too few new-pattern cases -> only the weighted strategies
    assert "new_pattern_only" not in strategies(data.iloc[:1100], "w", cfg, "retrain_concept")
    # other diagnoses: one standard training set
    assert list(strategies(data, "w", cfg, "retrain_needs_approval")) == ["standard"]


def test_concept_model_that_forgets_a_lot_needs_a_person(no_side_effects):
    champ = result(0.80, 0.51)
    g = check_gates(tr("retrain_concept", result(0.60, 0.75), champ))      # big gain, forgets old pattern
    assert g["passed"] and g["needs_approval"] and "forgets the old pattern" in g["approval_reason"]
    g = check_gates(tr("retrain_concept", result(0.78, 0.60), champ))      # small forgetting -> automatic
    assert g["passed"] and not g["needs_approval"]
    g = check_gates(tr("retrain_concept", result(0.40, 0.75), champ))      # beyond the hard limit
    assert not g["passed"]


def test_canary_routes_each_patient_in_a_batch():
    """A 10% canary must send ~10% of PATIENTS to the candidate, also inside batch requests."""
    from app import main
    from app.model_loader import LoadedModel
    from app.model_loader import model_service as ms
    from app.schemas import PatientInput
    saved = (ms.champion, ms.candidate, ms.canary_fraction)
    try:
        ms.champion = LoadedModel(_Model(0.2), {"model_version": "champ"})
        ms.candidate = LoadedModel(_Model(0.8), {"model_version": "cand"})
        ms.canary_fraction = 0.1
        p = PatientInput(age_years=50, gender=1, height=165, weight=70, ap_hi=130, ap_lo=85,
                         cholesterol=1, gluc=1, smoke=0, alco=0, active=1)
        outs, records, _ = main._score([p] * 2000, "/predict/batch")
        share = sum(o.model_version == "cand" for o in outs) / 2000
        assert 0.07 < share < 0.13
        assert all((o.probability == 0.8) == (o.model_version == "cand") for o in outs)   # right model per row
        assert [r["model_version"] for r in records] == [o.model_version for o in outs]
    finally:
        ms.champion, ms.candidate, ms.canary_fraction = saved


# ---------------------------------------------------------------- medical knowledge (monotone + clinical gate)
@pytest.fixture(scope="module")
def train_small():
    import pandas as pd
    return pd.read_parquet("data/processed/train.parquet").sample(8000, random_state=0)


def test_monotone_model_never_lowers_risk_for_worse_bp(train_small):
    import numpy as np
    import pandas as pd
    from common.features import INPUT_FEATURES, TARGET
    from training.train import candidates
    df = train_small.copy()
    young = df.age_years < 50
    df.loc[young, TARGET] = 1 - df.loc[young, TARGET]          # absurd labels on purpose
    m = candidates(monotone=True)["lightgbm"].fit(df[INPUT_FEATURES].astype(float), df[TARGET])
    base = pd.DataFrame([{"age_years": a, "gender": 1, "height": 168, "weight": 72, "ap_hi": hi, "ap_lo": lo,
                          "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1}
                         for a in (30, 45, 60) for hi, lo in ((110, 70), (130, 85), (150, 95), (170, 105))])
    risk = m.predict_proba(base[INPUT_FEATURES].astype(float))[:, 1].reshape(3, 4)
    assert (np.diff(risk, axis=1) >= -1e-9).all()             # each row: risk never falls as BP rises


def test_clinical_checks_catch_an_absurd_model(train_small):
    from common.features import INPUT_FEATURES, TARGET
    from mlops import clinical
    from mlops.pipeline import load_config
    from training.train import candidates
    cfg = load_config()["clinical"]
    X = train_small[INPUT_FEATURES].astype(float)
    sane = candidates(monotone=True)["lightgbm"].fit(X, train_small[TARGET])
    assert clinical.check(sane, cfg)["failures"] == []
    absurd = candidates()["lightgbm"].fit(X, 1 - train_small[TARGET])      # everything backwards
    assert len(clinical.check(absurd, cfg)["failures"]) >= 5


def test_clinical_failure_blocks_promotion(no_side_effects):
    cand = {**result(0.80, 0.80), "clinical": {"failures": ["45, severe hypertension: risk 20%, expected >= 60%"]}}
    g = check_gates(tr("retrain_concept", cand, result(0.80, 0.55)))
    assert not g["passed"]
    assert any(x["gate"].startswith("Clinical sanity") and not x["passed"] for x in g["results"])


def test_concept_gate_accepts_a_calibration_fix(no_side_effects):
    """Guideline change: ranking barely improves but predicted risk now matches reality."""
    champ = {**result(0.80, 0.78), "recent": {"roc_auc": 0.78, "recall": 0.70, "brier": 0.22}}
    cand = {**result(0.80, 0.785), "recent": {"roc_auc": 0.785, "recall": 0.72, "brier": 0.19}}
    assert check_gates(tr("retrain_concept", cand, champ))["passed"]


def test_calibration_drift_is_detected():
    import numpy as np
    import pandas as pd
    from monitoring.drift import analyse, load_config
    rng = np.random.default_rng(0)
    ref = pd.read_parquet("data/reference/reference.parquet").sample(1000, random_state=0)
    cur = ref.copy()
    cur["label"] = (rng.random(len(cur)) < 0.70).astype(int)     # 70% sick, model still says ~50%
    cur["label_source"] = "human"
    checks = {c["check"]: c for c in analyse(ref, cur, load_config(), reference_auc=0.80)["checks"]}
    assert checks["calibration"]["status"] == "critical"
    cur["label"] = (rng.random(len(cur)) < cur["prediction_proba"]).astype(int)   # calibrated
    checks = {c["check"]: c for c in analyse(ref, cur, load_config(), reference_auc=0.80)["checks"]}
    assert checks["calibration"]["status"] == "ok"


def test_diagnosis_uses_calibration_check():
    t = {"checks": [{"check": "psi_key_features", "status": "ok"}, {"check": "feature_drift_share", "status": "ok"},
                    {"check": "auc_drop", "status": "ok"}, {"check": "calibration", "status": "critical"},
                    {"check": "data_quality", "status": "ok"}]}
    assert diagnose(t)["diagnosis"] == "retrain_concept"


def test_guideline_scenario_only_adds_diagnoses():
    import numpy as np
    import pandas as pd
    from common.features import TARGET
    from simulate.traffic import apply_scenario
    df = pd.read_parquet("data/processed/test.parquet").sample(3000, random_state=0)
    out = apply_scenario(df, "guideline", np.random.default_rng(0))
    assert (out[TARGET] >= df.loc[out.index, TARGET]).all()          # nobody becomes healthy
    assert out[TARGET].mean() > df[TARGET].mean() + 0.1
