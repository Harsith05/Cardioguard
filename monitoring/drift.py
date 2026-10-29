"""Phase 2 core: compare a window of production data with the reference data.

`analyse(reference, current, cfg, ...)` is a pure function (DataFrames in, dict out),
so it is easy to test and reuse. It runs:

  1. Feature drift      - Evidently, per feature (Wasserstein / Jensen-Shannon)
  2. PSI                - on key clinical features
  3. Prediction drift   - Evidently, on the predicted probability
  4. Confidence         - mean max(p, 1-p) vs reference
  5. Performance        - ROC-AUC on human-labelled rows, if enough exist
  6. Data quality       - missing / out-of-range inputs

Each check gets a status (ok / warning / critical); the worst one wins.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from common.features import (BINARY_INPUTS, CATEGORICAL_INPUTS, INPUT_FEATURES,
                             NUMERIC_INPUTS, VALID_RANGES)

warnings.filterwarnings("ignore", category=SyntaxWarning)

CFG_PATH = Path(__file__).with_name("thresholds.yaml")
NUM_COLS = NUMERIC_INPUTS
CAT_COLS = CATEGORICAL_INPUTS + BINARY_INPUTS
RANK = {"ok": 0, "insufficient_data": 0, "warning": 1, "critical": 2}


def load_config(path: Path = CFG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


# ---------------------------------------------------------------- helpers
def add_bmi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
    return df


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10, categorical: bool = False) -> float:
    """Population Stability Index.

    Numeric: quantile bins taken from the reference. Categorical: one bin per category.
    """
    ref, cur = ref.dropna(), cur.dropna()
    if categorical:
        cats = sorted(set(ref.unique()) | set(cur.unique()))
        r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        ref, cur = ref.to_numpy(float), cur.to_numpy(float)
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def confidence(p: pd.Series) -> float:
    p = p.to_numpy(float)
    return float(np.mean(np.maximum(p, 1 - p)))


def auc_with_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 500, conf: float = 0.95,
                seed: int = 0) -> tuple[float, float, float]:
    """ROC-AUC with a percentile bootstrap confidence interval.

    With a few hundred labels the AUC swings by several points from sample noise
    alone, so a drop only counts when the whole interval sits below the reference.
    """
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], p[i]))
    a = (1 - conf) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    return auc, float(lo), float(hi)


def level(value: float, warn: float, crit: float) -> str:
    return "critical" if value >= crit else "warning" if value >= warn else "ok"


def quality_issues(df: pd.DataFrame) -> pd.Series:
    """True for rows with a missing value, an out-of-range value or ap_lo >= ap_hi."""
    bad = df[INPUT_FEATURES].isna().any(axis=1)
    for col, (lo, hi) in VALID_RANGES.items():
        bad |= ~df[col].between(lo, hi)
    bad |= df["ap_lo"] >= df["ap_hi"]
    return bad


# ---------------------------------------------------------------- Evidently
def evidently_drift(reference: pd.DataFrame, current: pd.DataFrame, cfg: dict):
    """Returns (per-feature results, prediction-drift result, Evidently snapshot)."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import ValueDrift
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    fd, pd_cfg = cfg["feature_drift"], cfg["prediction_drift"]
    cols = NUM_COLS + CAT_COLS + ["prediction_proba"]
    definition = DataDefinition(numerical_columns=NUM_COLS + ["prediction_proba"],
                                categorical_columns=CAT_COLS)
    report = Report([
        DataDriftPreset(columns=NUM_COLS + CAT_COLS,
                        num_method=fd["num_method"], num_threshold=fd["num_threshold"],
                        cat_method=fd["cat_method"], cat_threshold=fd["cat_threshold"]),
        ValueDrift(column="prediction_proba", method=pd_cfg["method"], threshold=pd_cfg["threshold"]),
        DataSummaryPreset(columns=cols),
    ])
    snapshot = report.run(Dataset.from_pandas(current[cols], data_definition=definition),
                          Dataset.from_pandas(reference[cols], data_definition=definition))

    features, prediction = [], None
    for m in snapshot.dict()["metrics"]:
        conf = m.get("config", {})
        if not conf.get("type", "").endswith(":ValueDrift"):
            continue
        score, thr = float(m["value"]), float(conf["threshold"])
        method = conf["method"]
        # distance methods: larger = more drift; p-value methods: smaller = more drift
        detected = score < thr if method in {"ks", "chisquare", "z", "fisher_exact", "g_test"} else score >= thr
        row = {"feature": conf["column"], "method": method, "score": round(score, 4),
               "threshold": thr, "detected": bool(detected)}
        if conf["column"] == "prediction_proba":
            prediction = row
        else:
            features.append(row)
    return features, prediction, snapshot


# ---------------------------------------------------------------- main analysis
def analyse(reference: pd.DataFrame, current: pd.DataFrame, cfg: dict | None = None,
            reference_auc: float | None = None, previous_prediction_drift: list[bool] | None = None,
            run_evidently: bool = True) -> dict:
    """
    reference: INPUT_FEATURES + prediction_proba (training rows)
    current:   INPUT_FEATURES + prediction_proba (+ label, label_source) (production window)
    previous_prediction_drift: detected flags of earlier runs, newest first
    """
    cfg = cfg or load_config()
    n = len(current)
    result = {"n_rows": n, "checks": [], "features": [], "snapshot": None}
    if n < cfg["min_rows"]:
        result.update(status="insufficient_data",
                      reasons=[f"Only {n} rows in the window; need at least {cfg['min_rows']}."])
        return result

    checks = []
    ref_b, cur_b = add_bmi(reference), add_bmi(current)

    # 1 + 3. Evidently feature drift and prediction drift
    feats, pred, snapshot = evidently_drift(reference, current, cfg) if run_evidently else ([], None, None)
    result["snapshot"] = snapshot

    # 2. PSI on key features (merged into the per-feature table)
    pcfg = cfg["psi"]
    psi_vals = {f: psi(ref_b[f], cur_b[f], pcfg["bins"], categorical=f in CAT_COLS)
                for f in pcfg["features"]}
    by_name = {f["feature"]: f for f in feats}
    for f, v in psi_vals.items():
        by_name.setdefault(f, {"feature": f, "method": None, "score": None, "threshold": None, "detected": False})
        by_name[f]["psi"] = round(v, 4)
    result["features"] = sorted(by_name.values(), key=lambda d: d["feature"])

    if feats:
        share = sum(f["detected"] for f in feats) / len(feats)
        fd = cfg["feature_drift"]
        drifted = [f["feature"] for f in feats if f["detected"]]
        checks.append({"check": "feature_drift_share", "value": round(share, 3),
                       "status": level(share, fd["share_warning"], fd["share_critical"]),
                       "detail": f"{len(drifted)} of {len(feats)} features drifted"
                                 + (f": {', '.join(drifted)}" if drifted else "")})

    worst_f = max(psi_vals, key=psi_vals.get)
    checks.append({"check": "psi_key_features", "value": round(psi_vals[worst_f], 3),
                   "status": level(psi_vals[worst_f], pcfg["warning"], pcfg["critical"]),
                   "detail": f"highest PSI: {worst_f}"})

    if pred:
        k = cfg["prediction_drift"]["consecutive_for_critical"]
        history = [pred["detected"]] + list(previous_prediction_drift or [])
        streak = 0
        for d in history:
            if not d:
                break
            streak += 1
        status = "critical" if streak >= k else "warning" if pred["detected"] else "ok"
        checks.append({"check": "prediction_drift", "value": pred["score"], "status": status,
                       "detail": f"detected {streak} window(s) in a row" if pred["detected"] else "no drift"})
        result["prediction_drift"] = pred

    # 4. Confidence
    c_ref, c_cur = confidence(reference["prediction_proba"]), confidence(current["prediction_proba"])
    drop = c_ref - c_cur
    ccfg = cfg["confidence"]
    checks.append({"check": "confidence_drop", "value": round(drop, 4),
                   "status": level(drop, ccfg["warning_drop"], ccfg["critical_drop"]),
                   "detail": f"mean confidence {c_cur:.1%} vs {c_ref:.1%} reference"})
    result["mean_confidence"] = round(c_cur, 4)
    result["mean_probability"] = round(float(current["prediction_proba"].mean()), 4)

    # 5. Performance on human-labelled rows only
    perf = cfg["performance"]
    labelled = current
    if "label" in current:
        labelled = current[current["label"].notna()]
        if "label_source" in current:
            labelled = labelled[labelled["label_source"] == "human"]
    else:
        labelled = current.iloc[0:0]
    result["n_labelled"] = int(len(labelled))
    if len(labelled) >= perf["min_labels"] and labelled["label"].nunique() == 2 and reference_auc:
        conf_level = perf.get("confidence_level", 0.95)
        auc, lo, hi = auc_with_ci(labelled["label"].astype(int).to_numpy(),
                                  labelled["prediction_proba"].to_numpy(float),
                                  perf.get("bootstrap", 500), conf_level)
        drop = reference_auc - auc
        significant = hi < reference_auc   # whole interval below the training AUC
        status = level(drop, perf["warning_auc_drop"], perf["critical_auc_drop"]) if significant else "ok"
        result.update(roc_auc=round(auc, 4), roc_auc_ci=[round(lo, 4), round(hi, 4)])
        note = "" if significant or drop < perf["warning_auc_drop"] else " (drop not significant; more labels needed)"
        checks.append({"check": "auc_drop", "value": round(drop, 4), "status": status,
                       "detail": f"ROC-AUC {auc:.3f} [{conf_level:.0%} CI {lo:.3f}-{hi:.3f}] vs {reference_auc:.3f} "
                                 f"at training, {len(labelled)} labelled rows{note}"})
    else:
        checks.append({"check": "auc_drop", "value": None, "status": "insufficient_data",
                       "detail": f"{len(labelled)} labelled rows; need {perf['min_labels']} with both classes"})

    # 5b. Calibration on the same rows: does the predicted risk still match how many patients are
    # really sick? Catches concept drift that keeps the ranking (AUC) but shifts the level of risk,
    # e.g. a new guideline that diagnoses disease earlier.
    cal = cfg.get("calibration")
    if cal and len(labelled) >= perf["min_labels"]:
        observed = float(labelled["label"].astype(int).mean())
        predicted = float(labelled["prediction_proba"].astype(float).mean())
        gap = observed - predicted
        se = np.sqrt(max(observed * (1 - observed), 1e-6) / len(labelled))
        significant = abs(gap) > cal.get("z", 2.58) * se            # 99% two-sided
        status = level(abs(gap), cal["warning_gap"], cal["critical_gap"]) if significant else "ok"
        result.update(observed_rate=round(observed, 4), calibration_gap=round(gap, 4))
        checks.append({"check": "calibration", "value": round(gap, 4), "status": status,
                       "detail": f"{observed:.0%} of confirmed patients are sick, the model predicted "
                                 f"{predicted:.0%} on average ({len(labelled)} labelled rows)"})
    elif cal:
        checks.append({"check": "calibration", "value": None, "status": "insufficient_data",
                       "detail": f"{len(labelled)} labelled rows; need {perf['min_labels']}"})

    # 6. Data quality
    bad_share = float(quality_issues(current).mean())
    q = cfg["data_quality"]
    checks.append({"check": "data_quality", "value": round(bad_share, 4),
                   "status": level(bad_share, q["warning_share"], q["critical_share"]),
                   "detail": f"{bad_share:.1%} of rows missing or out of range"})

    result["checks"] = checks
    result["status"] = max((c["status"] for c in checks), key=lambda s: RANK[s])
    result["reasons"] = [f"{c['check']}: {c['detail']}" for c in checks if c["status"] in ("warning", "critical")]
    return result
