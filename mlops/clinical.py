"""Clinical sanity checks ("golden patients") for any candidate model (Phase 6).

Statistical gates only ask "does it score well on recent labels?". If those labels are wrong or
strange, a model can score well and still be medically absurd (e.g. "high blood pressure =
healthy"). These checks ask "would a doctor accept these answers?":

  1. Textbook patients   clear-cut cases with a required risk range
                         (severe hypertension + smoker -> high; healthy 30-year-old -> low)
  2. Same patient, worse  raising blood pressure, cholesterol, glucose or age must not LOWER
                         the predicted risk (more than a small tolerance)

The cases live in mlops/config.yaml (section `clinical`) so a clinician can edit them.
"""
from __future__ import annotations

import pandas as pd

from common.features import INPUT_FEATURES

BASE = {"gender": 1, "height": 168, "weight": 72, "gluc": 1, "smoke": 0, "alco": 0, "active": 1}


def _row(case: dict) -> dict:
    return {**BASE, **{k: v for k, v in case.items() if k in INPUT_FEATURES}}


def _worse(base: dict, change: dict) -> dict:
    """Apply a change: "+30" adds to the patient's value, a plain number replaces it."""
    row = _row(base)
    for k, v in change.items():
        row[k] = row.get(k, 0) + float(v) if isinstance(v, str) else v
    return row


def check(model, cfg: dict) -> dict:
    """Returns {"n_checks", "failures": [text...], "cases": [...]} for one model."""
    cases = cfg["cases"]
    rows = [_row(c["patient"]) for c in cases]
    for base in cfg["worsening"]["patients"]:
        rows.append(_row(base))
        rows += [_worse(base, change) for change in cfg["worsening"]["changes"].values()]
    risk = model.predict_proba(pd.DataFrame(rows)[INPUT_FEATURES].astype(float))[:, 1]

    failures, out, i = [], [], 0
    for c in cases:
        r = float(risk[i]); i += 1
        ok = c.get("min_risk", 0) <= r <= c.get("max_risk", 1)
        out.append({"case": c["name"], "risk": round(r, 3), "ok": ok})
        if not ok:
            want = (f">= {c['min_risk']:.0%}" if "min_risk" in c else f"<= {c['max_risk']:.0%}")
            failures.append(f"{c['name']}: risk {r:.0%}, expected {want}")
    tol = cfg["worsening"]["tolerance"]
    n_pairs = 0
    for base in cfg["worsening"]["patients"]:
        r0 = float(risk[i]); i += 1
        for name, change in cfg["worsening"]["changes"].items():
            r1 = float(risk[i]); i += 1; n_pairs += 1
            if r1 < r0 - tol:
                failures.append(f"{base.get('label', 'patient')} with {name}: risk fell {r0:.0%} -> {r1:.0%}")
    return {"n_checks": len(cases) + n_pairs, "failures": failures, "cases": out}
