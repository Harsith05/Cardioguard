"""Paper experiment: does self-healing accept the right models? (Phase 6)

Two concept-drift scenarios from simulate/traffic.py, run through the REAL pipeline code
(strategies, model selection, promotion gates) without Docker:

  concept     diagnosis flipped for patients under 50 (medically impossible stress test)
  guideline   earlier diagnosis at BP >= 130/80 (realistic)

Two settings:
  before   unconstrained models, statistical gates only
  after    monotonic medical constraints + clinical sanity gate (current config)

    python -m mlops.experiment                 # ~5-10 min -> reports/self_healing/summary.md
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("reports/self_healing")


def main():
    os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp()}/exp.db"   # gate events go nowhere
    os.environ["MLFLOW_TRACKING_URI"] = ""
    import joblib
    from common.features import INPUT_FEATURES, TARGET
    from mlops import clinical
    from mlops.pipeline import _metrics, _recall_gap, check_gates, load_config, strategies
    from simulate.traffic import apply_scenario
    from training.train import candidates

    D = Path("data/processed")
    train, val = pd.read_parquet(D / "train.parquet"), pd.read_parquet(D / "val.parquet")
    test = pd.read_parquet(D / "test.parquet").sample(frac=1, random_state=0)
    champ = joblib.load("models/model.joblib")
    X = lambda d: d[INPUT_FEATURES].astype(float)                                   # noqa: E731

    def evaluate(model, recent, cfg):
        r = {"holdout": _metrics(val[TARGET], model.predict_proba(X(val))[:, 1]),
             "recent": _metrics(recent[TARGET], model.predict_proba(X(recent))[:, 1]),
             "recall_gap_sex": _recall_gap(model, val)}
        if "clinical" in cfg:
            r["clinical"] = clinical.check(model, cfg["clinical"])
        return r

    rows = []
    for scenario in ("concept", "guideline"):
        rng = np.random.default_rng(1)
        new_world = apply_scenario(test.iloc[:6000].assign(**{TARGET: test[TARGET].iloc[:6000]}), scenario, rng)
        recent_eval = new_world.iloc[:1500]
        # drift detection on 500 confirmed new-world patients (what the drift job sees)
        det = new_world.iloc[1500:2000]
        p = champ.predict_proba(X(det))[:, 1]
        from sklearn.metrics import roc_auc_score
        detect = {"auc": roc_auc_score(det[TARGET], p), "observed": det[TARGET].mean(), "predicted": p.mean()}
        for n_new in (1300, 3000):
            new_cases = new_world.iloc[2000:2000 + n_new]
            data = pd.concat([train[INPUT_FEATURES + [TARGET]].assign(source="training", w=1.0),
                              new_cases[INPUT_FEATURES + [TARGET]].assign(source="human_new", w=1.0)],
                             ignore_index=True)
            for setting in ("before", "after"):
                cfg = load_config()
                if setting == "before":
                    cfg["training"]["monotone"] = False
                    cfg.pop("clinical")
                champion = evaluate(champ, recent_eval, cfg)
                results = {}
                for strat, (rws, w) in strategies(data, "w", cfg, "retrain_concept").items():
                    for mt in cfg["training"]["candidates"]:
                        m = candidates(cfg["training"]["monotone"])[mt]
                        m.fit(X(rws), rws[TARGET], model__sample_weight=w)
                        results[f"{mt}/{strat}"] = evaluate(m, recent_eval, cfg)
                best = max(results, key=lambda k: (not (results[k].get("clinical") or {}).get("failures"),
                                                   results[k]["recent"]["roc_auc"]))
                g = check_gates({"diagnosis": "retrain_concept", "candidate": results[best], "champion": champion,
                                 "model_version": "exp", "registry_version": None})
                failed = [x["gate"].split(" (")[0].split(":")[0] for x in g["results"] if not x["passed"]]
                c = results[best]
                outcome = ("REJECTED: " + ", ".join(failed)) if not g["passed"] else \
                          ("needs approval" if g.get("needs_approval") else "auto-canary")
                rows.append({"scenario": scenario, "new cases": n_new, "setting": setting, "chosen": best,
                             "new-world AUC": round(c["recent"]["roc_auc"], 3),
                             "champion new-world AUC": round(champion["recent"]["roc_auc"], 3),
                             "Brier new": round(c["recent"]["brier"], 3), "champion Brier": round(champion["recent"]["brier"], 3),
                             "old-world AUC": round(c["holdout"]["roc_auc"], 3),
                             "clinical failures": len((c.get("clinical") or {}).get("failures", [])) if setting == "after"
                             else len(clinical.check(_refit(best, data, cfg, candidates, X, TARGET, strategies), load_config()["clinical"])["failures"]),
                             "outcome": outcome})
                print(rows[-1])
        rows.append({"scenario": scenario, "setting": "drift check",
                     "outcome": f"champion AUC {detect['auc']:.3f}; {detect['observed']:.0%} sick vs "
                                f"{detect['predicted']:.0%} predicted"})
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False)
    (OUT / "summary.md").write_text(df.fillna("").to_markdown(index=False))
    print(f"\nWrote {OUT / 'summary.md'}")


def _refit(best, data, cfg, candidates, X, TARGET, strategies):
    """'before' setting has no clinical gate: refit the chosen model once to count its failures."""
    mt, strat = best.split("/")
    rws, w = strategies(data, "w", cfg, "retrain_concept")[strat]
    m = candidates(False)[mt]
    m.fit(X(rws), rws[TARGET], model__sample_weight=w)
    return m


if __name__ == "__main__":
    main()
