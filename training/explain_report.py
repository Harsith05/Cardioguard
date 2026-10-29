"""SHAP figures for the paper (Phase 1 explainability).

Saves to reports/shap/:
  summary_beeswarm.png   - every patient's SHAP value per feature, coloured by feature value
  importance_bar.png     - mean |SHAP| per feature (global importance)
  importance_groups.png  - mean |SHAP| per readable group (blood pressure, body size, ...)
  waterfall_high.png     - one high-risk test patient, step by step from base value
  waterfall_low.png      - one low-risk test patient
  global_importance.json - the numbers behind the bar charts

Usage:
    python -m training.explain_report --n 2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from common.explain import EXPLAINED_FEATURES, LABELS, PipelineExplainer  # noqa: E402
from common.features import INPUT_FEATURES  # noqa: E402

OUT = Path("reports/shap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="test patients to explain")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    pipe = joblib.load("models/model.joblib")
    ref = pd.read_parquet("data/reference/reference.parquet")[INPUT_FEATURES]
    test = pd.read_parquet("data/processed/test.parquet")
    X = test[INPUT_FEATURES].sample(min(args.n, len(test)), random_state=0).reset_index(drop=True)

    ex = PipelineExplainer(pipe, background=ref)
    raw, base = ex._raw_shap(X)
    values = ex._per_feature(raw)                        # (n, 14) one-hot summed
    feat_values = ex.fe.transform(X)[EXPLAINED_FEATURES].round(1)  # numbers used for colouring
    names = [LABELS[f] for f in EXPLAINED_FEATURES]
    unit = "log-odds" if ex.output_space == "log_odds" else "probability"

    exp = shap.Explanation(values=values, base_values=np.full(len(X), base),
                           data=feat_values.values, feature_names=names)

    plt.figure()
    shap.plots.beeswarm(exp, max_display=14, show=False)
    plt.xlabel(f"SHAP value ({unit}): impact on predicted risk")
    plt.title("How each feature moves predicted cardiovascular risk")
    plt.tight_layout(); plt.savefig(OUT / "summary_beeswarm.png", dpi=200); plt.close()

    plt.figure()
    shap.plots.bar(exp, max_display=14, show=False)
    plt.title("Global feature importance (mean |SHAP|)")
    plt.tight_layout(); plt.savefig(OUT / "importance_bar.png", dpi=200); plt.close()

    gi = ex.global_importance(X)
    g = pd.DataFrame(gi["groups"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(g["label"], g["mean_abs_shap"], color="#0F5E66")
    ax.set_xlabel(f"mean |SHAP| ({unit})"); ax.set_title("Importance by clinical factor")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "importance_groups.png", dpi=200); plt.close(fig)

    proba = pipe.predict_proba(X)[:, 1]
    for tag, idx in [("high", int(np.argmax(proba))), ("low", int(np.argmin(proba)))]:
        plt.figure()
        shap.plots.waterfall(exp[idx], max_display=10, show=False)
        plt.title(f"{tag.capitalize()}-risk patient: predicted {proba[idx]:.0%}")
        plt.tight_layout(); plt.savefig(OUT / f"waterfall_{tag}.png", dpi=200, bbox_inches="tight"); plt.close()

    (OUT / "global_importance.json").write_text(json.dumps(gi, indent=2))
    # sanity check: SHAP values must add up to the model's output
    recon = base + values.sum(axis=1)
    recon_p = ex.to_probability(recon) if ex.output_space == "probability" else 1 / (1 + np.exp(-recon))
    print(f"Explained {len(X)} patients; max additivity error {np.abs(recon_p - proba).max():.2e}")
    print(f"Figures saved to {OUT}/")


if __name__ == "__main__":
    main()
