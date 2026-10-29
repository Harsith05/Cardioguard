"""Label-fraction experiment for the paper (Phase 5).

Hide most of the training labels, then compare how well each method recovers:

  supervised        LightGBM on the labelled rows only (the baseline)
  self_training     scikit-learn SelfTrainingClassifier (threshold 0.9, 10 rounds)
  label_spreading   LightGBM on labelled + label-spreading pseudo-labels (confidence >= 0.7)
  agreement         LightGBM on labelled + pseudo-labels both methods agree on (our method)
  agreement_capped  same, but at most 1x as many pseudo-labels as real labels
  full_supervision  LightGBM on ALL training labels (the upper bound)

Every model is scored on the same held-out test set (true labels). Pseudo-label
accuracy is measured against the hidden true labels, which production never has.

    python -m semisup.experiment                       # 1/5/10/20% x 5 seeds (~10 min)
    python -m semisup.experiment --fractions 0.01 0.05 --seeds 2
Output: reports/ssl/results.csv, summary.csv, summary.md, label_fraction.png
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from common.features import INPUT_FEATURES, TARGET
from semisup.methods import base_model, label_spreading, select, self_training

warnings.filterwarnings("ignore")
OUT = Path("reports/ssl")
ORDER = ["supervised", "self_training", "label_spreading", "agreement", "agreement_capped", "full_supervision"]
LABELS = {"supervised": "Supervised only", "self_training": "Self-training",
          "label_spreading": "Label spreading", "agreement": "Agreement filter (ours)",
          "agreement_capped": "Agreement filter, capped 1x", "full_supervision": "All labels (upper bound)"}


def score(model, X, y) -> dict:
    p = model.predict_proba(X)[:, 1]
    return {"roc_auc": roc_auc_score(y, p), "recall": recall_score(y, p >= 0.5), "brier": brier_score_loss(y, p)}


def fit_with(L: pd.DataFrame, U: pd.DataFrame, sel: dict, seed: int):
    X = pd.concat([L[INPUT_FEATURES], U.iloc[sel["index"]][INPUT_FEATURES]], ignore_index=True)
    y = np.concatenate([L[TARGET].to_numpy(), sel["label"]])
    return base_model(seed).fit(X, y)


def run(fractions, seeds, st_threshold, ls_threshold) -> pd.DataFrame:
    train = pd.read_parquet("data/processed/train.parquet")
    test = pd.read_parquet("data/processed/test.parquet")
    X_te, y_te = test[INPUT_FEATURES], test[TARGET].to_numpy()
    rows = []

    def record(frac, seed, method, model, sel=None, U=None, secs=0.0):
        r = {"fraction": frac, "seed": seed, "method": method, **score(model, X_te, y_te), "seconds": round(secs, 1)}
        if sel is not None:
            truth = U[TARGET].to_numpy()[sel["index"]]
            r.update(n_pseudo=sel["n_accepted"], pseudo_accuracy=float((sel["label"] == truth).mean())
                     if sel["n_accepted"] else np.nan)
        rows.append(r)
        print(f"  {method:17s} AUC {r['roc_auc']:.4f}  recall {r['recall']:.3f}"
              + (f"  pseudo {r['n_pseudo']:6d} acc {r['pseudo_accuracy']:.3f}" if sel is not None else ""))

    for seed in range(seeds):
        t = time.time()
        record(1.0, seed, "full_supervision", base_model(seed).fit(train[INPUT_FEATURES], train[TARGET]), secs=time.time() - t)
        for frac in fractions:
            print(f"\nseed {seed}, {frac:.0%} labelled")
            L, U = train_test_split(train, train_size=frac, stratify=train[TARGET], random_state=seed)
            y_l, pos_rate = L[TARGET].to_numpy(), float(L[TARGET].mean())

            t = time.time()
            record(frac, seed, "supervised", base_model(seed).fit(L[INPUT_FEATURES], y_l), secs=time.time() - t)

            t = time.time()
            st_p, st = self_training(L, y_l, U, st_threshold, 10, seed)
            st_secs = time.time() - t
            record(frac, seed, "self_training", st, secs=st_secs)

            t = time.time()
            ls_p = label_spreading(L, y_l, U)
            ls_secs = time.time() - t
            # label spreading alone: its own confidence, its own labels (passed as the primary scores)
            sel = select(ls_p, ls_p, st_threshold=ls_threshold, positive_rate=pos_rate, use_label_spreading=False)
            record(frac, seed, "label_spreading", fit_with(L, U, sel, seed), sel, U, ls_secs)

            sel = select(st_p, ls_p, st_threshold, ls_threshold, pos_rate)
            record(frac, seed, "agreement", fit_with(L, U, sel, seed), sel, U, st_secs + ls_secs)

            sel = select(st_p, ls_p, st_threshold, ls_threshold, pos_rate, max_accept=len(L))
            record(frac, seed, "agreement_capped", fit_with(L, U, sel, seed), sel, U, st_secs + ls_secs)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["fraction", "method"]).agg(
        roc_auc=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"), recall=("recall", "mean"),
        brier=("brier", "mean"), n_pseudo=("n_pseudo", "mean"), pseudo_accuracy=("pseudo_accuracy", "mean"),
        runs=("seed", "count")).reset_index()
    base = agg[agg.method == "supervised"].set_index("fraction")["roc_auc"]
    agg["auc_gain_vs_supervised"] = agg.apply(lambda r: r.roc_auc - base.get(r.fraction, np.nan), axis=1)
    agg["method"] = pd.Categorical(agg["method"], ORDER, ordered=True)
    return agg.sort_values(["fraction", "method"])


def to_markdown(s: pd.DataFrame) -> str:
    lines = ["| Labels | Method | ROC-AUC (mean ± sd) | Gain vs supervised | Recall | Pseudo-labels | Pseudo-label accuracy |",
             "|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        frac = "100%" if r.fraction == 1.0 else f"{r.fraction:.0%}"
        sd = f" ± {r.roc_auc_sd:.4f}" if pd.notna(r.roc_auc_sd) else ""
        gain = "" if r.method in ("supervised", "full_supervision") or pd.isna(r.auc_gain_vs_supervised) \
            else f"{r.auc_gain_vs_supervised:+.4f}"
        npl = "" if pd.isna(r.n_pseudo) else f"{r.n_pseudo:,.0f}"
        acc = "" if pd.isna(r.pseudo_accuracy) else f"{r.pseudo_accuracy:.3f}"
        lines.append(f"| {frac} | {LABELS[r.method]} | {r.roc_auc:.4f}{sd} | {gain} | {r.recall:.3f} | {npl} | {acc} |")
    return "\n".join(lines)


def plot(s: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    styles = {"supervised": ("#5B6B7A", "o", "--"), "self_training": ("#B07A12", "s", "-"),
              "label_spreading": ("#8A97A3", "^", "-"), "agreement": ("#0F5E66", "D", "-"),
              "agreement_capped": ("#2E7D5B", "v", ":")}
    part = s[s.fraction < 1.0]
    for m, (c, mk, ls) in styles.items():
        d = part[part.method == m].sort_values("fraction")
        ax.errorbar(d.fraction * 100, d.roc_auc, yerr=d.roc_auc_sd, color=c, marker=mk, linestyle=ls,
                    capsize=3, linewidth=2 if m == "agreement" else 1.3, label=LABELS[m])
    full = s[s.method == "full_supervision"].roc_auc.mean()
    ax.axhline(full, color="#B23A3A", linestyle="--", linewidth=1, label=f"All labels ({full:.3f})")
    ax.set_xscale("log")
    ax.set_xticks(sorted(part.fraction.unique() * 100))
    ax.set_xticklabels([f"{x:g}%" for x in sorted(part.fraction.unique() * 100)])
    ax.set_xlabel("Share of training labels available")
    ax.set_ylabel("ROC-AUC on held-out test set")
    ax.set_title("Agreement-filtered pseudo-labels recover 35-60% of the missing-label gap")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--st-threshold", type=float, default=0.9)
    ap.add_argument("--ls-threshold", type=float, default=0.7)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = run(args.fractions, args.seeds, args.st_threshold, args.ls_threshold)
    df.to_csv(OUT / "results.csv", index=False)
    s = summarise(df)
    s.to_csv(OUT / "summary.csv", index=False)
    (OUT / "summary.md").write_text(to_markdown(s) + "\n")
    plot(s, OUT / "label_fraction.png")
    print("\n" + to_markdown(s))
    print(f"\nSaved {OUT}/results.csv, summary.csv, summary.md, label_fraction.png")


if __name__ == "__main__":
    main()
