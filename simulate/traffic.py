"""Simulate production traffic so drift monitoring has something to watch.

Patients come from the held-out test set (never seen in training). A scenario
changes them before they are sent to the API, and a share of them later gets
its true diagnosis through /feedback, the way doctors would confirm cases.

Scenarios
  normal       unchanged test patients                          -> should stay OK
  bp_shift     systolic +20, diastolic +10 mmHg (e.g. a mis-calibrated BP device;
               the true diagnosis does not change)              -> data + performance drift
  aging        everyone 8 years older (an older clinic population) -> data drift
  cholesterol  far more "above normal" cholesterol readings,
               unrelated to the real diagnosis                  -> data + performance drift
  concept      inputs unchanged, but the diagnosis flips for patients under 50
               (stress test: medically impossible on purpose)   -> performance drift only
  guideline    inputs unchanged; a new guideline diagnoses disease earlier: 60% of
               undiagnosed patients with BP >= 130/80 are now diagnosed
               (realistic concept drift)                         -> calibration drift

Usage (API must be running):
    python -m simulate.traffic --scenario normal --n 1000 --label-rate 0.2
    python -m simulate.traffic --scenario bp_shift --n 1000
"""
from __future__ import annotations

import argparse
import time

import httpx
import numpy as np
import pandas as pd

from common.features import INPUT_FEATURES, TARGET, VALID_RANGES

SCENARIOS = ["normal", "bp_shift", "aging", "cholesterol", "concept", "guideline"]


def apply_scenario(df: pd.DataFrame, scenario: str, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    if scenario == "bp_shift":
        df["ap_hi"] = df["ap_hi"] + 20
        df["ap_lo"] = df["ap_lo"] + 10
    elif scenario == "aging":
        df["age_years"] = df["age_years"] + 8
    elif scenario == "cholesterol":
        df["cholesterol"] = rng.choice([1, 2, 3], len(df), p=[0.30, 0.35, 0.35])
    elif scenario == "concept":
        young = df["age_years"] < 50
        df.loc[young, TARGET] = 1 - df.loc[young, TARGET]
    elif scenario == "guideline":
        stage1 = ((df["ap_hi"] >= 130) | (df["ap_lo"] >= 80)) & (df[TARGET] == 0)
        df.loc[stage1 & (rng.random(len(df)) < 0.6), TARGET] = 1
    for col, (lo, hi) in VALID_RANGES.items():          # keep inputs valid for the API
        df[col] = df[col].clip(lo, hi)
    df = df[df["ap_lo"] < df["ap_hi"]]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=SCENARIOS, default="normal")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--label-rate", type=float, default=0.2, help="share of patients whose diagnosis is confirmed")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    test = pd.read_parquet("data/processed/test.parquet")
    df = apply_scenario(test.sample(args.n, random_state=int(rng.integers(1e9))), args.scenario, rng)

    sent, to_label = 0, []
    with httpx.Client(base_url=args.api, timeout=60) as client:
        health = client.get("/health").json()
        if not health.get("logging", {}).get("sql"):
            raise SystemExit(
                "The API is running but is NOT connected to PostgreSQL, so predictions and\n"
                "diagnoses would not be saved and drift checks have nothing to read.\n"
                "Check the API startup log for 'SQL logging disabled' and fix DATABASE_URL.")
        for start in range(0, len(df), 500):
            chunk = df.iloc[start:start + 500]
            patients = chunk[INPUT_FEATURES].astype(float).to_dict("records")
            for p in patients:  # integer-coded fields must be ints for the API
                for k in ("gender", "cholesterol", "gluc", "smoke", "alco", "active"):
                    p[k] = int(p[k])
            r = client.post("/predict/batch", json={"patients": patients})
            r.raise_for_status()
            preds = r.json()["predictions"]
            sent += len(preds)
            for pred, label in zip(preds, chunk[TARGET].astype(int)):
                if rng.random() < args.label_rate:
                    to_label.append((pred["request_id"], int(label)))

        # Diagnoses arrive after the predictions are stored (the API logs in the background)
        time.sleep(1.0)
        labelled = 0
        for rid, label in to_label:
            for _ in range(3):
                if client.post(f"/feedback/{rid}", json={"label": label}).status_code == 200:
                    labelled += 1
                    break
                time.sleep(0.5)

    print(f"Scenario '{args.scenario}': sent {sent} patients, confirmed {labelled} diagnoses.")


if __name__ == "__main__":
    main()
