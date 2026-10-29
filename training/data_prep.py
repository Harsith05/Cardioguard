"""Phase 1 - Step 1: clean the raw dataset and create train/val/test splits.

Usage:
    python -m training.data_prep --input data/raw/cardio_train.xlsx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common.features import INPUT_FEATURES, TARGET, VALID_RANGES

SEED = 42
OUT = Path("data/processed")


def load_raw(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    # Kaggle's original file is ';' separated
    return pd.read_csv(path, sep=None, engine="python")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"raw_rows": len(df)}
    df = df.copy()
    df["age_years"] = (df["age"] / 365.25).round(1)

    # 1) out-of-range values
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in VALID_RANGES.items():
        bad = ~df[col].between(lo, hi)
        log[f"dropped_{col}_out_of_range"] = int((bad & mask).sum())
        mask &= ~bad
    # 2) diastolic must be below systolic
    bad = df["ap_lo"] >= df["ap_hi"]
    log["dropped_ap_lo_ge_ap_hi"] = int((bad & mask).sum())
    mask &= ~bad
    df = df[mask]

    # 3) exact duplicates (same features + label) -> avoid leakage across splits
    before = len(df)
    df = df.drop_duplicates(subset=INPUT_FEATURES + [TARGET])
    log["dropped_duplicates"] = before - len(df)

    df = df[["id"] + INPUT_FEATURES + [TARGET]].reset_index(drop=True)
    log["clean_rows"] = len(df)
    log["removed_pct"] = round(100 * (1 - len(df) / log["raw_rows"]), 2)
    log["positive_rate"] = round(float(df[TARGET].mean()), 4)
    return df, log


def split(df: pd.DataFrame):
    train, temp = train_test_split(df, test_size=0.30, stratify=df[TARGET], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp[TARGET], random_state=SEED)
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/cardio_train.xlsx")
    args = ap.parse_args()

    df, log = clean(load_raw(Path(args.input)))
    train, val, test = split(df)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_parquet(OUT / f"{name}.parquet", index=False)
        log[f"{name}_rows"] = len(part)

    (OUT / "cleaning_report.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
