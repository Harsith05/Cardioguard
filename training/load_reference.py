"""Phase 1 - Step 3: copy the reference dataset into PostgreSQL.

Phase 2's drift job compares production rows (table `predictions`) against this
table. Re-running replaces the rows for the same model_version only.

Usage:
    python -m training.load_reference
"""
from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import create_engine, text

URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://cardio:cardio@localhost:5432/cardio")


def main():
    ref = pd.read_parquet("data/reference/reference.parquet")
    version = ref["model_version"].iloc[0]
    engine = create_engine(URL)
    with engine.begin() as conn:
        exists = conn.dialect.has_table(conn, "reference_data")
        if exists:
            conn.execute(text("DELETE FROM reference_data WHERE model_version = :v"), {"v": version})
    ref.to_sql("reference_data", engine, if_exists="append", index=False, chunksize=5000)
    print(f"Loaded {len(ref):,} reference rows for {version}")


if __name__ == "__main__":
    main()
