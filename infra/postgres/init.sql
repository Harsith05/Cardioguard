-- Runs once, on the first start of the Postgres container.
-- Extra databases used in later phases (MLflow in Phase 6, Airflow in Phase 6).
CREATE DATABASE mlflow;
CREATE DATABASE airflow;

-- The `predictions` table is created by the API on startup (SQLAlchemy).
-- Reference data for drift monitoring is loaded by training/load_reference.py.
