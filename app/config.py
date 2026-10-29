"""Settings read from environment variables (see .env.example)."""
import os
from pathlib import Path

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
MODEL_META_PATH = Path(os.getenv("MODEL_META_PATH", "models/model_meta.json"))
# Training sample used for SHAP background + global importance
REFERENCE_PATH = Path(os.getenv("REFERENCE_PATH", "data/reference/reference.parquet"))

# PostgreSQL in Docker; any SQLAlchemy URL works (tests use SQLite)
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg2://cardio:cardio@localhost:5432/cardio")
# Leave empty to disable raw-request logging to MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "cardioguard")
MAX_BATCH = int(os.getenv("MAX_BATCH", "1000"))
