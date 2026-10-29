import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Tests use a throwaway SQLite file instead of PostgreSQL, and skip MongoDB
TEST_DB = ROOT / "tests" / "_test.db"
if TEST_DB.exists():
    TEST_DB.unlink()
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB}"
os.environ["MONGO_URI"] = ""
os.environ["PUSHGATEWAY_URL"] = ""   # no Pushgateway in tests
os.environ["MLFLOW_TRACKING_URI"] = ""   # Phase 6: no registry in tests, models/model.joblib is the champion
os.environ["AIRFLOW_URL"] = ""
os.environ["CHAOS_ENABLED"] = "1"    # so the fault-injection test can run
# Keep test drift reports away from real ones in reports/drift
os.environ["DRIFT_REPORT_DIR"] = str(ROOT / "tests" / "_reports")
