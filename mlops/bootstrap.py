"""First start of Phase 6: make sure a champion exists. Safe to run every time.

  * MLflow configured and no champion yet -> register models/model.joblib, alias it "champion"
  * deployment row missing                 -> create it with that champion

    python -m mlops.bootstrap          (the compose "bootstrap" service runs this)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from sqlalchemy import create_engine

from mlops import deployment, registry

log = logging.getLogger("cardioguard.bootstrap")
MODEL, META = Path("models/model.joblib"), Path("models/model_meta.json")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    engine = create_engine(os.getenv("DATABASE_URL", "postgresql+psycopg2://cardio:cardio@localhost:5432/cardio"))
    meta = json.loads(META.read_text())
    state = deployment.get_state(engine)

    reg_version = None
    if registry.enabled():
        champ = registry.get("champion")
        if champ is None:
            reg_version = registry.register_files(MODEL, META)
            registry.set_alias("champion", reg_version)
            deployment.log_event(engine, "registered", meta["model_version"], reg_version, actor="bootstrap",
                                 reason="first champion from models/model.joblib")
            log.info("Registered %s as champion (MLflow version %s)", meta["model_version"], reg_version)
        else:
            reg_version = champ["version"]
            meta = {"model_version": champ["model_version"], "test": meta.get("test", {})
                    if meta.get("model_version") == champ["model_version"] else {}}
            log.info("Champion already registered: %s (MLflow version %s)", champ["model_version"], reg_version)
    else:
        log.info("MLFLOW_TRACKING_URI not set: serving models/model.joblib without a registry")

    if not state or not state.get("champion_version") or (reg_version and not state.get("champion_registry_version")):
        deployment.set_state(engine, status="stable", champion_version=meta["model_version"],
                             champion_registry_version=reg_version,
                             champion_auc=meta.get("test", {}).get("roc_auc"), candidate_version=None,
                             candidate_registry_version=None, canary_fraction=0.0, note="bootstrap")
        log.info("Deployment state: champion %s", meta["model_version"])
    else:
        log.info("Deployment state exists: %s, champion %s", state["status"], state["champion_version"])


if __name__ == "__main__":
    main()
