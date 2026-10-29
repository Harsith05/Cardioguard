"""MLflow model registry (Phase 6).

Every model is an MLflow model version of "cardioguard". Two aliases say what serves traffic:
  champion   the model answering (almost) all requests
  candidate  a new model on trial (canary) or waiting for approval

    from mlops import registry
    v = registry.register(pipeline, meta, metrics={"val_roc_auc": 0.80})
    registry.set_alias("candidate", v)
    model, meta = registry.load("champion")

The server lives at MLFLOW_TRACKING_URI (Docker: http://mlflow:5000). Artifacts go through
the server (--serve-artifacts), so clients need no shared disk.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("cardioguard.registry")
NAME = os.getenv("MLFLOW_MODEL_NAME", "cardioguard")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "cardioguard-retraining")


def tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "")


def enabled() -> bool:
    return bool(tracking_uri())


def _client():
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(tracking_uri())
    return MlflowClient(tracking_uri=tracking_uri())


# MLflow 3 saves sklearn models with skops (safer than pickle): loading refuses any type not
# explicitly trusted. We trust only types from these packages, and fail loudly otherwise.
TRUSTED_PREFIXES = ("common.", "sklearn.", "lightgbm.", "xgboost.", "numpy.", "scipy.", "collections.",
                    "builtins.", "pandas.")


def trusted_types(pipeline) -> list[str]:
    import skops.io as sio
    found = sio.get_untrusted_types(data=sio.dumps(pipeline))
    unknown = [t for t in found if not t.startswith(TRUSTED_PREFIXES)]
    if unknown:
        raise ValueError(f"Model contains types from unexpected packages, refusing to register: {unknown}")
    return found


def register(pipeline, meta: dict, metrics: dict | None = None, params: dict | None = None,
             tags: dict | None = None, run_name: str | None = None) -> int:
    """Log the sklearn pipeline + its metadata as a new registered version. Returns the version number."""
    import mlflow
    import mlflow.sklearn
    _client()
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name or meta.get("model_version")) as run:
        if params:
            mlflow.log_params({k: v for k, v in params.items() if v is not None})
        if metrics:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        mlflow.set_tags({"model_version": meta.get("model_version", ""), **(tags or {})})
        mlflow.log_dict(meta, "model_meta.json")
        types = trusted_types(pipeline)
        mlflow.log_dict({"trusted_types": types}, "trusted_types.json")
        info = mlflow.sklearn.log_model(
            sk_model=pipeline, name="model", registered_model_name=NAME, skops_trusted_types=types,
            # exact pins = the runtime image; skips slow dependency inference
            pip_requirements=["scikit-learn==1.9.1", "lightgbm==4.7.0", "xgboost-cpu==3.4.1", "pandas==3.0.5"])
        version = int(info.registered_model_version)
    client = _client()
    client.set_model_version_tag(NAME, str(version), "model_version", meta.get("model_version", ""))
    for k, v in (tags or {}).items():
        client.set_model_version_tag(NAME, str(version), k, str(v))
    log.info("Registered %s version %s (%s), run %s", NAME, version, meta.get("model_version"), run.info.run_id)
    return version


def set_alias(alias: str, version: int) -> None:
    _client().set_registered_model_alias(NAME, alias, str(version))


def delete_alias(alias: str) -> None:
    try:
        _client().delete_registered_model_alias(NAME, alias)
    except Exception:
        pass


def get(alias_or_version) -> dict | None:
    """{'version', 'run_id', 'model_version', 'tags'} for an alias or version number, or None."""
    client = _client()
    try:
        mv = (client.get_model_version(NAME, str(alias_or_version)) if str(alias_or_version).isdigit()
              else client.get_model_version_by_alias(NAME, alias_or_version))
    except Exception:
        return None
    return {"version": int(mv.version), "run_id": mv.run_id, "model_version": mv.tags.get("model_version"),
            "tags": dict(mv.tags)}


def tag(version: int, key: str, value) -> None:
    _client().set_model_version_tag(NAME, str(version), key, str(value))


def load(alias_or_version):
    """Returns (sklearn pipeline, meta dict)."""
    import mlflow
    import mlflow.sklearn
    info = get(alias_or_version)
    if info is None:
        raise LookupError(f"No model '{NAME}' with alias/version {alias_or_version!r}")
    _client()
    model = mlflow.sklearn.load_model(f"models:/{NAME}/{info['version']}")
    meta = mlflow.artifacts.load_dict(f"runs:/{info['run_id']}/model_meta.json")
    meta["registry_version"] = info["version"]
    return model, meta


def register_files(model_path: Path, meta_path: Path, tags: dict | None = None) -> int:
    """Register an existing models/model.joblib (used once, when the registry is empty)."""
    import joblib
    meta = json.loads(Path(meta_path).read_text())
    return register(joblib.load(model_path), meta, metrics={f"test_{k}": v for k, v in meta.get("test", {}).items()},
                    tags={"source": "bootstrap", **(tags or {})}, run_name=f"bootstrap-{meta['model_version']}")
