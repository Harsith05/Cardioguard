"""Model serving with a champion and an optional candidate (Phase 6).

  champion   serves every request, except...
  candidate  ...the share `canary_fraction` of requests during a canary release

The live switch is the `model_deployment` row (mlops/deployment.py). A background thread
reads it every DEPLOYMENT_POLL_SECONDS and loads/unloads models in memory when it changes,
so every API replica follows the same decision with no restart.

Models come from the MLflow registry when MLFLOW_TRACKING_URI is set; otherwise (or if the
registry is unreachable) the champion is models/model.joblib as in Phases 1-5.
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
from dataclasses import dataclass, field

import joblib
import pandas as pd

from app import config
from common.features import INPUT_FEATURES

log = logging.getLogger("cardioguard.model")
POLL_SECONDS = float(os.getenv("DEPLOYMENT_POLL_SECONDS", "15"))


@dataclass
class LoadedModel:
    model: object
    meta: dict
    registry_version: int | None = None
    explainer: object = None
    global_importance: dict | None = field(default=None, repr=False)

    @property
    def version(self) -> str:
        return self.meta.get("model_version", "unknown")

    @property
    def threshold(self) -> float:
        return float(self.meta.get("threshold", 0.5))


class ModelService:
    def __init__(self):
        self._lock = threading.Lock()
        self.champion: LoadedModel | None = None
        self.candidate: LoadedModel | None = None
        self.canary_fraction: float = 0.0
        self.status: str = "stable"
        self._engine = None
        self._stop = threading.Event()

    # ------------------------------------------------------------ loading
    def load(self, engine=None):
        """Called at startup: champion from the deployment state / registry, else from file."""
        self._engine = engine
        state = self._read_state()
        champ = None
        if state and state.get("champion_registry_version"):
            champ = self._load_registry(state["champion_registry_version"])
        if champ is None:
            champ = self._load_file()
        with self._lock:
            self.champion = champ
        if state:
            self._apply_candidate(state)
        self._publish_metrics()

    def _load_file(self) -> LoadedModel:
        model = joblib.load(config.MODEL_PATH)
        meta = json.loads(config.MODEL_META_PATH.read_text())
        return self._prepare(model, meta, None)

    def _load_registry(self, version) -> LoadedModel | None:
        try:
            from mlops import registry
            if not registry.enabled():
                return None
            model, meta = registry.load(version)
            return self._prepare(model, meta, int(meta.get("registry_version", version)))
        except Exception as e:
            log.error("Could not load model version %s from MLflow: %s", version, e)
            return None

    def _prepare(self, model, meta, registry_version) -> LoadedModel:
        self._check_environment(meta)
        explainer, global_imp = self._build_explainer(model)
        lm = LoadedModel(model, meta, registry_version, explainer, global_imp)
        log.info("Loaded model %s (MLflow version %s)", lm.version, registry_version or "-")
        return lm

    @staticmethod
    def _check_environment(meta: dict) -> None:
        """Warn when the model was saved with different library versions than this process has."""
        import importlib.metadata as md
        saved = meta.get("environment")
        if not saved:
            log.warning("Model has no recorded library versions (trained before Phase 4). "
                        "Retrain to be safe: docker compose --profile tools run --rm trainer")
            return
        for lib, saved_v in saved.items():
            if lib not in ("scikit-learn", "lightgbm", "xgboost"):
                continue
            try:
                now_v = md.version(lib)
            except md.PackageNotFoundError:
                now_v = md.version("xgboost-cpu") if lib == "xgboost" else "missing"
            if now_v != saved_v:
                log.warning("Model saved with %s %s but running %s %s. Retrain if predictions look wrong.",
                            lib, saved_v, lib, now_v)

    @staticmethod
    def _build_explainer(model):
        """SHAP is optional: if it fails, predictions keep working without explanations."""
        try:
            from common.explain import PipelineExplainer
            ref = pd.read_parquet(config.REFERENCE_PATH)[INPUT_FEATURES]
            explainer = PipelineExplainer(model, background=ref)
            sample = ref.sample(min(1000, len(ref)), random_state=0)
            return explainer, explainer.global_importance(sample)
        except Exception as e:
            log.warning("SHAP explanations disabled: %s", e)
            return None, None

    # ------------------------------------------------------------ hot reload
    def _read_state(self) -> dict | None:
        if self._engine is None:
            return None
        try:
            from mlops import deployment
            return deployment.get_state(self._engine)
        except Exception as e:
            log.debug("Deployment state unavailable: %s", e)
            return None

    def _apply_candidate(self, state: dict):
        want = state.get("candidate_registry_version") if state.get("status") in ("canary",) else None
        cand = self.candidate
        if want and (cand is None or cand.registry_version != want):
            cand = self._load_registry(want)
        elif not want:
            cand = None
        with self._lock:
            self.candidate = cand
            self.canary_fraction = float(state.get("canary_fraction") or 0.0) if cand else 0.0
            self.status = state.get("status") or "stable"

    def sync(self):
        """Bring the in-memory models in line with the deployment state (called by the watcher)."""
        state = self._read_state()
        if not state:
            return
        want = state.get("champion_registry_version")
        if want and (self.champion is None or self.champion.registry_version != want):
            # the candidate may already be loaded (promotion): reuse it instead of downloading again
            new = self.candidate if (self.candidate and self.candidate.registry_version == want) else self._load_registry(want)
            if new is not None:
                with self._lock:
                    old = self.champion.version if self.champion else None
                    self.champion = new
                log.warning("Champion switched %s -> %s", old, new.version)
        self._apply_candidate(state)
        self._publish_metrics()

    def start_watcher(self):
        def loop():
            while not self._stop.wait(POLL_SECONDS):
                try:
                    self.sync()
                except Exception:
                    log.exception("Deployment sync failed")
        threading.Thread(target=loop, name="deployment-watcher", daemon=True).start()

    def stop_watcher(self):
        self._stop.set()

    def _publish_metrics(self):
        from app import metrics
        metrics.set_models(self.champion, self.candidate, self.canary_fraction, self.status)

    # ------------------------------------------------------------ serving
    def choose(self) -> LoadedModel:
        """Pick the model for one request: candidate with probability canary_fraction."""
        with self._lock:
            champ, cand, frac = self.champion, self.candidate, self.canary_fraction
        if cand is not None and frac > 0 and random.random() < frac:
            return cand
        return champ

    def predict_with(self, lm: LoadedModel, rows: list[dict]) -> list[float]:
        X = pd.DataFrame(rows)[INPUT_FEATURES]
        return lm.model.predict_proba(X)[:, 1].tolist()

    # backwards-compatible helpers (champion)
    @property
    def ready(self) -> bool:
        return self.champion is not None

    @property
    def meta(self) -> dict:
        return self.champion.meta if self.champion else {}

    @property
    def version(self) -> str:
        return self.champion.version if self.champion else "unknown"

    @property
    def threshold(self) -> float:
        return self.champion.threshold if self.champion else 0.5

    @property
    def explainer(self):
        return self.champion.explainer if self.champion else None

    @property
    def global_importance(self):
        return self.champion.global_importance if self.champion else None

    def predict(self, rows: list[dict]) -> list[float]:
        return self.predict_with(self.champion, rows)

    def explain(self, rows: list[dict], lm: LoadedModel | None = None) -> list[dict]:
        lm = lm or self.champion
        if lm is None or lm.explainer is None:
            raise RuntimeError("Explanations are not available")
        return lm.explainer.explain(rows)


model_service = ModelService()
