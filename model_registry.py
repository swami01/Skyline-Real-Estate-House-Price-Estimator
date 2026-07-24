"""
A minimal model registry: every training run is saved as its own
versioned artifact, never overwriting a previous one. A separate
"production.json" pointer says which version is currently live. This
is a small, dependency-free version of what tools like MLflow's Model
Registry or SageMaker Model Registry do at scale.

Layout:
    models/
      registry.json         # append-only log of every trained version
      production.json        # {"active_version": "..."} -- the pointer
      versions/
        v1_20260718T114500Z/
          model.joblib
          metadata.json
        v2_20260719T091200Z/
          ...

Why this exists instead of just overwriting one model.joblib each time:
  - You can always roll back to a previous version if a new one turns
    out worse in practice, without retraining.
  - You get an audit trail: what changed, when, and how metrics moved
    between versions -- useful both for debugging and for explaining
    your own project's history later.
  - "Deploy" becomes a metadata operation (flip the pointer), not a
    file-copy operation -- which is the same shape real model-serving
    systems use, just without the infrastructure.
"""

import json
import os
import shutil
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VERSIONS_DIR = os.path.join(MODELS_DIR, "versions")
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")
PRODUCTION_PATH = os.path.join(MODELS_DIR, "production.json")


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _new_version_id() -> str:
    registry = _load_json(REGISTRY_PATH, [])
    n = len(registry) + 1
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"v{n}_{ts}"


def register_model(model, metadata: dict, promote_if_better_on="rmse") -> dict:
    """Saves a newly trained model as a new version, appends it to the
    registry, and promotes it to production only if it beats the
    current production model on `promote_if_better_on` (lower is
    better -- RMSE). If there's no production model yet, it's promoted
    automatically.

    Returns the registry entry for this version, including whether it
    was promoted.
    """
    import joblib  # local import so this module has no hard dependency at import time

    version_id = _new_version_id()
    version_dir = os.path.join(VERSIONS_DIR, version_id)
    os.makedirs(version_dir, exist_ok=True)

    model_path = os.path.join(version_dir, "model.joblib")
    joblib.dump(model, model_path, compress=3)

    metadata = dict(metadata)
    metadata["version_id"] = version_id
    metadata["created_at"] = datetime.now(timezone.utc).isoformat()
    _save_json(os.path.join(version_dir, "metadata.json"), metadata)

    registry = _load_json(REGISTRY_PATH, [])
    new_score = metadata["test_metrics"][promote_if_better_on]

    current_prod = _load_json(PRODUCTION_PATH, None)
    promote = current_prod is None
    if current_prod is not None:
        current_entry = next((r for r in registry if r["version_id"] == current_prod["active_version"]), None)
        if current_entry is not None:
            current_score = current_entry["test_metrics"][promote_if_better_on]
            promote = bool(new_score < current_score)  # lower RMSE is better; bool() avoids numpy.bool_

    entry = {
        "version_id": version_id,
        "model_name": metadata["model_name"],
        "test_metrics": metadata["test_metrics"],
        "created_at": metadata["created_at"],
        "promoted": promote,
    }
    registry.append(entry)
    _save_json(REGISTRY_PATH, registry)

    if promote:
        _save_json(PRODUCTION_PATH, {
            "active_version": version_id,
            "promoted_at": metadata["created_at"],
        })

    return entry


def load_production_model():
    """Loads whichever model is currently marked production. Raises
    FileNotFoundError with a clear message if nothing has been trained
    yet -- callers (the API) turn that into a proper 503 response.
    """
    import joblib

    prod = _load_json(PRODUCTION_PATH, None)
    if prod is None:
        raise FileNotFoundError("No production model registered. Run `python3 train.py` first.")

    version_dir = os.path.join(VERSIONS_DIR, prod["active_version"])
    model = joblib.load(os.path.join(version_dir, "model.joblib"))
    with open(os.path.join(version_dir, "metadata.json")) as f:
        metadata = json.load(f)
    return model, metadata


def list_versions():
    return _load_json(REGISTRY_PATH, [])


def get_production_info():
    return _load_json(PRODUCTION_PATH, None)


def rollback_to(version_id: str) -> dict:
    """Manually points production at a previous version -- e.g. if a
    newer model looks good on paper (lower test RMSE) but behaves
    badly in practice. Real registries support this; ours does too.
    """
    registry = _load_json(REGISTRY_PATH, [])
    if not any(r["version_id"] == version_id for r in registry):
        raise ValueError(f"Unknown version_id: {version_id}")

    prod = {"active_version": version_id, "promoted_at": datetime.now(timezone.utc).isoformat()}
    _save_json(PRODUCTION_PATH, prod)
    return prod
