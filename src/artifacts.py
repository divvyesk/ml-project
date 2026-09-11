"""Model bundle persistence (Weeks 12-15).

A bare pipeline is not enough to score a single vehicle: the Week 8 features
depend on ``reference_year`` and on the training-set ``frequency_map``. The
bundle keeps the fitted pipeline and those two values together so the
Streamlit app cannot silently recompute them and feed the model something it
never saw during training.
"""

import json
from datetime import datetime

import joblib

from .config import BUNDLE_PATH, FEATURE_COLUMNS, MODEL_DIR

BUNDLE_VERSION = 2


def save_bundle(model, model_name, fit_params, metrics, path=None):
    path = path or BUNDLE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "bundle_version": BUNDLE_VERSION,
        "model": model,
        "model_name": model_name,
        "reference_year": int(fit_params["reference_year"]),
        "frequency_map": dict(fit_params["frequency_map"]),
        "feature_columns": list(FEATURE_COLUMNS),
        "metrics": metrics,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, path)
    return path


def load_bundle(path=None):
    path = path or BUNDLE_PATH
    if not path.exists():
        raise FileNotFoundError(
            "Model bundle not found at {0}. Run `python scripts/run_all.py` "
            "first.".format(path)
        )

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(
            "{0} is not a model bundle. Delete it and re-run the pipeline.".format(path)
        )

    version = bundle.get("bundle_version", 0)
    if version != BUNDLE_VERSION:
        raise ValueError(
            "Model bundle version {0} was written by a different version of "
            "this project (expected {1}). Re-run the pipeline.".format(
                version, BUNDLE_VERSION
            )
        )
    return bundle


def save_baseline(model, model_name, directory=None):
    directory = directory or MODEL_DIR
    directory.mkdir(parents=True, exist_ok=True)
    safe = model_name.lower().replace(" ", "_")
    path = directory / "baseline_{0}.joblib".format(safe)
    joblib.dump(model, path)
    return path


def write_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def read_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
