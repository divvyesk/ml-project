"""Week 13/15: the single inference path used by the app and the scripts.

Everything that scores a vehicle goes through here, so the app and the
command line can never drift apart in how they build features.
"""

import pandas as pd

from .artifacts import load_bundle
from .config import BASE_COLUMNS, FEATURE_COLUMNS, TARGET
from .features import add_features

INPUT_COLUMNS = [c for c in BASE_COLUMNS if c != TARGET]


def prepare_input(records, bundle):
    """Turn user input into the exact feature frame the model was fitted on."""
    if isinstance(records, dict):
        records = [records]
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    else:
        frame = pd.DataFrame(list(records))

    missing = [c for c in INPUT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError("Missing input fields: {0}".format(missing))

    frame = frame[INPUT_COLUMNS].copy()
    frame["name"] = frame["name"].astype(str).map(lambda v: " ".join(v.split()))
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["km_driven"] = pd.to_numeric(frame["km_driven"], errors="coerce")

    frame, _, _ = add_features(
        frame,
        reference_year=bundle["reference_year"],
        frequency_map=bundle["frequency_map"],
    )
    return frame[FEATURE_COLUMNS]


def predict_price(records, bundle=None):
    """Predict selling price in rupees for one or many vehicles."""
    bundle = bundle or load_bundle()
    features = prepare_input(records, bundle)
    return bundle["model"].predict(features)


def explain_inputs(records, bundle):
    """Show the derived features behind a prediction (used in the viva demo)."""
    features = prepare_input(records, bundle)
    return features[["car_age", "km_per_year", "name_frequency", "brand", "model_family"]]
