"""Week 8: feature engineering.

Two values decided at training time must travel with the model:

* ``reference_year`` -- the year ``car_age`` is measured against
* ``frequency_map``  -- the training-set count for each exact car name

If either is recomputed at prediction time the model sees different numbers
from the ones it was trained on. Both are therefore returned by
:func:`add_features`, stored in the model bundle, and passed back in when a
single vehicle is scored.
"""

import pandas as pd


def extract_brand(name):
    """First token of the vehicle name, e.g. ``Maruti``."""
    if pd.isna(name):
        return "Unknown"
    tokens = str(name).split()
    return tokens[0] if tokens else "Unknown"


def extract_model_family(name):
    """Brand plus the first model token, e.g. ``Maruti Swift``.

    This is a lower-cardinality view of ``name`` (around 400 categories
    instead of 1,491) that still separates model lines within a brand.
    """
    if pd.isna(name):
        return "Unknown"
    tokens = str(name).split()
    if not tokens:
        return "Unknown"
    return " ".join(tokens[:2])


def build_frequency_map(names):
    """Count each exact car name. Built from training rows only."""
    return pd.Series(names).value_counts().to_dict()


def add_features(df, reference_year=None, frequency_map=None):
    """Add the Week 8 derived features.

    Returns ``(frame, reference_year, frequency_map)``. When either fitted
    value is passed in it is reused verbatim, which is what keeps training
    and prediction consistent.
    """
    df = df.copy()

    if reference_year is None:
        reference_year = int(pd.to_numeric(df["year"]).max())
    reference_year = int(reference_year)

    df["car_age"] = (reference_year - pd.to_numeric(df["year"])).clip(lower=0)
    df["brand"] = df["name"].map(extract_brand)
    df["model_family"] = df["name"].map(extract_model_family)

    # A car registered in the reference year has age 0; clip the denominator
    # so the first year of use is not divided by zero.
    df["km_per_year"] = pd.to_numeric(df["km_driven"]) / df["car_age"].clip(lower=1)

    if frequency_map is None:
        frequency_map = build_frequency_map(df["name"])

    # An unseen name legitimately has a training frequency of zero.
    df["name_frequency"] = df["name"].map(frequency_map).fillna(0).astype(float)

    return df, reference_year, frequency_map


def feature_catalogue():
    """Human-readable description of each engineered feature (Week 8/14)."""
    return pd.DataFrame(
        [
            {
                "feature": "car_age",
                "type": "numeric",
                "definition": "reference_year - year, clipped at 0",
                "rationale": "Depreciation is driven by age, not by the raw year label.",
            },
            {
                "feature": "brand",
                "type": "categorical",
                "definition": "First token of name",
                "rationale": "Brands hold value at very different rates.",
            },
            {
                "feature": "model_family",
                "type": "categorical",
                "definition": "First two tokens of name",
                "rationale": "Lower-cardinality view of name; groups trims of one model.",
            },
            {
                "feature": "km_per_year",
                "type": "numeric",
                "definition": "km_driven / max(car_age, 1)",
                "rationale": "Separates a lightly used old car from a hard-driven new one.",
            },
            {
                "feature": "name_frequency",
                "type": "numeric",
                "definition": "Count of the exact name in the training split",
                "rationale": "Proxy for how common (and how liquid) a model is.",
            },
        ]
    )
