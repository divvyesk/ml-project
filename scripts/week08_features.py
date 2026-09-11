"""Week 8 - Feature engineering and feature selection."""

import _bootstrap  # noqa: F401

import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    CLEAN_PATH,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    REPORT_DIR,
    TARGET,
    ensure_directories,
)
from src.data import load_csv
from src.features import add_features, feature_catalogue


def main():
    ensure_directories()

    clean = load_csv(CLEAN_PATH)
    df, reference_year, frequency_map = add_features(clean)

    df.to_csv(REPORT_DIR / "engineered_dataset.csv", index=False)
    feature_catalogue().to_csv(
        REPORT_DIR / "feature_engineering_summary.csv", index=False
    )

    # Feature selection evidence: correlation for numeric features,
    # cardinality and price spread for categorical ones.
    numeric_rows = []
    for column in NUMERIC_FEATURES:
        numeric_rows.append(
            {
                "feature": column,
                "kind": "numeric",
                "correlation_with_target": float(df[column].corr(df[TARGET])),
                "cardinality": int(df[column].nunique()),
                "decision": "keep",
            }
        )

    categorical_rows = []
    for column in CATEGORICAL_FEATURES:
        grouped = df.groupby(column)[TARGET].mean()
        categorical_rows.append(
            {
                "feature": column,
                "kind": "categorical",
                "correlation_with_target": float("nan"),
                "cardinality": int(df[column].nunique()),
                "decision": "keep",
                "price_spread_ratio": float(grouped.max() / max(grouped.min(), 1.0)),
            }
        )

    selection = pd.DataFrame(numeric_rows + categorical_rows)
    selection.to_csv(REPORT_DIR / "feature_selection.csv", index=False)

    print("Week 8 feature engineering complete.")
    print("Reference year: {0}".format(reference_year))
    print("Distinct car names in frequency map: {0}".format(len(frequency_map)))
    print("Model input columns ({0}): {1}".format(
        len(FEATURE_COLUMNS), ", ".join(FEATURE_COLUMNS)
    ))
    print()
    print(selection.to_string(index=False))


if __name__ == "__main__":
    main()
