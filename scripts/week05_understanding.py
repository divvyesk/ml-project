"""Week 5 - Dataset understanding, feature analysis and problem mapping.

Regenerates every table quoted in section 5 of the project document from the
raw CSV, so the document and the data can be checked against each other.
"""

import _bootstrap  # noqa: F401

import pandas as pd

from src.config import RAW_PATH, REPORT_DIR, TARGET, ensure_directories
from src.data import (
    categorical_summary,
    clean_text_columns,
    iqr_outlier_counts,
    load_csv,
    numeric_summary,
)


def main():
    ensure_directories()

    raw = clean_text_columns(load_csv(RAW_PATH))

    structure = pd.DataFrame(
        [
            ("Total Records", len(raw)),
            ("Total Attributes", raw.shape[1]),
            ("Numerical Attributes", 3),
            ("Categorical/Text Attributes", 5),
            ("Target Variable", TARGET),
            ("Missing Values", int(raw.isna().sum().sum())),
            ("Duplicate Records", int(raw.duplicated().sum())),
            ("Unique Car Names", int(raw["name"].nunique())),
        ],
        columns=["parameter", "value"],
    )
    structure.to_csv(REPORT_DIR / "week05_dataset_structure.csv", index=False)

    numeric_summary(raw).to_csv(REPORT_DIR / "week05_numeric_summary.csv")
    categorical_summary(raw, ["fuel", "seller_type", "transmission", "owner"]).to_csv(
        REPORT_DIR / "week05_categorical_summary.csv", index=False
    )

    top_names = (
        raw["name"]
        .value_counts()
        .head(10)
        .rename_axis("car_name")
        .reset_index(name="records")
    )
    top_names.to_csv(REPORT_DIR / "week05_top_car_names.csv", index=False)

    correlations = pd.DataFrame(
        [
            {
                "feature": column,
                "correlation_with_selling_price": float(
                    raw[column].corr(raw[TARGET])
                ),
            }
            for column in ["year", "km_driven"]
        ]
    )
    correlations.to_csv(REPORT_DIR / "week05_correlations.csv", index=False)

    iqr_outlier_counts(raw).to_csv(REPORT_DIR / "week05_outliers.csv", index=False)

    print("Week 5 dataset understanding complete.")
    print(structure.to_string(index=False))
    print()
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
