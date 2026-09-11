"""Week 7 - Exploratory data analysis with visualisations."""

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from src.config import CLEAN_PATH, PLOT_DIR, REPORT_DIR, ensure_directories
from src.data import categorical_summary, load_csv
from src.eda import correlation_table, group_price_table, run_eda
from src.features import add_features


def main():
    ensure_directories()

    clean = load_csv(CLEAN_PATH)
    df, reference_year, _ = add_features(clean)

    profile = pd.DataFrame(
        [
            ("records", len(df)),
            ("base_columns", clean.shape[1]),
            ("engineered_columns", df.shape[1] - clean.shape[1]),
            ("missing_values", int(df.isna().sum().sum())),
            ("exact_duplicates", int(clean.duplicated().sum())),
            ("unique_car_names", int(df["name"].nunique())),
            ("unique_brands", int(df["brand"].nunique())),
            ("unique_model_families", int(df["model_family"].nunique())),
            ("reference_year", reference_year),
        ],
        columns=["metric", "value"],
    )
    profile.to_csv(REPORT_DIR / "dataset_profile.csv", index=False)

    df[
        ["year", "selling_price", "km_driven", "car_age", "km_per_year"]
    ].describe().T.to_csv(REPORT_DIR / "numeric_summary.csv")

    categorical_summary(df).to_csv(
        REPORT_DIR / "categorical_summary.csv", index=False
    )
    group_price_table(df).to_csv(
        REPORT_DIR / "group_price_summary.csv", index=False
    )

    correlations = correlation_table(df)
    correlations.to_csv(REPORT_DIR / "correlation_summary.csv", index=False)

    log_price = np.log1p(df["selling_price"].clip(lower=1))
    skewness = pd.DataFrame(
        [
            ("selling_price", float(df["selling_price"].skew())),
            ("log1p(selling_price)", float(log_price.skew())),
        ],
        columns=["series", "skewness"],
    )
    skewness.to_csv(REPORT_DIR / "target_skewness.csv", index=False)

    created = run_eda(df, PLOT_DIR)

    print("Week 7 EDA complete. {0} plots written to {1}".format(
        len(created), PLOT_DIR
    ))
    print(profile.to_string(index=False))
    print()
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
