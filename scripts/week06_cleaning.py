"""Week 6 - Data cleaning, missing-value handling and preprocessing.

Writes `data/car_data_cleaned.csv`, which every later week reads. Running
this script is what makes the cleaned dataset reproducible instead of a file
that simply appeared alongside the raw one.
"""

import _bootstrap  # noqa: F401

import pandas as pd

from src.config import BASE_COLUMNS, CLEAN_PATH, RAW_PATH, REPORT_DIR, ensure_directories
from src.data import clean_data, invalid_value_report, iqr_outlier_counts, load_csv


def main():
    ensure_directories()

    raw = load_csv(RAW_PATH)

    missing_before = (
        raw.isna().sum().rename_axis("column").reset_index(name="missing_values")
    )
    missing_before.to_csv(REPORT_DIR / "week06_missing_values.csv", index=False)

    checks = invalid_value_report(raw)
    pd.DataFrame(
        sorted(checks.items()), columns=["check", "result"]
    ).to_csv(REPORT_DIR / "week06_invalid_values.csv", index=False)

    clean, stages = clean_data(raw)
    clean.to_csv(CLEAN_PATH, index=False)

    summary = pd.DataFrame(list(stages.items()), columns=["stage", "records"])
    summary.to_csv(REPORT_DIR / "week06_cleaning_summary.csv", index=False)

    iqr_outlier_counts(clean).to_csv(
        REPORT_DIR / "week06_outliers.csv", index=False
    )

    final = pd.DataFrame(
        [
            ("Original records", stages["original_records"]),
            ("Duplicate records removed", stages["removed_exact_duplicates"]),
            ("Invalid records removed", stages["removed_invalid_values"]),
            ("Missing/non-numeric removed", stages["removed_missing_or_non_numeric"]),
            ("Final cleaned records", stages["final_records"]),
            ("Missing values after cleaning", int(clean.isna().sum().sum())),
            ("Original columns retained", len(BASE_COLUMNS)),
        ],
        columns=["parameter", "result"],
    )
    final.to_csv(REPORT_DIR / "week06_final_dataset.csv", index=False)

    print("Week 6 cleaning complete. Wrote {0}".format(CLEAN_PATH))
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
