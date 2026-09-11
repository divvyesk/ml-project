"""Week 5-6: loading, validation and cleaning of the CarDekho dataset."""

import pandas as pd

from .config import (
    BASE_COLUMNS,
    MAX_VALID_YEAR,
    MIN_VALID_YEAR,
    RAW_NUMERIC_COLUMNS,
    RAW_TEXT_COLUMNS,
    TARGET,
)

# Column-name variants seen in the different public copies of the dataset.
RENAME_MAP = {
    "car_name": "name",
    "selling price": "selling_price",
    "sellingprice": "selling_price",
    "kms_driven": "km_driven",
    "km driven": "km_driven",
    "fuel_type": "fuel",
    "seller type": "seller_type",
}


def load_csv(path):
    """Read the CSV, normalise column names and keep the project schema."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns=RENAME_MAP)

    missing = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: {0}. Found: {1}".format(
                missing, list(df.columns)
            )
        )

    return df[BASE_COLUMNS].copy()


def clean_text_columns(df):
    """Strip surrounding spaces and collapse repeated internal whitespace."""
    df = df.copy()
    for col in RAW_TEXT_COLUMNS:
        df[col] = (
            df[col]
            .astype("object")
            .where(df[col].notna(), None)
            .map(lambda v: None if v is None else " ".join(str(v).split()))
        )
    return df


def invalid_value_report(df):
    """Week 6 invalid-value checks, reported before anything is dropped."""
    year = pd.to_numeric(df["year"], errors="coerce")
    price = pd.to_numeric(df[TARGET], errors="coerce")
    km = pd.to_numeric(df["km_driven"], errors="coerce")

    return {
        "negative_km_driven": int((km < 0).sum()),
        "non_positive_selling_price": int((price <= 0).sum()),
        "invalid_manufacturing_years": int(
            ((year < MIN_VALID_YEAR) | (year > MAX_VALID_YEAR)).sum()
        ),
        "non_numeric_values": int(
            year.isna().sum() + price.isna().sum() + km.isna().sum()
        ),
    }


def clean_data(df):
    """Return the cleaned frame plus a step-by-step record count summary.

    The order of operations matches the Week 6 write-up: text cleaning, type
    coercion, missing-value removal, invalid-value removal, then exact
    duplicate removal. Each stage is counted separately so the report can
    state honestly how many rows each step removed.
    """
    stages = []
    df = clean_text_columns(df)
    stages.append(("original_records", len(df)))

    for col in RAW_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=BASE_COLUMNS)
    stages.append(("removed_missing_or_non_numeric", before - len(df)))

    before = len(df)
    df = df[df["km_driven"] >= 0]
    df = df[df[TARGET] > 0]
    df = df[df["year"].between(MIN_VALID_YEAR, MAX_VALID_YEAR)]
    stages.append(("removed_invalid_values", before - len(df)))

    before = len(df)
    df = df.drop_duplicates()
    stages.append(("removed_exact_duplicates", before - len(df)))

    df = df.reset_index(drop=True)
    df["year"] = df["year"].astype(int)
    df[TARGET] = df[TARGET].astype(int)
    df["km_driven"] = df["km_driven"].astype(int)

    stages.append(("final_records", len(df)))
    return df, dict(stages)


def numeric_summary(df):
    """Descriptive statistics for the three numerical attributes."""
    return df[RAW_NUMERIC_COLUMNS].describe().T


def categorical_summary(df, columns=None):
    """Long-format value counts for the categorical attributes."""
    columns = columns or ["fuel", "seller_type", "transmission", "owner"]
    frames = []
    for col in columns:
        counts = (
            df[col]
            .value_counts(dropna=False)
            .rename_axis("category")
            .reset_index(name="count")
        )
        counts.insert(0, "feature", col)
        frames.append(counts)
    return pd.concat(frames, ignore_index=True)


def iqr_outlier_counts(df, columns=None):
    """Count IQR outliers without removing them (Week 5/6 decision)."""
    columns = columns or RAW_NUMERIC_COLUMNS
    rows = []
    for col in columns:
        series = df[col].astype(float)
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append(
            {
                "feature": col,
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_bound": float(low),
                "upper_bound": float(high),
                "outlier_count": int(((series < low) | (series > high)).sum()),
            }
        )
    return pd.DataFrame(rows)
