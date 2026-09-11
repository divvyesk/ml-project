"""Week 7: exploratory data analysis plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # scripts run headless; never open a GUI window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CORRELATION_COLUMNS = [
    "year",
    "selling_price",
    "km_driven",
    "car_age",
    "km_per_year",
    "name_frequency",
]


def _save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def run_eda(df, out_dir):
    """Write every Week 7 plot and return the list of files created."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created = []

    plt.figure(figsize=(9, 5))
    plt.hist(df["selling_price"], bins=40)
    plt.xlabel("Selling Price (Rs)")
    plt.ylabel("Frequency")
    plt.title("Selling Price Distribution")
    _save(out_dir / "target_distribution.png")
    created.append("target_distribution.png")

    plt.figure(figsize=(9, 5))
    plt.hist(np.log1p(df["selling_price"].clip(lower=1)), bins=40)
    plt.xlabel("log(1 + Selling Price)")
    plt.ylabel("Frequency")
    plt.title("Log-Transformed Selling Price Distribution")
    _save(out_dir / "target_log_distribution.png")
    created.append("target_log_distribution.png")

    plt.figure(figsize=(9, 5))
    plt.scatter(df["year"], df["selling_price"], alpha=0.35)
    plt.xlabel("Manufacturing Year")
    plt.ylabel("Selling Price (Rs)")
    plt.title("Manufacturing Year vs Selling Price")
    _save(out_dir / "year_vs_price.png")
    created.append("year_vs_price.png")

    plt.figure(figsize=(9, 5))
    plt.scatter(df["km_driven"], df["selling_price"], alpha=0.35)
    plt.xlabel("Kilometres Driven")
    plt.ylabel("Selling Price (Rs)")
    plt.title("Kilometres Driven vs Selling Price")
    _save(out_dir / "km_vs_price.png")
    created.append("km_vs_price.png")

    plt.figure(figsize=(9, 5))
    plt.scatter(df["car_age"], df["selling_price"], alpha=0.35)
    plt.xlabel("Car Age (years)")
    plt.ylabel("Selling Price (Rs)")
    plt.title("Car Age vs Selling Price")
    _save(out_dir / "car_age_vs_price.png")
    created.append("car_age_vs_price.png")

    for col in ["fuel", "transmission", "owner", "seller_type"]:
        label = col.replace("_", " ").title()
        averages = df.groupby(col)["selling_price"].mean().sort_values()
        plt.figure(figsize=(9, 5))
        averages.plot(kind="bar")
        plt.xlabel(label)
        plt.ylabel("Average Selling Price (Rs)")
        plt.title("Average Selling Price by {0}".format(label))
        plt.xticks(rotation=30, ha="right")
        _save(out_dir / "avg_price_by_{0}.png".format(col))
        created.append("avg_price_by_{0}.png".format(col))

    top_brands = df["brand"].value_counts().head(15).sort_values()
    plt.figure(figsize=(9, 6))
    top_brands.plot(kind="barh")
    plt.xlabel("Number of Records")
    plt.ylabel("Brand")
    plt.title("Top 15 Brands by Number of Records")
    _save(out_dir / "top_brands_by_count.png")
    created.append("top_brands_by_count.png")

    plt.figure(figsize=(9, 6))
    order = df.groupby("brand")["selling_price"].median().sort_values().tail(15)
    order.plot(kind="barh")
    plt.xlabel("Median Selling Price (Rs)")
    plt.ylabel("Brand")
    plt.title("Top 15 Brands by Median Selling Price")
    _save(out_dir / "top_brands_by_price.png")
    created.append("top_brands_by_price.png")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df[CORRELATION_COLUMNS].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
    )
    plt.title("Numerical Feature Correlation Heatmap")
    _save(out_dir / "correlation_heatmap.png")
    created.append("correlation_heatmap.png")

    return created


def correlation_table(df):
    """Correlation of each numeric feature with the target (Week 7 / 14)."""
    corr = df[CORRELATION_COLUMNS].corr()["selling_price"].drop("selling_price")
    return (
        corr.rename("correlation_with_selling_price")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("correlation_with_selling_price", ascending=False)
    )


def group_price_table(df, columns=None):
    """Average and median price per category (Week 7 observations)."""
    columns = columns or ["fuel", "seller_type", "transmission", "owner"]
    frames = []
    for col in columns:
        grouped = (
            df.groupby(col)["selling_price"]
            .agg(["count", "mean", "median"])
            .reset_index()
            .rename(columns={col: "category"})
        )
        grouped.insert(0, "feature", col)
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)
