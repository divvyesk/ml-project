"""Week 14 - Final report submission and documentation.

Reads the artifacts written by Weeks 5-13 and assembles
`reports/final_report.md` and `docs/MODEL_CARD.md`. Nothing here recomputes a
model, so the report always describes the model that is actually on disk.
"""

import _bootstrap  # noqa: F401

import pandas as pd

from src.artifacts import load_bundle, read_json
from src.config import (
    BASE_COLUMNS,
    CLEAN_PATH,
    CV_FOLDS,
    DOCS_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    ensure_directories,
)
from src.data import load_csv
from src.features import feature_catalogue
from src.reporting import build_final_report, build_model_card, read_csv

METRIC_ORDER = ["R2", "MAE", "RMSE", "MAPE", "R2_log", "R2_excl_top1pct"]


def build_context():
    bundle = load_bundle()
    summary = read_json(REPORT_DIR / "final_model_summary.json", {})
    clean = load_csv(CLEAN_PATH)

    metrics = summary.get("metrics") or bundle["metrics"]
    metrics_table = pd.DataFrame(
        [
            {"metric": key, "value": metrics[key]}
            for key in METRIC_ORDER
            if key in metrics
        ]
    )

    cleaning = read_csv(REPORT_DIR / "week06_cleaning_summary.csv")
    if cleaning is None:
        cleaning = pd.DataFrame(
            [{"stage": "final_records", "records": len(clean)}]
        )

    duplicates_removed = 0
    raw_records = len(clean)
    lookup = dict(zip(cleaning["stage"], cleaning["records"]))
    duplicates_removed = int(lookup.get("removed_exact_duplicates", 0))
    raw_records = int(lookup.get("original_records", len(clean)))

    baseline = read_csv(REPORT_DIR / "model_comparison.csv")
    tuned = read_csv(REPORT_DIR / "tuned_model_comparison.csv")
    importance = read_csv(REPORT_DIR / "feature_importance.csv")
    bands = read_csv(REPORT_DIR / "final_error_by_price_band.csv")

    if tuned is not None and "Best Params" in tuned.columns:
        tuned = tuned.drop(columns=["Best Params"])

    top_features = (
        importance.head(10)
        if importance is not None
        else pd.DataFrame([{"feature": "not available", "importance": 0.0}])
    )

    return {
        "created_at": bundle.get("created_at", "unknown"),
        "cleaning_summary": cleaning,
        "clean_records": len(clean),
        "raw_records": raw_records,
        "duplicates_removed": duplicates_removed,
        "missing_values": int(clean.isna().sum().sum()),
        "base_columns": BASE_COLUMNS,
        "feature_catalogue": feature_catalogue()[
            ["feature", "type", "definition", "rationale"]
        ],
        "reference_year": bundle["reference_year"],
        "baseline_results": baseline
        if baseline is not None
        else pd.DataFrame([{"Model": "not available"}]),
        "tuned_results": tuned
        if tuned is not None
        else pd.DataFrame([{"Model": "not available"}]),
        "final_model_name": bundle["model_name"],
        "final_metrics": metrics,
        "final_metrics_table": metrics_table,
        "error_bands": bands
        if bands is not None
        else pd.DataFrame([{"band": "not available"}]),
        "top_features": top_features,
        "top_feature_names": ", ".join(
            str(v).split("__")[-1] for v in top_features["feature"].head(5)
        ),
        "train_records": int(summary.get("train_records", 0)),
        "test_records": int(summary.get("test_records", 0)),
        "test_mean_price": float(clean["selling_price"].mean()),
        "electric_records": int((clean["fuel"] == "Electric").sum()),
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
    }


def main():
    ensure_directories()

    context = build_context()

    report_path = REPORT_DIR / "final_report.md"
    report_path.write_text(build_final_report(context), encoding="utf-8")

    card_path = DOCS_DIR / "MODEL_CARD.md"
    card_path.write_text(build_model_card(context), encoding="utf-8")

    print("Week 14 documentation complete.")
    print("  {0}".format(report_path))
    print("  {0}".format(card_path))
    print()
    print("Final model: {0}   Test R2: {1:.4f}".format(
        context["final_model_name"], context["final_metrics"]["R2"]
    ))


if __name__ == "__main__":
    main()
