"""Week 11 - Performance evaluation using appropriate metrics."""

import _bootstrap  # noqa: F401

import joblib

from src.artifacts import write_json
from src.config import MODEL_DIR, PLOT_DIR, REPORT_DIR, ensure_directories
from src.data import load_csv
from src.config import CLEAN_PATH
from src.evaluation import (
    error_by_price_band,
    evaluation_plots,
    model_comparison_plot,
)
from src.metrics import extended_metrics
from src.train import prepare_split, select_best, train_baselines
import pandas as pd


def main():
    ensure_directories()

    clean = load_csv(CLEAN_PATH)
    X_train, X_test, y_train, y_test, _ = prepare_split(clean)

    best_path = MODEL_DIR / "baseline_best_baseline.joblib"
    comparison_path = REPORT_DIR / "model_comparison.csv"

    if best_path.exists() and comparison_path.exists():
        results = pd.read_csv(comparison_path)
        model = joblib.load(best_path)
        best_name = results.sort_values("CV_R2_mean", ascending=False).iloc[0]["Model"]
    else:
        # Week 11 can be run on its own; retrain if Week 10 has not been run.
        results, fitted = train_baselines(X_train, y_train, X_test, y_test)
        results.to_csv(comparison_path, index=False)
        best_name, model = select_best(results, fitted)

    predictions = model.predict(X_test)
    metrics = extended_metrics(y_test, predictions)

    evaluation_plots(y_test, predictions, PLOT_DIR, prefix="baseline_")
    model_comparison_plot(results, PLOT_DIR)

    bands = error_by_price_band(y_test, predictions)
    bands.to_csv(REPORT_DIR / "error_by_price_band.csv", index=False)

    write_json(
        {
            "best_baseline_model": best_name,
            "test_records": int(len(X_test)),
            "metrics": metrics,
        },
        REPORT_DIR / "evaluation_summary.json",
    )

    print("Week 11 evaluation complete.")
    print("Best baseline: {0}".format(best_name))
    for key, value in metrics.items():
        print("  {0:<18} {1:.4f}".format(key, value))
    print()
    print(bands.to_string(index=False))


if __name__ == "__main__":
    main()
