"""Week 12 - Hyper-parameter tuning and model improvement.

Selection is by cross-validated R2 on the training split. The test set is
scored once, after the winner is chosen, and never used to pick between
candidates.
"""

import _bootstrap  # noqa: F401

import pandas as pd

from src.artifacts import save_bundle, write_json
from src.config import (
    BUNDLE_PATH,
    CLEAN_PATH,
    CV_FOLDS,
    PLOT_DIR,
    REPORT_DIR,
    ensure_directories,
)
from src.data import load_csv
from src.evaluation import error_by_price_band, evaluation_plots, feature_importance
from src.train import evaluate_stack, final_test_report, prepare_split, tune_models


def main():
    ensure_directories()

    clean = load_csv(CLEAN_PATH)
    X_train, X_test, y_train, y_test, fit_params = prepare_split(clean)

    results, tuned = tune_models(X_train, y_train, X_test, y_test)

    stack_row, stack_model = evaluate_stack(tuned, X_train, y_train, X_test, y_test)
    if stack_row is not None:
        results = pd.concat([results, pd.DataFrame([stack_row])], ignore_index=True)
        tuned["Stacked Ensemble"] = stack_model

    results = results.sort_values("CV_R2_mean", ascending=False).reset_index(drop=True)
    results.to_csv(REPORT_DIR / "tuned_model_comparison.csv", index=False)

    final_name = results.iloc[0]["Model"]
    final_model = tuned[final_name]

    metrics, predictions = final_test_report(final_model, X_test, y_test)

    evaluation_plots(y_test, predictions, PLOT_DIR, prefix="final_")
    error_by_price_band(y_test, predictions).to_csv(
        REPORT_DIR / "final_error_by_price_band.csv", index=False
    )

    importance = feature_importance(
        final_model, X_test, y_test, REPORT_DIR / "feature_importance.csv"
    )

    bundle_path = save_bundle(final_model, final_name, fit_params, metrics)

    write_json(
        {
            "final_model": final_name,
            "selection_rule": (
                "highest mean R2 across {0}-fold cross-validation on the "
                "training split".format(CV_FOLDS)
            ),
            "cv_r2_mean": float(results.iloc[0]["CV_R2_mean"]),
            "metrics": metrics,
            "train_records": int(len(X_train)),
            "test_records": int(len(X_test)),
            "reference_year": fit_params["reference_year"],
            "bundle_path": str(bundle_path),
        },
        REPORT_DIR / "final_model_summary.json",
    )

    print("Week 12 tuning complete.")
    print(results[["Model", "CV_R2_mean", "R2", "MAE", "RMSE"]].to_string(index=False))
    print()
    print("Final model: {0}".format(final_name))
    for key, value in metrics.items():
        print("  {0:<18} {1:.4f}".format(key, value))
    print()
    print("Saved bundle: {0}".format(BUNDLE_PATH))
    if importance is not None:
        print()
        print("Top features:")
        print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
