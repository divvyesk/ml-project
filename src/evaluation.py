"""Week 11: evaluation plots and feature importance."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def evaluation_plots(y_true, y_pred, out_dir, prefix=""):
    """Actual-vs-predicted, residual and error-distribution plots.

    ``prefix`` keeps the Week 11 baseline plots from being overwritten by the
    Week 12 tuned-model plots.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual Selling Price (Rs)")
    plt.ylabel("Predicted Selling Price (Rs)")
    plt.title("Actual vs Predicted Selling Price")
    plt.tight_layout()
    plt.savefig(out_dir / "{0}actual_vs_predicted.png".format(prefix), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.4)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Selling Price (Rs)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(out_dir / "{0}residuals.png".format(prefix), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=40)
    plt.xlabel("Residual (Rs)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "{0}residual_distribution.png".format(prefix), dpi=160)
    plt.close()


def model_comparison_plot(results, out_dir, filename="model_comparison.png"):
    """Bar chart of test R2 per model, for the Week 15 slides."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered = results.sort_values("R2")
    plt.figure(figsize=(9, 5))
    plt.barh(ordered["Model"], ordered["R2"])
    plt.xlabel("Test R2")
    plt.title("Model Comparison (Test R2)")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=160)
    plt.close()


def _inner_estimator(model):
    """Unwrap Pipeline -> TransformedTargetRegressor -> estimator."""
    step = model.named_steps["model"]
    return getattr(step, "regressor_", getattr(step, "regressor", step))


def feature_importance(model, X_test, y_test, output_path, top_n=30):
    """Feature importance for the final model.

    Tree models expose ``feature_importances_`` over the one-hot expanded
    matrix. Anything else (Ridge, the stack) falls back to permutation
    importance over the original input columns, which is slower but always
    available. Returns ``None`` if neither can be computed.
    """
    output_path = Path(output_path)
    estimator = _inner_estimator(model)

    if hasattr(estimator, "feature_importances_"):
        names = model.named_steps["preprocessor"].get_feature_names_out()
        frame = pd.DataFrame(
            {
                "feature": names,
                "importance": estimator.feature_importances_,
                "method": "tree_gain",
            }
        )
    else:
        result = permutation_importance(
            model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1
        )
        frame = pd.DataFrame(
            {
                "feature": list(X_test.columns),
                "importance": result.importances_mean,
                "method": "permutation",
            }
        )

    frame = frame.sort_values("importance", ascending=False).head(top_n)
    frame.to_csv(output_path, index=False)
    return frame


def error_by_price_band(y_true, y_pred, bands=None):
    """Break the error down by price band (Week 11 / 14 discussion)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    bands = bands or [0, 200000, 400000, 700000, 1200000, np.inf]
    labels = ["<2L", "2L-4L", "4L-7L", "7L-12L", ">12L"]

    frame = pd.DataFrame({"actual": y_true, "predicted": y_pred})
    frame["band"] = pd.cut(frame["actual"], bins=bands, labels=labels)
    frame["abs_error"] = (frame["actual"] - frame["predicted"]).abs()
    frame["pct_error"] = frame["abs_error"] / frame["actual"] * 100.0

    summary = (
        frame.groupby("band", observed=False)
        .agg(
            records=("actual", "size"),
            mean_actual=("actual", "mean"),
            mae=("abs_error", "mean"),
            mape=("pct_error", "mean"),
        )
        .reset_index()
    )
    return summary
