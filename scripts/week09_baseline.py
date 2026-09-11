"""Week 9 - Algorithm selection and baseline model development.

Two comparisons are run: the untransformed baseline, and the same models
fitted on log1p(price). The difference between the two tables is the
evidence for keeping the transform in every later week.
"""

import _bootstrap  # noqa: F401

import pandas as pd

from src.config import CLEAN_PATH, REPORT_DIR, ensure_directories
from src.data import load_csv
from src.train import prepare_split, train_baselines


def main():
    ensure_directories()

    clean = load_csv(CLEAN_PATH)
    X_train, X_test, y_train, y_test, fit_params = prepare_split(clean)

    raw_results, _ = train_baselines(
        X_train, y_train, X_test, y_test, use_log_target=False
    )
    raw_results.to_csv(
        REPORT_DIR / "baseline_no_transform.csv", index=False
    )

    log_results, _ = train_baselines(
        X_train, y_train, X_test, y_test, use_log_target=True
    )
    log_results.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    comparison = pd.merge(
        raw_results[["Model", "R2"]].rename(columns={"R2": "R2_raw_target"}),
        log_results[["Model", "R2"]].rename(columns={"R2": "R2_log_target"}),
        on="Model",
    )
    comparison["improvement"] = (
        comparison["R2_log_target"] - comparison["R2_raw_target"]
    )
    comparison.to_csv(REPORT_DIR / "target_transform_effect.csv", index=False)

    print("Week 9 baseline comparison complete.")
    print("Training rows: {0}   Test rows: {1}   Reference year: {2}".format(
        len(X_train), len(X_test), fit_params["reference_year"]
    ))
    print()
    print("Without target transform:")
    print(raw_results.to_string(index=False))
    print()
    print("With log1p target transform:")
    print(log_results.to_string(index=False))
    print()
    print("Effect of the transform:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
