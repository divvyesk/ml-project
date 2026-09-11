"""Week 10 - Model training and initial results.

Fits every baseline on the training split and saves each one, so the Week 11
evaluation can load a model instead of retraining it.
"""

import _bootstrap  # noqa: F401

from src.artifacts import save_baseline, write_json
from src.config import CLEAN_PATH, MODEL_DIR, REPORT_DIR, ensure_directories
from src.data import load_csv
from src.train import prepare_split, select_best, train_baselines


def main():
    ensure_directories()

    clean = load_csv(CLEAN_PATH)
    X_train, X_test, y_train, y_test, fit_params = prepare_split(clean)

    results, fitted = train_baselines(X_train, y_train, X_test, y_test)
    results.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    saved = []
    for name, model in fitted.items():
        saved.append(str(save_baseline(model, name)))

    best_name, best_model = select_best(results, fitted)
    save_baseline(best_model, "best_baseline")

    write_json(
        {
            "best_baseline_model": best_name,
            "selection_rule": "highest cross-validated R2 on the training split",
            "train_records": int(len(X_train)),
            "test_records": int(len(X_test)),
            "reference_year": fit_params["reference_year"],
            "saved_models": saved,
        },
        REPORT_DIR / "week10_training_summary.json",
    )

    print("Week 10 training complete. {0} models saved to {1}".format(
        len(saved), MODEL_DIR
    ))
    print(results.to_string(index=False))
    print()
    print("Best baseline by cross-validated R2: {0}".format(best_name))


if __name__ == "__main__":
    main()
