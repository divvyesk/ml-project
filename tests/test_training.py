import numpy as np
import pandas as pd
import pytest

from src.config import CLEAN_PATH, FEATURE_COLUMNS, TARGET
from src.data import load_csv
from src.metrics import extended_metrics, regression_metrics
from src.models import get_models
from src.train import prepare_split, train_baselines


def test_regression_metrics_are_exact_on_a_perfect_fit():
    y = np.array([100.0, 200.0, 300.0])
    result = regression_metrics(y, y)
    assert result["R2"] == pytest.approx(1.0)
    assert result["MAE"] == pytest.approx(0.0)
    assert result["RMSE"] == pytest.approx(0.0)


def test_rmse_is_the_square_root_of_mse():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    # MSE = (9 + 16) / 2 = 12.5 -> RMSE = 3.5355...
    assert regression_metrics(y_true, y_pred)["RMSE"] == pytest.approx(
        np.sqrt(12.5)
    )


def test_extended_metrics_adds_the_diagnostic_scores():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(50000, 900000, size=200)
    y_pred = y_true * rng.uniform(0.85, 1.15, size=200)
    result = extended_metrics(y_true, y_pred)
    for key in ("R2", "MAE", "RMSE", "R2_log", "R2_excl_top1pct", "MAPE"):
        assert key in result


def test_every_documented_algorithm_is_present():
    names = set(get_models().keys())
    for required in ("Linear Regression", "Decision Tree", "Random Forest",
                     "XGBoost"):
        assert required in names


def test_models_are_wrapped_in_the_log_target_transform():
    for estimator in get_models(use_log_target=True).values():
        assert hasattr(estimator, "regressor")
    for estimator in get_models(use_log_target=False).values():
        assert not hasattr(estimator, "regressor")


@pytest.mark.skipif(not CLEAN_PATH.exists(), reason="run week06_cleaning.py first")
def test_split_produces_the_documented_proportions():
    clean = load_csv(CLEAN_PATH)
    X_train, X_test, y_train, y_test, fit_params = prepare_split(clean)

    assert len(X_train) + len(X_test) == len(clean)
    assert abs(len(X_test) / len(clean) - 0.20) < 0.01
    assert list(X_train.columns) == FEATURE_COLUMNS
    assert fit_params["reference_year"] == int(
        clean.loc[X_train.index, "year"].max()
    )


@pytest.mark.skipif(not CLEAN_PATH.exists(), reason="run week06_cleaning.py first")
def test_frequency_map_is_fitted_on_training_rows_only():
    """The map must not know how often a name appears in the test split."""
    clean = load_csv(CLEAN_PATH)
    X_train, X_test, _, _, fit_params = prepare_split(clean)

    frequency_map = fit_params["frequency_map"]
    train_counts = X_train["name"].value_counts().to_dict()

    assert frequency_map == train_counts
    assert sum(frequency_map.values()) == len(X_train)


@pytest.mark.skipif(not CLEAN_PATH.exists(), reason="run week06_cleaning.py first")
def test_target_never_appears_among_the_model_inputs():
    clean = load_csv(CLEAN_PATH)
    X_train, X_test, _, _, _ = prepare_split(clean)
    assert TARGET not in X_train.columns
    assert TARGET not in X_test.columns


@pytest.mark.skipif(not CLEAN_PATH.exists(), reason="run week06_cleaning.py first")
def test_baselines_report_cross_validated_and_test_scores():
    clean = load_csv(CLEAN_PATH).sample(600, random_state=0)
    X_train, X_test, y_train, y_test, _ = prepare_split(clean)

    results, fitted = train_baselines(X_train, y_train, X_test, y_test)

    assert {"Model", "CV_R2_mean", "R2", "MAE", "RMSE"} <= set(results.columns)
    assert len(results) == len(fitted)
    # Ranked by the cross-validated score, not by the test score.
    assert results["CV_R2_mean"].is_monotonic_decreasing
    assert (results["MAE"] > 0).all()


@pytest.mark.skipif(not CLEAN_PATH.exists(), reason="run week06_cleaning.py first")
def test_predictions_stay_on_the_rupee_scale():
    """The log transform must be inverted before metrics are computed."""
    clean = load_csv(CLEAN_PATH).sample(600, random_state=0)
    X_train, X_test, y_train, y_test, _ = prepare_split(clean)
    results, fitted = train_baselines(X_train, y_train, X_test, y_test)

    predictions = fitted["Random Forest"].predict(X_test)
    assert predictions.min() > 1000  # a log-scale value would be near 12
    assert predictions.max() < 50000000
