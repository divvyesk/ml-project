"""Weeks 9-12: splitting, baseline training, tuning and model selection.

Order of operations matters and is fixed here:

1. split the cleaned rows into train and test,
2. fit ``name_frequency`` on the training rows only,
3. engineer features for both halves using those fitted values,
4. choose hyper-parameters by cross-validation *inside* the training half,
5. touch the test set once, at the end, to report the final numbers.
"""

import json

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .config import (
    CV_FOLDS,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from .features import add_features, build_frequency_map
from .metrics import extended_metrics, regression_metrics
from .models import TUNING_ITERATIONS, build_stack, get_models, get_tuning_spaces
from .pipeline import make_preprocessor


def make_model(estimator):
    return Pipeline(
        [
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ]
    )


def prepare_split(clean_df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split first, then engineer features, so nothing leaks across the line.

    Returns ``(X_train, X_test, y_train, y_test, fit_params)`` where
    ``fit_params`` holds the ``reference_year`` and ``frequency_map`` that
    must be reused at prediction time.
    """
    train_df, test_df = train_test_split(
        clean_df, test_size=test_size, random_state=random_state
    )

    reference_year = int(train_df["year"].max())
    frequency_map = build_frequency_map(train_df["name"])

    train_df, _, _ = add_features(train_df, reference_year, frequency_map)
    test_df, _, _ = add_features(test_df, reference_year, frequency_map)

    fit_params = {
        "reference_year": reference_year,
        "frequency_map": frequency_map,
    }

    return (
        train_df[FEATURE_COLUMNS],
        test_df[FEATURE_COLUMNS],
        train_df[TARGET],
        test_df[TARGET],
        fit_params,
    )


def train_baselines(X_train, y_train, X_test, y_test, use_log_target=True):
    """Week 9/10: fit each candidate and score it on the held-out test set.

    A cross-validated score on the training half is reported alongside, and
    that is the column used for model selection -- ranking by the test score
    would make the test set part of the fitting procedure.
    """
    rows = []
    fitted = {}

    for name, estimator in get_models(use_log_target).items():
        pipe = make_model(estimator)

        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=CV_FOLDS, scoring="r2", n_jobs=-1
        )

        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)

        rows.append(
            {
                "Model": name,
                "CV_R2_mean": float(cv_scores.mean()),
                "CV_R2_std": float(cv_scores.std()),
                **regression_metrics(y_test, predictions),
            }
        )
        fitted[name] = pipe

    results = pd.DataFrame(rows).sort_values("CV_R2_mean", ascending=False)
    return results.reset_index(drop=True), fitted


def tune_models(X_train, y_train, X_test, y_test, n_iter=None):
    """Week 12: randomized search, scored by cross-validation on the train half."""
    rows = []
    tuned = {}

    for name, space in get_tuning_spaces().items():
        estimator = get_models()[name]
        iterations = n_iter or TUNING_ITERATIONS.get(name, 15)
        search = RandomizedSearchCV(
            make_model(estimator),
            param_distributions=space,
            n_iter=iterations,
            scoring="r2",
            cv=CV_FOLDS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)

        predictions = search.best_estimator_.predict(X_test)
        rows.append(
            {
                "Model": name,
                "CV_R2_mean": float(search.best_score_),
                **regression_metrics(y_test, predictions),
                "Best Params": json.dumps(search.best_params_, default=str),
            }
        )
        tuned[name] = search.best_estimator_

    results = pd.DataFrame(rows).sort_values("CV_R2_mean", ascending=False)
    return results.reset_index(drop=True), tuned


def evaluate_stack(tuned_pipelines, X_train, y_train, X_test, y_test):
    """Week 12 improvement: blend the tuned models and score the blend.

    Returns ``(row, fitted_pipeline)`` or ``(None, None)`` if a stack cannot
    be built.
    """
    inner = {}
    for name, pipe in tuned_pipelines.items():
        inner[name] = pipe.named_steps["model"]

    stack = build_stack(inner)
    if stack is None:
        return None, None

    pipe = make_model(stack)
    cv_scores = cross_val_score(
        pipe, X_train, y_train, cv=CV_FOLDS, scoring="r2", n_jobs=-1
    )
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)

    row = {
        "Model": "Stacked Ensemble",
        "CV_R2_mean": float(cv_scores.mean()),
        **regression_metrics(y_test, predictions),
        "Best Params": "stack of tuned Ridge / Random Forest / XGBoost",
    }
    return row, pipe


def select_best(results, fitted):
    """Pick the model with the highest cross-validated training score."""
    best_name = results.sort_values("CV_R2_mean", ascending=False).iloc[0]["Model"]
    return best_name, fitted[best_name]


def final_test_report(model, X_test, y_test):
    """The single, final look at the held-out test set."""
    predictions = model.predict(X_test)
    return extended_metrics(y_test, predictions), predictions
