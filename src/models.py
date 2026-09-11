"""Week 9/12: the candidate algorithms and their search spaces.

Selling price spans Rs 20,000 to Rs 89,00,000 and is strongly right-skewed,
so every estimator is wrapped in a ``TransformedTargetRegressor`` that fits
on ``log1p(price)`` and inverts with ``expm1``. Predictions and all reported
metrics stay on the rupee scale; only the fitting happens in log space.
"""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from .config import RANDOM_STATE


def log_target(estimator):
    """Fit ``estimator`` against log1p(price) and invert on predict."""
    return TransformedTargetRegressor(
        regressor=estimator,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def base_estimators():
    """The four algorithms named in the project document, plus Ridge.

    Ridge is included because one-hot encoding ``name`` produces far more
    columns than rows, which is exactly the regime where unregularised
    ``LinearRegression`` becomes unstable.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(
            random_state=RANDOM_STATE,
            max_depth=20,
            min_samples_leaf=2,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def get_models(use_log_target=True):
    """Week 9/10 baselines, optionally without the log-target wrapper.

    ``use_log_target=False`` reproduces the untransformed Week 9 baseline so
    the Week 12 report can quantify what the transform actually bought.
    """
    if not use_log_target:
        return base_estimators()
    return {name: log_target(est) for name, est in base_estimators().items()}


# Search budget per model. Ridge has only seven candidates in total, so
# anything above that just repeats work.
TUNING_ITERATIONS = {
    "Ridge Regression": 7,
    "Random Forest": 10,
    "XGBoost": 15,
}


def get_tuning_spaces():
    """Search spaces for Week 12.

    Keys are prefixed ``model__regressor__`` because the estimator sits
    inside ``TransformedTargetRegressor``, which sits inside the Pipeline.
    """
    return {
        "Ridge Regression": {
            "model__regressor__alpha": [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0],
        },
        "Random Forest": {
            "model__regressor__n_estimators": [200, 300, 400],
            "model__regressor__max_depth": [None, 16, 24],
            "model__regressor__min_samples_leaf": [1, 2, 4],
            # Capped below 1.0: with `name` one-hot encoded there are more
            # columns than rows, and letting every split consider every
            # column makes the search an order of magnitude slower for no
            # measurable gain.
            "model__regressor__max_features": [0.3, 0.5, 0.7],
        },
        "XGBoost": {
            "model__regressor__n_estimators": [400, 700, 1000],
            "model__regressor__max_depth": [3, 4, 5, 6, 8],
            "model__regressor__learning_rate": [0.03, 0.05, 0.08, 0.12],
            "model__regressor__subsample": [0.7, 0.85, 1.0],
            "model__regressor__colsample_bytree": [0.5, 0.7, 0.9],
            "model__regressor__min_child_weight": [1, 3, 5],
            "model__regressor__reg_lambda": [1.0, 2.0, 5.0],
        },
    }


def build_stack(tuned_estimators):
    """Week 12 improvement: stack the tuned models behind a Ridge blender.

    ``tuned_estimators`` maps model name to a fitted
    ``TransformedTargetRegressor``; the inner regressors are re-used as
    stack members and the whole stack is wrapped in the log transform once.
    """
    members = []
    for name, estimator in tuned_estimators.items():
        inner = getattr(estimator, "regressor", estimator)
        members.append((name.lower().replace(" ", "_"), inner))

    if len(members) < 2:
        return None

    return log_target(
        StackingRegressor(
            estimators=members,
            final_estimator=Ridge(alpha=1.0),
            # 3 inner folds rather than 5: the stack is scored by an outer
            # cross-validation as well, so the nested cost multiplies.
            cv=3,
            n_jobs=1,
        )
    )
