"""Week 11: the three evaluation metrics named in the project document."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    """R2, MAE and RMSE on the rupee scale.

    ``squared=False`` was removed from ``mean_squared_error`` in recent
    scikit-learn versions, so RMSE is taken explicitly.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def extended_metrics(y_true, y_pred):
    """Rupee-scale metrics plus two diagnostics used in the Week 14 report.

    ``R2_log`` scores the prediction on the log scale the model is fitted on,
    and ``R2_excl_top1pct`` drops the most expensive 1% of the test set. The
    gap between these and plain ``R2`` shows how much of the error sits in a
    handful of luxury vehicles rather than in the bulk of the market.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    result = regression_metrics(y_true, y_pred)

    safe_pred = np.clip(y_pred, 1.0, None)
    result["R2_log"] = float(r2_score(np.log1p(y_true), np.log1p(safe_pred)))

    cutoff = np.quantile(y_true, 0.99)
    mask = y_true <= cutoff
    result["R2_excl_top1pct"] = float(r2_score(y_true[mask], y_pred[mask]))

    denominator = np.where(y_true == 0, np.nan, y_true)
    result["MAPE"] = float(
        np.nanmean(np.abs((y_true - y_pred) / denominator)) * 100.0
    )
    return result
