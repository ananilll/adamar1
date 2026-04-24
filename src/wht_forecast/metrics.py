"""
Forecast quality metrics.
"""

from typing import Dict

import numpy as np


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Compute standard forecast quality metrics.

    Parameters
    ----------
    actual : np.ndarray
        Actual (ground truth) values.
    predicted : np.ndarray
        Predicted values.

    Returns
    -------
    Dict[str, float]
        Dictionary with MAE, MSE, RMSE, MAPE.
    """
    mae = float(np.mean(np.abs(actual - predicted)))
    mse = float(np.mean((actual - predicted) ** 2))
    rmse = float(np.sqrt(mse))

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.where(actual != 0, actual, 1.0)
        mape = float(np.mean(np.abs((actual - predicted) / denom)) * 100)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
    }
