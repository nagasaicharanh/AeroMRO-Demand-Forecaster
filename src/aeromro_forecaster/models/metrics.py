from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(y_true, y_pred, y_train, seasonality: int = 7) -> float:
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= seasonality:
        raise ValueError("y_train length must be greater than seasonality")
    scale = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    if scale == 0:
        return float("inf")
    return mae(y_true, y_pred) / float(scale)


def coverage(y_true, lower, upper) -> float:
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))
