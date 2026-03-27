from __future__ import annotations

import numpy as np

from aeromro_forecaster.models.metrics import coverage, mae, mase, rmse


def test_metrics_known_values():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 3, 2])
    assert mae(y_true, y_pred) == 2 / 3
    assert round(rmse(y_true, y_pred), 6) == round(np.sqrt(2 / 3), 6)
    assert coverage(y_true, [0, 2, 4], [2, 2, 5]) == 2 / 3


def test_mase_uses_seasonal_scale():
    y_train = np.array([1, 2, 3, 4, 5, 6])
    assert mase([5], [4], y_train, seasonality=1) == 1.0
