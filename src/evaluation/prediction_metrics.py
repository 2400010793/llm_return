"""Basic out-of-sample prediction metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute common metrics after removing non-finite pairs."""
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual_array) & np.isfinite(predicted_array)
    if not mask.any():
        raise ValueError("No finite actual/predicted pairs")
    y_true, y_pred = actual_array[mask], predicted_array[mask]
    return {
        "n": float(mask.sum()),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "correlation": float(np.corrcoef(y_true, y_pred)[0, 1])
        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0
        else float("nan"),
    }
