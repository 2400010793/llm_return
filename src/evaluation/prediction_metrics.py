"""Basic out-of-sample prediction metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
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


def return_prediction_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    historical_mean: float,
) -> dict[str, float]:
    """Evaluate continuous returns against zero and historical-mean forecasts."""
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual_array) & np.isfinite(predicted_array)
    if not mask.any():
        raise ValueError("No finite actual/predicted pairs")
    y_true, y_pred = actual_array[mask], predicted_array[mask]
    metrics = regression_metrics(y_true, y_pred)
    model_sse = float(np.square(y_true - y_pred).sum())
    zero_sse = float(np.square(y_true).sum())
    mean_sse = float(np.square(y_true - historical_mean).sum())
    metrics.update({
        "oos_r2_vs_zero": 1.0 - model_sse / zero_sse if zero_sse > 0 else float("nan"),
        "oos_r2_vs_historical_mean": (
            1.0 - model_sse / mean_sse if mean_sse > 0 else float("nan")
        ),
        "direction_accuracy": float(((y_true > 0) == (y_pred > 0)).mean()),
        "prediction_mean": float(y_pred.mean()),
        "prediction_std": float(y_pred.std(ddof=0)),
        "actual_mean": float(y_true.mean()),
    })
    return metrics


def daily_rank_ic(
    frame: pd.DataFrame,
    *,
    date_column: str = "entry_date",
    actual_column: str = "actual_return",
    prediction_column: str = "prediction",
    min_stocks: int = 5,
    method: str = "spearman",
) -> dict[str, float]:
    """Summarize daily cross-sectional information coefficients."""
    if method not in {"spearman", "pearson"}:
        raise ValueError("IC method must be spearman or pearson")
    required = {date_column, actual_column, prediction_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing Rank IC columns: {', '.join(sorted(missing))}")
    values: list[float] = []
    for _, group in frame.dropna(subset=list(required)).groupby(date_column):
        if len(group) < min_stocks:
            continue
        correlation = group[actual_column].corr(
            group[prediction_column], method=method
        )
        if pd.notna(correlation):
            values.append(float(correlation))
    if not values:
        return {
            "rank_ic_mean": float("nan"), "rank_ic_std": float("nan"),
            "rank_ic_information_ratio": float("nan"), "rank_ic_positive_rate": float("nan"),
            "rank_ic_days": 0.0,
        }
    array = np.asarray(values, dtype=float)
    std = float(array.std(ddof=1)) if len(array) > 1 else float("nan")
    return {
        "rank_ic_mean": float(array.mean()),
        "rank_ic_std": std,
        "rank_ic_information_ratio": float(array.mean() / std) if std > 0 else float("nan"),
        "rank_ic_positive_rate": float((array > 0).mean()),
        "rank_ic_days": float(len(array)),
    }
