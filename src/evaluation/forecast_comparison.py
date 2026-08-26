"""Dependent out-of-sample forecast comparisons.

The functions in this module deliberately operate on one loss observation per
trading date.  Cross-sectional stock or announcement losses should be averaged
within date before calling the DM or MCS routines.  This prevents the very large
number of announcements from being mistaken for independent time observations.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


Alternative = Literal["two-sided", "greater", "less"]
MCSMethod = Literal["range", "max"]


def _as_loss_frame(losses: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    frame = losses.copy() if isinstance(losses, pd.DataFrame) else pd.DataFrame(losses)
    if frame.ndim != 2 or frame.shape[1] < 2:
        raise ValueError("forecast comparison requires at least two models")
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if len(frame) < 3:
        raise ValueError("forecast comparison requires at least three complete dates")
    if frame.columns.duplicated().any():
        raise ValueError("model names must be unique")
    return frame


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm step-down family-wise-error adjustment."""
    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("p-values must be a finite one-dimensional sequence")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[position]))
        adjusted[position] = running
    return adjusted


def benjamini_hochberg_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjustment."""
    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("p-values must be a finite one-dimensional sequence")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 1.0
    total = len(values)
    for reverse_rank in range(total - 1, -1, -1):
        position = order[reverse_rank]
        candidate = values[position] * total / (reverse_rank + 1)
        running = min(running, candidate)
        adjusted[position] = min(1.0, running)
    return adjusted


def _newey_west_long_run_variance(values: np.ndarray, lag: int) -> float:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    n = len(centered)
    if lag < 0 or lag >= n:
        raise ValueError("HAC lag must satisfy 0 <= lag < n_observations")
    variance = float(np.dot(centered, centered) / n)
    for offset in range(1, lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / n)
        weight = 1.0 - offset / (lag + 1.0)
        variance += 2.0 * weight * covariance
    # Finite samples can produce a slightly negative HAC estimate.
    return max(0.0, variance)


def diebold_mariano_test(
    losses_a: Iterable[float],
    losses_b: Iterable[float],
    *,
    hac_lag: int = 5,
    horizon: int = 1,
    alternative: Alternative = "two-sided",
    small_sample: bool = True,
) -> dict[str, Any]:
    """Test equal predictive accuracy for two aligned daily loss series.

    The differential is ``loss_a - loss_b``.  A positive value therefore
    favors model B.  Bartlett/Newey-West weights estimate serial dependence and
    the optional Harvey-Leybourne-Newbold correction is applied for finite T.
    """
    a = np.asarray(list(losses_a), dtype=float)
    b = np.asarray(list(losses_b), dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("DM losses must be aligned one-dimensional arrays")
    mask = np.isfinite(a) & np.isfinite(b)
    differential = a[mask] - b[mask]
    n = len(differential)
    if n < 3:
        raise ValueError("DM test requires at least three paired dates")
    if horizon < 1:
        raise ValueError("forecast horizon must be positive")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be two-sided, greater, or less")

    mean = float(np.mean(differential))
    long_run_variance = _newey_west_long_run_variance(differential, hac_lag)
    standard_error = float(np.sqrt(long_run_variance / n))
    if standard_error == 0.0:
        statistic = 0.0 if mean == 0.0 else float(np.copysign(np.inf, mean))
    else:
        statistic = mean / standard_error

    correction = 1.0
    if small_sample:
        correction_squared = (n + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / n) / n
        correction = float(np.sqrt(max(0.0, correction_squared)))
        statistic *= correction

    if alternative == "two-sided":
        p_value = float(2.0 * student_t.sf(abs(statistic), df=n - 1))
    elif alternative == "greater":
        p_value = float(student_t.sf(statistic, df=n - 1))
    else:
        p_value = float(student_t.cdf(statistic, df=n - 1))
    favored = "b" if mean > 0 else "a" if mean < 0 else "tie"
    return {
        "n_dates": n,
        "mean_loss_a": float(np.mean(a[mask])),
        "mean_loss_b": float(np.mean(b[mask])),
        "mean_loss_difference_a_minus_b": mean,
        "long_run_variance": long_run_variance,
        "standard_error": standard_error,
        "hac_lag": int(hac_lag),
        "horizon": int(horizon),
        "small_sample_correction": correction,
        "alternative": alternative,
        "statistic": float(statistic),
        "p_value": p_value,
        "favored": favored,
    }


def pairwise_dm_tests(
    losses: pd.DataFrame | np.ndarray,
    *,
    hac_lag: int = 5,
    horizon: int = 1,
    alternative: Alternative = "two-sided",
) -> pd.DataFrame:
    """Run all pairwise DM tests and adjust their p-values."""
    frame = _as_loss_frame(losses)
    rows: list[dict[str, Any]] = []
    for model_a, model_b in combinations(frame.columns, 2):
        result = diebold_mariano_test(
            frame[model_a], frame[model_b], hac_lag=hac_lag,
            horizon=horizon, alternative=alternative,
        )
        favored = model_b if result["favored"] == "b" else model_a if result["favored"] == "a" else "tie"
        rows.append({"model_a": str(model_a), "model_b": str(model_b), "favored_model": str(favored), **result})
    output = pd.DataFrame(rows)
    output["p_value_holm"] = holm_adjust(output["p_value"])
    output["p_value_bh"] = benjamini_hochberg_adjust(output["p_value"])
    return output.sort_values(["p_value_holm", "p_value", "model_a", "model_b"], kind="stable").reset_index(drop=True)


def _circular_block_indices(
    n: int, block_length: int, repetitions: int, rng: np.random.Generator,
) -> np.ndarray:
    if block_length < 1 or block_length > n:
        raise ValueError("block_length must satisfy 1 <= block_length <= n_observations")
    blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(repetitions, blocks))
    offsets = np.arange(block_length)
    return ((starts[..., None] + offsets) % n).reshape(repetitions, -1)[:, :n]


def _safe_studentize(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    )
    output = np.zeros(numerator.shape, dtype=float)
    valid = denominator > 0.0
    output[valid] = numerator[valid] / denominator[valid]
    output[~valid & (numerator != 0.0)] = np.copysign(np.inf, numerator[~valid & (numerator != 0.0)])
    return output


def model_confidence_set(
    losses: pd.DataFrame | np.ndarray,
    *,
    alpha: float = 0.10,
    bootstrap_reps: int = 5_000,
    block_length: int = 5,
    seed: int = 42,
    method: MCSMethod = "range",
) -> dict[str, Any]:
    """Hansen-Lunde-Nason style sequential Model Confidence Set.

    ``range`` uses the maximum absolute pairwise loss-difference statistic and
    ``max`` uses each model's loss relative to the active-set average.  Lower
    loss is better.  Bootstrap p-values are recomputed after every elimination.
    """
    frame = _as_loss_frame(losses)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if bootstrap_reps < 100:
        raise ValueError("bootstrap_reps must be at least 100")
    if method not in {"range", "max"}:
        raise ValueError("MCS method must be range or max")
    rng = np.random.default_rng(seed)
    all_models = [str(column) for column in frame.columns]
    frame.columns = all_models
    active = list(all_models)
    history: list[dict[str, Any]] = []

    while len(active) > 1:
        values = frame[active].to_numpy(dtype=float)
        means = values.mean(axis=0)
        indices = _circular_block_indices(
            len(values), block_length, bootstrap_reps, rng,
        )
        boot_means = values[indices].mean(axis=1)
        centered_boot_means = boot_means - means

        if method == "range":
            observed_differences = means[:, None] - means[None, :]
            boot_differences = (
                centered_boot_means[:, :, None] - centered_boot_means[:, None, :]
            )
            standard_errors = boot_differences.std(axis=0, ddof=1)
            observed_t = _safe_studentize(observed_differences, standard_errors)
            bootstrap_t = _safe_studentize(boot_differences, standard_errors)
            statistic = float(np.max(np.abs(observed_t)))
            bootstrap_statistics = np.max(np.abs(bootstrap_t), axis=(1, 2))
            elimination_scores = np.max(observed_t, axis=1)
        else:
            observed_differences = means - means.mean()
            boot_differences = centered_boot_means - centered_boot_means.mean(axis=1, keepdims=True)
            standard_errors = boot_differences.std(axis=0, ddof=1)
            observed_t = _safe_studentize(observed_differences, standard_errors)
            bootstrap_t = _safe_studentize(boot_differences, standard_errors)
            statistic = float(np.max(observed_t))
            bootstrap_statistics = np.max(bootstrap_t, axis=1)
            elimination_scores = observed_t

        p_value = float((1 + np.sum(bootstrap_statistics >= statistic)) / (bootstrap_reps + 1))
        worst_index = int(np.argmax(elimination_scores))
        worst_model = active[worst_index]
        rejected = p_value < alpha
        history.append({
            "step": len(history) + 1,
            "active_models": list(active),
            "mean_losses": {name: float(value) for name, value in zip(active, means)},
            "statistic": statistic,
            "p_value": p_value,
            "worst_model": worst_model,
            "rejected_equal_predictive_ability": rejected,
        })
        if not rejected:
            break
        active.remove(worst_model)

    return {
        "method": method,
        "alpha": float(alpha),
        "bootstrap_reps": int(bootstrap_reps),
        "block_length": int(block_length),
        "seed": int(seed),
        "n_dates": int(len(frame)),
        "included_models": active,
        "excluded_models": [model for model in all_models if model not in active],
        "history": history,
    }


def classification_losses(
    actual: Iterable[float], probabilities: Iterable[float], *, threshold: float = 0.5,
) -> pd.DataFrame:
    """Return additive binary losses for aligned stock-day predictions."""
    returns = np.asarray(list(actual), dtype=float)
    probability = np.asarray(list(probabilities), dtype=float)
    if returns.shape != probability.shape or returns.ndim != 1:
        raise ValueError("classification actuals and probabilities must align")
    labels = (returns > 0.0).astype(float)
    probability = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return pd.DataFrame({
        "log_loss": -(labels * np.log(probability) + (1.0 - labels) * np.log(1.0 - probability)),
        "brier": np.square(probability - labels),
        "zero_one": ((probability > threshold) != labels).astype(float),
    })


def regression_losses(
    actual: Iterable[float], predictions: Iterable[float], *, huber_delta: float = 0.01,
) -> pd.DataFrame:
    """Return additive continuous-return losses for stock-day predictions."""
    actual_values = np.asarray(list(actual), dtype=float)
    predicted_values = np.asarray(list(predictions), dtype=float)
    if actual_values.shape != predicted_values.shape or actual_values.ndim != 1:
        raise ValueError("regression actuals and predictions must align")
    if huber_delta <= 0.0:
        raise ValueError("huber_delta must be positive")
    error = predicted_values - actual_values
    absolute = np.abs(error)
    huber = np.where(
        absolute <= huber_delta,
        0.5 * np.square(error),
        huber_delta * (absolute - 0.5 * huber_delta),
    )
    return pd.DataFrame({"squared": np.square(error), "absolute": absolute, "huber": huber})


def daily_average_loss(
    dates: Iterable[Any], losses: Iterable[float], *, name: str,
) -> pd.Series:
    """Average cross-sectional losses into one ordered observation per date."""
    frame = pd.DataFrame({"date": pd.to_datetime(list(dates), errors="coerce"), "loss": list(losses)})
    frame["loss"] = pd.to_numeric(frame["loss"], errors="coerce")
    frame = frame.dropna(subset=["date", "loss"])
    if frame.empty:
        raise ValueError("daily loss aggregation has no finite observations")
    result = frame.groupby("date", sort=True, observed=True)["loss"].mean()
    result.name = name
    return result
