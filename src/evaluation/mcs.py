"""Model Confidence Set style tests for out-of-sample classification losses.

This implements a reproducible block-bootstrap elimination procedure inspired by
Hansen, Lunde and Nason (2011). It uses the maximum loss-difference statistic,
removes the worst model when rejected, and returns the surviving confidence set.
It is intended for dependent daily/event observations; ``block_length`` should
match the chosen event clustering horizon.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def _blocks(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if block_length <= 1:
        return rng.integers(0, n, size=n)
    starts = rng.integers(0, n, size=int(np.ceil(n / block_length)))
    indices = np.concatenate([(start + np.arange(block_length)) % n for start in starts])
    return indices[:n]


def model_confidence_set(
    losses: np.ndarray | pd.DataFrame,
    *,
    model_names: Iterable[str] | None = None,
    alpha: float = 0.10,
    bootstrap_reps: int = 1000,
    block_length: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a block-bootstrap MCS elimination test on loss columns.

    ``losses`` has shape ``(observations, models)``; lower loss is better.
    Missing rows are removed listwise so all models use the same observations.
    The reported ``p_value`` is the bootstrap p-value for the maximum
    studentized loss difference. This is a practical MCS-style implementation,
    not a claim of exact asymptotic calibration for overlapping event windows.
    """
    frame = losses.copy() if isinstance(losses, pd.DataFrame) else pd.DataFrame(np.asarray(losses, dtype=float))
    if frame.shape[1] < 2:
        raise ValueError("MCS requires at least two models")
    if model_names is not None:
        names = list(model_names)
        if len(names) != frame.shape[1]:
            raise ValueError("model_names must match the number of loss columns")
        frame.columns = names
    else:
        frame.columns = [str(column) for column in frame.columns]
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if len(frame) < 2:
        raise ValueError("MCS requires at least two complete observations")
    if not 0 < alpha < 1 or bootstrap_reps < 100:
        raise ValueError("alpha must be in (0, 1) and bootstrap_reps must be at least 100")

    rng = np.random.default_rng(seed)
    active = list(frame.columns)
    history: list[dict[str, Any]] = []
    while len(active) > 1:
        values = frame[active].to_numpy(dtype=float)
        means = values.mean(axis=0)
        centered = values - means
        # The statistic compares each model with the best model in the current set.
        best = int(np.argmin(means))
        differences = means - means[best]
        se = centered.std(axis=0, ddof=1) / np.sqrt(len(values))
        safe_se = np.where(se > 0, se, np.inf)
        observed = float(np.max(differences / safe_se))
        bootstrap_statistics = np.empty(bootstrap_reps, dtype=float)
        for repetition in range(bootstrap_reps):
            sampled = centered[_blocks(len(values), block_length, rng)]
            boot_means = sampled.mean(axis=0)
            boot_best = int(np.argmin(boot_means))
            boot_se = sampled.std(axis=0, ddof=1) / np.sqrt(len(values))
            boot_safe_se = np.where(boot_se > 0, boot_se, np.inf)
            bootstrap_statistics[repetition] = np.max((boot_means - boot_means[boot_best]) / boot_safe_se)
        p_value = float((1 + np.sum(bootstrap_statistics >= observed)) / (bootstrap_reps + 1))
        worst = active[int(np.argmax(means))]
        reject = p_value < alpha
        history.append({"active_models": list(active), "worst_model": worst, "statistic": observed, "p_value": p_value, "rejected": reject})
        if not reject:
            break
        active.remove(worst)

    return {
        "alpha": float(alpha),
        "bootstrap_reps": int(bootstrap_reps),
        "block_length": int(block_length),
        "seed": int(seed),
        "n_observations": int(len(frame)),
        "included_models": active,
        "excluded_models": [name for name in frame.columns if name not in active],
        "history": history,
    }


def classification_loss_matrix(actual: Iterable[float], predictions: dict[str, Iterable[float]], *, loss: str = "log_loss") -> pd.DataFrame:
    """Build aligned per-observation classification losses for MCS."""
    y = (np.asarray(list(actual), dtype=float) > 0).astype(float)
    result: dict[str, np.ndarray] = {}
    for name, values in predictions.items():
        probability = np.clip(np.asarray(list(values), dtype=float), 1e-7, 1 - 1e-7)
        if probability.shape != y.shape:
            raise ValueError("all prediction arrays must match actual length")
        if loss == "log_loss":
            value = -(y * np.log(probability) + (1 - y) * np.log(1 - probability))
        elif loss == "brier":
            value = (probability - y) ** 2
        else:
            raise ValueError("loss must be log_loss or brier")
        result[name] = value
    return pd.DataFrame(result)
