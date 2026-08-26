"""Unified, out-of-sample evaluation for binary stock-news classification."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


PRIMARY_METRIC = "accuracy"
SECONDARY_METRICS = (
    "auc",
    "balanced_accuracy",
    "f1",
    "accuracy",
    "log_loss",
    "brier",
    "mcc",
)


def evaluate_binary_classification(
    actual_three_day_returns: Iterable[float],
    probabilities: Iterable[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return the complete fixed metric set for the paper's weak label.

    Accuracy is the primary metric to match the paper's reported
    out-of-sample classification results. AUC is retained as a supplementary
    threshold-independent ranking metric. The paper uses a strict 0.5 cutoff:
    probabilities greater than 0.5 are positive and all others are negative.
    Missing observations are removed only from the evaluation set, never
    imputed as a class.
    """
    returns = np.asarray(list(actual_three_day_returns), dtype=float)
    probs = np.asarray(list(probabilities), dtype=float)
    if returns.shape != probs.shape:
        raise ValueError("actual returns and probabilities must have equal length")
    mask = np.isfinite(returns) & np.isfinite(probs)
    if not mask.any():
        raise ValueError("no finite classification observations")
    actual = (returns[mask] > 0).astype(int)
    probs = np.clip(probs[mask], 1e-7, 1.0 - 1e-7)
    predicted = (probs > threshold).astype(int)
    positive_rate = float(actual.mean())
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    result = {
        "n": float(mask.sum()),
        "positive_rate": positive_rate,
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(actual, predicted)) if len(np.unique(actual)) > 1 else float("nan"),
        "log_loss": float(log_loss(actual, probs, labels=[0, 1])),
        "brier": float(brier_score_loss(actual, probs)),
        "auc": float(roc_auc_score(actual, probs)) if len(np.unique(actual)) == 2 else float("nan"),
        "majority_accuracy": float(max(positive_rate, 1.0 - positive_rate)),
        "predicted_positive_rate": float(predicted.mean()),
    }
    result["accuracy_lift_vs_majority"] = result["accuracy"] - result["majority_accuracy"]
    return result


def summarize_classification_stability(
    rows: Iterable[dict[str, Any]],
    *,
    group_by: tuple[str, ...] = ("representation", "classifier", "tuned"),
) -> list[dict[str, Any]]:
    """Aggregate repeated-seed test results without hiding seed dispersion."""
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return []
    numeric = [name for name in ("auc", "balanced_accuracy", "f1", "accuracy", "log_loss", "brier", "mcc") if name in frame]
    grouped = frame.groupby(list(group_by), dropna=False, sort=True)
    output: list[dict[str, Any]] = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_by, keys))
        record["n_seeds"] = int(len(group))
        for metric in numeric:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            record[f"{metric}_min"] = float(values.min())
            record[f"{metric}_max"] = float(values.max())
        output.append(record)
    return output


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in the original order."""
    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Holm adjustment requires finite one-dimensional p-values")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[position])
        running = max(running, candidate)
        adjusted[position] = running
    return adjusted.tolist()


def paired_classification_comparison(
    actual_returns: Iterable[float],
    probabilities_a: Iterable[float],
    probabilities_b: Iterable[float],
    clusters: Iterable[Any],
    *,
    threshold: float = 0.5,
    n_bootstrap: int = 2_000,
    seed: int = 42,
    metrics: tuple[str, ...] = ("accuracy", "auc", "balanced_accuracy", "mcc"),
) -> dict[str, Any]:
    """Compare B minus A using a paired cluster bootstrap and exact McNemar.

    Clusters are sampled with replacement, preserving every announcement in a
    sampled trading date. All arrays are filtered by one common finite mask so
    the comparison can never silently use different test universes.
    """
    returns = np.asarray(list(actual_returns), dtype=float)
    probs_a = np.asarray(list(probabilities_a), dtype=float)
    probs_b = np.asarray(list(probabilities_b), dtype=float)
    cluster_values = np.asarray(list(clusters), dtype=object)
    if not (returns.shape == probs_a.shape == probs_b.shape == cluster_values.shape):
        raise ValueError("paired comparison inputs must have equal shapes")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    mask = np.isfinite(returns) & np.isfinite(probs_a) & np.isfinite(probs_b)
    mask &= pd.notna(cluster_values)
    if not mask.any():
        raise ValueError("no common finite paired observations")
    returns, probs_a, probs_b = returns[mask], probs_a[mask], probs_b[mask]
    cluster_values = cluster_values[mask]
    unique_clusters, cluster_codes = np.unique(cluster_values.astype(str), return_inverse=True)
    cluster_indices = [np.flatnonzero(cluster_codes == code) for code in range(len(unique_clusters))]

    metrics_a = evaluate_binary_classification(returns, probs_a, threshold=threshold)
    metrics_b = evaluate_binary_classification(returns, probs_b, threshold=threshold)
    unknown = set(metrics).difference(metrics_a)
    if unknown:
        raise ValueError(f"unknown comparison metrics: {sorted(unknown)}")
    point = {metric: metrics_b[metric] - metrics_a[metric] for metric in metrics}

    rng = np.random.default_rng(seed)
    draws = {metric: np.empty(n_bootstrap, dtype=float) for metric in metrics}
    for bootstrap_index in range(n_bootstrap):
        sampled_codes = rng.integers(0, len(unique_clusters), size=len(unique_clusters))
        indices = np.concatenate([cluster_indices[code] for code in sampled_codes])
        sample_a = evaluate_binary_classification(
            returns[indices], probs_a[indices], threshold=threshold
        )
        sample_b = evaluate_binary_classification(
            returns[indices], probs_b[indices], threshold=threshold
        )
        for metric in metrics:
            draws[metric][bootstrap_index] = sample_b[metric] - sample_a[metric]

    actual = returns > 0
    correct_a = (probs_a > threshold) == actual
    correct_b = (probs_b > threshold) == actual
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    discordant = a_only + b_only
    mcnemar_p = float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "direction": "b_minus_a",
        "n": int(len(returns)),
        "n_clusters": int(len(unique_clusters)),
        "threshold": threshold,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "delta": point,
        "clustered_95_ci": {
            metric: [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
            for metric, values in draws.items()
        },
        "mcnemar_accuracy": {
            "a_correct_b_wrong": a_only,
            "a_wrong_b_correct": b_only,
            "discordant": discordant,
            "exact_p_value": mcnemar_p,
        },
    }
