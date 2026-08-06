"""Unified, out-of-sample evaluation for binary stock-news classification."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
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


PRIMARY_METRIC = "auc"
SECONDARY_METRICS = (
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
