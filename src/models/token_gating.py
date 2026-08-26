"""Leakage-safe hard gates for contextualized prompt-token vectors.

The gate selects global prompt positions using training rows only.  Selected
token vectors keep every hidden coordinate and are flattened in original token
order; no token-vector mean pooling is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TokenGate:
    method: str
    token_count: int
    hidden_size: int
    selected_positions: np.ndarray
    scores: np.ndarray

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float32)
        expected = self.token_count * self.hidden_size
        if values.ndim != 2 or values.shape[1] != expected:
            raise ValueError(
                f"token gate expected a 2-D matrix with {expected} columns; "
                f"found {values.shape}"
            )
        tokens = values.reshape(len(values), self.token_count, self.hidden_size)
        selected = tokens[:, self.selected_positions, :]
        return np.ascontiguousarray(selected.reshape(len(values), -1), dtype=np.float32)


def _moments(
    matrix: np.ndarray,
    *,
    token_count: int,
    hidden_size: int,
    labels: np.ndarray | None,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    total = np.zeros((token_count, hidden_size), dtype=np.float64)
    total_sq = np.zeros_like(total)
    count = 0
    for start in range(0, len(matrix), batch_size):
        stop = min(start + batch_size, len(matrix))
        values = np.asarray(matrix[start:stop], dtype=np.float32)
        if labels is not None:
            values = values[labels[start:stop]]
        if not len(values):
            continue
        tokens = values.reshape(len(values), token_count, hidden_size)
        total += tokens.sum(axis=0, dtype=np.float64)
        total_sq += np.square(tokens, dtype=np.float64).sum(axis=0)
        count += len(tokens)
    if count == 0:
        raise ValueError("token gate received no eligible training rows")
    return total, total_sq, count


def fit_token_gate(
    matrix: np.ndarray,
    targets: np.ndarray,
    *,
    token_count: int,
    keep_tokens: int,
    method: str = "fisher",
    batch_size: int = 4096,
) -> TokenGate:
    """Fit a global token-position gate using training rows only.

    ``fisher`` ranks positions by the mean squared standardized difference
    between positive and negative class centroids. ``variance`` is an
    unsupervised training-only baseline. ``logistic_l1`` learns a sparse
    supervised gate from each group's mean, standard deviation, and RMS while
    preserving every coordinate of selected groups in the output. Ties are
    resolved by group position.
    """
    values = np.asarray(matrix)
    numeric_targets = np.asarray(targets, dtype=float)
    if values.ndim != 2 or len(values) != len(numeric_targets):
        raise ValueError("token gate matrix and targets must have equal rows")
    if token_count < 1 or values.shape[1] % token_count:
        raise ValueError("flattened token dimension must be divisible by token_count")
    if keep_tokens < 1 or keep_tokens > token_count:
        raise ValueError(f"keep_tokens must be in 1..{token_count}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    hidden_size = values.shape[1] // token_count
    finite = np.isfinite(numeric_targets)
    key = method.lower().replace("-", "_")
    if key == "variance":
        total, total_sq, count = _moments(
            values, token_count=token_count, hidden_size=hidden_size,
            labels=finite, batch_size=batch_size,
        )
        mean = total / count
        variance = np.maximum(total_sq / count - mean * mean, 0.0)
        scores = variance.mean(axis=1)
    elif key == "fisher":
        positive = finite & (numeric_targets > 0)
        negative = finite & (numeric_targets <= 0)
        if not positive.any() or not negative.any():
            raise ValueError("Fisher token gate requires both target classes")
        pos_sum, pos_sq, pos_count = _moments(
            values, token_count=token_count, hidden_size=hidden_size,
            labels=positive, batch_size=batch_size,
        )
        neg_sum, neg_sq, neg_count = _moments(
            values, token_count=token_count, hidden_size=hidden_size,
            labels=negative, batch_size=batch_size,
        )
        pos_mean, neg_mean = pos_sum / pos_count, neg_sum / neg_count
        pos_var = np.maximum(pos_sq / pos_count - pos_mean * pos_mean, 0.0)
        neg_var = np.maximum(neg_sq / neg_count - neg_mean * neg_mean, 0.0)
        standardized = np.square(pos_mean - neg_mean) / (pos_var + neg_var + 1e-12)
        scores = standardized.mean(axis=1)
    elif key == "logistic_l1":
        eligible = np.flatnonzero(finite)
        labels = (numeric_targets[eligible] > 0).astype(np.int8)
        if len(np.unique(labels)) < 2:
            raise ValueError("L1 logistic token gate requires both target classes")
        summaries: list[np.ndarray] = []
        for start in range(0, len(eligible), batch_size):
            positions = eligible[start:start + batch_size]
            tokens = np.asarray(values[positions], dtype=np.float32).reshape(
                len(positions), token_count, hidden_size
            )
            mean = tokens.mean(axis=2)
            std = tokens.std(axis=2)
            rms = np.sqrt(np.square(tokens).mean(axis=2))
            summaries.append(np.stack((mean, std, rms), axis=2).reshape(len(tokens), -1))
        design = np.concatenate(summaries, axis=0)
        scaler = StandardScaler()
        design = scaler.fit_transform(design)
        selector = LogisticRegression(
            penalty="l1", solver="liblinear", C=0.1,
            class_weight="balanced", max_iter=1000, random_state=0,
        )
        selector.fit(design, labels)
        coefficients = selector.coef_.reshape(token_count, 3)
        scores = np.linalg.norm(coefficients, axis=1)
    else:
        raise ValueError(
            "token gate method must be one of: fisher, variance, logistic_l1"
        )
    if not np.isfinite(scores).all():
        raise ValueError("token gate produced non-finite position scores")
    ranked = np.lexsort((np.arange(token_count), -scores))
    selected = np.sort(ranked[:keep_tokens]).astype(np.int64)
    return TokenGate(
        method=key,
        token_count=token_count,
        hidden_size=hidden_size,
        selected_positions=selected,
        scores=np.asarray(scores, dtype=np.float64),
    )


def fit_streaming_token_gate(
    chunks: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    token_count: int,
    keep_tokens: int,
    method: str = "fisher",
) -> TokenGate:
    """Fit a variance/Fisher gate without materializing all rows at once.

    Each chunk contains a 2-D flattened token matrix and its numeric targets.
    The computation is algebraically equivalent to the moment-based branches
    of :func:`fit_token_gate`; only training chunks should be supplied.
    """
    key = method.lower().replace("-", "_")
    if key not in {"fisher", "variance"}:
        raise ValueError("streaming token gate supports only fisher or variance")
    if token_count < 1:
        raise ValueError("token_count must be positive")
    if keep_tokens < 1 or keep_tokens > token_count:
        raise ValueError(f"keep_tokens must be in 1..{token_count}")

    sums: dict[str, np.ndarray] = {}
    squared_sums: dict[str, np.ndarray] = {}
    counts = {"all": 0, "positive": 0, "negative": 0}
    hidden_size: int | None = None
    for matrix, targets in chunks:
        values = np.asarray(matrix, dtype=np.float32)
        numeric_targets = np.asarray(targets, dtype=float)
        if values.ndim != 2 or len(values) != len(numeric_targets):
            raise ValueError("each token chunk and target chunk must have equal rows")
        if values.shape[1] % token_count:
            raise ValueError("flattened token dimension must be divisible by token_count")
        current_hidden = values.shape[1] // token_count
        if hidden_size is None:
            hidden_size = current_hidden
            for name in counts:
                sums[name] = np.zeros((token_count, hidden_size), dtype=np.float64)
                squared_sums[name] = np.zeros_like(sums[name])
        elif current_hidden != hidden_size:
            raise ValueError("token chunks have inconsistent hidden sizes")
        tokens = values.reshape(len(values), token_count, current_hidden)
        finite = np.isfinite(numeric_targets)
        masks = {"all": finite}
        if key == "fisher":
            masks.update({
                "positive": finite & (numeric_targets > 0),
                "negative": finite & (numeric_targets <= 0),
            })
        for name, mask in masks.items():
            selected = tokens[mask]
            if not len(selected):
                continue
            sums[name] += selected.sum(axis=0, dtype=np.float64)
            squared_sums[name] += np.square(selected, dtype=np.float64).sum(axis=0)
            counts[name] += len(selected)

    if hidden_size is None or counts["all"] == 0:
        raise ValueError("streaming token gate received no eligible training rows")
    if key == "variance":
        mean = sums["all"] / counts["all"]
        variance = np.maximum(squared_sums["all"] / counts["all"] - mean * mean, 0.0)
        scores = variance.mean(axis=1)
    else:
        if counts["positive"] == 0 or counts["negative"] == 0:
            raise ValueError("Fisher token gate requires both target classes")
        positive_mean = sums["positive"] / counts["positive"]
        negative_mean = sums["negative"] / counts["negative"]
        positive_variance = np.maximum(
            squared_sums["positive"] / counts["positive"] - positive_mean * positive_mean,
            0.0,
        )
        negative_variance = np.maximum(
            squared_sums["negative"] / counts["negative"] - negative_mean * negative_mean,
            0.0,
        )
        standardized = np.square(positive_mean - negative_mean) / (
            positive_variance + negative_variance + 1e-12
        )
        scores = standardized.mean(axis=1)
    if not np.isfinite(scores).all():
        raise ValueError("streaming token gate produced non-finite position scores")
    ranked = np.lexsort((np.arange(token_count), -scores))
    selected_positions = np.sort(ranked[:keep_tokens]).astype(np.int64)
    return TokenGate(
        method=key,
        token_count=token_count,
        hidden_size=hidden_size,
        selected_positions=selected_positions,
        scores=np.asarray(scores, dtype=np.float64),
    )