"""Training helpers for frozen prompt-token variable-selection models."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from src.data.prompt_token_embeddings import PromptTokenEmbeddingStore
from src.models.dynamic_token_gating import TokenVariableSelectionNetwork


@dataclass(frozen=True)
class TargetTransform:
    task: str
    mean: float
    scale: float

    @classmethod
    def fit(cls, values: np.ndarray, *, task: str) -> "TargetTransform":
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            raise ValueError("cannot fit a target transform without finite targets")
        if task == "classification":
            unique = set(np.unique(finite).tolist())
            if not unique.issubset({0.0, 1.0}):
                raise ValueError(f"classification targets must be binary; found {sorted(unique)}")
            return cls(task=task, mean=0.0, scale=1.0)
        if task != "regression":
            raise ValueError("task must be regression or classification")
        scale = float(np.std(finite, ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("regression targets must have positive finite scale")
        return cls(task=task, mean=float(np.mean(finite)), scale=scale)

    def encode_tensor(self, values: torch.Tensor) -> torch.Tensor:
        if self.task == "classification":
            return values
        return (values - self.mean) / self.scale

    def decode_array(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if self.task == "classification":
            return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))
        return values * self.scale + self.mean


@dataclass(frozen=True)
class TokenPrediction:
    row_indexes: np.ndarray
    predictions: np.ndarray
    weights: np.ndarray
    mean_loss: float


@dataclass(frozen=True)
class TokenRepresentation:
    """Frozen gate outputs for a downstream, independently fitted predictor."""

    row_indexes: np.ndarray
    representations: np.ndarray
    weights: np.ndarray


def seed_torch(seed: int) -> None:
    """Seed Python, NumPy, and torch without requiring a CUDA device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_loss(task: str) -> nn.Module:
    if task == "regression":
        return nn.SmoothL1Loss(reduction="mean", beta=1.0)
    if task == "classification":
        return nn.BCEWithLogitsLoss(reduction="mean")
    raise ValueError("task must be regression or classification")


def train_token_epoch(
    model: TokenVariableSelectionNetwork,
    store: PromptTokenEmbeddingStore,
    selected_by_row: np.ndarray,
    targets_by_row: np.ndarray,
    *,
    transform: TargetTransform,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    seed: int,
    gradient_clip_norm: float,
) -> float:
    """Train one shard-streamed epoch and return row-weighted mean loss."""
    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    criterion = make_loss(transform.task)
    model.train()
    total_loss = 0.0
    total_rows = 0
    for batch in store.iter_batches(
        selected_by_row, targets_by_row, batch_size=batch_size,
        shuffle=True, seed=seed,
    ):
        tokens = torch.from_numpy(batch.values).to(device=device)
        targets = torch.from_numpy(batch.targets).to(device=device)
        encoded = transform.encode_tensor(targets)
        optimizer.zero_grad(set_to_none=True)
        output = model(tokens)
        loss = criterion(output.prediction, encoded)
        if not torch.isfinite(loss):
            raise RuntimeError("dynamic token training produced non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        rows = len(batch.targets)
        total_loss += float(loss.detach().cpu()) * rows
        total_rows += rows
    if total_rows == 0:
        raise ValueError("training selection contains no finite rows")
    return total_loss / total_rows


def predict_token_store(
    model: TokenVariableSelectionNetwork,
    store: PromptTokenEmbeddingStore,
    selected_by_row: np.ndarray,
    targets_by_row: np.ndarray,
    *,
    transform: TargetTransform,
    device: torch.device,
    batch_size: int,
) -> TokenPrediction:
    """Predict selected rows and retain auditable token weights."""
    criterion = make_loss(transform.task)
    model.eval()
    rows: list[np.ndarray] = []
    raw_predictions: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    total_loss = 0.0
    total_rows = 0
    with torch.no_grad():
        for batch in store.iter_batches(
            selected_by_row, targets_by_row, batch_size=batch_size,
            shuffle=False, seed=0,
        ):
            tokens = torch.from_numpy(batch.values).to(device=device)
            targets = torch.from_numpy(batch.targets).to(device=device)
            encoded = transform.encode_tensor(targets)
            output = model(tokens)
            loss = criterion(output.prediction, encoded)
            count = len(batch.targets)
            total_loss += float(loss.detach().cpu()) * count
            total_rows += count
            rows.append(batch.row_indexes)
            raw_predictions.append(output.prediction.detach().cpu().numpy())
            weights.append(output.weights.detach().cpu().numpy())
    if not total_rows:
        raise ValueError("prediction selection contains no finite rows")
    row_values = np.concatenate(rows).astype(np.int64, copy=False)
    raw_values = np.concatenate(raw_predictions)
    weight_values = np.concatenate(weights).astype(np.float32, copy=False)
    order = np.argsort(row_values, kind="stable")
    return TokenPrediction(
        row_indexes=row_values[order],
        predictions=transform.decode_array(raw_values[order]),
        weights=weight_values[order],
        mean_loss=total_loss / total_rows,
    )


def encode_token_store(
    model: TokenVariableSelectionNetwork,
    store: PromptTokenEmbeddingStore,
    selected_by_row: np.ndarray,
    targets_by_row: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> TokenRepresentation:
    """Encode finite-target rows with a frozen gate, ordered by ``row_index``.

    The model's auxiliary prediction head is evaluated as part of ``forward``
    but its output is intentionally discarded.  Only the gated representation
    is exposed to the separate downstream prediction stage.
    """
    model.eval()
    rows: list[np.ndarray] = []
    representations: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    with torch.no_grad():
        for batch in store.iter_batches(
            selected_by_row, targets_by_row, batch_size=batch_size,
            shuffle=False, seed=0,
        ):
            output = model(torch.from_numpy(batch.values).to(device=device))
            rows.append(batch.row_indexes)
            representations.append(
                output.representation.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            weights.append(
                output.weights.detach().cpu().numpy().astype(np.float32, copy=False)
            )
    if not rows:
        raise ValueError("encoding selection contains no finite rows")
    row_values = np.concatenate(rows).astype(np.int64, copy=False)
    representation_values = np.concatenate(representations)
    weight_values = np.concatenate(weights)
    order = np.argsort(row_values, kind="stable")
    return TokenRepresentation(
        row_indexes=row_values[order],
        representations=representation_values[order],
        weights=weight_values[order],
    )


def token_weight_summary(
    weights: np.ndarray, tokens: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Summarize gate distributions and detect static/collapsed behavior."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(tokens):
        raise ValueError("weights must have shape [rows, len(tokens)]")
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("weights must be non-empty and finite")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("each token-weight row must sum to one")
    entropy = -np.sum(values * np.log(np.clip(values, 1e-12, 1.0)), axis=1)
    normalized_entropy = entropy / math.log(values.shape[1]) if values.shape[1] > 1 else entropy
    position_std = values.std(axis=0, ddof=0)
    return {
        "rows": int(len(values)),
        "token_count": int(values.shape[1]),
        "entropy_mean": float(entropy.mean()),
        "entropy_std": float(entropy.std(ddof=0)),
        "normalized_entropy_mean": float(normalized_entropy.mean()),
        "mean_position_std_across_announcements": float(position_std.mean()),
        "max_position_std_across_announcements": float(position_std.max()),
        "argmax_position_counts": np.bincount(
            values.argmax(axis=1), minlength=values.shape[1]
        ).astype(int).tolist(),
        "positions": [
            {
                "position_zero_based": position,
                "position_one_based": position + 1,
                "token": str(tokens[position]),
                "mean": float(values[:, position].mean()),
                "std": float(position_std[position]),
                "min": float(values[:, position].min()),
                "max": float(values[:, position].max()),
            }
            for position in range(values.shape[1])
        ],
    }
