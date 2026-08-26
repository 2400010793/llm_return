"""CPU GraphSAGE-style binary classifier for pre-aggregated stock neighbors."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def weighted_accuracy(
    labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> float:
    actual = np.asarray(labels, dtype=np.int8)
    predicted = np.asarray(probabilities) > 0.5
    values = np.asarray(weights, dtype=float)
    return float(np.average(predicted == actual, weights=values))


class _StockGraphSAGE(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_size: int,
        dropout: float,
        use_neighbors: bool,
    ) -> None:
        super().__init__()
        self.input_dimension = input_dimension
        self.use_neighbors = use_neighbors
        self.self_projection = nn.Linear(input_dimension, hidden_size)
        self.neighbor_projection = (
            nn.Linear(input_dimension, hidden_size, bias=False)
            if use_neighbors else None
        )
        self.normalization = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, max(16, hidden_size // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_size // 2), 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        own = values[:, :self.input_dimension]
        hidden = self.self_projection(own)
        if self.use_neighbors:
            neighbor = values[:, self.input_dimension:]
            hidden = hidden + self.neighbor_projection(neighbor)
        hidden = self.dropout(self.activation(self.normalization(hidden)))
        return self.output(hidden).squeeze(-1)


class StockGraphSAGEClassifier:
    """One-hop mean GraphSAGE with weighted training and validation stopping."""

    def __init__(
        self,
        *,
        input_dimension: int,
        use_neighbors: bool,
        hidden_size: int = 64,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 30,
        patience: int = 4,
        batch_size: int = 2048,
        gradient_clip_norm: float = 1.0,
        random_state: int = 42,
        num_threads: int = 1,
    ) -> None:
        self.input_dimension = int(input_dimension)
        self.use_neighbors = bool(use_neighbors)
        self.hidden_size = int(hidden_size)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.random_state = int(random_state)
        self.num_threads = int(num_threads)

    def _validate_features(self, values: Any) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        expected = self.input_dimension * (2 if self.use_neighbors else 1)
        if matrix.ndim != 2 or matrix.shape[1] != expected:
            raise ValueError(f"expected feature shape [rows, {expected}]; found {matrix.shape}")
        if not len(matrix) or not np.isfinite(matrix).all():
            raise ValueError("features must be non-empty and finite")
        return np.ascontiguousarray(matrix)

    def _seed(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.set_num_threads(max(1, self.num_threads))

    def _new_model(self) -> _StockGraphSAGE:
        return _StockGraphSAGE(
            self.input_dimension,
            self.hidden_size,
            self.dropout,
            self.use_neighbors,
        )

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any | None = None,
        validation_data: tuple[Any, Any, Any] | None = None,
    ) -> "StockGraphSAGEClassifier":
        features = self._validate_features(X)
        labels = np.asarray(y, dtype=np.float32)
        if labels.shape != (len(features),) or not np.isin(labels, (0.0, 1.0)).all():
            raise ValueError("labels must be a finite binary vector")
        weights = (
            np.ones(len(features), dtype=np.float32)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float32)
        )
        if weights.shape != labels.shape or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("sample weights must be positive and finite")
        validation = None
        if validation_data is not None:
            validation_features = self._validate_features(validation_data[0])
            validation_labels = np.asarray(validation_data[1], dtype=np.float32)
            validation_weights = np.asarray(validation_data[2], dtype=np.float32)
            if validation_labels.shape != (len(validation_features),):
                raise ValueError("validation labels have the wrong shape")
            if validation_weights.shape != validation_labels.shape:
                raise ValueError("validation weights have the wrong shape")
            validation = (validation_features, validation_labels, validation_weights)

        self._seed()
        model = self._new_model()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(features),
                torch.from_numpy(labels),
                torch.from_numpy(weights),
            ),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        best_score = -np.inf
        best_epoch = 0
        best_state = None
        stale = 0
        history: list[dict[str, float]] = []
        for epoch in range(1, self.max_epochs + 1):
            model.train()
            total_loss = 0.0
            total_weight = 0.0
            for batch_features, batch_labels, batch_weights in loader:
                optimizer.zero_grad(set_to_none=True)
                losses = criterion(model(batch_features), batch_labels)
                loss = (losses * batch_weights).sum() / batch_weights.sum()
                if not torch.isfinite(loss):
                    raise RuntimeError("GraphSAGE produced a non-finite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                optimizer.step()
                total_loss += float((losses.detach() * batch_weights).sum())
                total_weight += float(batch_weights.sum())
            record = {"epoch": float(epoch), "train_loss": total_loss / total_weight}
            if validation is not None:
                probabilities = self._predict_model(model, validation[0])
                score = weighted_accuracy(validation[1], probabilities, validation[2])
                record["validation_weighted_accuracy"] = score
                if score > best_score + 1e-12:
                    best_score = score
                    best_epoch = epoch
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                    stale = 0
                else:
                    stale += 1
                if stale >= self.patience:
                    history.append(record)
                    break
            history.append(record)
        if validation is not None and best_state is not None:
            model.load_state_dict(best_state)
        else:
            best_epoch = len(history)
            best_score = float("nan")
        model.eval()
        self.model_ = model
        self.history_ = history
        self.best_epoch_ = int(best_epoch)
        self.best_validation_weighted_accuracy_ = float(best_score)
        self.classes_ = np.array([0, 1], dtype=np.int8)
        return self

    def _predict_model(self, model: nn.Module, features: np.ndarray) -> np.ndarray:
        values: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(features), self.batch_size):
                logits = model(torch.from_numpy(features[start:start + self.batch_size]))
                values.append(torch.sigmoid(logits).numpy())
        return np.concatenate(values).astype(np.float64, copy=False)

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("classifier must be fitted before prediction")
        positive = self._predict_model(self.model_, self._validate_features(X))
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(np.int8)
