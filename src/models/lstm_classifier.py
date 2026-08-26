"""A small, leakage-safe LSTM classifier for frozen representations.

The current pooled-embedding runner presents one frozen embedding per
announcement to a classifier.  This wrapper therefore uses the paper-style
one-step sequence shape ``[row, 1, features]``.  It is intentionally exposed
through the scikit-learn estimator interface so the existing chronological
validation and artifact code can treat it like the other classifiers.

This is an architectural baseline, not a cross-day recurrent model: the
sequence length is explicitly one and no future row is ever supplied to the
network.  A stock-day lookback model should be implemented as a separate
sequence builder rather than silently changing the meaning of the existing
row-level classification matrix.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class _OneStepLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.recurrent = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence, _state = self.recurrent(values)
        return self.output(sequence[:, -1, :]).squeeze(-1)


class LSTMClassifier(BaseEstimator, ClassifierMixin):
    """Fit a deterministic one-step LSTM binary classifier on dense features.

    ``fit`` accepts ``[rows, features]`` and internally reshapes it to
    ``[rows, 1, features]``.  Inputs are expected to be finite and already
    transformed by the caller's training-window-only preprocessor.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 5,
        batch_size: int = 512,
        gradient_clip_norm: float = 1.0,
        random_state: int = 42,
        device: str = "auto",
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.gradient_clip_norm = gradient_clip_norm
        self.random_state = random_state
        self.device = device

    @staticmethod
    def _validate_features(values: Any) -> np.ndarray:
        features = np.asarray(values, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"LSTMClassifier expects a 2-D matrix; found {features.shape}")
        if len(features) == 0 or features.shape[1] == 0:
            raise ValueError("LSTMClassifier requires at least one row and one feature")
        if not np.isfinite(features).all():
            raise ValueError("LSTMClassifier received non-finite features")
        return np.ascontiguousarray(features)

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(self.device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("LSTMClassifier requested CUDA but it is unavailable")
        return resolved

    def fit(self, X: Any, y: Any) -> "LSTMClassifier":
        features = self._validate_features(X)
        if self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("hidden_size and num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.epochs < 1 or self.batch_size < 1 or self.gradient_clip_norm <= 0:
            raise ValueError("epochs, batch_size, and gradient_clip_norm must be positive")

        labels = np.asarray(y, dtype=np.float32)
        if labels.ndim != 1 or len(labels) != len(features):
            raise ValueError("LSTMClassifier features and labels must have equal rows")
        if not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
            raise ValueError("LSTMClassifier labels must be finite binary values")
        if len(np.unique(labels)) < 2:
            raise ValueError("LSTMClassifier training requires both classes")

        _seed_everything(int(self.random_state))
        device = self._resolve_device()
        model = _OneStepLSTM(
            input_size=int(features.shape[1]),
            hidden_size=int(self.hidden_size),
            num_layers=int(self.num_layers),
            dropout=float(self.dropout),
        ).to(device)
        positive = float(labels.sum())
        negative = float(len(labels) - positive)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negative / positive], dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(self.learning_rate), weight_decay=float(self.weight_decay)
        )
        tensors = torch.from_numpy(features[:, None, :])
        targets = torch.from_numpy(labels)
        generator = torch.Generator()
        generator.manual_seed(int(self.random_state))
        loader = DataLoader(
            TensorDataset(tensors, targets),
            batch_size=int(self.batch_size),
            shuffle=True,
            generator=generator,
            num_workers=0,
        )

        history: list[float] = []
        model.train()
        for _epoch in range(int(self.epochs)):
            total_loss = 0.0
            total_rows = 0
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(device=device)
                batch_labels = batch_labels.to(device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_features), batch_labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("LSTMClassifier produced a non-finite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(self.gradient_clip_norm)
                )
                optimizer.step()
                count = len(batch_labels)
                total_loss += float(loss.detach().cpu()) * count
                total_rows += count
            history.append(total_loss / total_rows)

        model.cpu()
        model.eval()
        self.model_ = model
        self.classes_ = np.array([0, 1], dtype=np.int8)
        self.n_features_in_ = int(features.shape[1])
        self.sequence_length_ = 1
        self.n_rows_fit_ = int(len(features))
        self.n_positive_fit_ = int(positive)
        self.n_negative_fit_ = int(negative)
        self.loss_history_ = tuple(float(value) for value in history)
        self.training_device_ = str(device)
        return self

    def _predict_positive(self, X: Any) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("LSTMClassifier must be fitted before prediction")
        features = self._validate_features(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"LSTMClassifier expected {self.n_features_in_} features; "
                f"found {features.shape[1]}"
            )
        values: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(features), int(self.batch_size)):
                batch = torch.from_numpy(features[start:start + int(self.batch_size), None, :])
                logits = self.model_(batch)
                values.append(torch.sigmoid(logits).numpy())
        return np.concatenate(values).astype(np.float64, copy=False)

    def predict_proba(self, X: Any) -> np.ndarray:
        positive = self._predict_positive(X)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X: Any) -> np.ndarray:
        return (self._predict_positive(X) >= 0.5).astype(np.int8)