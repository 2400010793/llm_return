"""Leakage-safe dynamic token gating for downstream classifiers.

The transformer is trained only on the rows supplied to :meth:`fit`.  It learns
an announcement-specific softmax distribution over contextualized prompt-token
vectors and exposes the gated representation to a separate scikit-learn
classifier.  The auxiliary head is used only to train the gate; it is not the
classifier evaluated by the rolling runner.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class DynamicTokenGateTransformer:
    """Fit a dynamic token gate and export fixed-size representations.

    Parameters are intentionally explicit so a fitted instance can be stored in
    a joblib preprocessor bundle.  ``matrix`` is the flattened form
    ``[rows, token_count * hidden_size]``; no labels are accepted by
    :meth:`transform`, which makes accidental test-label use impossible at the
    transformation boundary.
    """

    def __init__(
        self,
        token_count: int,
        hidden_size: int,
        *,
        gate_hidden_size: int = 64,
        representation_size: int = 64,
        gate_mode: str = "dynamic",
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 10,
        batch_size: int = 256,
        gradient_clip_norm: float = 1.0,
        random_state: int = 42,
        device: str = "auto",
    ) -> None:
        if token_count < 1 or hidden_size < 1:
            raise ValueError("token_count and hidden_size must be positive")
        if gate_mode not in {"uniform", "static", "dynamic"}:
            raise ValueError("gate_mode must be uniform, static, or dynamic")
        if gate_hidden_size < 1 or representation_size < 1:
            raise ValueError("gate_hidden_size and representation_size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if epochs < 1 or batch_size < 1 or gradient_clip_norm <= 0:
            raise ValueError("epochs, batch_size, and gradient_clip_norm must be positive")
        self.token_count = int(token_count)
        self.hidden_size = int(hidden_size)
        self.gate_hidden_size = int(gate_hidden_size)
        self.representation_size = int(representation_size)
        self.gate_mode = gate_mode
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.random_state = int(random_state)
        self.device = str(device)

    @property
    def output_dimension(self) -> int:
        return self.representation_size

    def _validate_matrix(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float32)
        expected = self.token_count * self.hidden_size
        if values.ndim != 2 or values.shape[1] != expected:
            raise ValueError(
                f"dynamic token gate expected shape [rows, {expected}]; found {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("dynamic token gate received non-finite features")
        return values

    def _resolve_device(self):
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(self.device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("dynamic token gate requested CUDA but it is unavailable")
        return resolved

    def fit(self, matrix: np.ndarray, targets: np.ndarray) -> "DynamicTokenGateTransformer":
        """Fit the gate using only finite training targets and their features."""
        values = self._validate_matrix(matrix)
        numeric_targets = np.asarray(targets, dtype=float)
        if numeric_targets.ndim != 1 or len(numeric_targets) != len(values):
            raise ValueError("dynamic token gate features and targets must have equal rows")
        finite = np.isfinite(numeric_targets)
        if not finite.any():
            raise ValueError("dynamic token gate received no finite training targets")
        labels = (numeric_targets[finite] > 0).astype(np.float32)
        if len(np.unique(labels)) < 2:
            raise ValueError("dynamic token gate training requires both target classes")

        import torch
        from torch import nn
        from src.models.dynamic_token_gating import TokenVariableSelectionNetwork
        from src.models.dynamic_token_training import seed_torch

        seed_torch(self.random_state)
        device = self._resolve_device()
        model = TokenVariableSelectionNetwork(
            self.hidden_size,
            self.token_count,
            gate_hidden_size=self.gate_hidden_size,
            representation_size=self.representation_size,
            gate_mode=self.gate_mode,
            dropout=self.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
        tokens = values[finite].reshape(-1, self.token_count, self.hidden_size)
        rng = np.random.default_rng(self.random_state)
        losses: list[float] = []
        model.train()
        for _epoch in range(1, self.epochs + 1):
            order = rng.permutation(len(tokens))
            total_loss = 0.0
            total_rows = 0
            for start in range(0, len(order), self.batch_size):
                batch_indices = order[start:start + self.batch_size]
                batch_tokens = torch.from_numpy(tokens[batch_indices]).to(device=device)
                batch_labels = torch.from_numpy(labels[batch_indices]).to(device=device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch_tokens)
                loss = criterion(output.prediction, batch_labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("dynamic token gate produced a non-finite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.gradient_clip_norm,
                )
                optimizer.step()
                count = len(batch_indices)
                total_loss += float(loss.detach().cpu()) * count
                total_rows += count
            losses.append(total_loss / total_rows)

        model.eval()
        model.cpu()
        self.model_ = model
        self.loss_history_ = tuple(float(value) for value in losses)
        self.n_rows_fit_ = int(finite.sum())
        self.n_positive_fit_ = int(labels.sum())
        self.n_negative_fit_ = int(len(labels) - labels.sum())
        self.training_device_ = str(device)
        return self

    def transform(
        self, matrix: np.ndarray, *, return_weights: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Transform features with the frozen gate; labels are never required."""
        if not hasattr(self, "model_"):
            raise RuntimeError("dynamic token gate must be fitted before transform")
        values = self._validate_matrix(matrix)
        import torch

        model = self.model_
        model.eval()
        tokens = values.reshape(-1, self.token_count, self.hidden_size)
        representations: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(tokens), self.batch_size):
                batch = torch.from_numpy(tokens[start:start + self.batch_size])
                output = model(batch)
                representations.append(
                    output.representation.cpu().numpy().astype(np.float32, copy=False)
                )
                if return_weights:
                    weights.append(
                        output.weights.cpu().numpy().astype(np.float32, copy=False)
                    )
        representation = (
            np.concatenate(representations, axis=0)
            if representations else np.empty((0, self.representation_size), dtype=np.float32)
        )
        if not return_weights:
            return np.ascontiguousarray(representation, dtype=np.float32)
        weight_values = (
            np.concatenate(weights, axis=0)
            if weights else np.empty((0, self.token_count), dtype=np.float32)
        )
        return (
            np.ascontiguousarray(representation, dtype=np.float32),
            np.ascontiguousarray(weight_values, dtype=np.float32),
        )

    def audit(self, matrix: np.ndarray, *, scope: str) -> dict[str, Any]:
        """Return a compact, finite audit of the frozen gate's weights."""
        _representation, weights = self.transform(matrix, return_weights=True)
        if not len(weights):
            raise ValueError("cannot audit a dynamic token gate on zero rows")
        entropy = -np.sum(
            weights * np.log(np.clip(weights, 1e-12, 1.0)), axis=1,
        )
        position_std = weights.std(axis=0, dtype=np.float64)
        return {
            "scope": scope,
            "gate_mode": self.gate_mode,
            "token_count": self.token_count,
            "hidden_size": self.hidden_size,
            "representation_size": self.representation_size,
            "output_dimension": self.output_dimension,
            "epochs": self.epochs,
            "loss_history": list(self.loss_history_),
            "training_rows": self.n_rows_fit_,
            "training_positive_rows": self.n_positive_fit_,
            "training_negative_rows": self.n_negative_fit_,
            "training_device": self.training_device_,
            "weight_rows": int(len(weights)),
            "weights_sum_max_error": float(np.max(np.abs(weights.sum(axis=1) - 1.0))),
            "entropy_mean": float(entropy.mean()),
            "entropy_std": float(entropy.std(ddof=0)),
            "normalized_entropy_mean": float(
                (entropy / np.log(self.token_count)).mean()
                if self.token_count > 1 else entropy.mean()
            ),
            "mean_position_std_across_announcements": float(position_std.mean()),
            "max_position_std_across_announcements": float(position_std.max()),
            "argmax_position_counts": np.bincount(
                weights.argmax(axis=1), minlength=self.token_count,
            ).astype(int).tolist(),
        }
