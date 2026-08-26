"""TFT-inspired variable selection for frozen contextualized token vectors.

This is deliberately not a full Temporal Fusion Transformer: announcements are
not treated as a time sequence.  Prompt-token positions are the variables.  A
small gated residual network produces announcement-specific softmax weights,
then a weighted token representation feeds a continuous-return head.  Frozen
embedding tensors are the only model inputs; no Transformer is fine-tuned.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TokenGateOutput:
    """Prediction plus auditable per-announcement token weights."""

    prediction: torch.Tensor
    representation: torch.Tensor
    weights: torch.Tensor


class GatedResidualNetwork(nn.Module):
    """Compact TFT-style gated residual block with optional context."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, output_size)
        self.hidden = nn.Linear(input_size, hidden_size)
        self.context = nn.Linear(output_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(hidden_size, output_size)
        self.activation = nn.ELU()
        self.normalization = nn.LayerNorm(output_size)

    def forward(
        self, values: torch.Tensor, context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.hidden(values)
        if context is not None:
            hidden = hidden + self.context(context)
        hidden = self.activation(hidden)
        gated = self.value(hidden) * torch.sigmoid(self.gate(hidden))
        return self.normalization(self.input_projection(values) + gated)


class TokenVariableSelectionNetwork(nn.Module):
    """Pool fixed token embeddings with uniform, static, or dynamic gates.

    ``dynamic`` is the TFT-inspired treatment: each announcement gets its own
    token distribution. ``static`` learns one global distribution and
    ``uniform`` is the no-selection ablation.  Padding/special positions can be
    excluded with a boolean mask and always receive exactly zero weight.
    """

    def __init__(
        self,
        hidden_size: int,
        token_count: int,
        *,
        gate_hidden_size: int = 64,
        representation_size: int = 64,
        gate_mode: str = "dynamic",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or token_count < 1:
            raise ValueError("hidden_size and token_count must be positive")
        if gate_hidden_size < 1 or representation_size < 1:
            raise ValueError("gate and representation sizes must be positive")
        if gate_mode not in {"uniform", "static", "dynamic"}:
            raise ValueError("gate_mode must be uniform, static, or dynamic")
        self.hidden_size = hidden_size
        self.token_count = token_count
        self.gate_mode = gate_mode
        self.position_embedding = nn.Parameter(
            torch.zeros(token_count, representation_size)
        )
        self.token_projection = GatedResidualNetwork(
            hidden_size, gate_hidden_size, representation_size
        )
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_size, representation_size), nn.ELU(),
            nn.LayerNorm(representation_size),
        )
        self.dynamic_score = nn.Sequential(
            nn.Linear(representation_size * 2, gate_hidden_size), nn.ELU(),
            nn.Linear(gate_hidden_size, 1),
        )
        self.static_logits = nn.Parameter(torch.zeros(token_count))
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(representation_size), nn.Dropout(dropout),
            nn.Linear(representation_size, 1),
        )

    def _validate(
        self, tokens: torch.Tensor, token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (
            self.token_count, self.hidden_size
        ):
            raise ValueError(
                "tokens must have shape "
                f"[batch, {self.token_count}, {self.hidden_size}]"
            )
        if token_mask is None:
            mask = torch.ones(
                tokens.shape[:2], dtype=torch.bool, device=tokens.device
            )
        else:
            if token_mask.shape != tokens.shape[:2]:
                raise ValueError("token_mask must have shape [batch, token_count]")
            mask = token_mask.to(device=tokens.device, dtype=torch.bool)
        if not torch.all(mask.any(dim=1)):
            raise ValueError("every announcement must retain at least one token")
        return mask

    def forward(
        self, tokens: torch.Tensor, token_mask: torch.Tensor | None = None,
    ) -> TokenGateOutput:
        mask = self._validate(tokens, token_mask)
        numeric_mask = mask.unsqueeze(-1).to(dtype=tokens.dtype)
        denominator = numeric_mask.sum(dim=1).clamp_min(1.0)
        context = self.context_projection(
            (tokens * numeric_mask).sum(dim=1) / denominator
        )
        transformed = self.token_projection(tokens, context.unsqueeze(1))
        transformed = transformed + self.position_embedding.unsqueeze(0)

        if self.gate_mode == "uniform":
            logits = torch.zeros(
                tokens.shape[:2], dtype=tokens.dtype, device=tokens.device
            )
        elif self.gate_mode == "static":
            logits = self.static_logits.unsqueeze(0).expand(tokens.shape[0], -1)
        else:
            expanded_context = context.unsqueeze(1).expand(-1, self.token_count, -1)
            logits = self.dynamic_score(
                torch.cat((transformed, expanded_context), dim=-1)
            ).squeeze(-1)
        logits = logits.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(logits, dim=1)
        representation = torch.sum(transformed * weights.unsqueeze(-1), dim=1)
        prediction = self.prediction_head(representation).squeeze(-1)
        return TokenGateOutput(prediction, representation, weights)
