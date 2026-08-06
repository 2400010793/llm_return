"""Optional local Transformer embeddings.

The module imports torch/transformers lazily so TF-IDF works without downloading
or installing a local language model.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def mean_pool(last_hidden_state: "object", attention_mask: "object") -> "object":
    """Mean-pool token embeddings while ignoring padding tokens."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def encode_local_transformer(
    texts: Iterable[str],
    model_name: str = "hfl/chinese-roberta-wwm-ext",
    *,
    batch_size: int = 32,
    max_length: int = 256,
    device: str | None = None,
    local_files_only: bool = False,
) -> np.ndarray:
    """Encode texts with a frozen local Transformer model.

    The first call downloads model weights from Hugging Face. No model is
    downloaded merely by importing this module.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Local embeddings require torch and transformers; install them after the TF-IDF baseline works."
        ) from exc

    values = list(texts)
    if not values:
        return np.empty((0, 0), dtype=np.float32)
    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only).to(target).eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            encoded = tokenizer(
                values[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(target) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            batches.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(batches, axis=0)
