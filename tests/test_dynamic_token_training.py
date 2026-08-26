import numpy as np
import json
from pathlib import Path
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.data.prompt_token_embeddings import PromptTokenEmbeddingStore
from src.models.dynamic_token_gating import TokenVariableSelectionNetwork
from src.models.dynamic_token_training import (
    TargetTransform,
    encode_token_store,
    predict_token_store,
    token_weight_summary,
    train_token_epoch,
)


def test_regression_target_transform_round_trip() -> None:
    transform = TargetTransform.fit(np.array([-1.0, 0.0, 2.0]), task="regression")
    raw = np.array([-0.5, 0.25])
    encoded = transform.encode_tensor(torch.tensor(raw, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(transform.decode_array(encoded), raw, atol=1e-6)


def test_classification_transform_decodes_logits_to_probabilities() -> None:
    transform = TargetTransform.fit(np.array([0.0, 1.0]), task="classification")
    np.testing.assert_allclose(transform.decode_array(np.array([0.0])), [0.5])


def test_token_weight_summary_detects_static_uniform_weights() -> None:
    summary = token_weight_summary(
        np.full((4, 2), 0.5, dtype=np.float32), ["甲", "乙"]
    )
    assert summary["normalized_entropy_mean"] == pytest.approx(1.0)
    assert summary["mean_position_std_across_announcements"] == 0.0
    assert summary["positions"][0]["token"] == "甲"


def test_streamed_training_and_prediction_round_trip(tmp_path: Path) -> None:
    directory = tmp_path / "shard-0" / "roberta" / "masked_short"
    directory.mkdir(parents=True)
    rng = np.random.default_rng(42)
    np.save(
        directory / "prompt_token_embeddings.npy",
        rng.normal(size=(6, 2, 3)).astype(np.float32),
    )
    pd.DataFrame({"row_index": np.arange(1, 7)}).to_json(
        directory / "metadata.jsonl", orient="records", lines=True
    )
    (directory / "prompt_tokens.json").write_text(
        json.dumps({"text": "甲乙", "tokens": ["甲", "乙"]}), encoding="utf-8"
    )
    (directory / "summary.json").write_text(json.dumps({"rows": 6}), encoding="utf-8")
    store = PromptTokenEmbeddingStore(
        tmp_path, model="roberta", variant="masked_short",
        expected_shards=1, expected_rows=6,
        expected_token_count=2, expected_hidden_size=3,
    )
    selected = np.array([False, True, True, True, True, True, True])
    targets = np.array([np.nan, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    transform = TargetTransform.fit(targets[1:], task="regression")
    model = TokenVariableSelectionNetwork(
        3, 2, representation_size=4, gate_hidden_size=4, gate_mode="dynamic"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_token_epoch(
        model, store, selected, targets,
        transform=transform, optimizer=optimizer, device=torch.device("cpu"),
        batch_size=3, seed=42, gradient_clip_norm=1.0,
    )
    prediction = predict_token_store(
        model, store, selected, targets, transform=transform,
        device=torch.device("cpu"), batch_size=2,
    )
    assert np.isfinite(loss)
    assert prediction.row_indexes.tolist() == [1, 2, 3, 4, 5, 6]
    assert prediction.weights.shape == (6, 2)
    np.testing.assert_allclose(prediction.weights.sum(axis=1), 1.0, atol=1e-6)
    encoded = encode_token_store(
        model, store, selected, targets, device=torch.device("cpu"), batch_size=2,
    )
    assert encoded.row_indexes.tolist() == [1, 2, 3, 4, 5, 6]
    assert encoded.representations.shape == (6, 4)
    assert encoded.weights.shape == (6, 2)
    np.testing.assert_allclose(encoded.weights.sum(axis=1), 1.0, atol=1e-6)
