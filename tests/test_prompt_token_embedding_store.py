import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.prompt_token_embeddings import PromptTokenEmbeddingStore


def write_shard(root: Path, shard: int, rows: list[int], values: np.ndarray) -> None:
    directory = root / f"shard-{shard}" / "roberta" / "masked_short"
    directory.mkdir(parents=True)
    np.save(directory / "prompt_token_embeddings.npy", values.astype(np.float32))
    pd.DataFrame({"row_index": rows}).to_json(
        directory / "metadata.jsonl", orient="records", lines=True
    )
    (directory / "prompt_tokens.json").write_text(
        json.dumps({"text": "xy", "tokens": ["x", "y"]}), encoding="utf-8"
    )
    (directory / "summary.json").write_text(
        json.dumps({"rows": len(rows)}), encoding="utf-8"
    )


def test_store_streams_selected_rows_in_reproducible_batches(tmp_path: Path) -> None:
    write_shard(tmp_path, 0, [1, 3], np.arange(8).reshape(2, 2, 2))
    write_shard(tmp_path, 1, [2, 4], np.arange(8, 16).reshape(2, 2, 2))
    store = PromptTokenEmbeddingStore(
        tmp_path, model="roberta", variant="masked_short",
        expected_shards=2, expected_rows=4,
        expected_token_count=2, expected_hidden_size=2,
    )
    selected = np.array([False, True, True, False, True])
    targets = np.array([np.nan, 0.1, 0.2, 0.3, np.nan])
    batches = list(store.iter_batches(
        selected, targets, batch_size=1, shuffle=False, seed=42,
    ))
    assert [batch.row_indexes.tolist() for batch in batches] == [[1], [2]]
    assert store.count_selected(selected, targets) == 2
    np.testing.assert_allclose(
        np.concatenate([batch.targets for batch in batches]), [0.1, 0.2]
    )


def test_store_rejects_cross_shard_duplicate_rows(tmp_path: Path) -> None:
    write_shard(tmp_path, 0, [1], np.zeros((1, 2, 2)))
    write_shard(tmp_path, 1, [1], np.ones((1, 2, 2)))
    with pytest.raises(ValueError, match="across"):
        PromptTokenEmbeddingStore(
            tmp_path, model="roberta", variant="masked_short", expected_shards=2,
        )


def test_selected_position_view_streams_only_requested_tokens(tmp_path: Path) -> None:
    values = np.arange(24).reshape(2, 3, 4)
    directory = tmp_path / "shard-0" / "roberta" / "masked_short"
    directory.mkdir(parents=True)
    np.save(directory / "prompt_token_embeddings.npy", values.astype(np.float32))
    pd.DataFrame({"row_index": [1, 2]}).to_json(
        directory / "metadata.jsonl", orient="records", lines=True
    )
    (directory / "prompt_tokens.json").write_text(
        json.dumps({"text": "xyz", "tokens": ["x", "y", "z"]}), encoding="utf-8"
    )
    (directory / "summary.json").write_text(json.dumps({"rows": 2}), encoding="utf-8")
    store = PromptTokenEmbeddingStore(
        tmp_path, model="roberta", variant="masked_short",
        expected_shards=1, expected_rows=2,
    )
    view = store.select_positions([0, 2])
    selected = np.array([False, True, True])
    targets = np.array([np.nan, 0.1, -0.2])
    batches = list(view.iter_batches(
        selected, targets, batch_size=2, shuffle=False, seed=0,
    ))
    assert view.token_count == 2
    assert view.prompt_tokens == ("x", "z")
    np.testing.assert_array_equal(batches[0].values, values[:, [0, 2], :])


def test_selected_position_view_rejects_invalid_positions(tmp_path: Path) -> None:
    write_shard(tmp_path, 0, [1], np.zeros((1, 2, 2)))
    store = PromptTokenEmbeddingStore(
        tmp_path, model="roberta", variant="masked_short", expected_shards=1,
    )
    with pytest.raises(ValueError, match="unique"):
        store.select_positions([0, 0])
    with pytest.raises(ValueError, match="0..1"):
        store.select_positions([2])
