import json

import numpy as np
import pandas as pd
import pytest

from src.data.pooled_embeddings import align_embeddings_to_panel, load_pooled_embeddings


def write_part(root, shard, rows, matrix, *, external_prompt_tokens=False):
    directory = root / f"shard-{shard}" / "roberta" / "short"
    directory.mkdir(parents=True)
    prompt_tokens = np.stack([matrix, matrix + 10], axis=1)
    arrays = {
        "prompt_mean": matrix,
        "title_mean": matrix + 1,
        "body_mean": matrix + 2,
        "title_body_mean": matrix + 3,
        "full_mean": matrix + 4,
    }
    if not external_prompt_tokens:
        arrays["prompt_token_embeddings"] = prompt_tokens
    else:
        np.save(directory / "prompt_token_embeddings.npy", prompt_tokens)
    np.savez_compressed(
        directory / "short_pooling.npz",
        **arrays,
    )
    (directory / "metadata.jsonl").write_text(
        "".join(json.dumps({"row_index": row}) + "\n" for row in rows), encoding="utf-8"
    )
    (directory / "summary.json").write_text(
        json.dumps({
            "model": "roberta",
            "variant": "short",
            "rows": len(rows),
            "outputs": {
                "prompt_token_embeddings": [len(rows), 2, matrix.shape[1]],
                "prompt_mean": list(matrix.shape),
                "title_mean": list(matrix.shape),
                "body_mean": list(matrix.shape),
                "title_body_mean": list(matrix.shape),
                "full_mean": list(matrix.shape),
            },
        }), encoding="utf-8"
    )


def test_loads_modulo_shards_and_sorts_by_row_index(tmp_path):
    write_part(tmp_path, 0, [1, 3], np.array([[10, 11], [30, 31]], dtype=np.float32))
    write_part(tmp_path, 1, [2, 4], np.array([[20, 21], [40, 41]], dtype=np.float32))
    loaded = load_pooled_embeddings(
        tmp_path, model="roberta", variant="short", feature="body_mean",
        require_complete_rows=4,
    )
    assert loaded.metadata["row_index"].tolist() == [1, 2, 3, 4]
    assert loaded.matrix.tolist() == [[12, 13], [22, 23], [32, 33], [42, 43]]


def test_loads_prompt_mean_pooling(tmp_path):
    write_part(tmp_path, 0, [1], np.array([[10, 11]], dtype=np.float32))
    loaded = load_pooled_embeddings(
        tmp_path, model="roberta", variant="short", feature="prompt_mean"
    )
    assert loaded.matrix.tolist() == [[10, 11]]


def test_aligns_panel_by_row_index_not_panel_order(tmp_path):
    write_part(tmp_path, 0, [1, 2], np.array([[10, 11], [20, 21]], dtype=np.float32))
    loaded = load_pooled_embeddings(tmp_path, model="roberta", variant="short", feature="title_mean")
    panel = pd.DataFrame({"row_index": [2, 1], "target": [1, 0]})
    frame, matrix = align_embeddings_to_panel(panel, loaded)
    assert frame["row_index"].tolist() == [2, 1]
    assert matrix.tolist() == [[21, 22], [11, 12]]


def test_concatenates_segments_without_averaging_them(tmp_path):
    write_part(tmp_path, 0, [1], np.array([[10, 11]], dtype=np.float32))
    loaded = load_pooled_embeddings(
        tmp_path, model="roberta", variant="short", feature="title_body_concat"
    )
    assert loaded.matrix.shape == (1, 4)
    assert loaded.matrix.tolist() == [[11, 12, 12, 13]]


def test_flattens_all_prompt_tokens_in_token_then_hidden_order(tmp_path):
    write_part(tmp_path, 0, [1], np.array([[10, 11]], dtype=np.float32))
    loaded = load_pooled_embeddings(
        tmp_path, model="roberta", variant="short", feature="prompt_tokens_flat"
    )
    assert loaded.matrix.shape == (1, 4)
    assert loaded.matrix.tolist() == [[10, 11, 20, 21]]
    assert loaded.component_shapes == ((2, 2),)


def test_loads_prompt_tokens_from_external_memmap(tmp_path):
    write_part(
        tmp_path, 0, [1], np.array([[10, 11]], dtype=np.float32),
        external_prompt_tokens=True,
    )
    loaded = load_pooled_embeddings(
        tmp_path, model="roberta", variant="short", feature="prompt_tokens_flat"
    )
    assert loaded.matrix.shape == (1, 4)
    assert loaded.matrix.tolist() == [[10, 11, 20, 21]]
    assert loaded.component_shapes == ((2, 2),)


def test_concatenates_title_body_and_full_in_declared_order(tmp_path):
    write_part(tmp_path, 0, [1], np.array([[10, 11]], dtype=np.float32))
    loaded = load_pooled_embeddings(
        tmp_path, model="roberta", variant="short", feature="title_body_full_concat"
    )
    assert loaded.matrix.shape == (1, 6)
    assert loaded.matrix.tolist() == [[11, 12, 12, 13, 14, 15]]
    assert loaded.component_shapes == ((2,), (2,), (2,))


def test_rejects_duplicate_embedding_row_index(tmp_path):
    write_part(tmp_path, 0, [1], np.array([[10, 11]], dtype=np.float32))
    write_part(tmp_path, 1, [1], np.array([[20, 21]], dtype=np.float32))
    with pytest.raises(ValueError, match="duplicate embedding row_index"):
        load_pooled_embeddings(tmp_path, model="roberta", variant="short", feature="full_mean")


def test_rejects_incomplete_discovered_shard(tmp_path):
    write_part(tmp_path, 0, [1], np.array([[10, 11]], dtype=np.float32))
    incomplete = tmp_path / "shard-1" / "roberta" / "short"
    incomplete.mkdir(parents=True)
    (incomplete / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete embedding shard outputs"):
        load_pooled_embeddings(tmp_path, model="roberta", variant="short", feature="full_mean")


def test_rejects_non_contiguous_shard_ids(tmp_path):
    write_part(tmp_path, 0, [1], np.array([[10, 11]], dtype=np.float32))
    write_part(tmp_path, 2, [2], np.array([[20, 21]], dtype=np.float32))
    with pytest.raises(ValueError, match="missing shard IDs: \\[1\\]"):
        load_pooled_embeddings(tmp_path, model="roberta", variant="short", feature="full_mean")


def test_rejects_gapped_complete_row_index_range(tmp_path):
    write_part(tmp_path, 0, [1, 3], np.array([[10, 11], [30, 31]], dtype=np.float32))
    with pytest.raises(ValueError, match="must exactly cover 1..2"):
        load_pooled_embeddings(
            tmp_path, model="roberta", variant="short", feature="full_mean",
            require_complete_rows=2,
        )


def test_rejects_estimated_matrix_before_loading_large_feature(tmp_path):
    write_part(tmp_path, 0, [1], np.array([[10, 11]], dtype=np.float32))
    with pytest.raises(ValueError, match="exceeding.*safety limit"):
        load_pooled_embeddings(
            tmp_path, model="roberta", variant="short", feature="prompt_tokens_flat",
            max_matrix_gib=1e-12,
        )