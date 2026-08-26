import json
from pathlib import Path

from scripts.build_sina_cninfo_method_prompt_token_classification_manifest import (
    MODELS,
    VARIANTS,
    build_rows,
    complete,
)


def write_complete_tree(root: Path, *, shards: int = 2, rows_per_shard: int = 2) -> None:
    for model in MODELS:
        for variant in VARIANTS:
            for shard in range(shards):
                directory = root / f"shard-{shard}" / model / variant
                directory.mkdir(parents=True)
                (directory / "summary.json").write_text(
                    json.dumps({"rows": rows_per_shard}), encoding="utf-8"
                )
                for name in (
                    "metadata.jsonl",
                    "short_pooling.npz",
                    "prompt_token_embeddings.npy",
                    "prompt_input_ids.npy",
                    "prompt_tokens.json",
                ):
                    (directory / name).write_bytes(b"x")


def test_complete_requires_all_prompt_token_files(tmp_path: Path):
    directory = tmp_path / "shard-0" / "roberta" / "short"
    directory.mkdir(parents=True)
    for name in (
        "summary.json",
        "metadata.jsonl",
        "short_pooling.npz",
        "prompt_token_embeddings.npy",
        "prompt_input_ids.npy",
        "prompt_tokens.json",
    ):
        (directory / name).write_bytes(b"x")
    assert complete(tmp_path, "roberta", "short", 1)
    (directory / "prompt_token_embeddings.npy").unlink()
    assert not complete(tmp_path, "roberta", "short", 1)


def test_builds_sixteen_prompt_token_logistic_pca_tasks(tmp_path: Path):
    embedding_root = tmp_path / "embeddings"
    output_root = tmp_path / "outputs"
    write_complete_tree(embedding_root)
    rows = build_rows(
        embedding_root,
        output_root,
        expected_shards=2,
        expected_rows=4,
    )
    assert len(rows) == 16
    assert {row["model"] for row in rows} == set(MODELS)
    assert {row["variant"] for row in rows} == set(VARIANTS)
    assert {row["feature"] for row in rows} == {"prompt_tokens_flat"}
    assert {row["classifier"] for row in rows} == {"logistic"}
    assert {row["reducer"] for row in rows} == {"pca"}
    assert {row["components"] for row in rows} == {128}
    assert all(
        str(row["output"]).endswith(
            "prompt_token_classification/"
            f"{row['model']}_{row['variant']}_prompt_tokens_flat_logistic_pca_128.json"
        )
        for row in rows
    )
