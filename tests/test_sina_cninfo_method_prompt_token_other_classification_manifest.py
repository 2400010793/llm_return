import json
from pathlib import Path

from scripts.build_sina_cninfo_method_prompt_token_classification_manifest import (
    MODELS,
    VARIANTS,
)
from scripts.build_sina_cninfo_method_prompt_token_other_classification_manifest import (
    CLASSIFIERS,
    build_rows,
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


def test_builds_32_additional_classifier_tasks(tmp_path: Path):
    embedding_root = tmp_path / "embeddings"
    output_root = tmp_path / "outputs"
    write_complete_tree(embedding_root)
    rows = build_rows(
        embedding_root,
        output_root,
        expected_shards=2,
        expected_rows=4,
    )
    assert len(rows) == 32
    assert {row["classifier"] for row in rows} == set(CLASSIFIERS)
    assert {row["feature"] for row in rows} == {"prompt_tokens_flat"}
    assert {row["reducer"] for row in rows} == {"pca"}
    assert {row["components"] for row in rows} == {128}
    assert [row["task_id"] for row in rows] == list(range(32))
