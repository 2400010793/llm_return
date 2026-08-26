from pathlib import Path

from scripts.build_sina_cninfo_method_downstream_manifests import (
    MODELS,
    VARIANTS,
    complete,
)


def test_complete_requires_every_shard(tmp_path: Path):
    for shard in range(2):
        directory = tmp_path / f"shard-{shard}" / "roberta" / "short"
        directory.mkdir(parents=True)
        for name in ("summary.json", "metadata.jsonl", "short_pooling.npz"):
            (directory / name).write_bytes(b"x")
    assert complete(tmp_path, "roberta", "short", 2)
    (tmp_path / "shard-1/roberta/short/summary.json").unlink()
    assert not complete(tmp_path, "roberta", "short", 2)


def test_manifest_matrix_constants_cover_four_models_and_five_variants():
    assert MODELS == ("roberta", "bge_m3", "ckip_bert", "xlm_roberta_large")
    assert VARIANTS == ("plain", "short", "masked_short", "long", "masked_long")
