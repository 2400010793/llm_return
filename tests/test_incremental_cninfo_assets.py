import json
from pathlib import Path

import numpy as np

from scripts.build_cninfo_prompt_bundle import main as build_prompt_bundle
from scripts.merge_incremental_pooled_embeddings import main as merge_embeddings
from src.data.pooled_embeddings import load_pooled_embeddings


def test_prompt_bundle_accepts_global_row_offset(tmp_path, monkeypatch):
    source = tmp_path / "historical.jsonl"
    source.write_text(
        "".join(
            json.dumps({
                "row_index": row_index,
                "document_id": f"doc-{row_index}",
                "stock_id": "000001",
                "stock_name": "平安银行",
                "announcement_date": "2017-01-01",
                "title_clean_final": "公告",
                "text_model": "平安银行发布公告。",
            }, ensure_ascii=False) + "\n"
            for row_index in (6, 7)
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle"
    monkeypatch.setattr(
        "sys.argv", [
            "build_cninfo_prompt_bundle.py",
            "--input", str(source),
            "--short-output-dir", str(bundle / "short"),
            "--long-output-dir", str(bundle / "long"),
            "--manifest", str(bundle / "manifest.json"),
            "--row-offset", "5",
            "--num-shards", "2",
            "--part-size", "10",
        ],
    )
    build_prompt_bundle()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_index_start"] == 6
    assert manifest["row_index_end"] == 7
    records = [
        json.loads(line)
        for path in sorted((bundle / "short").glob("shard-*/*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert sorted(record["row_index"] for record in records) == [6, 7]


def _write_embedding_root(root: Path, row_indices: list[int], values: list[float]) -> None:
    for shard, (row_index, value) in enumerate(zip(row_indices, values)):
        part = root / f"shard-{shard}" / "roberta" / "short"
        part.mkdir(parents=True)
        matrix = np.array([[value, value + 1]], dtype=np.float32)
        np.savez_compressed(part / "short_pooling.npz", full_mean=matrix)
        (part / "metadata.jsonl").write_text(
            json.dumps({"row_index": row_index}) + "\n", encoding="utf-8"
        )
        (part / "summary.json").write_text(json.dumps({
            "model": "roberta", "variant": "short", "rows": 1,
            "prompt_condition": "task_prompt", "outputs": {"full_mean": [1, 2]},
        }), encoding="utf-8")


def test_embedding_merge_reuses_frozen_shards_without_copying(tmp_path, monkeypatch):
    existing = tmp_path / "existing"
    increment = tmp_path / "increment"
    output = tmp_path / "combined"
    _write_embedding_root(existing, [1, 2], [1, 2])
    _write_embedding_root(increment, [3, 4], [3, 4])
    monkeypatch.setattr(
        "sys.argv", [
            "merge_incremental_pooled_embeddings.py",
            "--existing-root", str(existing),
            "--incremental-root", str(increment),
            "--output-root", str(output),
            "--existing-rows", "2",
            "--incremental-rows", "2",
            "--models", "roberta",
            "--variants", "short",
        ],
    )
    merge_embeddings()
    assert (output / "shard-0").is_symlink()
    assert (output / "shard-2").is_symlink()
    loaded = load_pooled_embeddings(
        output, model="roberta", variant="short", feature="full_mean",
        require_complete_rows=4,
    )
    assert loaded.metadata["row_index"].tolist() == [1, 2, 3, 4]
    np.testing.assert_array_equal(loaded.matrix[:, 0], [1, 2, 3, 4])
