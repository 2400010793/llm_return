import json
import sys
from pathlib import Path

from scripts.compose_pooled_embedding_variants import main


def make_variant(root: Path, shard: int, model: str, variant: str) -> Path:
    output = root / f"shard-{shard}" / model / variant
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps({"rows": 1}))
    (output / "metadata.jsonl").write_text('{"row_index": 1}\n')
    (output / "short_pooling.npz").write_bytes(b"npz")
    return output


def test_compose_uses_short_and_clean_masked_sources(tmp_path, monkeypatch):
    short_root = tmp_path / "short"
    masked_root = tmp_path / "masked"
    output_root = tmp_path / "combined"
    for model in ("roberta", "bge_m3"):
        make_variant(short_root, 0, model, "short")
        make_variant(masked_root, 0, model, "masked_short")
    monkeypatch.setattr(sys, "argv", [
        "compose_pooled_embedding_variants.py",
        "--short-root", str(short_root),
        "--masked-root", str(masked_root),
        "--output-root", str(output_root),
    ])

    main()

    for model in ("roberta", "bge_m3"):
        short = output_root / "shard-0" / model / "short"
        masked = output_root / "shard-0" / model / "masked_short"
        assert short.is_symlink() and short.resolve().is_dir()
        assert masked.is_symlink() and masked.resolve().is_dir()
    summary = json.loads((output_root / "compose_summary.json").read_text())
    assert summary["shards"] == 1
