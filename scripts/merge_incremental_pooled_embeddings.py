"""Create a zero-copy pooled-embedding view from frozen and incremental shards.

The output contains symlinks to the existing 2018--2026 shard directories and
new shard directories whose metadata already uses global row indices. No old
embedding array is read, copied, or re-encoded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pooled_embeddings import discover_pooled_parts


def _shard_number(path: Path) -> int:
    return int(path.name.split("-", 1)[1])


def _shards(root: Path) -> list[Path]:
    result = sorted(
        (path for path in root.glob("shard-*") if path.is_dir()),
        key=_shard_number,
    )
    ids = [_shard_number(path) for path in result]
    if ids != list(range(len(ids))):
        raise ValueError(f"{root} must contain contiguous shard IDs starting at zero; found {ids}")
    return result


def _metadata_rows(parts: list[Path]) -> list[int]:
    rows: list[int] = []
    for part in parts:
        with (part / "metadata.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    rows.append(int(value["row_index"]))
    return rows


def _validate_range(root: Path, models: tuple[str, ...], variants: tuple[str, ...], start: int, end: int) -> None:
    expected = np.arange(start, end + 1, dtype=np.int64)
    for model in models:
        for variant in variants:
            parts = discover_pooled_parts(root, model, variant)
            actual = np.asarray(_metadata_rows(parts), dtype=np.int64)
            if not np.array_equal(np.sort(actual), expected):
                raise ValueError(
                    f"{root} {model}/{variant} row range mismatch: "
                    f"expected {start}..{end}, found {len(actual)} rows"
                )
            if len(np.unique(actual)) != len(actual):
                raise ValueError(f"duplicate row_index values in {root} {model}/{variant}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-root", type=Path, required=True)
    parser.add_argument("--incremental-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--existing-rows", type=int, required=True)
    parser.add_argument("--incremental-rows", type=int, required=True)
    parser.add_argument("--models", default="roberta,bge_m3")
    parser.add_argument("--variants", default="short,masked_short")
    args = parser.parse_args()
    if args.existing_rows < 1 or args.incremental_rows < 1:
        raise ValueError("existing-rows and incremental-rows must be positive")
    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite merge output: {args.output_root}")
    existing_shards = _shards(args.existing_root)
    incremental_shards = _shards(args.incremental_root)
    if not existing_shards or not incremental_shards:
        raise ValueError("both embedding roots must contain shard directories")
    _validate_range(args.existing_root, models, variants, 1, args.existing_rows)
    _validate_range(
        args.incremental_root, models, variants,
        args.existing_rows + 1,
        args.existing_rows + args.incremental_rows,
    )

    args.output_root.mkdir(parents=True)
    for source in existing_shards:
        target = args.output_root / source.name
        target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=True)
    offset = len(existing_shards)
    for source in incremental_shards:
        target = args.output_root / f"shard-{offset + _shard_number(source)}"
        target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=True)

    summary = {
        "existing_root": str(args.existing_root),
        "incremental_root": str(args.incremental_root),
        "output_root": str(args.output_root),
        "existing_rows": args.existing_rows,
        "incremental_rows": args.incremental_rows,
        "rows": args.existing_rows + args.incremental_rows,
        "existing_shards": len(existing_shards),
        "incremental_shards": len(incremental_shards),
        "copy_policy": "symlink shard directories; do not copy or re-encode frozen arrays",
        "models": list(models),
        "variants": list(variants),
    }
    (args.output_root / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
