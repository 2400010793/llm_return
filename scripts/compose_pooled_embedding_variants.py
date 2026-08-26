"""Compose a pooled-embedding root from variant-specific source roots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def shard_ids(root: Path) -> list[int]:
    ids = sorted(
        int(path.name.split("-", 1)[1])
        for path in root.glob("shard-*")
        if path.is_dir()
    )
    if ids != list(range(len(ids))):
        raise ValueError(f"non-contiguous shard IDs in {root}: {ids}")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-root", type=Path, required=True)
    parser.add_argument("--masked-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", default="roberta,bge_m3")
    args = parser.parse_args()
    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    if not models:
        raise ValueError("at least one model is required")
    short_root = args.short_root.resolve()
    masked_root = args.masked_root.resolve()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {args.output_root}")
    short_ids = shard_ids(short_root)
    masked_ids = shard_ids(masked_root)
    if not short_ids or short_ids != masked_ids:
        raise ValueError("short and masked roots must have the same non-empty shard IDs")

    staging = args.output_root.parent / f".{args.output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    try:
        for shard in short_ids:
            for model in models:
                target_base = staging / f"shard-{shard}" / model
                target_base.mkdir(parents=True, exist_ok=True)
                sources = {
                    "short": short_root / f"shard-{shard}" / model / "short",
                    "masked_short": masked_root / f"shard-{shard}" / model / "masked_short",
                }
                for variant, source in sources.items():
                    for required in ("summary.json", "metadata.jsonl", "short_pooling.npz"):
                        if not (source / required).is_file():
                            raise FileNotFoundError(f"missing {source / required}")
                    target = target_base / variant
                    target.symlink_to(
                        os.path.relpath(source, target.parent), target_is_directory=True
                    )
        summary = {
            "short_root": str(short_root),
            "masked_root": str(masked_root),
            "output_root": str(args.output_root),
            "models": list(models),
            "variants": ["short", "masked_short"],
            "shards": len(short_ids),
            "copy_policy": "variant directories are relative symlinks",
        }
        (staging / "compose_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staging.replace(args.output_root)
        print(json.dumps(summary, ensure_ascii=False))
    except BaseException:
        if staging.exists():
            for path in sorted(staging.rglob("*"), reverse=True):
                if path.is_symlink() or path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            staging.rmdir()
        raise


if __name__ == "__main__":
    main()
