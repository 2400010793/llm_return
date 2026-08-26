"""Validate Qwen candidate shards and expose them to pooled-model runners.

The source candidate job stores one NPY per text form.  This script creates a
separate, resumable pooled-embedding tree without modifying any source shard.
Row indices are converted from the source's 0-based convention to the panel's
persisted 1-based convention.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_json(temporary, orient="records", lines=True, force_ascii=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        # Deliberately uncompressed: conversion is I/O-bound and downstream
        # loaders would otherwise pay decompression cost for every model cell.
        np.savez(handle, full_mean=np.asarray(matrix, dtype=np.float32))
    os.replace(temporary, path)


def complete_output(directory: Path, *, rows: int, dimension: int) -> bool:
    required = (
        directory / "short_pooling.npz",
        directory / "metadata.jsonl",
        directory / "summary.json",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        summary = json.loads(required[2].read_text(encoding="utf-8"))
        return (
            int(summary["rows"]) == rows
            and summary["outputs"]["full_mean"] == [rows, dimension]
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--views", default="short,masked_short,plain")
    parser.add_argument("--expected-parts", type=int, default=71)
    parser.add_argument("--expected-rows", type=int, default=350577)
    parser.add_argument("--expected-dimension", type=int, default=4096)
    args = parser.parse_args()
    views = [value.strip() for value in args.views.split(",") if value.strip()]
    if not views or len(set(views)) != len(views):
        raise ValueError("views must be a non-empty unique comma list")

    seen_rows: list[np.ndarray] = []
    converted = skipped = 0
    per_view_rows = {view: 0 for view in views}
    for part_id in range(args.expected_parts):
        source = args.input_root / f"part-{part_id:05d}"
        metadata_path = source / "metadata.parquet"
        missing = [
            str(path) for path in (
                metadata_path, *(source / f"{view}.npy" for view in views)
            ) if not path.is_file()
        ]
        if missing:
            raise ValueError(f"incomplete Qwen source part {part_id}: {missing}")
        metadata = pd.read_parquet(metadata_path)
        if "row_index" not in metadata:
            raise ValueError(f"row_index missing from {metadata_path}")
        raw_rows = pd.to_numeric(metadata["row_index"], errors="raise").to_numpy(
            dtype=np.int64
        )
        if len(np.unique(raw_rows)) != len(raw_rows):
            raise ValueError(f"duplicate row_index values in {metadata_path}")
        seen_rows.append(raw_rows)
        pooled_metadata = metadata.copy()
        pooled_metadata["source_row_index_zero_based"] = raw_rows
        pooled_metadata["row_index"] = raw_rows + 1

        for view in views:
            source_matrix = np.load(source / f"{view}.npy", mmap_mode="r")
            if source_matrix.shape != (len(metadata), args.expected_dimension):
                raise ValueError(
                    f"unexpected {view} shape in {source}: {source_matrix.shape}"
                )
            if not np.isfinite(source_matrix).all():
                raise ValueError(f"non-finite {view} values in {source}")
            destination = (
                args.output_root / f"shard-{part_id}" / "qwen3_embedding_8b" / view
            )
            if complete_output(
                destination, rows=len(metadata), dimension=args.expected_dimension
            ):
                skipped += 1
            else:
                atomic_npz(destination / "short_pooling.npz", source_matrix)
                atomic_jsonl(destination / "metadata.jsonl", pooled_metadata)
                atomic_json(destination / "summary.json", {
                    "format_version": "qwen_candidate_pooled_adapter_v1",
                    "model": "qwen3_embedding_8b",
                    "variant": view,
                    "prompt_condition": "candidate_text",
                    "source": str(source),
                    "rows": int(len(metadata)),
                    "outputs": {"full_mean": [int(len(metadata)), args.expected_dimension]},
                })
                converted += 1
            per_view_rows[view] += len(metadata)

    combined = np.concatenate(seen_rows)
    expected = np.arange(args.expected_rows, dtype=np.int64)
    actual = np.sort(combined)
    if not np.array_equal(actual, expected):
        missing = np.setdiff1d(expected, actual, assume_unique=False)[:10].tolist()
        extra = np.setdiff1d(actual, expected, assume_unique=False)[:10].tolist()
        raise ValueError(f"Qwen row coverage mismatch: missing={missing}, extra={extra}")
    if any(rows != args.expected_rows for rows in per_view_rows.values()):
        raise ValueError(f"per-view row totals are incomplete: {per_view_rows}")
    summary = {
        "format_version": "qwen_candidate_pooled_adapter_v1",
        "source_root": str(args.input_root),
        "output_root": str(args.output_root),
        "views": views,
        "parts": args.expected_parts,
        "rows": args.expected_rows,
        "dimension": args.expected_dimension,
        "row_index_conversion": "source zero-based + 1 = panel row_index",
        "per_view_rows": per_view_rows,
        "converted_cells": converted,
        "resumed_cells": skipped,
    }
    atomic_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
