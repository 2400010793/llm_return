"""Validate an embedding matrix and its row-level text metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-input-chars", type=int, default=12000)
    parser.add_argument("--expected-dimensions", type=int)
    parser.add_argument(
        "--allow-duplicate-vectors",
        action="store_true",
        help="Report exact duplicate rows as a warning instead of a failed check",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    metadata = pd.read_parquet(args.metadata)
    matrix = np.load(args.matrix, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"embedding matrix must be 2D, got {matrix.shape}")

    texts = [
        ("" if pd.isna(value) else str(value))[: args.max_input_chars]
        for value in frame[args.text_column]
    ]
    expected_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    metadata_hashes = metadata["text_sha256"].astype(str).tolist()
    row_count_ok = len(frame) == len(metadata) == matrix.shape[0]
    dimension_ok = args.expected_dimensions is None or matrix.shape[1] == args.expected_dimensions
    hash_alignment_ok = expected_hashes == metadata_hashes

    values = np.asarray(matrix, dtype=np.float32)
    finite_rows = np.isfinite(values).all(axis=1)
    norms = np.linalg.norm(values, axis=1)
    zero_rows = np.flatnonzero(norms == 0).astype(int).tolist()
    duplicate_rows = int(len(values) - len({row.tobytes() for row in values}))
    checks = {
        "row_count": row_count_ok,
        "expected_dimensions": dimension_ok,
        "text_hash_alignment": hash_alignment_ok,
        "all_finite": bool(finite_rows.all()),
        "no_zero_vectors": not zero_rows,
        "no_duplicate_vectors": duplicate_rows == 0 or args.allow_duplicate_vectors,
    }
    report = {
        "input": str(args.input.resolve()),
        "matrix": str(args.matrix.resolve()),
        "metadata": str(args.metadata.resolve()),
        "shape": [int(value) for value in matrix.shape],
        "dtype": str(matrix.dtype),
        "max_input_chars": args.max_input_chars,
        "checks": checks,
        "passed": all(checks.values()),
        "nonfinite_rows": np.flatnonzero(~finite_rows).astype(int).tolist(),
        "zero_rows": zero_rows,
        "duplicate_row_count": duplicate_rows,
        "duplicate_vectors_allowed": args.allow_duplicate_vectors,
        "norm": {
            "min": float(norms.min()),
            "median": float(np.median(norms)),
            "mean": float(norms.mean()),
            "max": float(norms.max()),
        },
        "matrix_sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
