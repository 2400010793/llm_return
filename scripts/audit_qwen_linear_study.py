"""Audit the frozen Qwen candidate store before return experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts", type=int, default=71)
    parser.add_argument("--rows", type=int, default=350577)
    parser.add_argument("--dimension", type=int, default=4096)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel, columns=["row_index", "stock_id", "entry_date"])
    if len(panel) != args.rows or panel["row_index"].duplicated().any():
        raise ValueError("panel row count or row_index uniqueness failed")
    panel_rows = np.sort(pd.to_numeric(panel["row_index"]).to_numpy(np.int64))
    if not np.array_equal(panel_rows, np.arange(1, args.rows + 1, dtype=np.int64)):
        raise ValueError("panel row_index is not the expected one-based range")

    seen: list[np.ndarray] = []
    view_rows = {view: 0 for view in ("plain", "short", "masked_short")}
    for part_id in range(args.parts):
        part = args.source_root / f"part-{part_id:05d}"
        metadata = pd.read_parquet(part / "metadata.parquet")
        rows = pd.to_numeric(metadata["row_index"], errors="raise").to_numpy(np.int64)
        if len(np.unique(rows)) != len(rows):
            raise ValueError(f"duplicate source row_index in {part}")
        seen.append(rows)
        for view in view_rows:
            matrix = np.load(part / f"{view}.npy", mmap_mode="r")
            expected = (len(metadata), args.dimension)
            if matrix.shape != expected:
                raise ValueError(f"{part}/{view} shape {matrix.shape} != {expected}")
            for start in range(0, len(matrix), 256):
                if not np.isfinite(np.asarray(matrix[start:start + 256])).all():
                    raise ValueError(f"non-finite values in {part}/{view} at row {start}")
            view_rows[view] += len(matrix)

    combined = np.concatenate(seen)
    if not np.array_equal(np.sort(combined), np.arange(args.rows, dtype=np.int64)):
        raise ValueError("Qwen source row_index is not a complete zero-based range")
    dates = pd.to_datetime(panel["entry_date"], errors="raise")
    result = {
        "status": "passed", "parts": args.parts, "rows": args.rows,
        "dimension": args.dimension, "views": view_rows,
        "row_index_alignment": "Qwen zero-based + 1 equals panel one-based",
        "stocks": int(panel["stock_id"].nunique()),
        "date_start": str(dates.min().date()), "date_end": str(dates.max().date()),
        "years": sorted(int(year) for year in dates.dt.year.unique()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
