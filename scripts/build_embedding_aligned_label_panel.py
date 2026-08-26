"""Subset the cleaned label panel to exactly the row_index set in an embedding matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    panel = pd.read_parquet(args.panel)
    metadata = pd.read_parquet(args.metadata, columns=["row_index"])
    if metadata["row_index"].duplicated().any():
        raise ValueError("embedding metadata contains duplicate row_index")
    panel["row_index"] = pd.to_numeric(panel["row_index"], errors="raise").astype("int64")
    selected = panel[panel["row_index"].isin(metadata["row_index"])].copy()
    if len(selected) != len(metadata) or selected["row_index"].duplicated().any():
        raise ValueError(f"panel/embedding row mismatch: panel={len(selected)} embedding={len(metadata)}")
    selected = selected.set_index("row_index").loc[metadata["row_index"].to_numpy()].reset_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output, index=False)
    summary = {
        "format_version": "embedding_aligned_label_panel_v1",
        "source_panel": str(args.panel),
        "embedding_metadata": str(args.metadata),
        "rows": int(len(selected)),
        "row_index_min": int(selected.row_index.min()),
        "row_index_max": int(selected.row_index.max()),
        "ordered_like_embedding_metadata": True,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
