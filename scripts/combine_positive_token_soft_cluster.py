"""Combine and audit nine direction-token soft-cluster test folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.fold_root.glob("20??/stock_day_predictions.parquet"))
    if len(files) != 9:
        raise ValueError(f"expected 9 completed folds under {args.fold_root}; found {len(files)}")
    years = [int(path.parent.name) for path in files]
    if years != list(range(2018, 2027)):
        raise ValueError(f"unexpected test years: {years}")
    frames = [pd.read_parquet(path) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    duplicates = combined.duplicated(["stock_id", "entry_date"])
    if duplicates.any():
        raise ValueError(f"combined predictions contain {int(duplicates.sum())} duplicate stock-days")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.sort_values(["entry_date", "stock_id"]).to_parquet(args.output, index=False)
    audit = {
        "format_version": "positive_token_soft_cluster_combined_v1",
        "folds": len(files), "years": years, "rows": len(combined),
        "date_start": str(pd.to_datetime(combined["entry_date"]).min().date()),
        "date_end": str(pd.to_datetime(combined["entry_date"]).max().date()),
        "prediction_columns": [
            column for column in combined.columns if column.startswith("prediction_")
        ],
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
