"""Split rolling predictions by test year and build an evaluation manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--factor-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--year-column", default="test_year")
    args = parser.parse_args()

    predictions = pd.read_parquet(args.predictions)
    if args.year_column not in predictions:
        raise ValueError(f"predictions missing {args.year_column}")
    input_root = args.output_root / "inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    rows = []
    numeric_year = pd.to_numeric(predictions[args.year_column], errors="coerce")
    for year in sorted(int(value) for value in numeric_year.dropna().unique()):
        yearly = predictions.loc[numeric_year.eq(year)].copy()
        if yearly.empty:
            continue
        factor_id = f"{args.factor_id}_{year}"
        prediction_path = input_root / f"{factor_id}.parquet"
        yearly.to_parquet(prediction_path, index=False)
        rows.append({
            "task_id": len(rows),
            "source_task_id": year,
            "predictions": str(prediction_path),
            "factor_id": factor_id,
            "output_dir": str(args.output_root / str(year)),
        })
    if not rows:
        raise ValueError("no yearly prediction groups found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "predictions": str(args.predictions),
        "factor_id": args.factor_id,
        "years": [row["source_task_id"] for row in rows],
        "tasks": len(rows),
        "output": str(args.output),
        "output_root": str(args.output_root),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
