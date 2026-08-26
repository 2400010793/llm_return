"""Audit yearly CNINFO shards for the selected classification stock universe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock-ids-from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2017)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    panel = pd.read_parquet(args.stock_ids_from, columns=["stock_id"])
    stock_ids = sorted(panel["stock_id"].astype(str).str.strip().str.zfill(6).unique())
    expected = len(stock_ids) * (args.end_year - args.start_year + 1)
    complete = partial = missing = invalid = records = 0
    incomplete_examples: list[str] = []

    for stock_id in stock_ids:
        for year in range(args.start_year, args.end_year + 1):
            path = args.output_dir / f"{stock_id}_{year}.json"
            if not path.exists():
                missing += 1
                if len(incomplete_examples) < 20:
                    incomplete_examples.append(str(path))
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                invalid += 1
                if len(incomplete_examples) < 20:
                    incomplete_examples.append(str(path))
                continue
            query = payload.get("query", {}) if isinstance(payload, dict) else {}
            query_matches = (
                query.get("start_date") == f"{year}-01-01"
                and query.get("end_date") == f"{year}-12-31"
                and not query.get("index_only", False)
                and not query.get("focus_only", False)
            )
            if payload.get("status") == "complete" and query_matches:
                complete += 1
                records += len(payload.get("records", []))
            else:
                partial += 1
                if len(incomplete_examples) < 20:
                    incomplete_examples.append(str(path))

    summary = {
        "status": "complete" if complete == expected else "partial",
        "stocks": len(stock_ids),
        "years": [args.start_year, args.end_year],
        "expected_stock_years": expected,
        "complete_stock_years": complete,
        "partial_stock_years": partial,
        "missing_stock_years": missing,
        "invalid_stock_years": invalid,
        "records_in_complete_files": records,
        "incomplete_examples": incomplete_examples,
        "output_dir": str(args.output_dir),
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
