"""Audit annual CNINFO shards for a CSV stock universe."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--final-end-date", required=True)
    parser.add_argument("--expected-stocks", type=int, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    with args.stocks.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("active", "1") == "1"]
    stock_ids = sorted({row["stock_id"].strip().zfill(6) for row in rows})
    if len(stock_ids) != args.expected_stocks:
        raise ValueError(f"expected {args.expected_stocks} stocks, got {len(stock_ids)}")

    expected = len(stock_ids) * (args.end_year - args.start_year + 1)
    complete = missing = invalid = partial = records = error_records = 0
    incomplete_examples: list[dict[str, str]] = []
    for stock_id in stock_ids:
        for year in range(args.start_year, args.end_year + 1):
            path = args.output_dir / f"{stock_id}_{year}.json"
            if not path.exists():
                missing += 1
                reason = "missing"
            else:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    invalid += 1
                    reason = "invalid_json"
                else:
                    query = payload.get("query", {}) if isinstance(payload, dict) else {}
                    rows = payload.get("records", []) if isinstance(payload, dict) else []
                    errors = sum(
                        isinstance(row, dict) and row.get("content_type") == "collection_error"
                        for row in rows
                    ) if isinstance(rows, list) else 0
                    end_date = args.final_end_date if year == args.end_year else f"{year}-12-31"
                    query_matches = (
                        query.get("start_date") == f"{year}-01-01"
                        and query.get("end_date") == end_date
                        and not query.get("index_only", False)
                        and not query.get("focus_only", False)
                    )
                    if payload.get("status") == "complete" and query_matches and not errors:
                        complete += 1
                        records += len(rows)
                        continue
                    partial += 1
                    error_records += errors
                    reason = "partial_or_query_mismatch"
            if len(incomplete_examples) < 50:
                incomplete_examples.append(
                    {"stock_id": stock_id, "year": str(year), "path": str(path), "reason": reason}
                )

    summary = {
        "status": "complete" if complete == expected else "partial",
        "stocks": len(stock_ids),
        "years": [args.start_year, args.end_year],
        "final_end_date": args.final_end_date,
        "expected_stock_years": expected,
        "complete_stock_years": complete,
        "missing_stock_years": missing,
        "partial_stock_years": partial,
        "invalid_stock_years": invalid,
        "records_in_complete_files": records,
        "collection_error_records": error_records,
        "incomplete_examples": incomplete_examples,
        "output_dir": str(args.output_dir),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
