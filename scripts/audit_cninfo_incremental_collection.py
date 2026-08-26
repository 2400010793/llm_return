"""Audit one date-bounded CNINFO collection shard for every selected stock."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--expected-stocks", type=int, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    with args.stocks.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("active", "1") == "1"]
    stock_ids = sorted({row["stock_id"].strip().zfill(6) for row in rows})
    if len(stock_ids) != args.expected_stocks:
        raise ValueError(f"expected {args.expected_stocks} stocks, got {len(stock_ids)}")

    complete = missing = invalid = partial = records = error_records = 0
    incomplete_examples: list[dict[str, str]] = []
    for stock_id in stock_ids:
        path = args.output_dir / f"{stock_id}_{args.suffix}.json"
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
                query_matches = (
                    query.get("start_date") == args.start_date
                    and query.get("end_date") == args.end_date
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
        if len(incomplete_examples) < 30:
            incomplete_examples.append({"stock_id": stock_id, "path": str(path), "reason": reason})

    summary = {
        "status": "complete" if complete == args.expected_stocks else "partial",
        "date_range": [args.start_date, args.end_date],
        "suffix": args.suffix,
        "expected_stocks": args.expected_stocks,
        "complete_stocks": complete,
        "missing_stocks": missing,
        "partial_stocks": partial,
        "invalid_stocks": invalid,
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
