"""Deduplicate cleaned CNINFO records while preserving cross-stock records.

The deduplication key is (stock_id, text_hash). Therefore, identical text linked
 to two different stocks is retained once for each stock, as required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text.preprocess_zh import text_hash


def iter_records(input_dirs: list[Path]):
    for input_dir in input_dirs:
        for path in sorted(input_dir.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSON: {path}:{line_number}") from exc
                    if isinstance(record, dict):
                        yield path, record


def selected_stock_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    import pandas as pd

    frame = pd.read_parquet(path, columns=["stock_id"])
    return {
        value
        for value in frame["stock_id"].astype(str).str.strip().str.zfill(6)
        if re.fullmatch(r"\d{6}", value)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--stock-ids-from", type=Path, help="optional parquet panel used to retain only selected stocks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    seen: set[tuple[str, str]] = set()
    allowed_stock_ids = selected_stock_ids(args.stock_ids_from)
    source_files: set[str] = set()
    input_records = output_records = duplicate_records = filtered_records = 0
    stock_counts: Counter[str] = Counter()
    duplicate_by_stock: Counter[str] = Counter()

    with temporary.open("w", encoding="utf-8") as output:
        for source_path, record in iter_records(args.input_dir):
            input_records += 1
            source_files.add(source_path.name)
            stock_id = str(record.get("stock_id", "")).strip().zfill(6)
            if allowed_stock_ids is not None and stock_id not in allowed_stock_ids:
                filtered_records += 1
                continue
            record_hash = str(record.get("text_hash", "")).strip()
            if not record_hash:
                record_hash = text_hash(str(record.get("text", "") or ""))
            key = (stock_id, record_hash)
            if key in seen:
                duplicate_records += 1
                duplicate_by_stock[stock_id] += 1
                continue
            seen.add(key)
            record["dedup_key"] = f"{stock_id}:{record_hash}"
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_records += 1
            stock_counts[stock_id] += 1
    os.replace(temporary, args.output)

    summary = {
        "input_dirs": [str(path) for path in args.input_dir],
        "input_files": len(source_files),
        "input_records": input_records,
        "records_filtered_by_stock_universe": filtered_records,
        "output_records": output_records,
        "duplicate_records_removed": duplicate_records,
        "duplicate_rate": duplicate_records / input_records if input_records else 0.0,
        "stocks_after_dedup": len(stock_counts),
        "cross_stock_policy": "preserve identical text once per stock_id; deduplication key is (stock_id, text_hash)",
        "duplicate_records_by_stock_top20": duplicate_by_stock.most_common(20),
        "output": str(args.output),
    }
    summary_temporary = args.summary.with_suffix(args.summary.suffix + f".tmp.{os.getpid()}")
    summary_temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(summary_temporary, args.summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
