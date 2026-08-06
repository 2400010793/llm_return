"""Run bounded, sequential batches of stock-specific collection.

This runner deliberately avoids unbounded background crawling. It reads a stock
CSV, processes a limited number of batches, waits between batches, and leaves a
manifest for incremental deduplication.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_BATCH_PAUSE = 60.0


def load_stocks(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("active", "1") == "1"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", default="data/stocks_sample.csv")
    parser.add_argument("--output-dir", default="data/interim/stock_batches")
    parser.add_argument("--manifest", default="data/interim/collector_manifest_all_stocks.json")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--batch-pause", type=float, default=DEFAULT_BATCH_PAUSE)
    parser.add_argument("--news-detail-limit", type=int, default=4)
    parser.add_argument("--stock-detail-limit", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stocks = load_stocks(Path(args.stocks))
    if args.batch_size < 1 or args.max_batches < 1:
        raise ValueError("batch-size and max-batches must be positive")
    batches = [stocks[i : i + args.batch_size] for i in range(0, len(stocks), args.batch_size)]
    selected = batches[: args.max_batches]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch_no, batch in enumerate(selected, start=1):
        codes = ",".join(row["stock_id"] for row in batch)
        xueqiu = {row["stock_id"]: row["xueqiu_symbol"] for row in batch if row.get("xueqiu_symbol")}
        names = {row["stock_id"]: row["stock_name"] for row in batch if row.get("stock_name")}
        output = output_dir / f"batch_{batch_no:04d}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        command = [
            sys.executable,
            "scripts/collect_browser_visible.py",
            "--output",
            str(output),
            "--guba-codes",
            codes,
            "--max-stocks",
            str(len(batch)),
            "--xueqiu-symbols-json",
            json.dumps(xueqiu, ensure_ascii=False),
            "--stock-names-json",
            json.dumps(names, ensure_ascii=False),
            "--news-detail-limit",
            str(args.news_detail_limit),
            "--stock-detail-limit",
            str(args.stock_detail_limit),
            "--manifest",
            args.manifest,
        ]
        print(json.dumps({"batch": batch_no, "stocks": codes.split(","), "output": str(output), "dry_run": args.dry_run}, ensure_ascii=False))
        if not args.dry_run:
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                diagnostic = {}
                if output.exists():
                    payload = json.loads(output.read_text(encoding="utf-8"))
                    errors = [r for r in payload.get("records", []) if r.get("content_type") == "collection_error"]
                    diagnostic = {"error_count": len(errors), "errors": errors}
                print(json.dumps({"batch": batch_no, "status": "stopped", "reason": "collector exited non-zero; no retry", **diagnostic}, ensure_ascii=False), file=sys.stderr)
                break
            if output.exists():
                payload = json.loads(output.read_text(encoding="utf-8"))
                errors = [r for r in payload.get("records", []) if r.get("content_type") == "collection_error"]
                if errors:
                    print(json.dumps({"batch": batch_no, "status": "completed_with_errors", "error_count": len(errors), "errors": errors}, ensure_ascii=False), file=sys.stderr)
        if batch_no < len(selected) and not args.dry_run:
            time.sleep(args.batch_pause)


if __name__ == "__main__":
    main()
