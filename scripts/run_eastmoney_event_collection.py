"""Collect event-oriented Eastmoney stock news and notices for the 90 additions.

This is a bounded, sequential wrapper. It uses only visible stock quote pages,
does not visit the global focus channel, and stops without retrying on access
control responses.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", default="data/stock_universe_90_template.csv")
    parser.add_argument("--expected-stocks", type=int, default=90)
    parser.add_argument("--output-dir", default="data/interim/stock90_events_eastmoney")
    parser.add_argument("--manifest", default="data/interim/collector_manifest_stock90_events.json")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--batch-pause", type=float, default=60.0)
    parser.add_argument("--page-pause", type=float, default=3.0)
    parser.add_argument("--stock-detail-limit", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.page_pause < 3 or args.batch_size < 1:
        raise ValueError("page-pause must be at least 3 seconds and batch-size must be positive")

    with Path(args.stocks).open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("active", "1") == "1"]
    if len(rows) != args.expected_stocks:
        raise ValueError(f"expected exactly {args.expected_stocks} active stocks, got {len(rows)}")

    # Reuse the bounded runner so the event collection has the same stop/no-retry policy.
    command = [
        sys.executable, "scripts/run_100_stock_collection.py",
        "--stocks", args.stocks, "--expected-stocks", str(args.expected_stocks),
        "--output-dir", args.output_dir, "--manifest", args.manifest,
        "--batch-size", str(args.batch_size), "--max-batches", "{batches}",
        "--batch-pause", str(args.batch_pause), "--page-pause", str(args.page_pause),
        "--stock-detail-limit", str(args.stock_detail_limit),
    ]
    batches = (len(rows) + args.batch_size - 1) // args.batch_size
    command[command.index("{batches}")] = str(batches)
    if args.dry_run:
        command.append("--dry-run")
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()