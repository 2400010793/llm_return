"""Bounded Eastmoney-only collection runner for a stock universe.

The input CSV must contain the requested number of active rows with stock_id, stock_name, exchange,
industry, and optionally xueqiu_symbol. This script does not collect the global
focus channel and does not bypass access controls.
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


REQUIRED_COLUMNS = {"stock_id", "stock_name", "exchange", "industry", "active"}


def load_universe(path: Path, expected_stocks: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        rows = [row for row in reader if row.get("active", "1") == "1"]
    if len(rows) != expected_stocks:
        raise ValueError(f"expected exactly {expected_stocks} active stocks, got {len(rows)}")
    ids = [row["stock_id"].strip() for row in rows]
    if any(len(code) != 6 or not code.isdigit() for code in ids):
        raise ValueError("every stock_id must be a six-digit code")
    if len(set(ids)) != len(ids):
        raise ValueError("stock_id values must be unique")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", required=True, help="CSV containing the requested number of active stocks")
    parser.add_argument("--expected-stocks", type=int, default=90)
    parser.add_argument("--output-dir", default="data/interim/stock100_eastmoney")
    parser.add_argument("--manifest", default="data/interim/collector_manifest_stock100_eastmoney.json")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--start-batch", type=int, default=1, help="从第几个批次开始，批次编号从 1 开始")
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--batch-pause", type=float, default=60.0)
    parser.add_argument("--page-pause", type=float, default=3.0)
    parser.add_argument("--forum-detail-limit", type=int, default=10)
    parser.add_argument("--stock-detail-limit", type=int, default=6)
    parser.add_argument("--stock-link-limit", type=int, default=20)
    parser.add_argument("--research-detail-limit", type=int, default=0)
    parser.add_argument("--stock-links-only", action="store_true", help="只采集股票页可见的公告、新闻和研报")
    parser.add_argument("--xueqiu-feeds", nargs="*", choices=("news", "announcement"), default=[], help="同时采集雪球可见资讯/公告")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.page_pause < 3.0:
        raise ValueError("page-pause must be at least 3 seconds")
    if args.batch_size < 1 or args.start_batch < 1 or args.max_batches < 1:
        raise ValueError("batch-size, start-batch, and max-batches must be positive")

    rows = load_universe(Path(args.stocks), args.expected_stocks)
    batches = [rows[i : i + args.batch_size] for i in range(0, len(rows), args.batch_size)]
    if args.start_batch > len(batches):
        raise ValueError(f"start-batch {args.start_batch} exceeds {len(batches)} batches")
    selected = batches[args.start_batch - 1 : args.start_batch - 1 + args.max_batches]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for offset, batch in enumerate(selected):
        batch_no = args.start_batch + offset
        codes = ",".join(row["stock_id"] for row in batch)
        symbols = {row["stock_id"]: row.get("xueqiu_symbol", "") for row in batch if row.get("xueqiu_symbol")}
        names = {row["stock_id"]: row["stock_name"] for row in batch}
        output = output_dir / f"batch_{batch_no:04d}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        command = [
            sys.executable,
            "scripts/collect_browser_visible.py",
            "--output", str(output),
            "--guba-codes", codes,
            "--max-stocks", str(len(batch)),
            "--xueqiu-symbols-json", json.dumps(symbols, ensure_ascii=False),
            "--stock-names-json", json.dumps(names, ensure_ascii=False),
            "--stock-detail-limit", str(args.stock_detail_limit),
            "--stock-link-limit", str(args.stock_link_limit),
            "--research-detail-limit", str(args.research_detail_limit),
            "--forum-detail-limit", str(args.forum_detail_limit),
            "--pause-seconds", str(args.page_pause),
            "--manifest", args.manifest,
        ]
        if args.stock_links_only:
            command.append("--stock-links-only")
        if args.xueqiu_feeds:
            command.extend(["--xueqiu-feeds", *args.xueqiu_feeds])
        print(json.dumps({"batch": batch_no, "stocks": [r["stock_id"] for r in batch], "output": str(output), "dry_run": args.dry_run}, ensure_ascii=False), flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            print(json.dumps({"status": "failed", "batch": batch_no, "reason": "collector failed or access control detected; continue to next batch"}, ensure_ascii=False), file=sys.stderr, flush=True)
        if offset < len(selected) - 1:
            time.sleep(args.batch_pause)


if __name__ == "__main__":
    main()
