"""Bounded batch runner for stock Guba and Xueqiu pages."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", default="data/stocks_sample.csv")
    parser.add_argument("--output-dir", default="data/interim/social_batches")
    parser.add_argument("--manifest", default="data/interim/collector_manifest_social.json")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--start-batch", type=int, default=1, help="从第几个批次开始，便于失败后续跑")
    parser.add_argument("--batch-pause", type=float, default=60.0)
    parser.add_argument("--forum-detail-limit", type=int, default=10)
    parser.add_argument("--pause-seconds", type=float, default=3.0)
    parser.add_argument("--xueqiu-sort-types", nargs="+", choices=("new", "hot"), default=["hot"])
    parser.add_argument("--xueqiu-max-pages", type=int, default=20)
    parser.add_argument("--xueqiu-max-records", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.start_batch < 1:
        raise ValueError("start-batch must be positive")
    with Path(args.stocks).open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("active", "1") == "1"]
    all_batches = [rows[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]
    batches = all_batches[args.start_batch - 1 : args.start_batch - 1 + args.max_batches]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for offset, batch in enumerate(batches):
        no = args.start_batch + offset
        codes = ",".join(r["stock_id"] for r in batch)
        symbols = {r["stock_id"]: r["xueqiu_symbol"] for r in batch}
        names = {r["stock_id"]: r["stock_name"] for r in batch}
        out = Path(args.output_dir) / f"batch_{no:04d}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        cmd = [sys.executable, "scripts/collect_browser_visible.py", "--output", str(out), "--guba-codes", codes, "--max-stocks", str(len(batch)), "--xueqiu-symbols-json", json.dumps(symbols, ensure_ascii=False), "--stock-names-json", json.dumps(names, ensure_ascii=False), "--xueqiu-sort-types", *args.xueqiu_sort_types, "--xueqiu-max-pages", str(args.xueqiu_max_pages), "--xueqiu-max-records", str(args.xueqiu_max_records), "--stock-link-limit", "0", "--stock-detail-limit", "0", "--forum-detail-limit", str(args.forum_detail_limit), "--pause-seconds", str(args.pause_seconds), "--manifest", args.manifest]
        print(json.dumps({"batch": no, "stocks": codes.split(","), "output": str(out), "dry_run": args.dry_run}, ensure_ascii=False))
        if not args.dry_run:
            completed = subprocess.run(cmd, check=False)
            if completed.returncode != 0:
                diagnostic = {}
                if out.exists():
                    payload = json.loads(out.read_text(encoding="utf-8"))
                    errors = [r for r in payload.get("records", []) if r.get("content_type") == "collection_error"]
                    diagnostic = {"error_count": len(errors), "errors": errors}
                print(json.dumps({"batch": no, "status": "stopped", "reason": "collector exited non-zero; no retry", **diagnostic}, ensure_ascii=False), file=sys.stderr)
                break
            if out.exists():
                payload = json.loads(out.read_text(encoding="utf-8"))
                errors = [r for r in payload.get("records", []) if r.get("content_type") == "collection_error"]
                if errors:
                    print(json.dumps({"batch": no, "status": "completed_with_errors", "error_count": len(errors), "errors": errors}, ensure_ascii=False), file=sys.stderr)
            if no < len(batches):
                time.sleep(args.batch_pause)


if __name__ == "__main__":
    main()
