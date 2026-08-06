"""Run low-rate CNINFO collection for the paper stock universe.

Each stock is isolated in its own subprocess so one unavailable company page
does not discard the other stocks.  Organization IDs use the observed CNINFO
code convention, with explicit overrides for stocks whose IDs differ.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


OVERRIDES = {
    "000001": "gssz0000001",
    "600519": "gssh0600519",
    "300750": "GD165627",
    "601229": "9900010207",
}


def load_stocks(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("active", "1") == "1"]
    if len({row.get("stock_id", "") for row in rows}) != len(rows):
        raise ValueError("stock universe contains duplicate stock_id values")
    return rows


def org_id(row: dict[str, str]) -> str:
    code = row["stock_id"].strip()
    if code in OVERRIDES:
        return OVERRIDES[code]
    # CNINFO's observed IDs include the exchange marker plus a seven-digit
    # numeric suffix: e.g. gssh0600000 and gssz0000002.
    prefix = "gssh" if row.get("exchange") == "SH" else "gssz"
    return prefix + code.zfill(7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", default="data/stock_universe_paper_100.csv")
    parser.add_argument("--expected-stocks", type=int, default=100)
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2026-08-01")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--start-index", type=int, default=1, help="1-based stock index to start from")
    parser.add_argument("--workers", type=int, default=30, help="number of concurrent stock collectors")
    parser.add_argument("--focus-only", action="store_true", help="only download research-priority announcement titles")
    parser.add_argument("--resolve-stock", action="store_true", help="resolve each stock through CNINFO's visible company field")
    parser.add_argument("--max-pages", type=int, default=0, help="visible list pages per stock; 0 means all")
    parser.add_argument("--page-pause-seconds", type=float, default=3.0)
    parser.add_argument("--pause-seconds", type=float, default=6.0)
    parser.add_argument("--stock-pause-seconds", type=float, default=30.0)
    parser.add_argument("--output-dir", default="data/interim/cninfo_paper_100")
    parser.add_argument("--manifest", default="data/interim/cninfo_paper_100_manifest.json")
    parser.add_argument("--raw-dir", default="data/raw/cninfo_paper_100")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 2000 or args.pause_seconds < 0.1 or args.page_pause_seconds < 0.1 or args.max_pages < 0 or args.stock_pause_seconds < 0 or args.start_index < 1 or args.workers < 1:
        raise ValueError("limit must be 1..2000; pauses must be at least 0.1; start-index and workers must be positive")
    rows = load_stocks(Path(args.stocks))
    if len(rows) != args.expected_stocks:
        raise ValueError(f"expected {args.expected_stocks} active stocks, got {len(rows)}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.raw_dir).mkdir(parents=True, exist_ok=True)
    jobs = []
    for index, row in enumerate(rows, 1):
        if index < args.start_index:
            continue
        code = row["stock_id"]
        output = output_dir / f"{code}_2018_2026.json"
        if output.exists():
            try:
                payload = json.loads(output.read_text(encoding="utf-8"))
                existing_records = payload.get("records", []) if isinstance(payload, dict) else []
                if isinstance(existing_records, list) and existing_records:
                    print(json.dumps({"index": index, "stock": code, "status": "skipped_complete", "records": len(existing_records)}, ensure_ascii=False), flush=True)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        spec = f"{code}:{org_id(row)}:{row['stock_name']}"
        command = [
            sys.executable, "scripts/collect_cninfo_announcements.py",
            "--stock", spec, "--start-date", args.start_date, "--end-date", args.end_date,
            "--limit", str(args.limit), "--pause-seconds", str(args.pause_seconds),
            "--max-pages", str(args.max_pages), "--page-pause-seconds", str(args.page_pause_seconds),
            "--focus-only" if args.focus_only else "",
            "--resolve-stock" if args.resolve_stock else "",
            "--stock-pause-seconds", "0", "--output", str(output), "--manifest",
            str(Path(args.manifest).with_name(f"{code}_manifest.json") if args.workers > 1 else Path(args.manifest)),
            "--raw-dir", args.raw_dir,
        ]
        command = [part for part in command if part]
        print(json.dumps({"index": index, "stock": spec, "output": str(output), "dry_run": args.dry_run}, ensure_ascii=False), flush=True)
        jobs.append((index, code, command))
    if args.dry_run:
        return
    for offset in range(0, len(jobs), args.workers):
        batch = jobs[offset:offset + args.workers]
        processes = [(index, code, subprocess.Popen(command)) for index, code, command in batch]
        for index, code, process in processes:
            returncode = process.wait()
            if returncode != 0:
                print(json.dumps({"index": index, "stock_id": code, "status": "failed_no_retry", "returncode": returncode}, ensure_ascii=False), file=sys.stderr, flush=True)
        if offset + args.workers < len(jobs):
            time.sleep(args.stock_pause_seconds)


if __name__ == "__main__":
    main()