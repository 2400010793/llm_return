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
import re
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


def stock_ids_from_embedding_input(path: Path) -> set[str]:
    files = sorted(path.rglob("part-*.jsonl")) if path.is_dir() else [path]
    stock_ids: set[str] = set()
    for file_path in files:
        with file_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                stock_id = str(record.get("stock_id_alignment", "")).strip()
                if re.fullmatch(r"\d{6}", stock_id):
                    stock_ids.add(stock_id)
    return stock_ids


def org_id(row: dict[str, str]) -> str:
    code = row["stock_id"].strip()
    if code in OVERRIDES:
        return OVERRIDES[code]
    # CNINFO's observed IDs include the exchange marker plus a seven-digit
    # numeric suffix: e.g. gssh0600000 and gssz0000002.
    prefix = "gssh" if row.get("exchange") == "SH" else "gssz"
    return prefix + code.zfill(7)


def reusable_complete_output(
    path: Path, *, start_date: str, end_date: str,
    index_only: bool, focus_only: bool,
) -> tuple[bool, int]:
    """Return whether a prior output exactly satisfies the current query."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False, 0
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        return False, 0
    query = payload.get("query", {})
    records = payload.get("records", [])
    if not isinstance(query, dict) or not isinstance(records, list):
        return False, 0
    query_matches = (
        query.get("start_date") == start_date
        and query.get("end_date") == end_date
        and bool(query.get("index_only", False)) == index_only
        and bool(query.get("focus_only", False)) == focus_only
    )
    has_collection_error = any(
        isinstance(record, dict)
        and record.get("content_type") == "collection_error"
        for record in records
    )
    return query_matches and not has_collection_error, len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", default="data/stock_universe_paper_100.csv")
    parser.add_argument("--stock-ids-from", default="", help="derive the active stock subset from cleaned embedding JSONL input")
    parser.add_argument("--expected-stocks", type=int, default=100)
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2026-08-01")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--start-index", type=int, default=1, help="1-based stock index to start from")
    parser.add_argument(
        "--end-index", type=int, default=0,
        help="1-based inclusive stock index to stop at; 0 means the final stock",
    )
    parser.add_argument("--retry-failed-log", default="", help="retry only stock IDs marked failed_no_retry in a prior runner log")
    parser.add_argument("--workers", type=int, default=30, help="number of concurrent stock collectors")
    parser.add_argument("--stock-timeout-seconds", type=float, default=1800.0, help="maximum runtime for one stock subprocess")
    parser.add_argument("--focus-only", action="store_true", help="only download research-priority announcement titles")
    parser.add_argument("--index-only", action="store_true", help="collect visible announcement indexes without detail/PDF requests")
    parser.add_argument("--resolve-stock", action="store_true", help="resolve each stock through CNINFO's visible company field")
    parser.add_argument("--max-pages", type=int, default=0, help="visible list pages per stock; 0 means all")
    parser.add_argument("--page-pause-seconds", type=float, default=3.0)
    parser.add_argument("--pause-seconds", type=float, default=6.0)
    parser.add_argument("--stock-pause-seconds", type=float, default=30.0)
    parser.add_argument("--output-dir", default="data/interim/cninfo_paper_100")
    parser.add_argument("--output-suffix", default="2018_2026", help="suffix used in per-stock output filenames")
    parser.add_argument("--manifest", default="data/interim/cninfo_paper_100_manifest.json")
    parser.add_argument("--raw-dir", default="data/raw/cninfo_paper_100")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or args.pause_seconds < 0.1 or args.page_pause_seconds < 0.1 or args.max_pages < 0 or args.stock_pause_seconds < 0 or args.start_index < 1 or args.end_index < 0 or args.workers < 1 or args.stock_timeout_seconds <= 0:
        raise ValueError("limit and end-index must be non-negative; pauses must be at least 0.1; start-index and workers must be positive; stock timeout must be positive")
    if args.end_index and args.end_index < args.start_index:
        raise ValueError("end-index must be zero or greater than or equal to start-index")
    rows = load_stocks(Path(args.stocks))
    if args.stock_ids_from:
        selected_ids = stock_ids_from_embedding_input(Path(args.stock_ids_from))
        rows = [row for row in rows if row.get("stock_id") in selected_ids]
        if len(rows) != args.expected_stocks:
            raise ValueError(f"expected {args.expected_stocks} stocks from embedding input, got {len(rows)}")
    if args.retry_failed_log:
        log_text = Path(args.retry_failed_log).read_text(encoding="utf-8", errors="replace")
        failed = set(re.findall(r'"status":\s*"failed_no_retry"[^\n]*?"stock_id":\s*"(\d{6})"', log_text))
        # The runner prints stock_id before status in some versions.
        failed.update(re.findall(r'"stock_id":\s*"(\d{6})"[^\n]*?"status":\s*"failed_no_retry"', log_text))
        rows = [row for row in rows if row.get("stock_id") in failed]
        if not rows:
            print(json.dumps({"status": "no_failed_stocks_found", "log": args.retry_failed_log}), flush=True)
            return
    if not args.retry_failed_log and len(rows) != args.expected_stocks:
        raise ValueError(f"expected {args.expected_stocks} active stocks, got {len(rows)}")
    if args.start_index > len(rows) or args.end_index > len(rows):
        raise ValueError(
            f"stock index range {args.start_index}..{args.end_index or len(rows)} "
            f"exceeds the {len(rows)} available stocks"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.raw_dir).mkdir(parents=True, exist_ok=True)
    jobs = []
    for index, row in enumerate(rows, 1):
        if index < args.start_index:
            continue
        if args.end_index and index > args.end_index:
            break
        code = row["stock_id"]
        output = output_dir / f"{code}_{args.output_suffix}.json"
        if output.exists():
            reusable, record_count = reusable_complete_output(
                output, start_date=args.start_date, end_date=args.end_date,
                index_only=args.index_only, focus_only=args.focus_only,
            )
            if reusable:
                print(json.dumps({
                    "index": index, "stock": code, "status": "skipped_complete",
                    "records": record_count,
                }, ensure_ascii=False), flush=True)
                continue
        spec = f"{code}:{org_id(row)}:{row['stock_name']}"
        command = [
            sys.executable, "scripts/collect_cninfo_announcements.py",
            "--stock", spec, "--start-date", args.start_date, "--end-date", args.end_date,
            "--limit", str(args.limit), "--pause-seconds", str(args.pause_seconds),
            "--max-pages", str(args.max_pages), "--page-pause-seconds", str(args.page_pause_seconds),
            "--focus-only" if args.focus_only else "",
            "--index-only" if args.index_only else "",
            "--resolve-stock" if args.resolve_stock else "",
            "--stock-pause-seconds", "0", "--output", str(output), "--manifest",
            str(output_dir / "manifests" / f"{code}.json"),
            "--raw-dir", args.raw_dir,
        ]
        command = [part for part in command if part]
        print(json.dumps({"index": index, "stock": spec, "output": str(output), "dry_run": args.dry_run}, ensure_ascii=False), flush=True)
        jobs.append((index, code, command))
    if args.dry_run:
        return
    active: list[dict] = []
    next_job = 0
    while next_job < len(jobs) or active:
        while next_job < len(jobs) and len(active) < args.workers:
            index, code, command = jobs[next_job]
            active.append({"index": index, "code": code, "started": time.monotonic(), "process": subprocess.Popen(command)})
            next_job += 1
        if not active:
            break
        time.sleep(0.5)
        still_active: list[dict] = []
        for item in active:
            process = item["process"]
            returncode = process.poll()
            if returncode is None and time.monotonic() - item["started"] > args.stock_timeout_seconds:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                print(json.dumps({"index": item["index"], "stock_id": item["code"], "status": "failed_timeout", "timeout_seconds": args.stock_timeout_seconds}, ensure_ascii=False), file=sys.stderr, flush=True)
            elif returncode is None:
                still_active.append(item)
            elif returncode != 0:
                print(json.dumps({"index": item["index"], "stock_id": item["code"], "status": "failed_no_retry", "returncode": returncode}, ensure_ascii=False), file=sys.stderr, flush=True)
        active = still_active
        if active and next_job < len(jobs):
            time.sleep(args.stock_pause_seconds)


if __name__ == "__main__":
    main()
