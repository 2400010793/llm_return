"""Run a conservative, resumable first phase of paper-scale news collection.

The current permitted public-source implementation collects visible stock-page
news/notices. It does not claim to provide a historical 3--5 year archive;
that requires a source with stable historical search and publication timestamps.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


FIELDS = ("stock_id", "stock_name", "exchange", "xueqiu_symbol", "industry", "active")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return [row for row in reader if row.get("active", "1") == "1"]


def build_universe(paths: list[Path], size: int) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        for row in load_rows(path):
            code = row["stock_id"].strip()
            if code in seen:
                continue
            seen.add(code)
            result.append({field: row.get(field, "") for field in FIELDS})
            if len(result) == size:
                return result
    raise ValueError(f"the supplied lists contain only {len(result)} unique active stocks; need {size}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", action="append", default=["data/stock_universe_90_template.csv"])
    parser.add_argument("--expected-stocks", type=int, default=100)
    parser.add_argument("--universe-output", default="data/stock_universe_paper_100.csv")
    parser.add_argument("--output-dir", default="data/interim/paper_scale_media")
    parser.add_argument("--manifest", default="data/interim/collector_manifest_paper_scale_media.json")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--batch-pause", type=float, default=60.0)
    parser.add_argument("--page-pause", type=float, default=6.0)
    parser.add_argument("--stock-detail-limit", type=int, default=3)
    parser.add_argument("--forum-detail-limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.expected_stocks < 1 or args.batch_size < 1 or args.page_pause < 3:
        raise ValueError("expected-stocks and batch-size must be positive; page-pause must be at least 3")

    paths = [Path(value) for value in args.universe]
    paths += [Path("data/stock_universe_100_template.csv"), Path("data/stock_universe_missing_3.csv"), Path("data/stock_universe_additional_2.csv")]
    rows = build_universe(paths, args.expected_stocks)
    universe_path = Path(args.universe_output)
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    with universe_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"universe": str(universe_path), "stocks": len(rows), "source_scope": "visible Eastmoney stock pages; not a 3-5 year historical archive"}, ensure_ascii=False), flush=True)
    command = [
        sys.executable, "scripts/run_100_stock_collection.py",
        "--stocks", str(universe_path), "--expected-stocks", str(args.expected_stocks),
        "--output-dir", args.output_dir, "--manifest", args.manifest,
        "--batch-size", str(args.batch_size), "--max-batches", str((len(rows) + args.batch_size - 1) // args.batch_size),
        "--batch-pause", str(args.batch_pause), "--page-pause", str(args.page_pause),
        "--stock-detail-limit", str(args.stock_detail_limit), "--forum-detail-limit", str(args.forum_detail_limit),
    ]
    if args.dry_run:
        command.append("--dry-run")
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()