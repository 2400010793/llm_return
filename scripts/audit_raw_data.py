"""Audit a raw news or price table.

Example:
    python scripts/audit_raw_data.py data/raw/news.parquet --kind news
"""

from __future__ import annotations

import argparse
import json

from src.data.audit import audit_news, audit_prices
from src.data.ingest import read_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--kind", choices=("news", "prices"), required=True)
    args = parser.parse_args()
    table = read_table(args.path)
    report = audit_news(table) if args.kind == "news" else audit_prices(table)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
