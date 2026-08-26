"""Select deduplicated, auditable historical Sina article seeds from crawler output."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

HISTORICAL_URL_RE = re.compile(r"^https?://finance\.sina\.com\.cn/(?:t|s|e|y)/\d+\.html$", re.I)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                yield value


def choose(rows: Iterable[Dict[str, Any]], min_chars: int) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        if row.get("body_quality") != "valid":
            continue
        if row.get("published_year") not in (2000, 2001):
            continue
        if not HISTORICAL_URL_RE.fullmatch(str(row.get("url", ""))):
            continue
        if int(row.get("body_chars", 0)) < min_chars:
            continue
        groups[row.get("body_sha256") or row.get("article_id")].append(row)

    selected = []
    for group in groups.values():
        selected.append(sorted(group, key=lambda row: (-int(row.get("body_chars", 0)), row.get("url", "")))[0])
    return sorted(selected, key=lambda row: (int(row["published_year"]), row.get("published_date") or "", row.get("url", "")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Select historical Sina article seeds")
    parser.add_argument("--articles", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-body-chars", type=int, default=250)
    parser.add_argument("--target-per-year", type=int, default=20)
    args = parser.parse_args()
    rows = choose(read_jsonl(Path(args.articles)), args.min_body_chars)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            seed = {"seed_id": "historical_%04d" % index, "seed_year": row["published_year"], "seed_path": urlparse(row["url"]).path.split("/", 2)[1], "url": row["url"], "title": row.get("title"), "published_date": row.get("published_date"), "body_chars": row.get("body_chars"), "body_sha256": row.get("body_sha256"), "stock_match_count": row.get("stock_match_count", 0)}
            handle.write(json.dumps(seed, ensure_ascii=False, separators=(",", ":")) + "\n")
    counts = defaultdict(int)
    for row in rows:
        counts[str(row["published_year"])] += 1
    year_counts = dict(sorted(counts.items()))
    shortages = {
        year: max(args.target_per_year - count, 0)
        for year, count in year_counts.items()
        if count < args.target_per_year
    }
    print(json.dumps({"selected": len(rows), "year_counts": year_counts, "shortages": shortages, "target_per_year": args.target_per_year, "min_body_chars": args.min_body_chars, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
