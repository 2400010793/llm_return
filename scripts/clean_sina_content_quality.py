#!/usr/bin/env python3
"""Stream a content-screened Sina archive into a single-stock clean corpus."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clean_sina_news import (
    build_sina_target_text,
    clean_sina_article_text,
    normalize_sina_timestamp,
)


def load_stock_names(path: Path | None) -> tuple[dict[str, str], str | None]:
    if path is None:
        return {}, None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = {str(row.get("stock_id", "")).zfill(6): str(row.get("stock_name", "")) for row in rows}
    dates = sorted({str(row.get("as_of_date", "")) for row in rows if row.get("as_of_date")})
    return names, dates[0] if len(dates) == 1 else None


def exclusion_record(record: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "article_id": record.get("article_id"),
        "url": record.get("url"),
        "published_at": record.get("published_at"),
        "coverage_year": record.get("coverage_year"),
        "screen_stock_ids": record.get("screen_stock_ids"),
        "reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗内容筛选后的新浪财经 JSONL，并生成单股票语料")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stock-universe", type=Path)
    parser.add_argument("--min-body-chars", type=int, default=50)
    parser.add_argument(
        "--single-stock-only",
        action="store_true",
        help="Keep only articles whose screened target list contains exactly one stock",
    )
    args = parser.parse_args()
    if args.min_body_chars < 1:
        parser.error("--min-body-chars must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "sina_single_stock" if args.single_stock_only else "sina_stock_target"
    output = args.output_dir / f"{prefix}_clean.jsonl"
    excluded = args.output_dir / f"{prefix}_excluded.jsonl"
    summary_path = args.output_dir / f"{prefix}_clean.summary.json"
    stock_names, universe_as_of = load_stock_names(args.stock_universe)

    counters: dict[str, int] = {
        "input_rows": 0,
        "excluded_no_target_stock": 0,
        "excluded_multiple_target_stocks": 0,
        "excluded_invalid_timestamp": 0,
        "excluded_short_after_cleaning": 0,
        "excluded_duplicate_text_stock": 0,
        "input_target_articles": 0,
        "input_single_stock_articles": 0,
        "input_multi_stock_articles": 0,
        "expanded_target_rows": 0,
        "output_rows": 0,
        "output_single_stock_rows": 0,
        "output_multi_stock_rows": 0,
    }
    years: dict[str, int] = {}
    stocks: dict[str, int] = {}
    actions: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()

    with (
        args.input.open(encoding="utf-8", errors="strict") as source,
        output.open("w", encoding="utf-8") as destination,
        excluded.open("w", encoding="utf-8") as rejected,
    ):
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            counters["input_rows"] += 1
            record = json.loads(line)
            targets = record.get("screen_stock_ids")
            if not isinstance(targets, list) or not targets:
                counters["excluded_no_target_stock"] += 1
                rejected.write(json.dumps(exclusion_record(record, "no_target_stock"), ensure_ascii=False) + "\n")
                continue
            targets = list(dict.fromkeys(str(target).zfill(6) for target in targets))
            if args.single_stock_only and len(targets) != 1:
                counters["excluded_multiple_target_stocks"] += 1
                rejected.write(json.dumps(exclusion_record(record, "multiple_target_stocks"), ensure_ascii=False) + "\n")
                continue
            counters["input_target_articles"] += 1
            if len(targets) == 1:
                counters["input_single_stock_articles"] += 1
            else:
                counters["input_multi_stock_articles"] += 1
            counters["expanded_target_rows"] += len(targets)

            try:
                published_at = normalize_sina_timestamp(record.get("published_at"))
            except (TypeError, ValueError):
                counters["excluded_invalid_timestamp"] += 1
                rejected.write(json.dumps(exclusion_record(record, "invalid_timestamp"), ensure_ascii=False) + "\n")
                continue

            cleaned = clean_sina_article_text(record.get("title"), record.get("body"))
            if cleaned["body_clean_chars"] < args.min_body_chars:
                counters["excluded_short_after_cleaning"] += 1
                rejected.write(json.dumps(exclusion_record(record, "short_after_cleaning"), ensure_ascii=False) + "\n")
                continue

            is_multi_stock = len(targets) > 1
            for stock_id in targets:
                key = (stock_id, cleaned["text_hash"])
                if key in seen:
                    counters["excluded_duplicate_text_stock"] += 1
                    duplicate = exclusion_record(record, "duplicate_text_stock")
                    duplicate["stock_id"] = stock_id
                    rejected.write(json.dumps(duplicate, ensure_ascii=False) + "\n")
                    continue
                seen.add(key)

                stock_name = stock_names.get(stock_id, "")
                article_id = str(record.get("article_id") or "")
                row = {
                    "article_id": article_id,
                    "stock_id": stock_id,
                    "stock_name": stock_name,
                    "published_at": published_at,
                    "publication_date": published_at[:10],
                    "source": record.get("source", "sina_finance"),
                    "content_type": record.get("content_type", "financial_news"),
                    "url": record.get("url"),
                    "coverage_year": record.get("coverage_year"),
                    "collected_at": record.get("collected_at"),
                    "body_truncated": bool(record.get("body_truncated")),
                    **cleaned,
                }
                if not args.single_stock_only:
                    target_text = build_sina_target_text(
                        stock_id=stock_id,
                        stock_name=stock_name,
                        title=cleaned["title_clean"],
                        body=cleaned["body_clean"],
                    )
                    row.update(
                        {
                            "document_id": f"{article_id}:{stock_id}",
                            "target_stock_ids": targets,
                            "target_stock_count": len(targets),
                            "is_multi_stock_article": is_multi_stock,
                            **target_text,
                        }
                    )
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                counters["output_rows"] += 1
                counters["output_multi_stock_rows" if is_multi_stock else "output_single_stock_rows"] += 1
                year = str(record.get("coverage_year"))
                years[year] = years.get(year, 0) + 1
                stocks[stock_id] = stocks.get(stock_id, 0) + 1
                for action in cleaned["cleaning_actions"]:
                    actions[action] = actions.get(action, 0) + 1

    summary = {
        "input": str(args.input.resolve()),
        "output": str(output.resolve()),
        "excluded": str(excluded.resolve()),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "policy": {
            "target_field": "screen_stock_ids",
            "require_exactly_one_target_stock": args.single_stock_only,
            "expand_multi_stock_articles": not args.single_stock_only,
            "model_text_fields": ["text"] if args.single_stock_only else ["target_prompt", "text_model"],
            "min_body_chars": args.min_body_chars,
            "deduplicate_on": ["stock_id", "text_hash"],
            "stock_universe": str(args.stock_universe.resolve()) if args.stock_universe else None,
            "stock_universe_as_of": universe_as_of,
            "historical_constituent_warning": bool(universe_as_of),
        },
        "counts": counters,
        "years": dict(sorted(years.items())),
        "unique_stocks": len(stocks),
        "top_stocks": dict(sorted(stocks.items(), key=lambda item: (-item[1], item[0]))[:30]),
        "cleaning_actions": dict(sorted(actions.items(), key=lambda item: (-item[1], item[0]))),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
