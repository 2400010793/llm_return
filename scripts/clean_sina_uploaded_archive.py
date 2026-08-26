#!/usr/bin/env python3
"""Strictly clean a newly uploaded Sina archive for return-prediction use."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clean_sina_news import clean_sina_article_text, normalize_sina_timestamp


_STOCK_ID = re.compile(r"\d{6}")
_URL_DATE_PATTERNS = (
    re.compile(r"/(20\d{2})[-/](\d{1,2})[-/](\d{1,2})(?:/|$)"),
    re.compile(r"/(20\d{2})(\d{2})(\d{2})(?:/|$)"),
)
_PAGE_URL_PATTERNS = (
    ("stock_quote_page", re.compile(r"/realstock/company/", re.IGNORECASE)),
    ("fund_quote_page", re.compile(r"/fund/quotes/", re.IGNORECASE)),
    ("topic_page", re.compile(r"/(?:zt_d|focus)(?:/|$)", re.IGNORECASE)),
    ("esg_index_page", re.compile(r"/esg/?$", re.IGNORECASE)),
)
_PAGE_TITLE_PATTERNS = (
    ("stock_quote_page", re.compile(r"股票股价[,，]行情[,，]新闻[,，]财报数据")),
    ("fund_quote_page", re.compile(r"基金行情.*新浪财经")),
    ("opinion_index_page", re.compile(r"^意见领袖_新浪财经")),
    ("site_index_page", re.compile(r"^(?:新浪财经|财经首页)(?:_新浪网)?$")),
)


class ParquetSink:
    """Write dictionaries in bounded batches without retaining the corpus."""

    def __init__(self, path: Path, batch_size: int = 1000) -> None:
        self.path = path
        self.batch_size = batch_size
        self.rows: list[dict[str, Any]] = []
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        if self.writer is None:
            table = pa.Table.from_pylist(self.rows)
            self.schema = table.schema
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        else:
            assert self.schema is not None
            table = pa.Table.from_pylist(self.rows, schema=self.schema)
        self.writer.write_table(table)
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()


def load_stock_universe(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        result = {
            str(row.get("stock_id", "")).zfill(6): str(row.get("stock_name", "")).strip()
            for row in rows
            if row.get("stock_id") and row.get("stock_name")
        }
    if not result:
        raise ValueError(f"stock universe is empty: {path}")
    return result


def extract_url_date(url: str) -> date | None:
    path = urlparse(url).path
    for pattern in _URL_DATE_PATTERNS:
        match = pattern.search(path)
        if match:
            try:
                return date(*(int(part) for part in match.groups()))
            except ValueError:
                return None
    return None


def page_rejection_reason(title: str, url: str) -> str | None:
    path = urlparse(url).path
    for reason, pattern in _PAGE_URL_PATTERNS:
        if pattern.search(path):
            return reason
    for reason, pattern in _PAGE_TITLE_PATTERNS:
        if pattern.search(title.strip()):
            return reason
    return None


def _contains_stock_code(text: str, stock_id: str) -> bool:
    return re.search(rf"(?<!\d){re.escape(stock_id)}(?!\d)", text) is not None


def resolve_stock_evidence(
    record: dict[str, Any],
    *,
    title: str,
    body: str,
    stock_names: dict[str, str],
) -> tuple[dict[str, str] | None, str | None]:
    matches = record.get("stock_matches")
    if not isinstance(matches, list) or len(matches) != 1 or not isinstance(matches[0], dict):
        return None, "not_exactly_one_raw_stock"

    raw_match = matches[0]
    stock_id = str(raw_match.get("stock_id") or "").strip().zfill(6)
    if not _STOCK_ID.fullmatch(stock_id):
        return None, "invalid_stock_id"
    canonical_name = stock_names.get(stock_id)
    if not canonical_name:
        return None, "stock_outside_universe"

    raw_name = str(raw_match.get("stock_name") or "").strip()
    names = list(dict.fromkeys(name for name in (canonical_name, raw_name) if name))
    text = f"{title} {body}"
    code_in_title = _contains_stock_code(title, stock_id)
    code_in_body = _contains_stock_code(body, stock_id)
    long_names = [name for name in names if len(name) >= 4]
    name_in_title = next((name for name in long_names if name in title), None)
    name_in_body = next((name for name in long_names if name in body), None)

    if code_in_title:
        evidence = "code_in_title"
    elif code_in_body:
        evidence = "code_in_body"
    elif name_in_title:
        evidence = "unambiguous_name_in_title"
    elif name_in_body:
        evidence = "unambiguous_name_in_body"
    else:
        return None, "no_reliable_stock_evidence"

    return {
        "stock_id": stock_id,
        "stock_name": canonical_name,
        "stock_evidence": evidence,
        "raw_stock_match_method": str(raw_match.get("stock_match_method") or "unknown"),
    }, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rejection_row(
    record: dict[str, Any],
    *,
    line_number: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "line_number": line_number,
        "article_id": record.get("article_id"),
        "url": record.get("url"),
        "published_at": record.get("published_at"),
        "title": record.get("title"),
        "stock_matches": record.get("stock_matches"),
        "reason": reason,
    }


def merge_with_existing(new_path: Path, existing_path: Path, output_path: Path) -> dict[str, int]:
    existing = pd.read_parquet(existing_path)
    new = pd.read_parquet(new_path)
    existing = existing.copy()
    new = new.copy()
    existing["dataset_origin"] = "existing_clean"
    new["dataset_origin"] = "upload_20260818"

    existing_ids = set(existing["article_id"].astype(str))
    existing_urls = set(existing["url"].fillna("").astype(str))
    existing_keys = set(zip(existing["stock_id"].astype(str), existing["text_hash"].astype(str)))
    new_ids = new["article_id"].astype(str)
    new_urls = new["url"].fillna("").astype(str)
    new_keys = list(zip(new["stock_id"].astype(str), new["text_hash"].astype(str)))
    keep = ~new_ids.isin(existing_ids) & ~new_urls.isin(existing_urls)
    keep &= pd.Series([key not in existing_keys for key in new_keys], index=new.index)
    added = new.loc[keep].copy()
    merged = pd.concat([existing, added], ignore_index=True, sort=False)
    if merged["article_id"].astype(str).duplicated().any():
        raise ValueError("merged output contains duplicate article_id values")
    merged.to_parquet(output_path, index=False, compression="zstd")
    return {
        "existing_rows": int(len(existing)),
        "new_clean_rows": int(len(new)),
        "new_rows_added": int(len(added)),
        "new_rows_suppressed_as_existing": int(len(new) - len(added)),
        "merged_rows": int(len(merged)),
    }


def clean_archive(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "sina_single_stock_clean_new.jsonl"
    parquet_path = args.output_dir / "sina_single_stock_clean_new.parquet"
    rejected_path = args.output_dir / "sina_single_stock_rejected.jsonl"
    summary_path = args.output_dir / "sina_single_stock_clean.summary.json"
    stock_names = load_stock_universe(args.stock_universe)
    cutoff = date.fromisoformat(args.cutoff_date)

    counts = Counter()
    years = Counter()
    stocks = Counter()
    evidence_counts = Counter()
    raw_match_methods = Counter()
    cleaning_actions = Counter()
    seen_article_ids: set[str] = set()
    seen_urls: set[str] = set()
    seen_text_stock: set[tuple[str, str]] = set()
    sink = ParquetSink(parquet_path, args.parquet_batch_size)

    with (
        args.input.open(encoding="utf-8", errors="strict") as source,
        jsonl_path.open("w", encoding="utf-8") as destination,
        rejected_path.open("w", encoding="utf-8") as rejected,
    ):
        for line_number, line in enumerate(source, 1):
            if args.max_rows and line_number > args.max_rows:
                break
            counts["input_rows"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["rejected_invalid_json"] += 1
                rejected.write(json.dumps({"line_number": line_number, "reason": "invalid_json"}) + "\n")
                continue

            reason: str | None = None
            title_raw = str(record.get("title") or "")
            body_raw = str(record.get("body") or "")
            url = str(record.get("url") or "")
            article_id = str(record.get("article_id") or "").strip()
            if not article_id or not url:
                reason = "missing_identity"
            elif "\ufffd" in title_raw or "\ufffd" in body_raw:
                reason = "replacement_character"
            elif bool(record.get("body_truncated")):
                reason = "truncated_body"
            elif (page_reason := page_rejection_reason(title_raw, url)) is not None:
                reason = page_reason

            published_at: str | None = None
            published_date: date | None = None
            if reason is None:
                try:
                    published_at = normalize_sina_timestamp(record.get("published_at"))
                    published_date = datetime.fromisoformat(published_at).date()
                except (TypeError, ValueError):
                    reason = "invalid_timestamp"
            if reason is None and published_date is not None:
                if published_date.year < args.min_year or published_date > cutoff:
                    reason = "date_out_of_range"
                else:
                    url_date = extract_url_date(url)
                    if url_date is not None and abs((url_date - published_date).days) > 1:
                        reason = "url_date_conflict"

            cleaned: dict[str, Any] | None = None
            evidence: dict[str, str] | None = None
            if reason is None:
                cleaned = clean_sina_article_text(title_raw, body_raw)
                if cleaned["body_clean_chars"] < args.min_body_chars:
                    reason = "short_after_cleaning"
            if reason is None and cleaned is not None:
                evidence, reason = resolve_stock_evidence(
                    record,
                    title=cleaned["title_clean"],
                    body=cleaned["body_clean"],
                    stock_names=stock_names,
                )

            if reason is None and cleaned is not None and evidence is not None:
                key = (evidence["stock_id"], cleaned["text_hash"])
                if article_id in seen_article_ids:
                    reason = "duplicate_article_id"
                elif url in seen_urls:
                    reason = "duplicate_url"
                elif key in seen_text_stock:
                    reason = "duplicate_clean_text_stock"

            if reason is not None:
                counts[f"rejected_{reason}"] += 1
                rejected.write(
                    json.dumps(
                        _rejection_row(record, line_number=line_number, reason=reason),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            assert cleaned is not None and evidence is not None and published_at is not None
            assert published_date is not None
            seen_article_ids.add(article_id)
            seen_urls.add(url)
            seen_text_stock.add((evidence["stock_id"], cleaned["text_hash"]))
            row = {
                "article_id": article_id,
                "stock_id": evidence["stock_id"],
                "stock_name": evidence["stock_name"],
                "published_at": published_at,
                "publication_date": published_date.isoformat(),
                "source": record.get("source", "sina_finance"),
                "content_type": record.get("content_type", "financial_news"),
                "url": url,
                "coverage_year": published_date.year,
                "collected_at": record.get("collected_at"),
                "body_truncated": False,
                **cleaned,
                "stock_evidence": evidence["stock_evidence"],
                "raw_stock_match_method": evidence["raw_stock_match_method"],
                "source_depth": record.get("depth"),
            }
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.append(row)
            counts["output_rows"] += 1
            years[str(published_date.year)] += 1
            stocks[evidence["stock_id"]] += 1
            evidence_counts[evidence["stock_evidence"]] += 1
            raw_match_methods[evidence["raw_stock_match_method"]] += 1
            cleaning_actions.update(cleaned["cleaning_actions"])

    sink.close()
    if counts["output_rows"] == 0 and parquet_path.exists():
        parquet_path.unlink()

    summary: dict[str, Any] = {
        "input": str(args.input.resolve()),
        "input_bytes": args.input.stat().st_size,
        "input_sha256": _sha256(args.input),
        "outputs": {
            "jsonl": str(jsonl_path.resolve()),
            "parquet": str(parquet_path.resolve()) if parquet_path.exists() else None,
            "rejected": str(rejected_path.resolve()),
        },
        "policy": {
            "cutoff_date": cutoff.isoformat(),
            "min_year": args.min_year,
            "min_body_chars_after_cleaning": args.min_body_chars,
            "exclude_replacement_character": True,
            "exclude_truncated_body": True,
            "exclude_quote_topic_and_index_pages": True,
            "url_date_tolerance_days": 1,
            "stock_universe": str(args.stock_universe.resolve()),
            "stock_evidence": (
                "explicit six-digit code, or a stock name of at least four characters, "
                "in the cleaned title/body"
            ),
            "deduplicate_on": ["article_id", "url", "stock_id+cleaned_text_hash"],
        },
        "counts": dict(sorted(counts.items())),
        "years": dict(sorted(years.items())),
        "unique_stocks": len(stocks),
        "top_stocks": dict(stocks.most_common(30)),
        "stock_evidence": dict(sorted(evidence_counts.items())),
        "raw_stock_match_methods": dict(sorted(raw_match_methods.items())),
        "cleaning_actions": dict(cleaning_actions.most_common()),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if args.existing_clean is not None and parquet_path.exists():
        merged_path = args.output_dir / "sina_single_stock_clean_merged.parquet"
        summary["merge"] = merge_with_existing(parquet_path, args.existing_clean, merged_path)
        summary["outputs"]["merged_parquet"] = str(merged_path.resolve())
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stock-universe", type=Path, required=True)
    parser.add_argument("--cutoff-date", required=True, help="Latest admissible publication date (YYYY-MM-DD)")
    parser.add_argument("--min-year", type=int, default=2010)
    parser.add_argument("--min-body-chars", type=int, default=120)
    parser.add_argument("--existing-clean", type=Path)
    parser.add_argument("--parquet-batch-size", type=int, default=1000)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional smoke-test limit; 0 reads all rows")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.min_body_chars < 1 or args.parquet_batch_size < 1:
        raise SystemExit("min-body-chars and parquet-batch-size must be positive")
    clean_archive(args)


if __name__ == "__main__":
    main()
