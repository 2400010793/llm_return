#!/usr/bin/env python3
"""Re-attribute and clean the complete Sina archive without trusting page chrome."""

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
from typing import Any, Iterable
import unicodedata

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.clean_sina_uploaded_archive import extract_url_date, page_rejection_reason
from src.data.clean_sina_news import (
    build_sina_target_text,
    clean_sina_article_text,
    normalize_sina_timestamp,
)


_STOCK_ID = re.compile(r"\d{6}")
_EXPLICIT_CODE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


class ParquetSink:
    """Write dictionaries in bounded batches."""

    def __init__(self, path: Path, batch_size: int) -> None:
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


def normalize_entity(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value or "")).strip()


def load_stock_names(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        names = {
            str(row.get("stock_id", "")).zfill(6): normalize_entity(str(row.get("stock_name", "")))
            for row in rows
            if row.get("stock_id") and row.get("stock_name")
        }
    if not names:
        raise ValueError(f"stock universe is empty: {path}")
    return names


def load_return_codes(path: Path | None, stock_names: dict[str, str]) -> set[str]:
    if path is None:
        return set(stock_names)
    names = pq.ParquetFile(path).schema.names
    codes = {match.group(1) for name in names if (match := re.fullmatch(r"(\d{6})\.[A-Z]+", name))}
    if not codes:
        raise ValueError(f"no six-digit return tickers found in: {path}")
    return codes


def stock_name_aliases(canonical_name: str, raw_names: Iterable[str]) -> list[str]:
    names = [normalize_entity(canonical_name)]
    names.extend(normalize_entity(name) for name in raw_names)
    result: list[str] = []
    for name in names:
        if not name:
            continue
        result.append(name)
        without_st = re.sub(r"^(?:S\*ST|SST|\*ST|ST)", "", name, flags=re.IGNORECASE)
        if len(without_st) >= 4:
            result.append(without_st)
    return list(dict.fromkeys(result))


def _contains_code(text: str, stock_id: str) -> bool:
    return re.search(rf"(?<!\d){re.escape(stock_id)}(?!\d)", text) is not None


def resolve_stocks_from_clean_text(
    record: dict[str, Any],
    *,
    title: str,
    body: str,
    stock_names: dict[str, str],
    return_codes: set[str],
) -> tuple[list[dict[str, str]], Counter[str]]:
    """Verify raw candidates against cleaned text and discover explicit codes."""

    raw_by_code: dict[str, list[dict[str, Any]]] = {}
    matches = record.get("stock_matches")
    if isinstance(matches, list):
        for item in matches:
            if not isinstance(item, dict):
                continue
            stock_id = str(item.get("stock_id") or "").strip().zfill(6)
            if _STOCK_ID.fullmatch(stock_id):
                raw_by_code.setdefault(stock_id, []).append(item)

    normalized_title = normalize_entity(title)
    normalized_body = normalize_entity(body)
    explicit_codes = set(_EXPLICIT_CODE.findall(f"{normalized_title} {normalized_body}"))
    candidate_codes = set(raw_by_code) | explicit_codes
    resolved: list[dict[str, str]] = []
    candidate_counts: Counter[str] = Counter()

    for stock_id in sorted(candidate_codes):
        candidate_counts["candidate_stock_rows"] += 1
        if stock_id not in return_codes:
            candidate_counts["candidate_outside_returns"] += 1
            continue
        raw_items = raw_by_code.get(stock_id, [])
        raw_names = [str(item.get("stock_name") or "") for item in raw_items]
        canonical_name = stock_names.get(stock_id, "") or next(
            (normalize_entity(name) for name in raw_names if normalize_entity(name)),
            "",
        )

        code_in_title = _contains_code(normalized_title, stock_id)
        code_in_body = _contains_code(normalized_body, stock_id)
        aliases = [alias for alias in stock_name_aliases(canonical_name, raw_names) if len(alias) >= 4]
        name_in_title = next((alias for alias in aliases if alias in normalized_title), None)
        name_in_body = next((alias for alias in aliases if alias in normalized_body), None)
        if code_in_title:
            evidence = "code_in_title"
            confidence = "gold"
        elif code_in_body:
            evidence = "code_in_body"
            confidence = "gold"
        elif name_in_title:
            evidence = "unambiguous_name_in_title"
            confidence = "gold"
        elif name_in_body:
            evidence = "unambiguous_name_in_body"
            confidence = "silver"
        else:
            candidate_counts["candidate_without_clean_text_evidence"] += 1
            continue

        raw_methods = sorted(
            {str(item.get("stock_match_method") or "unknown") for item in raw_items}
        )
        resolved.append(
            {
                "stock_id": stock_id,
                "stock_name": canonical_name,
                "stock_evidence": evidence,
                "stock_confidence": confidence,
                "raw_stock_match_methods": ",".join(raw_methods) if raw_methods else "explicit_code_scan",
            }
        )
        candidate_counts[f"resolved_{evidence}"] += 1
    return resolved, candidate_counts


def rejection_row(
    record: dict[str, Any],
    *,
    line_number: int,
    reason: str,
    resolved_stock_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "line_number": line_number,
        "article_id": record.get("article_id"),
        "url": record.get("url"),
        "published_at": record.get("published_at"),
        "title": record.get("title"),
        "raw_stock_match_count": len(record.get("stock_matches") or []),
        "resolved_stock_ids": resolved_stock_ids or [],
        "reason": reason,
    }


def merge_with_existing(new_path: Path, existing_path: Path, output_path: Path) -> dict[str, int]:
    existing = pd.read_parquet(existing_path).copy()
    new = pd.read_parquet(new_path).copy()
    existing["dataset_origin"] = existing.get("dataset_origin", "existing_gold")
    new["dataset_origin"] = "all_news_reattributed_20260818"

    if "source_article_id" not in existing:
        existing["source_article_id"] = existing["article_id"].astype(str)
    if "document_id" not in existing:
        existing["document_id"] = existing["article_id"].astype(str)
    if "target_stock_count" not in existing:
        existing["target_stock_count"] = 1
    if "is_multi_stock_article" not in existing:
        existing["is_multi_stock_article"] = False

    existing_url_stock = set(zip(existing["url"].fillna("").astype(str), existing["stock_id"].astype(str)))
    existing_text_stock = set(zip(existing["text_hash"].astype(str), existing["stock_id"].astype(str)))
    keep = []
    for row in new.itertuples(index=False):
        url_stock = (str(row.url or ""), str(row.stock_id))
        text_stock = (str(row.text_hash), str(row.stock_id))
        keep.append(url_stock not in existing_url_stock and text_stock not in existing_text_stock)
    added = new.loc[keep].copy()
    merged = pd.concat([existing, added], ignore_index=True, sort=False)
    if merged["article_id"].astype(str).duplicated().any():
        raise ValueError("merged output contains duplicate derived article_id values")
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
    jsonl_path = args.output_dir / "sina_all_news_clean_expanded.jsonl"
    parquet_path = args.output_dir / "sina_all_news_clean_expanded.parquet"
    rejected_path = args.output_dir / "sina_all_news_rejected.jsonl"
    summary_path = args.output_dir / "sina_all_news_clean.summary.json"
    stock_names = load_stock_names(args.stock_universe)
    return_codes = load_return_codes(args.returns, stock_names)
    cutoff = date.fromisoformat(args.cutoff_date)

    counts: Counter[str] = Counter()
    years: Counter[str] = Counter()
    stocks: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    cleaning_actions: Counter[str] = Counter()
    seen_source_stock: set[tuple[str, str]] = set()
    seen_url_stock: set[tuple[str, str]] = set()
    seen_text_stock: set[tuple[str, str]] = set()
    sink = ParquetSink(parquet_path, args.parquet_batch_size)

    with (
        args.input.open(encoding="utf-8", errors="replace") as source,
        jsonl_path.open("w", encoding="utf-8") as destination,
        rejected_path.open("w", encoding="utf-8") as rejected,
    ):
        for line_number, line in enumerate(source, 1):
            if args.max_rows and line_number > args.max_rows:
                break
            counts["input_lines"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["rejected_invalid_json"] += 1
                rejected.write(json.dumps({"line_number": line_number, "reason": "invalid_json"}) + "\n")
                continue
            counts["input_rows"] += 1

            title_raw = str(record.get("title") or "")
            body_raw = str(record.get("body") or "")
            url = str(record.get("url") or "")
            source_article_id = str(record.get("article_id") or "").strip()
            reason: str | None = None
            if not source_article_id or not url:
                reason = "missing_identity"
            elif "\ufffd" in title_raw or "\ufffd" in body_raw:
                reason = "replacement_character"
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
            resolved: list[dict[str, str]] = []
            if reason is None:
                cleaned = clean_sina_article_text(title_raw, body_raw)
                if cleaned["body_clean_chars"] < args.min_body_chars:
                    reason = "short_after_cleaning"
            if reason is None and cleaned is not None:
                resolved, candidate_counts = resolve_stocks_from_clean_text(
                    record,
                    title=cleaned["title_clean"],
                    body=cleaned["body_clean"],
                    stock_names=stock_names,
                    return_codes=return_codes,
                )
                counts.update(candidate_counts)
                if not resolved:
                    reason = "no_verified_stock"

            if reason is not None:
                counts[f"rejected_{reason}"] += 1
                rejected.write(
                    json.dumps(
                        rejection_row(record, line_number=line_number, reason=reason),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            assert cleaned is not None and published_at is not None and published_date is not None
            counts["accepted_source_articles"] += 1
            if len(resolved) > 1:
                counts["accepted_multi_stock_articles"] += 1
            else:
                counts["accepted_single_stock_articles"] += 1
            counts["expanded_candidate_rows"] += len(resolved)
            target_ids = [item["stock_id"] for item in resolved]
            is_multi = len(target_ids) > 1

            kept_for_article = 0
            for evidence in resolved:
                stock_id = evidence["stock_id"]
                source_stock = (source_article_id, stock_id)
                url_stock = (url, stock_id)
                text_stock = (cleaned["text_hash"], stock_id)
                duplicate_reason = None
                if source_stock in seen_source_stock:
                    duplicate_reason = "duplicate_source_article_stock"
                elif url_stock in seen_url_stock:
                    duplicate_reason = "duplicate_url_stock"
                elif text_stock in seen_text_stock:
                    duplicate_reason = "duplicate_clean_text_stock"
                if duplicate_reason:
                    counts[f"rejected_{duplicate_reason}"] += 1
                    duplicate = rejection_row(
                        record,
                        line_number=line_number,
                        reason=duplicate_reason,
                        resolved_stock_ids=[stock_id],
                    )
                    rejected.write(json.dumps(duplicate, ensure_ascii=False) + "\n")
                    continue

                seen_source_stock.add(source_stock)
                seen_url_stock.add(url_stock)
                seen_text_stock.add(text_stock)
                derived_id = (
                    source_article_id if not is_multi else f"{source_article_id}:{stock_id}"
                )
                target_text = build_sina_target_text(
                    stock_id=stock_id,
                    stock_name=evidence["stock_name"],
                    title=cleaned["title_clean"],
                    body=cleaned["body_clean"],
                )
                row = {
                    "article_id": derived_id,
                    "document_id": derived_id,
                    "source_article_id": source_article_id,
                    "stock_id": stock_id,
                    "stock_name": evidence["stock_name"],
                    "published_at": published_at,
                    "publication_date": published_date.isoformat(),
                    "source": record.get("source", "sina_finance"),
                    "content_type": record.get("content_type", "financial_news"),
                    "url": url,
                    "coverage_year": published_date.year,
                    "collected_at": record.get("collected_at"),
                    "body_truncated": bool(record.get("body_truncated")),
                    **cleaned,
                    **target_text,
                    "target_stock_ids": target_ids,
                    "target_stock_count": len(target_ids),
                    "is_multi_stock_article": is_multi,
                    "stock_evidence": evidence["stock_evidence"],
                    "stock_confidence": evidence["stock_confidence"],
                    "raw_stock_match_methods": evidence["raw_stock_match_methods"],
                    "source_depth": record.get("depth"),
                }
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.append(row)
                kept_for_article += 1
                counts["output_rows"] += 1
                years[str(published_date.year)] += 1
                stocks[stock_id] += 1
                evidence_counts[evidence["stock_evidence"]] += 1
                confidence_counts[evidence["stock_confidence"]] += 1
                cleaning_actions.update(cleaned["cleaning_actions"])
                if bool(record.get("body_truncated")):
                    counts["output_truncated_source_rows"] += 1
            if kept_for_article == 0:
                counts["source_articles_fully_deduplicated"] += 1

    sink.close()
    summary: dict[str, Any] = {
        "input": str(args.input.resolve()),
        "outputs": {
            "jsonl": str(jsonl_path.resolve()),
            "parquet": str(parquet_path.resolve()) if parquet_path.exists() else None,
            "rejected": str(rejected_path.resolve()),
        },
        "policy": {
            "cutoff_date": cutoff.isoformat(),
            "min_year": args.min_year,
            "min_body_chars_after_cleaning": args.min_body_chars,
            "trust_raw_stock_links": False,
            "stock_attribution": "verify raw candidates against cleaned text; also scan explicit codes",
            "expand_multi_stock_articles": True,
            "keep_truncated_when_clean_prefix_is_sufficient": True,
            "deduplicate_on": [
                "source_article_id+stock_id",
                "url+stock_id",
                "stock_id+cleaned_text_hash",
            ],
            "stock_universe": str(args.stock_universe.resolve()),
            "returns": str(args.returns.resolve()) if args.returns else None,
        },
        "counts": dict(sorted(counts.items())),
        "years": dict(sorted(years.items())),
        "unique_stocks": len(stocks),
        "top_stocks": dict(stocks.most_common(30)),
        "stock_evidence": dict(sorted(evidence_counts.items())),
        "stock_confidence": dict(sorted(confidence_counts.items())),
        "cleaning_actions": dict(cleaning_actions.most_common()),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if args.existing_clean is not None and parquet_path.exists():
        merged_path = args.output_dir / "sina_all_news_clean_merged.parquet"
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
    parser.add_argument("--returns", type=Path)
    parser.add_argument("--existing-clean", type=Path)
    parser.add_argument("--cutoff-date", required=True)
    parser.add_argument("--min-year", type=int, default=2010)
    parser.add_argument("--min-body-chars", type=int, default=120)
    parser.add_argument("--parquet-batch-size", type=int, default=1000)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.min_body_chars < 1 or args.parquet_batch_size < 1:
        raise SystemExit("min-body-chars and parquet-batch-size must be positive")
    clean_archive(args)


if __name__ == "__main__":
    main()
