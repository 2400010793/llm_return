"""Stream-filter the merged Sina archive into auditable research candidates.

This deliberately does not infer new stock entities from article text. It only
trusts the existing ``stock_matches`` field and emits separate broad and strict
candidates so the raw archive remains untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

URL_YEAR = re.compile(r"/(20\d{2})(?:[-/]\d{2}(?:[-/]\d{2})?|/)")
CODE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
NAV_MARKERS = (
    "24小时客户服务热线", "版权声明", "联系我们", "新浪财经", "首页",
    "个股资料", "行情走势", "公司概况", "公司公告", "历史新闻列表",
)
LIST_PATH_MARKERS = ("vCB_AllNewsStock", "/corp/go.php", "/corp/view.php")


def parse_stock_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values: Iterable[Any] = value
    elif isinstance(value, dict):
        values = value.keys()
    else:
        values = [value]
    result: list[str] = []
    for item in values:
        if isinstance(item, dict):
            text = " ".join(str(item.get(k, "")) for k in ("stock_id", "code", "ticker", "symbol"))
        else:
            text = str(item)
        result.extend(CODE.findall(text))
    return sorted(set(result))


def url_year(url: str) -> int | None:
    match = URL_YEAR.search(urlparse(url).path)
    return int(match.group(1)) if match else None


def is_list_page(record: dict[str, Any], body: str) -> bool:
    url = str(record.get("url", ""))
    if any(marker in url for marker in LIST_PATH_MARKERS):
        return True
    marker_hits = sum(marker in body for marker in NAV_MARKERS)
    return marker_hits >= 4


def clean_record(record: dict[str, Any], stock_names: dict[str, str], cutoff: date, content_only: bool = False) -> tuple[str, dict[str, Any]]:
    body = str(record.get("body") or "")
    title = str(record.get("title") or "")
    published = str(record.get("published_at") or "")[:10]
    reasons: list[str] = []
    if not record.get("article_id"):
        reasons.append("missing_article_id")
    if not record.get("url"):
        reasons.append("missing_url")
    if "\ufffd" in title or "\ufffd" in body:
        reasons.append("replacement_character")
    if len(body.strip()) < 100:
        reasons.append("body_too_short")
    if record.get("body_truncated") is True:
        reasons.append("body_truncated")
    try:
        published_date = date.fromisoformat(published)
    except ValueError:
        published_date = None
        reasons.append("invalid_date")
    if published_date and (published_date < date(2010, 1, 1) or published_date > cutoff):
        reasons.append("date_out_of_range")
    uyear = url_year(str(record.get("url", "")))
    if uyear and published_date and uyear != published_date.year:
        reasons.append("url_date_conflict")
    if is_list_page(record, body):
        reasons.append("list_or_navigation_page")
    stocks = parse_stock_ids(record.get("stock_matches"))
    # Use the title as the entity evidence. The body often contains unrelated
    # recommendation/navigation links and would reintroduce the same leak.
    entity_text = title
    # The crawler's stock_matches field can contain page-navigation links.
    # Retain a code only when the article itself also contains that code or
    # the corresponding company name. This removes cases such as an article
    # about 000920 being tagged with an unrelated navigation stock.
    target_stocks = sorted(
        stock for stock in set(stocks) & set(stock_names)
        if stock in entity_text or (len(stock_names[stock]) >= 4 and stock_names[stock] in entity_text)
    )
    if not target_stocks:
        reasons.append("no_target_stock")
    elif len(target_stocks) != 1:
        reasons.append("multiple_target_stocks")
    if "000001" in target_stocks:
        reasons.append("high_frequency_000001")
    # Content-only mode intentionally does not use the three research-entity
    # restrictions: current CSI500 membership, exact-one relation, or title
    # entity evidence. Stock matches remain in the output for later remapping.
    entity_reasons = {"no_target_stock", "multiple_target_stocks", "high_frequency_000001"}
    status = "reject" if (set(reasons) - entity_reasons if content_only else reasons) else "keep"
    output = dict(record)
    output["screen_stock_ids"] = target_stocks
    output["screen_url_year"] = uyear
    output["screen_reasons"] = reasons
    output["screen_status"] = status
    return status, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff", default="2026-08-12")
    parser.add_argument("--content-only", action="store_true", help="不按当前CSI500、单股票或标题实体筛选，仅做正文/日期/页面质量筛选")
    args = parser.parse_args()
    cutoff = date.fromisoformat(args.cutoff)
    stock_names: dict[str, str] = {}
    # Some Windows CSV exports include a UTF-8 BOM; utf-8-sig keeps the first
    # header name as ``stock_id`` instead of ``\ufeffstock_id``.
    with args.universe.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("active", "1") == "1":
                stock_names[str(row.get("stock_id", "")).zfill(6)] = str(row.get("stock_name", "")).strip()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    broad_path = args.output_dir / "articles_screened_no_replacement.jsonl"
    strict_path = args.output_dir / ("articles_screened_content_quality.jsonl" if args.content_only else "articles_screened_single_stock.jsonl")
    reject_path = args.output_dir / "articles_rejected.jsonl"
    counts = Counter()
    years = Counter()
    stocks = Counter()
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    with args.input.open(encoding="utf-8", errors="replace") as source, broad_path.open("w", encoding="utf-8") as broad, strict_path.open("w", encoding="utf-8") as strict, reject_path.open("w", encoding="utf-8") as rejected:
        for line_no, line in enumerate(source, 1):
            counts["input_lines"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["bad_json"] += 1
                continue
            if not isinstance(record, dict):
                counts["non_object"] += 1
                continue
            status, screened = clean_record(record, stock_names, cutoff, args.content_only)
            if record.get("article_id") in seen_ids:
                counts["duplicate_article_id"] += 1
            else:
                seen_ids.add(str(record.get("article_id")))
            if record.get("url") in seen_urls:
                counts["duplicate_url"] += 1
            else:
                seen_urls.add(str(record.get("url")))
            reasons = screened["screen_reasons"]
            counts["kept_broad"] += int("replacement_character" not in reasons and "invalid_date" not in reasons and "date_out_of_range" not in reasons and "list_or_navigation_page" not in reasons and "body_too_short" not in reasons)
            counts["kept_strict"] += int(status == "keep")
            for reason in reasons:
                counts["reject_" + reason] += 1
            if status == "keep":
                strict.write(json.dumps(screened, ensure_ascii=False, separators=(",", ":")) + "\n")
                year = str(screened.get("published_at", ""))[:4]
                years[year] += 1
                for stock in screened["screen_stock_ids"]:
                    stocks[stock] += 1
            if "replacement_character" not in reasons and "invalid_date" not in reasons and "date_out_of_range" not in reasons and "list_or_navigation_page" not in reasons and "body_too_short" not in reasons:
                broad.write(json.dumps(screened, ensure_ascii=False, separators=(",", ":")) + "\n")
            if status != "keep":
                rejected.write(json.dumps(screened, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "input": str(args.input), "universe": str(args.universe), "cutoff": args.cutoff,
        "allowed_stocks": len(stock_names), "counts": dict(counts),
        "strict_years": dict(sorted(years.items())), "strict_stocks": dict(stocks.most_common()),
        "outputs": {"broad": str(broad_path), "strict_single_stock": str(strict_path), "rejected": str(reject_path)},
        "policy": {"content_only": args.content_only, "trust_existing_stock_matches_only": not args.content_only, "exclude_list_pages": True, "exclude_replacement_chars": True, "exclude_date_conflicts": True, "exclude_truncated_body": True, "entity_restrictions_removed": args.content_only},
    }
    summary_path = args.output_dir / "screening_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
