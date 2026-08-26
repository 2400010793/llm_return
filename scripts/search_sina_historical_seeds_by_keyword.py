"""Discover Sina historical article seeds from public browser keyword searches.

This script intentionally starts from search queries, not existing article seeds.
It uses visible browser pages only and does not call private APIs or bypass access
controls. Search coverage is auditable per year and per keyword.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from playwright.async_api import TimeoutError as PlaywrightTimeoutError, async_playwright

ARTICLE_RE = re.compile(r"^https?://(?:finance\.sina\.com\.cn|cj\.sina\.com\.cn)/(?:t|s|e|y)/\d+\.html$", re.I)
DATED_ARTICLE_RE = re.compile(
    r"^https?://finance\.sina\.com\.cn/(?:[^/?#]+/){1,3}20\d{2}-\d{2}-\d{2}/doc-[^/?#]+\.shtml$",
    re.I,
)
DATE_RE = re.compile(r"(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日")
BLOCK_MARKERS = ("验证码", "访问异常", "请求过于频繁", "安全验证", "captcha")
NAV_MARKERS = ("新浪财经_", "股票首页", "行情中心", "登录", "注册")
DEFAULT_KEYWORDS = ("股市", "股票", "上市公司", "证券", "财经", "业绩", "重组", "公告")


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def body_hash(value: str) -> str:
    return hashlib.sha256(clean(value).encode("utf-8")).hexdigest()


def parse_date(text: str):
    match = DATE_RE.search(text)
    if not match:
        return None, None, None
    year, month, day = map(int, match.groups())
    return f"{year:04d}-{month:02d}-{day:02d}", year, month


def canonical(url: str) -> str | None:
    url = url.split("#", 1)[0]
    return url.replace("http://", "https://", 1) if (ARTICLE_RE.fullmatch(url) or DATED_ARTICLE_RE.fullmatch(url)) else None


async def search_year(browser, year: int, keywords: list[str], args, semaphore) -> list[dict]:
    async with semaphore:
        page = await browser.new_page()
        records: list[dict] = []
        seen_urls: set[str] = set()
        try:
            for keyword in keywords:
                query = f"site:finance.sina.com.cn {year}年 {keyword} 新浪财经 新闻"
                search_url = "https://cn.bing.com/search?q=" + quote(query)
                try:
                    await page.goto(search_url, wait_until="domcontentloaded", timeout=args.timeout_ms)
                    links = await page.locator("a").evaluate_all("els => els.map(a => a.href)")
                except (PlaywrightTimeoutError, Exception) as exc:
                    records.append({"year": year, "keyword": keyword, "status": "search_error", "error": type(exc).__name__})
                    continue
                candidates = []
                for href in links:
                    url = canonical(href or "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        candidates.append(url)
                for url in candidates[: args.results_per_query]:
                    try:
                        await page.goto(url, wait_until="domcontentloaded", timeout=args.timeout_ms)
                        text = clean(await page.locator("body").inner_text())
                        title = clean(await page.title())
                        if any(marker.lower() in text.lower() for marker in BLOCK_MARKERS):
                            records.append({"url": url, "year_query": year, "keyword": keyword, "status": "blocked_marker"})
                            continue
                        published_at, published_year, published_month = parse_date(text)
                        valid = bool(published_at and published_year == year and len(text) >= args.min_body_chars and not any(marker.lower() in title.lower() for marker in NAV_MARKERS))
                        records.append({"url": url, "year_query": year, "keyword": keyword, "status": "ok", "quality_status": "valid" if valid else "low_quality", "title": title, "published_at": published_at, "published_year": published_year, "published_month": published_month, "body_chars": len(text), "body_sha256": body_hash(text), "body": text[: args.max_body_chars], "checked_at": datetime.now(timezone.utc).isoformat()})
                    except (PlaywrightTimeoutError, Exception) as exc:
                        records.append({"url": url, "year_query": year, "keyword": keyword, "status": "article_error", "error": type(exc).__name__})
        finally:
            await page.close()
        return records


def select(records: list[dict], years: list[int], per_year: int) -> list[dict]:
    selected = []
    for year in years:
        pool = [x for x in records if x.get("quality_status") == "valid" and x.get("published_year") == year]
        pool.sort(key=lambda x: (x.get("body_chars", 0), x.get("published_at") or "", x.get("url", "")), reverse=True)
        seen: set[str] = set()
        chosen = []
        for row in pool:
            if row.get("body_sha256") in seen:
                continue
            seen.add(row.get("body_sha256"))
            item = dict(row)
            item["seed_id"] = f"{year}_{len(chosen) + 1:03d}"
            item["selection_status"] = "accepted" if len(chosen) < per_year else "overflow"
            chosen.append(item)
            if len(chosen) >= per_year:
                break
        selected.extend(chosen)
    return selected


async def main(args) -> None:
    years = list(range(args.start_year, args.end_year + 1))
    keywords = args.keyword or list(DEFAULT_KEYWORDS)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not args.headed)
        try:
            chunks = await asyncio.gather(*(search_year(browser, year, keywords, args, semaphore) for year in years))
        finally:
            await browser.close()
    records = [row for chunk in chunks for row in chunk]
    selected = select(records, years, args.per_year)
    payload = {"start_year": args.start_year, "end_year": args.end_year, "keywords": keywords, "concurrency": args.concurrency, "records": records, "selected": selected, "year_counts": {str(year): sum(1 for row in selected if row.get("published_year") == year) for year in years}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.catalog_output)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["seed_id", "published_year", "published_month", "title", "url", "published_at", "body_chars", "body_sha256", "keyword", "selection_status"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in selected)
    root_dir = output.parent
    root_files = {}
    for year in years:
        urls = sorted({row["url"] for row in selected if row.get("published_year") == year})
        root_path = root_dir / f"keyword_roots_{year}.txt"
        root_path.write_text(("\n".join(urls) + "\n") if urls else "", encoding="utf-8")
        root_files[str(year)] = str(root_path)
    payload["root_files"] = root_files
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "catalog_output": str(csv_path), "searched_years": len(years), "records": len(records), "year_counts": payload["year_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从公开浏览器关键词搜索中发现新浪历史文章种子")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--per-year", type=int, default=100)
    parser.add_argument("--keyword", action="append", help="重复指定关键词；默认使用财经股票事件词")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--results-per-query", type=int, default=10)
    parser.add_argument("--min-body-chars", type=int, default=500)
    parser.add_argument("--max-body-chars", type=int, default=12000)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--catalog-output", required=True)
    parser.add_argument("--headed", action="store_true")
    asyncio.run(main(parser.parse_args()))
