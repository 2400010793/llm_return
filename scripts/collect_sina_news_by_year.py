"""Collect visible Sina Finance article pages for selected calendar years.

This collector uses the rendered public listing and article pages only. It does
not call private APIs or bypass access controls. Year selection is derived from
the date embedded in each visible article URL, then the article page is opened
sequentially to capture its visible text.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError, async_playwright


BASE_URL = "https://finance.sina.com.cn/roll/c/56592.shtml"
BLOCK_MARKERS = ("验证码", "访问异常", "请求过于频繁", "安全验证", "captcha")
DATE_IN_URL = re.compile(r"/(20\d{2})-(\d{2})-(\d{2})/")


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def article_year(url: str) -> int | None:
    match = DATE_IN_URL.search(url)
    return int(match.group(1)) if match else None


def article_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]


async def ensure_allowed(page: Page) -> None:
    text = await page.locator("body").inner_text()
    marker = next((x for x in BLOCK_MARKERS if x.lower() in text.lower()), None)
    if marker:
        raise RuntimeError(f"{marker} detected at {page.url}")


async def listing_articles(page: Page) -> list[dict]:
    return await page.locator('a[href*="finance.sina.com.cn/"]').evaluate_all(
        """els => { const seen = new Set(); return els.map(a => ({
        title: (a.innerText || '').trim(), url: a.href,
        })).filter(x => x.title && /\\/20\\d{2}-\\d{2}-\\d{2}\\//.test(x.url)
        && !seen.has(x.url) && (seen.add(x.url), true)); }"""
    )


async def article_detail(page: Page, item: dict, year: int) -> dict:
    await page.goto(item["url"], wait_until="domcontentloaded")
    await page.wait_for_timeout(1200)
    await ensure_allowed(page)
    text = await page.locator("body").inner_text()
    lines = [clean(x) for x in text.splitlines() if clean(x)]
    timestamp = next((x for x in lines if re.search(r"20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}", x)), None)
    return {
        "source": "sina_finance",
        "content_type": "financial_news",
        "article_id": article_id(item["url"]),
        "year": year,
        "title": item["title"],
        "url": item["url"],
        "published_at_display": timestamp,
        "body": " ".join(lines)[:50000],
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


async def collect_years(
    page: Page,
    years: set[int],
    *,
    max_pages: int,
    detail_limit: int,
    pause_seconds: float,
) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    page_number = 1
    while page_number <= max_pages and len(records) < detail_limit * len(years):
        url = BASE_URL if page_number == 1 else f"{BASE_URL}?page={page_number}"
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(1200)
        await ensure_allowed(page)
        items = await listing_articles(page)
        if not items:
            break
        page_years = {article_year(x["url"]) for x in items} - {None}
        for item in items:
            year = article_year(item["url"])
            if year not in years or item["url"] in seen:
                continue
            seen.add(item["url"])
            per_year = sum(1 for x in records if x["year"] == year)
            if per_year >= detail_limit:
                continue
            try:
                records.append(await article_detail(page, item, year))
            except PlaywrightTimeoutError:
                records.append({"source": "sina_finance", "content_type": "collection_error", "year": year, "title": item["title"], "url": item["url"], "error": "article timeout; no retry"})
            await asyncio.sleep(pause_seconds)
            if len(records) >= detail_limit * len(years):
                break
        if len(records) >= detail_limit * len(years):
            break
        # The listing is reverse chronological. Once every target year is older
        # than the requested range, later pages cannot add a target article.
        if page_years and max(page_years) < min(years):
            break
        page_number += 1
    return records


async def main(args: argparse.Namespace) -> None:
    years = set(args.years)
    if not years or any(year < 2000 or year > 2100 for year in years):
        raise ValueError("years must contain valid calendar years")
    if args.max_pages < 1 or args.detail_limit < 1 or args.pause_seconds < 0:
        raise ValueError("max-pages and detail-limit must be positive; pause must be non-negative")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not args.headed)
        page = await browser.new_page()
        records = await collect_years(page, years, max_pages=args.max_pages, detail_limit=args.detail_limit, pause_seconds=args.pause_seconds)
        await browser.close()
    output.write_text(json.dumps({"source": "sina_finance", "years": sorted(years), "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "years": sorted(years), "records": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", required=True, help="按年份采集，例如 --years 2018 2019 2020")
    parser.add_argument("--max-pages", type=int, default=100, help="每个列表序列最多访问页数")
    parser.add_argument("--detail-limit", type=int, default=100, help="每个年份最多采集正文数量")
    parser.add_argument("--pause-seconds", type=float, default=3.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--headed", action="store_true")
    asyncio.run(main(parser.parse_args()))
