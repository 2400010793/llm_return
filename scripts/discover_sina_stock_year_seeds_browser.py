"""Bounded visible-browser discovery of Sina articles by stock and year.

Searches public rendered result pages, opens visible Sina URLs, and records
validated year/stock matches. It does not call private APIs or bypass access
controls.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlparse

from playwright.async_api import TimeoutError as PlaywrightTimeoutError, async_playwright

SINA_HOSTS = {"finance.sina.com.cn", "cj.sina.com.cn", "vip.stock.finance.sina.com.cn", "stock.finance.sina.com.cn"}
ARTICLE_RE = re.compile(r"^https?://(?:finance|cj)\.sina\.com\.cn/(?:t|s|e|y|roll|stock|view|globe|special|focus|wm|jjxw|jhzx)/[^?#]+\.(?:html|shtml)$", re.I)
REPORT_RE = re.compile(r"^https?://(?:vip\.stock|stock)\.finance\.sina\.com\.cn/(?:corp/view|stock/go\.php)/.+", re.I)
DATE_RE = re.compile(r"(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日(?:[^0-9]{0,12}(\d{1,2}):(\d{2}))?")
BLOCK_MARKERS = ("验证码", "访问异常", "请求过于频繁", "安全验证", "captcha")


def canonical(url: str) -> str | None:
    url = url.split("#", 1)[0]
    parsed = urlparse(url)
    if parsed.hostname not in SINA_HOSTS or not (ARTICLE_RE.fullmatch(url) or REPORT_RE.fullmatch(url)):
        return None
    return url.replace("http://", "https://", 1)


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def parse_date(text: str, target_year: int) -> str | None:
    for match in DATE_RE.finditer(text):
        if int(match.group(1)) != target_year:
            continue
        result = f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
        if match.group(4):
            result += f" {int(match.group(4)):02d}:{match.group(5)}"
        return result
    return None


async def search(page, query: str, timeout_ms: int) -> list[str]:
    try:
        # Search pages can keep an analytics/resource request open.  The
        # visible document is usable after commit, so do not discard it when
        # domcontentloaded exceeds the network timeout.
        await page.goto("https://duckduckgo.com/?q=" + quote(query), wait_until="commit", timeout=timeout_ms)
    except PlaywrightTimeoutError:
        pass
    await page.wait_for_timeout(500)
    links = await page.locator("a").evaluate_all("els => els.map(a => a.href)")
    return list(dict.fromkeys(filter(None, (canonical(url) for url in links))))


async def inspect(page, url: str, stock: dict[str, str], year: int, query: str, args) -> dict:
    base = {"url": url, "stock_id": stock["stock_id"], "stock_name": stock["stock_name"], "year_requested": year, "query": query, "discovery_method": "browser_visible_stock_year_search", "checked_at": datetime.now(timezone.utc).isoformat()}
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=args.timeout_ms)
        await page.wait_for_timeout(args.render_wait_ms)
        text = clean(await page.locator("body").inner_text())
        title = clean(await page.title())
        published = parse_date(text, year)
        stock_hit = stock["stock_name"] in text or stock["stock_id"] in text
        blocked = any(marker.lower() in text.lower() for marker in BLOCK_MARKERS)
        valid = bool(published and stock_hit and len(text) >= args.min_body_chars and not blocked)
        return {**base, "status": "validated" if valid else "rejected", "title": title, "published_at": published, "body_chars_visible": len(text), "body_chars_stored": min(len(text), args.max_body_chars), "body_truncated": len(text) > args.max_body_chars, "stock_hit": stock_hit, "block_marker": blocked, "body_preview": text[:300]}
    except PlaywrightTimeoutError:
        return {**base, "status": "timeout"}
    except Exception as exc:
        return {**base, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


async def run(args) -> None:
    with Path(args.stocks).open(encoding="utf-8-sig", newline="") as stream:
        stocks = [{"stock_id": row["stock_id"].zfill(6), "stock_name": row["stock_name"]} for row in csv.DictReader(stream) if row.get("stock_id") and row.get("stock_name")]
    stocks = stocks[: args.max_stocks]
    years = list(range(args.start_year, args.end_year + 1))
    records, seen = [], set()
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not args.headed)
        search_page, article_page = await browser.new_page(), await browser.new_page()
        try:
            for stock in stocks:
                for year in years:
                    query = f"site:finance.sina.com.cn {stock['stock_name']} {stock['stock_id']} {year} 年报 股票 新闻"
                    try:
                        urls = await search(search_page, query, args.timeout_ms)
                    except Exception as exc:
                        records.append({"stock_id": stock["stock_id"], "stock_name": stock["stock_name"], "year_requested": year, "query": query, "status": "search_error", "error": str(exc)})
                        continue
                    for url in urls[: args.max_results_per_query]:
                        if url in seen:
                            continue
                        seen.add(url)
                        item = await inspect(article_page, url, stock, year, query, args)
                        if item["status"] == "validated" or args.keep_rejected:
                            records.append(item)
                        if args.pause_seconds:
                            await asyncio.sleep(args.pause_seconds)
        finally:
            await browser.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "stocks": stocks, "years": years, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "visited": len(seen), "validated": sum(x.get("status") == "validated" for x in records)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="模拟浏览器按热门股票和年份发现新浪财经报道 seed")
    parser.add_argument("--stocks", required=True)
    parser.add_argument("--start-year", type=int, default=2002)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--max-stocks", type=int, default=10)
    parser.add_argument("--max-results-per-query", type=int, default=8)
    parser.add_argument("--min-body-chars", type=int, default=500)
    parser.add_argument("--max-body-chars", type=int, default=12000)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--render-wait-ms", type=int, default=500)
    parser.add_argument("--pause-seconds", type=float, default=1.5)
    parser.add_argument("--keep-rejected", action="store_true")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.start_year > args.end_year or args.max_stocks < 1:
        parser.error("年份或股票数量参数无效")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()