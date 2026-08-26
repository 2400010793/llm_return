"""Discover year-specific Sina news seeds from company news pages.

This is a bounded browser discovery step only.  It follows visible public
company-news lists, writes one JSON record as soon as an article is inspected,
and does not use private APIs or bypass access controls.  The resulting seed
JSONL can be handed to the compliant urllib crawler for formal collection.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlparse

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError, async_playwright

ARTICLE_RE = re.compile(r"https?://finance\.sina\.com\.cn/(?:[^/?#]+/)+20\d{2}(?:-\d{2}-\d{2}|\d{4,8})/[^?#]+\.(?:html|shtml)", re.I)
DATE_RE = re.compile(r"/(20\d{2})(?:-(\d{2})-(\d{2})|\d{4,8})/")
BLOCK_MARKERS = ("验证码", "访问异常", "请求过于频繁", "安全验证", "captcha")


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def article_year(url: str) -> int | None:
    match = DATE_RE.search(url)
    return int(match.group(1)) if match else None


def read_stocks(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            code = (row.get("stock_id") or row.get("code") or row.get("symbol") or "").strip()
            code = re.sub(r"^(?:sh|sz)", "", code, flags=re.I).zfill(6)
            if re.fullmatch(r"\d{6}", code):
                rows.append({"stock_id": code, "stock_name": (row.get("stock_name") or row.get("name") or "").strip()})
    if not rows:
        raise ValueError(f"股票文件没有可识别的 stock_id/code/symbol 列: {path}")
    return rows


def news_url(code: str) -> str:
    market = "sh" if code.startswith("6") else "sz"
    return f"https://vip.stock.finance.sina.com.cn/corp/go.php/vCB_AllNewsStock/symbol/{market}{code}.phtml"


async def inspect_stock(page: Page, stock: dict[str, str], args, output) -> dict:
    code = stock["stock_id"]
    base = {"stock_id": code, "stock_name": stock["stock_name"], "list_url": news_url(code), "checked_at": datetime.now(timezone.utc).isoformat()}
    try:
        articles = []
        seen = set()
        pages_visited = 0
        year_counts: Counter[int] = Counter()
        for page_number in range(1, args.max_pages_per_stock + 1):
            list_url = base["list_url"] if page_number == 1 else "http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php?" + urlencode({"symbol": ("sh" if code.startswith("6") else "sz") + code, "Page": page_number})
            await page.goto(list_url, wait_until="domcontentloaded", timeout=args.timeout_ms)
            await page.wait_for_timeout(args.render_wait_ms)
            text = clean(await page.locator("body").inner_text())
            if any(marker.lower() in text.lower() for marker in BLOCK_MARKERS):
                return {**base, "status": "blocked_marker", "articles": len(articles), "pages": pages_visited}
            pages_visited += 1
            links = await page.locator("a").evaluate_all("els => els.map(a => ({title:(a.innerText||'').trim(), url:a.href}))")
            page_articles = 0
            for item in links:
                url = item.get("url", "").split("#", 1)[0]
                if not ARTICLE_RE.fullmatch(url) or url in seen:
                    continue
                seen.add(url)
                year = article_year(url)
                if year not in args.years:
                    continue
                if len(articles) >= args.per_stock:
                    break
                if year_counts[year] >= args.per_year:
                    continue
                article = {"source": "sina_finance", "content_type": "stock_news_seed", "stock_id": code, "stock_name": stock["stock_name"], "year": year, "title": clean(item.get("title", "")), "url": url, "discovered_from": list_url, "discovered_at": datetime.now(timezone.utc).isoformat()}
                output.write(json.dumps(article, ensure_ascii=False) + "\n")
                output.flush()
                articles.append(article)
                page_articles += 1
                year_counts[year] += 1
                if len(articles) >= args.per_stock or all(year_counts[year] >= args.per_year for year in args.years):
                    break
            if len(articles) >= args.per_stock or all(year_counts[year] >= args.per_year for year in args.years) or not links:
                break
            # Stop when a page has no article links at all; otherwise continue
            # through the visible public pagination to reach older years.
            if not any(ARTICLE_RE.fullmatch(item.get("url", "").split("#", 1)[0]) for item in links):
                break
        return {**base, "status": "ok", "articles": len(articles), "pages": pages_visited, "year_counts": dict(year_counts)}
    except PlaywrightTimeoutError:
        return {**base, "status": "timeout", "articles": []}
    except Exception as exc:
        return {**base, "status": "error", "error": f"{type(exc).__name__}: {exc}", "articles": []}


async def main(args: argparse.Namespace) -> None:
    stocks = read_stocks(Path(args.stocks))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    lock = asyncio.Lock()
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=not args.headed)
        pages = [await browser.new_page() for _ in range(args.workers)]
        queue: asyncio.Queue[dict[str, str] | None] = asyncio.Queue()
        for stock in stocks:
            queue.put_nowait(stock)
        with output_path.open("w", encoding="utf-8") as output:
            async def worker(page: Page) -> None:
                while True:
                    stock = await queue.get()
                    if stock is None:
                        queue.task_done()
                        return
                    summary = await inspect_stock(page, stock, args, output)
                    async with lock:
                        summaries.append(summary)
                    queue.task_done()
                    await asyncio.sleep(args.pause_seconds)
            tasks = [asyncio.create_task(worker(page)) for page in pages]
            await queue.join()
            for _ in tasks:
                queue.put_nowait(None)
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await browser.close()
        except Exception:
            pass
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"stocks": len(stocks), "years": args.years, "workers": args.workers, "summaries": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stocks": len(stocks), "articles": sum(int(x.get("articles", 0)) for x in summaries), "output": str(output_path), "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", required=True)
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--per-stock", type=int, default=100)
    parser.add_argument("--per-year", type=int, default=100, help="每只股票每个年份最多保留多少条 seed")
    parser.add_argument("--max-pages-per-stock", type=int, default=100, help="每个股票资讯页最多翻页数")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    parser.add_argument("--render-wait-ms", type=int, default=300)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.per_stock < 1 or args.per_year < 1 or args.max_pages_per_stock < 1 or args.pause_seconds < 0:
        parser.error("workers/per-stock/per-year/max-pages-per-stock 必须为正数，pause 不能为负数")
    asyncio.run(main(args))
