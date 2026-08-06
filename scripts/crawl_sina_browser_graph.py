"""Automated browser BFS/DFS crawler for a bounded Sina historical probe.

This crawler uses normal Playwright page navigation and visible HTML only. It
never calls private APIs or attempts to bypass CAPTCHAs, login walls, robots,
or other access controls. Use it from a network where normal browser access is
permitted. The same script supports BFS and DFS so the traversal comparison is
reproducible.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

from playwright.async_api import TimeoutError as PlaywrightTimeoutError, async_playwright

from crawl_sina_bfs_fast import (
    ALLOWED_HOSTS,
    BLOCK_MARKERS,
    DATE_RE,
    PageParser,
    first_meta,
    load_catalog,
    matches,
    merge_stock_matches,
    normalize_url,
    sha,
    STOCK_LINK_RE,
    year_from_timestamp,
)

DEFAULT_ROOTS = [
    "https://finance.sina.com.cn/",
    "https://finance.sina.com.cn/roll/c/56592.shtml",
    "https://finance.sina.com.cn/stock/",
    "https://cj.sina.com.cn/",
]


def clean_url(base: str, href: str) -> str | None:
    return normalize_url(base, href)


def body_and_record(url: str, depth: int, html_text: str, args, raw_path: Path, catalog: list[dict[str, str]]) -> tuple[dict, list[str]]:
    parser = PageParser()
    parser.feed(html_text)
    body = parser.body
    links = list(dict.fromkeys(filter(None, (clean_url(url, href) for href in parser.links))))
    if any(marker.lower() in body.lower() for marker in BLOCK_MARKERS):
        return {"status": "blocked_marker", "url": url, "depth": depth, "body_chars": len(body)}, links
    raw_path.write_text(html_text, encoding="utf-8")
    published = first_meta(parser.meta, ("article:published_time", "bytedance:published_time", "publishdate", "pubdate", "date"))
    if not published:
        match = DATE_RE.search(body)
        published = match.group(0) if match else None
    title = first_meta(parser.meta, ("og:title",)) or parser.title
    page_ids = list(dict.fromkeys(STOCK_LINK_RE.findall(html_text)))
    article_ids = {item["stock_id"] for item in parser.stock_links}
    selected_ids = page_ids if args.paper_stock_matching else [item for item in page_ids if item in article_ids]
    link_matches = [{"stock_id": code, "stock_name": "", "stock_match_method": "sina_stock_link"} for code in selected_ids]
    stock_matches = merge_stock_matches(link_matches, matches(title, body, catalog))
    record = None
    if published and len(body) >= args.min_body_chars and (not args.exactly_one_stock or len(stock_matches) == 1):
        record = {
            "source": "sina_finance",
            "content_type": "financial_news",
            "article_id": sha(url),
            "url": url,
            "depth": depth,
            "title": title,
            "published_at": published,
            "stock_matches": stock_matches,
            "body_chars": len(body),
            "body": body[: args.max_body_chars],
            "raw_html": str(raw_path),
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
    return {
        "status": "ok",
        "url": url,
        "depth": depth,
        "title": title,
        "published_at": published,
        "body_chars": len(body),
        "stock_link_count": len(selected_ids),
        "record": record,
    }, links


async def crawl(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(args.stock_catalog)
    roots = args.root or DEFAULT_ROOTS
    frontier = deque((url, 0) for url in roots) if args.strategy == "bfs" else [(url, 0) for url in reversed(roots)]
    queued = set(roots)
    visited: set[str] = set()
    records: list[dict] = []
    statuses: dict[str, int] = {}
    manifest_path = output.with_suffix(".manifest.jsonl")
    started = time.monotonic()

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=not args.headed)
        context = await browser.new_context(user_agent=args.user_agent)
        page = await context.new_page()
        with manifest_path.open("a", encoding="utf-8") as manifest:
            while frontier and len(visited) < args.max_pages and time.monotonic() - started < args.max_seconds:
                item = frontier.popleft() if args.strategy == "bfs" else frontier.pop()
                url, depth = item
                if url in visited:
                    continue
                visited.add(url)
                base = {"url": url, "depth": depth, "fetched_at": datetime.now(timezone.utc).isoformat()}
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=args.timeout * 1000)
                    await page.wait_for_timeout(args.render_wait_ms)
                    html_text = await page.content()
                    raw_path = raw_dir / f"{sha(url)}.html"
                    result, links = body_and_record(url, depth, html_text, args, raw_path, catalog)
                    result.update(base)
                    if result.get("record"):
                        records.append(result.pop("record"))
                    status = result.get("status", "ok")
                    statuses[status] = statuses.get(status, 0) + 1
                    manifest.write(json.dumps(result, ensure_ascii=False) + "\n")
                    if depth < args.max_depth:
                        children = [(child, depth + 1) for child in links if child not in queued]
                        for child, child_depth in children:
                            queued.add(child)
                            if args.strategy == "bfs":
                                frontier.append((child, child_depth))
                            else:
                                frontier.append((child, child_depth))
                except PlaywrightTimeoutError as exc:
                    statuses["timeout"] = statuses.get("timeout", 0) + 1
                    manifest.write(json.dumps({**base, "status": "timeout", "error": str(exc)}, ensure_ascii=False) + "\n")
                except Exception as exc:
                    statuses["error"] = statuses.get("error", 0) + 1
                    manifest.write(json.dumps({**base, "status": "error", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False) + "\n")
                manifest.flush()
                if args.pause_seconds:
                    await asyncio.sleep(args.pause_seconds)
        await browser.close()

    year_counts: dict[str, int] = {}
    stock_counts: dict[str, int] = {}
    for record in records:
        year = year_from_timestamp(record.get("published_at"))
        if year:
            year_counts[year] = year_counts.get(year, 0) + 1
        for stock in record.get("stock_matches", []):
            stock_counts[stock["stock_id"]] = stock_counts.get(stock["stock_id"], 0) + 1
    result = {
        "source": "sina_finance",
        "traversal": args.strategy,
        "roots": roots,
        "visited": len(visited),
        "records": records,
        "queued_remaining": len(frontier),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "statuses": statuses,
        "coverage": {
            "article_year_counts": dict(sorted(year_counts.items())),
            "earliest_article_year": min(year_counts) if year_counts else None,
            "latest_article_year": max(year_counts) if year_counts else None,
            "stock_article_rate": round(sum(bool(r.get("stock_matches")) for r in records) / len(records), 4) if records else 0,
            "stock_match_counts": dict(sorted(stock_counts.items(), key=lambda item: (-item[1], item[0]))),
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="浏览器环境下可复现的新浪 BFS/DFS 小规模爬虫")
    parser.add_argument("--strategy", choices=("bfs", "dfs"), required=True)
    parser.add_argument("--root", action="append")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    parser.add_argument("--render-wait-ms", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--min-body-chars", type=int, default=120)
    parser.add_argument("--max-body-chars", type=int, default=50000)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stock-catalog")
    parser.add_argument("--exactly-one-stock", action="store_true")
    parser.add_argument("--paper-stock-matching", action="store_true")
    parser.add_argument("--user-agent", default="llm-return-research/0.1 (academic prototype)")
    args = parser.parse_args()
    if args.max_pages < 1 or args.max_depth < 0 or args.max_seconds <= 0 or args.pause_seconds < 0:
        parser.error("参数范围无效")
    result = asyncio.run(crawl(args))
    print(json.dumps({"output": args.output, "traversal": args.strategy, "visited": result["visited"], "articles": len(result["records"]), "coverage": result["coverage"], "statuses": result["statuses"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
