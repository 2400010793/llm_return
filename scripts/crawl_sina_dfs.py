"""Bounded depth-first Sina HTML crawler for comparing traversal order.

This is an experiment, not a bypass: it keeps the same allowed hosts and
robots.txt policy as the BFS crawler, but follows one link branch deeply before
backtracking. The output records the oldest article year actually discovered.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from crawl_sina_bfs_fast import (
    ALLOWED_HOSTS,
    DEFAULT_ROOTS,
    BLOCK_MARKERS,
    DATE_RE,
    PageParser,
    first_meta,
    load_catalog,
    matches,
    merge_stock_matches,
    normalize_url,
    robots_allowed,
    sha,
    STOCK_LINK_RE,
    year_from_timestamp,
)


def crawl(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(args.stock_catalog)
    stack = [(root, 0) for root in reversed(args.root or DEFAULT_ROOTS)]
    queued = {url for url, _ in stack}
    visited: set[str] = set()
    records: list[dict] = []
    manifest_path = output.with_suffix(".manifest.jsonl")
    robot_cache = {}
    from threading import Lock
    robot_lock = Lock()
    started = time.monotonic()

    with manifest_path.open("a", encoding="utf-8") as manifest:
        while stack and len(visited) < args.max_pages and time.monotonic() - started < args.max_seconds:
            url, depth = stack.pop()
            if url in visited:
                continue
            visited.add(url)
            base = {"url": url, "depth": depth, "fetched_at": datetime.now(timezone.utc).isoformat()}
            result = dict(base)
            try:
                if not robots_allowed(url, args.user_agent, robot_cache, robot_lock):
                    result["status"] = "robots_denied"
                    manifest.write(json.dumps(result, ensure_ascii=False) + "\n")
                    continue
                request = Request(url, headers={"User-Agent": args.user_agent, "Accept": "text/html,application/xhtml+xml"})
                with urlopen(request, timeout=args.timeout) as response:
                    payload = response.read()
                    content_type = response.headers.get("Content-Type", "")
                if content_type and "html" not in content_type.lower():
                    result["status"] = "not_html"
                    manifest.write(json.dumps(result, ensure_ascii=False) + "\n")
                    continue
                decoded = payload.decode("utf-8", errors="replace")
                parser = PageParser()
                parser.feed(decoded)
                body = parser.body
                if any(marker.lower() in body.lower() for marker in BLOCK_MARKERS):
                    result["status"] = "blocked_marker"
                    manifest.write(json.dumps(result, ensure_ascii=False) + "\n")
                    continue
                raw_path = raw_dir / f"{sha(url)}.html"
                raw_path.write_bytes(payload)
                links = list(dict.fromkeys(filter(None, (normalize_url(url, href) for href in parser.links))))
                published = first_meta(parser.meta, ("article:published_time", "bytedance:published_time", "publishdate", "pubdate", "date"))
                if not published:
                    match = DATE_RE.search(body)
                    published = match.group(0) if match else None
                title = first_meta(parser.meta, ("og:title",)) or parser.title
                page_ids = list(dict.fromkeys(STOCK_LINK_RE.findall(decoded)))
                article_ids = {item["stock_id"] for item in parser.stock_links}
                link_matches = [{"stock_id": code, "stock_name": "", "stock_match_method": "sina_stock_link"} for code in page_ids if args.paper_stock_matching or code in article_ids]
                stock_matches = merge_stock_matches(link_matches, matches(title, body, catalog))
                if published and len(body) >= args.min_body_chars and (not args.exactly_one_stock or len(stock_matches) == 1):
                    stored_body = body[:args.max_body_chars]
                    records.append({"source": "sina_finance", "content_type": "financial_news", "article_id": sha(url), "url": url, "depth": depth, "title": title, "published_at": published, "stock_matches": stock_matches, "body": stored_body, "body_chars_full": len(body), "body_chars_stored": len(stored_body), "body_truncated": len(stored_body) < len(body), "raw_html": str(raw_path), "collected_at": base["fetched_at"]})
                result.update({"status": "ok", "links": len(links), "published_at": published, "body_chars": len(body)})
                manifest.write(json.dumps(result, ensure_ascii=False) + "\n")
                if depth < args.max_depth:
                    # Reverse because stack.pop() visits the first HTML link first.
                    for child in reversed(links):
                        if child not in queued:
                            queued.add(child)
                            stack.append((child, depth + 1))
            except Exception as exc:
                result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                manifest.write(json.dumps(result, ensure_ascii=False) + "\n")
            manifest.flush()
            if args.pause_seconds:
                time.sleep(args.pause_seconds)

    year_counts: dict[str, int] = {}
    for record in records:
        year = year_from_timestamp(record.get("published_at"))
        if year:
            year_counts[year] = year_counts.get(year, 0) + 1
    result = {
        "source": "sina_finance",
        "traversal": "dfs",
        "roots": args.root or list(DEFAULT_ROOTS),
        "visited": len(visited),
        "records": records,
        "queued_remaining": len(stack),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "coverage": {"article_year_counts": dict(sorted(year_counts.items())), "earliest_article_year": min(year_counts) if year_counts else None, "latest_article_year": max(year_counts) if year_counts else None},
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="有边界、遵守 robots.txt 的新浪 DFS 新闻爬虫")
    parser.add_argument("--root", action="append")
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--min-body-chars", type=int, default=120)
    parser.add_argument("--max-body-chars", type=int, default=12000, help="正文最多保存字符数；用于控制后续模型输入长度")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stock-catalog")
    parser.add_argument("--exactly-one-stock", action="store_true")
    parser.add_argument("--paper-stock-matching", action="store_true")
    parser.add_argument("--user-agent", default="llm-return-research/0.1 (academic prototype)")
    args = parser.parse_args()
    if args.max_pages < 1 or args.max_depth < 0 or args.max_seconds <= 0 or args.pause_seconds < 0:
        parser.error("参数范围无效")
    result = crawl(args)
    print(json.dumps({"output": args.output, "traversal": "dfs", "visited": result["visited"], "articles": len(result["records"]), "coverage": result["coverage"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
