"""Discover year-stratified Sina article roots from public hotspot pages.

The input is an independent set of current hotspot article URLs.  Each hotspot
is crawled in its own browser task; discovered public Sina article URLs are
classified by the publication date rendered on the page.  The output contains
one root file per year so later seed selection does not reuse an uneven legacy
seed pool.

This script uses visible public pages only.  It does not call private APIs or
bypass CAPTCHA, login, robots, or other access controls.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urlparse

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError, async_playwright

ALLOWED_HOSTS = {"finance.sina.com.cn", "cj.sina.com.cn"}
ARTICLE_RE = re.compile(
    r"^https?://(?:finance\.sina\.com\.cn|cj\.sina\.com\.cn)/(?:t|s|e|y)/\d+\.html$",
    re.I,
)
ROOT_RE = re.compile(
    r"^https?://finance\.sina\.com\.cn/(?:[^/?#]+/){1,3}20\d{2}-\d{2}-\d{2}/doc-[^/?#]+\.shtml$",
    re.I,
)
DATE_RE = re.compile(r"(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日")
BLOCK_MARKERS = ("验证码", "访问异常", "请求过于频繁", "安全验证", "captcha")


def canonical_url(value: str) -> str | None:
    value = urldefrag(value.strip())[0]
    parsed = urlparse(value)
    if parsed.hostname not in ALLOWED_HOSTS:
        return None
    if not (ARTICLE_RE.fullmatch(value) or ROOT_RE.fullmatch(value)):
        return None
    return value.replace("http://", "https://", 1)


def parse_year(text: str) -> int | None:
    match = DATE_RE.search(text)
    return int(match.group(1)) if match else None


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


async def inspect(page: Page, url: str, timeout_ms: int, render_wait_ms: int) -> tuple[dict, list[str]]:
    checked_at = datetime.now(timezone.utc).isoformat()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        if render_wait_ms:
            await page.wait_for_timeout(render_wait_ms)
        text = clean(await page.locator("body").inner_text())
        title = clean(await page.title())
        links = await page.locator("a").evaluate_all("els => els.map(a => a.href)")
        children = list(dict.fromkeys(filter(None, (canonical_url(x or "") for x in links))))
        if any(marker.lower() in text.lower() for marker in BLOCK_MARKERS):
            return {"url": url, "status": "blocked_marker", "title": title, "checked_at": checked_at}, []
        return {
            "url": url,
            "status": "ok",
            "title": title,
            "published_year": parse_year(text),
            "body_chars": len(text),
            "links_found": len(children),
            "checked_at": checked_at,
        }, children
    except PlaywrightTimeoutError:
        return {"url": url, "status": "timeout", "checked_at": checked_at}, []
    except Exception as exc:
        return {
            "url": url,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "checked_at": checked_at,
        }, []


async def crawl_root(browser, root: str, args, semaphore: asyncio.Semaphore) -> list[dict]:
    async with semaphore:
        page = await browser.new_page()
        queue = deque([(root, 0)])
        queued = {root}
        visited: set[str] = set()
        records: list[dict] = []
        try:
            while queue and len(visited) < args.max_pages_per_root:
                url, depth = queue.popleft()
                if url in visited:
                    continue
                visited.add(url)
                record, children = await inspect(page, url, args.timeout_ms, args.render_wait_ms)
                record.update({"root_url": root, "depth": depth})
                records.append(record)
                if depth < args.max_depth:
                    for child in children:
                        if child not in queued:
                            queued.add(child)
                            queue.append((child, depth + 1))
                if args.pause_seconds:
                    await asyncio.sleep(args.pause_seconds)
        finally:
            await page.close()
        return records


def write_outputs(args, roots: list[str], records: list[dict]) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    by_url: dict[str, dict] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        year = record.get("published_year")
        if not isinstance(year, int) or not args.start_year <= year <= args.end_year:
            continue
        if record.get("body_chars", 0) < args.min_body_chars:
            continue
        # Prefer the shallowest observation and retain only one row per URL.
        old = by_url.get(record["url"])
        if old is None or (record.get("depth", 99), record.get("root_url", "")) < (
            old.get("depth", 99), old.get("root_url", "")
        ):
            by_url[record["url"]] = record

    year_roots: dict[int, list[str]] = defaultdict(list)
    for url, record in by_url.items():
        year_roots[int(record["published_year"])].append(url)
    for year in year_roots:
        year_roots[year].sort()
        path = output.parent / f"sina_roots_{year}.txt"
        path.write_text("\n".join(year_roots[year]) + "\n", encoding="utf-8")

    catalog_path = output.parent / f"{output.stem}_catalog.csv"
    fields = ["published_year", "url", "title", "body_chars", "depth", "root_url", "checked_at"]
    with catalog_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in sorted(by_url.values(), key=lambda x: (x["published_year"], x["url"])):
            writer.writerow({field: record.get(field, "") for field in fields})

    payload = {
        "source_roots": roots,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "records_crawled": len(records),
        "unique_year_roots": len(by_url),
        "year_counts": {str(year): len(year_roots.get(year, [])) for year in range(args.start_year, args.end_year + 1)},
        "year_root_files": {str(year): str(output.parent / f"sina_roots_{year}.txt") for year in sorted(year_roots)},
        "records": records,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "catalog_output": str(catalog_path), "records_crawled": len(records), "unique_year_roots": len(by_url), "year_counts": payload["year_counts"]}, ensure_ascii=False))


async def main(args) -> None:
    roots = []
    for line in Path(args.root_file).read_text(encoding="utf-8").splitlines():
        url = canonical_url(line)
        if url and url not in roots:
            roots.append(url)
    if not roots:
        raise ValueError("root file 中没有合法的新浪公开文章 URL")
    semaphore = asyncio.Semaphore(args.concurrency)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not args.headed)
        try:
            chunks = await asyncio.gather(*(crawl_root(browser, root, args, semaphore) for root in roots))
        finally:
            await browser.close()
    write_outputs(args, roots, [row for chunk in chunks for row in chunk])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从2026热点公开页面构建按年份分层的新浪根URL")
    parser.add_argument("--root-file", required=True, help="每行一个独立的2026热点新浪文章URL")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--max-pages-per-root", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--pause-seconds", type=float, default=0.3)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--render-wait-ms", type=int, default=500)
    parser.add_argument("--min-body-chars", type=int, default=500)
    parser.add_argument("--output", required=True)
    parser.add_argument("--headed", action="store_true")
    asyncio.run(main(parser.parse_args()))
