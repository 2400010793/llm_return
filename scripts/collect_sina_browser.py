"""Low-rate browser collector for a bounded Sina Finance article probe.

It collects visible listing links, opens article pages, and stores useful
records only when title, publication time, and a sufficiently long article body
are available. It does not call private APIs or bypass access controls.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from playwright.async_api import TimeoutError as PlaywrightTimeoutError, async_playwright

BASE_URL = "https://finance.sina.com.cn/roll/c/56592.shtml"
BLOCK_MARKERS = ("验证码", "访问异常", "请求过于频繁", "安全验证", "captcha")
STOCK_CODE_RE = re.compile(r"/realstock/company/(?:sh|sz)(\d{6})/", re.I)


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def article_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]


async def listing(page, page_number: int) -> list[dict[str, str]]:
    url = BASE_URL if page_number == 1 else f"{BASE_URL}?page={page_number}"
    await page.goto(url, wait_until="domcontentloaded")
    await page.wait_for_timeout(700)
    items = await page.locator('a[href*="finance.sina.com.cn/"]').evaluate_all(
        """els => { const seen = new Set(); return els.map(a => ({
        title: (a.innerText || '').trim(), url: a.href
        })).filter(x => x.title && /\\/20\\d{2}-\\d{2}-\\d{2}\\//.test(x.url)
        && !seen.has(x.url) && (seen.add(x.url), true)); }"""
    )
    return items


async def collect_article(page, item: dict[str, str], max_body_chars: int) -> dict:
    await page.goto(item["url"], wait_until="domcontentloaded")
    await page.wait_for_timeout(500)
    body_text = await page.locator("body").inner_text()
    lowered = body_text.lower()
    marker = next((x for x in BLOCK_MARKERS if x.lower() in lowered), None)
    if marker:
        return {"status": "blocked", "url": item["url"], "error": marker}
    meta = await page.locator("meta").evaluate_all(
        """els => Object.fromEntries(els.map(e => [
        e.getAttribute('property') || e.getAttribute('name'), e.getAttribute('content')
        ]).filter(x => x[0] && x[1]))"""
    )
    body_locator = page.locator("#artibody, #article_content, .article-content").first
    if await body_locator.count():
        body = await body_locator.inner_text()
    else:
        body = body_text
    body = clean(body)
    title = clean(meta.get("og:title") or await page.title() or item["title"])
    published_at = meta.get("article:published_time") or meta.get("bytedance:published_time") or meta.get("publishdate")
    stock_links = await body_locator.locator('a[href*="realstock/company/"]').evaluate_all(
        """els => { const seen = new Set(); return els.map(a => ({
        stock_name: (a.innerText || '').trim(), url: a.href
        })).map(x => { const m=x.url.match(/\\/(?:sh|sz)(\\d{6})\\//i); return m ? {...x, stock_id:m[1], stock_match_method:'sina_stock_link'} : null; })
        .filter(x => x && x.stock_name && x.stock_name !== '最近访问'
        && !seen.has(x.stock_id) && (seen.add(x.stock_id), true)); }"""
    ) if await body_locator.count() else []
    return {
        "status": "ok" if published_at and len(body) >= 120 else "low_quality",
        "source": "sina_finance",
        "content_type": "financial_news",
        "article_id": article_id(item["url"]),
        "url": item["url"],
        "title": title,
        "published_at": published_at,
        "body_chars": len(body),
        "body": body[:max_body_chars],
        "stock_matches": stock_links,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


async def main_async(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    seen: set[str] = set()
    started = datetime.now(timezone.utc)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=not args.headed)
        page = await browser.new_page()
        for page_number in range(1, args.max_pages + 1):
            items = await listing(page, page_number)
            for item in items:
                if item["url"] in seen or len(records) >= args.limit:
                    continue
                seen.add(item["url"])
                try:
                    record = await collect_article(page, item, args.max_body_chars)
                except PlaywrightTimeoutError as exc:
                    record = {"status": "timeout", "url": item["url"], "error": str(exc)}
                records.append(record)
                if args.pause_seconds:
                    await asyncio.sleep(args.pause_seconds)
            if len(records) >= args.limit:
                break
        await browser.close()
    payload = {"source": "sina_finance", "listing_pages": args.max_pages, "records": records, "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds()}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    good = [x for x in records if x.get("status") == "ok"]
    print(json.dumps({"output": str(output), "attempted": len(records), "useful": len(good), "elapsed_seconds": round(payload["elapsed_seconds"], 2), "useful_rate": round(len(good) / len(records), 3) if records else 0}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    parser.add_argument("--max-body-chars", type=int, default=50000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--headed", action="store_true")
    asyncio.run(main_async(parser.parse_args()))
