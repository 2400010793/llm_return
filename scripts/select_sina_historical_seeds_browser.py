"""Discover and select high-quality historical Sina article seeds with a browser.

The script follows only links rendered in public HTML pages. It is intended for
small, auditable seed discovery, not unrestricted crawling: it respects a page
limit, a delay, and the user's supplied roots. Candidates are ranked to prefer
exactly one stock relation, long article bodies, target-year dates, and diverse
quarters. It does not call private APIs or bypass robots, CAPTCHAs, logins, or
other access controls.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urlparse

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError, async_playwright

ALLOWED_HOSTS = {"finance.sina.com.cn", "cj.sina.com.cn"}
ARTICLE_URL_RE = re.compile(
    r"^https?://(?:finance\.sina\.com\.cn|cj\.sina\.com\.cn)/"
    r"(?:t|s|e|y|roll|stock|view|globe|special|focus|wm|jjxw|jhzx)/"
    r"[^?#]+\.(?:html|shtml)$",
    re.I,
)
ROOT_ARTICLE_URL_RE = re.compile(r"^https?://(?:finance\.sina\.com\.cn|cj\.sina\.com\.cn)/(?:[^/?#]+/)+20\d{2}(?:-\d{2}-\d{2}|\d{4,8})/[^?#]+\.(?:html|shtml)$|^https?://(?:finance\.sina\.com\.cn|cj\.sina\.com\.cn)/(?:[^/?#]+/)+20\d{2}(?:-\d{2}-\d{2})/[^?#]+\.shtml$", re.I)
DATE_RE = re.compile(r"(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日(?:[^0-9]{0,12}(\d{1,2}):(\d{2}))?")
ISO_DATE_RE = re.compile(r"(20\d{2})[-/]([0-9]{1,2})[-/]([0-9]{1,2})(?:[^0-9]{0,12}([0-9]{1,2}):([0-9]{2}))?")
STOCK_LINK_RE = re.compile(r"/(?:realstock/company/)(?:sh|sz)(\d{6})/", re.I)
LEGACY_STOCK_RE = re.compile(r"[?&]symbol=(?:sh|sz)?(\d{6})", re.I)
BLOCK_MARKERS = ("验证码", "访问异常", "请求过于频繁", "安全验证", "captcha")
NAVIGATION_MARKERS = ("新浪财经_", "股票首页", "港股", "美股", "新浪期货", "财经面对面", "个人理财", "保险频道", "财经会议")


def canonical_url(url: str) -> str | None:
    url = urldefrag(url)[0]
    parsed = urlparse(url)
    if parsed.hostname not in ALLOWED_HOSTS or not (ARTICLE_URL_RE.fullmatch(url) or ROOT_ARTICLE_URL_RE.fullmatch(url)):
        return None
    return url.replace("http://", "https://", 1)


def article_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]


def body_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()


def parse_date(text: str) -> tuple[str | None, int | None, int | None]:
    match = DATE_RE.search(text)
    if not match:
        match = ISO_DATE_RE.search(text)
    if not match:
        return None, None, None
    year, month, day, hour, minute = match.groups()
    value = f"{year}-{int(month):02d}-{int(day):02d}"
    if hour and minute:
        value += f" {int(hour):02d}:{minute}"
    return value, int(year), int(month)


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def path_prefix(url: str) -> str:
    return urlparse(url).path.strip("/").split("/", 1)[0]


def stock_ids(html_text: str, visible_text: str, stock_catalog: dict[str, str]) -> list[str]:
    values = set(STOCK_LINK_RE.findall(html_text)) | set(LEGACY_STOCK_RE.findall(html_text))
    combined = f"{visible_text} {html_text}"
    for code, name in stock_catalog.items():
        if name and name in combined:
            values.add(code)
    return sorted(values)


async def rendered_page(page: Page, url: str, timeout_ms: int, render_wait_ms: int) -> dict:
    await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    if render_wait_ms:
        await page.wait_for_timeout(render_wait_ms)
    html_text = await page.content()
    visible_text = await page.locator("body").inner_text()
    title = clean_text(await page.title())
    return {"html": html_text, "text": clean_text(visible_text), "title": title}


async def inspect(page: Page, url: str, parent_url: str | None, depth: int, args, catalog: dict[str, str], discovery_method: str = "related_link") -> tuple[dict, list[str]]:
    base = {"url": url, "parent_url": parent_url, "depth": depth, "discovery_method": discovery_method, "checked_at": datetime.now(timezone.utc).isoformat()}
    try:
        page_data = await rendered_page(page, url, args.timeout_ms, args.render_wait_ms)
        text = page_data["text"]
        title = page_data["title"]
        if any(marker.lower() in text.lower() for marker in BLOCK_MARKERS):
            return {**base, "status": "blocked_marker", "title": title}, []
        links = await page.locator("a").evaluate_all("els => els.map(a => a.href)")
        children = list(dict.fromkeys(filter(None, (canonical_url(x) for x in links))))
        published_at, year, month = parse_date(text)
        stocks = stock_ids(page_data["html"], text, catalog)
        body_chars = len(text)
        model_text = text[: args.max_body_chars]
        navigation = any(marker.lower() in title.lower() for marker in NAVIGATION_MARKERS)
        valid = bool(year and published_at and body_chars >= args.min_body_chars and not navigation)
        quality = "valid" if valid else "low_quality"
        if valid and body_chars < args.long_body_chars:
            quality = "short_news"
        record = {**base, "status": "ok", "quality_status": quality, "title": title,
                  "published_at": published_at, "published_year": year, "published_month": month,
                  "body_chars": body_chars, "body_chars_for_model": len(model_text),
                  "body_truncated": len(model_text) < body_chars, "body_sha256": body_hash(text), "stock_ids": stocks,
                  "model_text": model_text,
                  "stock_count": len(stocks), "path_prefix": path_prefix(url), "links_found": len(children)}
        return record, children
    except PlaywrightTimeoutError:
        return {**base, "status": "timeout"}, []
    except Exception as exc:
        return {**base, "status": "error", "error": f"{type(exc).__name__}: {exc}"}, []


def rank(candidate: dict, target_year: int) -> tuple:
    exact_one = candidate["stock_count"] == 1
    target = candidate.get("published_year") == target_year
    long_body = candidate.get("body_chars", 0) >= 800
    # Exact-one stock is deliberately the dominant criterion.
    return (int(exact_one), int(target), int(long_body), min(candidate.get("body_chars", 0), 5000), -candidate.get("depth", 99))


GENERIC_TITLE_TOKENS = {
    "关于", "公司", "市场", "证券", "股票", "股市", "中国", "年度", "新闻",
    "十大", "盘点", "公告", "记者", "新浪", "财经", "分析", "评论", "今日",
}


def title_tokens(title: str) -> set[str]:
    """Return coarse Chinese/Latin title tokens for diversity scoring."""
    tokens: set[str] = set()
    for value in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}", title or ""):
        if re.fullmatch(r"[\u4e00-\u9fff]+", value):
            tokens.update(value[index:index + 2] for index in range(len(value) - 1))
        elif value not in GENERIC_TITLE_TOKENS:
            tokens.add(value.lower())
    return {value for value in tokens if value not in GENERIC_TITLE_TOKENS}


def title_overlap(left: dict, right: dict) -> float:
    a = title_tokens(left.get("title", ""))
    b = title_tokens(right.get("title", ""))
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def diversity_gain(candidate: dict, chosen: list[dict], stock_counts: dict[str, int], used_paths: set[str], used_quarters: set[int]) -> tuple:
    """Score a candidate for farthest-first, low-relatedness selection."""
    stock = candidate["stock_ids"][0] if candidate.get("stock_count") == 1 else None
    quarter = (candidate.get("published_month", 1) - 1) // 3 + 1
    max_overlap = max((title_overlap(candidate, item) for item in chosen), default=0.0)
    return (
        int(stock is not None and stock not in stock_counts),
        int(candidate.get("path_prefix") not in used_paths),
        int(quarter not in used_quarters),
        -max_overlap,
        *rank(candidate, candidate.get("published_year", 0)),
    )


def select_seeds(
    records: list[dict],
    years: list[int],
    per_year: int,
    max_per_stock: int,
    single_stock_only: bool = True,
) -> list[dict]:
    usable = [
        x for x in records
        if x.get("quality_status") in {"valid", "short_news"}
        and x.get("published_year") in years
        and (not single_stock_only or x.get("stock_count") == 1)
    ]
    by_year: dict[int, list[dict]] = defaultdict(list)
    for item in usable:
        by_year[item["published_year"]].append(item)
    selected: list[dict] = []
    for year in years:
        pool = sorted(by_year[year], key=lambda x: rank(x, year), reverse=True)
        chosen: list[dict] = []
        seen_hashes: set[str] = set()
        stock_counts: dict[str, int] = defaultdict(int)
        used_paths: set[str] = set()
        used_quarters: set[int] = set()
        # Greedy farthest-first selection: prefer a new stock, URL section,
        # quarter, and title vocabulary instead of near-duplicate summaries.
        remaining = list(pool)
        while remaining and len(chosen) < per_year:
            eligible = []
            for item in remaining:
                if item["body_sha256"] in seen_hashes:
                    continue
                stock = item["stock_ids"][0] if item["stock_count"] == 1 else None
                if stock and stock_counts[stock] >= max_per_stock:
                    continue
                eligible.append(item)
            if not eligible:
                break
            item = max(eligible, key=lambda value: diversity_gain(value, chosen, stock_counts, used_paths, used_quarters))
            remaining.remove(item)
            chosen.append(item)
            seen_hashes.add(item["body_sha256"])
            stock = item["stock_ids"][0] if item["stock_count"] == 1 else None
            if stock:
                stock_counts[stock] += 1
            used_paths.add(item.get("path_prefix", ""))
            used_quarters.add((item.get("published_month", 1) - 1) // 3 + 1)
        for index, item in enumerate(chosen, 1):
            item = dict(item)
            item["seed_id"] = f"{year}_{index:02d}"
            item["selection_score"] = list(rank(item, year))
            item["selection_strategy"] = "farthest_first_title_stock_path_quarter"
            item["selection_status"] = "accepted" if len(chosen) >= per_year else "accepted_under_quota"
            selected.append(item)
    return selected


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["seed_id", "published_year", "published_month", "path_prefix", "title", "url", "published_at", "stock_ids", "stock_count", "body_chars", "body_sha256", "depth", "parent_url", "quality_status", "selection_status"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            csv_row = {field: row.get(field, "") for field in fields}
            csv_row["stock_ids"] = ",".join(row.get("stock_ids", []))
            writer.writerow(csv_row)


async def main(args: argparse.Namespace) -> None:
    years = sorted(set(args.years))
    candidate_metadata: dict[str, dict] = {}
    candidate_urls: list[str] = []
    if args.candidate_file:
        payload = json.loads(Path(args.candidate_file).read_text(encoding="utf-8"))
        for item in payload.get("records", []):
            url = canonical_url(item.get("url", ""))
            if url:
                candidate_urls.append(url)
                candidate_metadata[url] = item
    root_values = list(args.root or [])
    if args.root_file:
        root_values.extend(Path(args.root_file).read_text(encoding="utf-8").splitlines())
    roots = [canonical_url(x.strip()) for x in root_values if x.strip()]
    roots.extend(candidate_urls)
    roots = list(dict.fromkeys(roots))
    roots = [x for x in roots if x]
    if not roots:
        raise ValueError("至少需要一个合法的新浪历史文章 --root、--root-file 或 --candidate-file")
    catalog: dict[str, str] = {}
    if args.stock_catalog:
        with Path(args.stock_catalog).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("stock_id") and row.get("stock_name"):
                    catalog[row["stock_id"].zfill(6)] = row["stock_name"]
    queue: asyncio.Queue[tuple[str, str | None, int] | None] = asyncio.Queue()
    queued = set(roots)
    visited: set[str] = set()
    records: list[dict] = []
    stream_output = Path(args.stream_output or (str(args.output) + ".records.jsonl"))
    stream_output.parent.mkdir(parents=True, exist_ok=True)
    stream_handle = stream_output.open("w", encoding="utf-8")
    for root in roots:
        queue.put_nowait((root, None, 0))
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not args.headed)
        pages = [await browser.new_page() for _ in range(args.workers)]

        async def worker(page: Page) -> None:
            while True:
                job = await queue.get()
                if job is None:
                    queue.task_done()
                    return
                url, parent, depth = job
                if len(visited) >= args.max_pages or url in visited:
                    queue.task_done()
                    continue
                visited.add(url)
                discovery_method = candidate_metadata.get(url, {}).get("discovery_method", "related_link")
                record, children = await inspect(page, url, parent, depth, args, catalog, discovery_method)
                records.append(record)
                # Write and flush immediately: a completed page is durable even
                # if a later page or the browser process fails.
                stream_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream_handle.flush()
                if depth < args.max_depth:
                    for child in children:
                        if child not in queued:
                            queued.add(child)
                            queue.put_nowait((child, url, depth + 1))
                queue.task_done()
                if args.pause_seconds:
                    await asyncio.sleep(args.pause_seconds)

        tasks = [asyncio.create_task(worker(page)) for page in pages]
        try:
            await queue.join()
        finally:
            for _ in tasks:
                queue.put_nowait(None)
            await asyncio.gather(*tasks, return_exceptions=True)
            # A completed Chromium driver can disconnect during shutdown on
            # Windows.  The collected records are still valid, so shutdown
            # failure must not discard the output or turn a successful crawl
            # into a failed run.
            try:
                await browser.close()
            except Exception:
                pass
            stream_handle.close()
    selected = select_seeds(
        records,
        years,
        args.per_year,
        args.max_per_stock,
        single_stock_only=not args.allow_multi_stock,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"years": years, "per_year": args.per_year, "visited": len(visited), "records": records, "selected": selected}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(Path(args.catalog_output), selected)
    summary = {str(year): sum(1 for x in selected if x.get("published_year") == year) for year in years}
    print(json.dumps({"output": str(output), "stream_output": str(stream_output), "catalog_output": args.catalog_output, "workers": args.workers, "visited": len(visited), "selected": summary, "single_stock_selected": sum(x.get("stock_count") == 1 for x in selected), "single_stock_only": not args.allow_multi_stock}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用可见浏览器页面发现并选择新浪历史文章种子")
    parser.add_argument("--root", action="append", default=[], help="历史新浪文章 URL，可重复")
    parser.add_argument("--root-file", help="每行一个历史新浪文章 URL")
    parser.add_argument("--candidate-file", help="候选 URL JSON 文件；格式为 records[].url，可与 --root 同时使用")
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--per-year", type=int, default=20)
    parser.add_argument("--max-per-stock", type=int, default=1, help="同一股票最多保留多少个 seed；默认 1 以降低相关性")
    parser.add_argument("--allow-multi-stock", action="store_true", help="允许多股票文章作为补足；默认只选择恰好匹配一只股票的文章")
    parser.add_argument("--max-pages", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8, help="并行浏览器页面数；默认 8，避免历史站点压力过高")
    parser.add_argument("--pause-seconds", type=float, default=3.0)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--render-wait-ms", type=int, default=500)
    parser.add_argument("--min-body-chars", type=int, default=500)
    parser.add_argument("--long-body-chars", type=int, default=800)
    parser.add_argument("--max-body-chars", type=int, default=12000, help="种子正文最多保存字符数；超过部分仍保留原始 HTML")
    parser.add_argument("--stock-catalog")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stream-output", help="逐页追加 JSONL；默认 OUTPUT.records.jsonl")
    parser.add_argument("--catalog-output", required=True)
    parser.add_argument("--headed", action="store_true")
    parsed = parser.parse_args()
    if not parsed.root and not parsed.root_file and not parsed.candidate_file:
        parser.error("至少需要一个 --root、--root-file 或 --candidate-file")
    if parsed.workers < 1 or parsed.max_pages < 1 or parsed.max_depth < 0:
        parser.error("workers、max-pages 必须为正数，max-depth 不能为负数")
    asyncio.run(main(parsed))
