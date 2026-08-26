"""Collect a bounded set of visible CNINFO announcements.

This is an official-source announcement collector, not a general web scraper.
It uses the visible stock disclosure page, applies the date filter through the
page controls, reads only linked detail pages, and stops on access-control or
navigation failures. It does not call hidden APIs, bypass login/CAPTCHA, or
retry failed requests.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.async_api import APIRequestContext, Page, TimeoutError as PlaywrightTimeoutError, async_playwright


BLOCK_MARKERS = ("验证码", "访问异常", "请求过于频繁", "安全验证", "captcha")
DEFAULT_PAUSE = 5.0
FOCUS_TITLE_PATTERN = re.compile(
    r"业绩|年报|半年报|季报|季度报告|业绩预告|业绩快报|利润分配|分红|回购|增持|减持|股东减持|"
    r"重大合同|合同|中标|投资|收购|资产重组|并购|关联交易|担保|诉讼|仲裁|处罚|监管|问询|"
    r"风险提示|停牌|复牌|发行|定增|配股|可转债|债券|股东大会|董事会|监事会|换届|辞职|聘任|审计",
)


class AccessControlError(RuntimeError):
    pass


def build_output_payload(
    args: argparse.Namespace, records: list[dict], collected_at: str,
) -> dict:
    has_collection_error = any(
        isinstance(record, dict)
        and record.get("content_type") == "collection_error"
        for record in records
    )
    return {
        "status": "partial" if has_collection_error else "complete",
        "query": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "index_only": bool(args.index_only),
            "focus_only": bool(args.focus_only),
        },
        "collected_at": collected_at,
        "records": records,
    }


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stable_key(value: str) -> str:
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()


def canonical_detail_url(value: str) -> str:
    value = value.strip()
    if value.startswith("https://www.cninfo.com.cn"):
        return value[len("https://www.cninfo.com.cn"):]
    if value.startswith("http://www.cninfo.com.cn"):
        return value[len("http://www.cninfo.com.cn"):]
    return value


def parse_stock(value: str) -> dict[str, str]:
    parts = [part.strip() for part in value.split(":", 2)]
    if len(parts) != 3 or not re.fullmatch(r"\d{6}", parts[0]):
        raise ValueError("stock must use CODE:ORG_ID:NAME, for example 000001:gssz0000001:平安银行")
    return {"stock_id": parts[0], "org_id": parts[1], "stock_name": parts[2]}


async def ensure_allowed(page: Page) -> None:
    text = (await page.locator("body").inner_text()).lower()
    marker = next((item for item in BLOCK_MARKERS if item.lower() in text), None)
    if marker:
        raise AccessControlError(f"{marker} detected at {page.url}")


async def set_date_input(page: Page, placeholder: str, value: str) -> None:
    field = page.locator(f'input[placeholder="{placeholder}"]').first
    await field.wait_for(state="visible", timeout=15000)
    await field.click(force=True)
    await field.press("Control+A")
    await field.type(value, delay=80)
    await field.press("Enter")
    await page.wait_for_timeout(500)


async def apply_filter(page: Page, start_date: str, end_date: str) -> None:
    await set_date_input(page, "开始日期", start_date)
    await set_date_input(page, "结束日期", end_date)
    await page.keyboard.press("Escape")
    query_button = page.get_by_role("button", name="查询", exact=True)
    await query_button.wait_for(state="visible", timeout=15000)
    await query_button.click(force=True)
    await page.wait_for_timeout(3000)
    await ensure_allowed(page)


async def open_stock_page(page: Page, url: str) -> None:
    """Open a visible stock page with recovery for slow CNINFO navigation.

    The site can leave the initial document load pending while the Vue page is
    already usable.  The date-range control is the readiness signal, not the
    ``domcontentloaded`` event.
    """
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            await page.goto(url, wait_until="commit", timeout=15000)
        except PlaywrightTimeoutError as error:
            last_error = error
        try:
            await page.wait_for_selector('input[placeholder="开始日期"]', state="visible", timeout=20000)
            return
        except PlaywrightTimeoutError as error:
            last_error = error
            await page.goto("about:blank", wait_until="commit", timeout=10000)
            await asyncio.sleep(2.0 * (attempt + 1))
    raise PlaywrightTimeoutError(f"CNINFO stock page did not expose date controls after 3 attempts: {url}; {last_error}")


async def resolve_stock_by_visible_field(page: Page, stock: dict[str, str]) -> str:
    """Resolve a stock page through CNINFO's visible company-switch field.

    This avoids relying only on locally inferred organization IDs. The field
    accepts a code or company name and the selected visible result determines
    the organization ID used by the announcement page.
    """
    seed = "https://www.cninfo.com.cn/new/disclosure/stock?stockCode=000001&orgId=gssz0000001#latestAnnouncement"
    await page.goto(seed, wait_until="domcontentloaded")
    await page.wait_for_timeout(1500)
    field = page.locator('input[placeholder="请输入您要切换公司的代码、简称、拼音"]').first
    if not await field.count():
        raise PlaywrightTimeoutError("visible company-switch field not found")
    await field.fill(stock["stock_id"])
    await page.wait_for_timeout(800)
    await field.press("Enter")
    await page.wait_for_timeout(1800)
    parsed = parse_qs(urlparse(page.url).query)
    resolved_code = parsed.get("stockCode", [""])[0]
    resolved_org = parsed.get("orgId", [""])[0]
    if resolved_code != stock["stock_id"] or not resolved_org:
        raise RuntimeError(f"visible stock lookup did not resolve {stock['stock_id']}: {page.url}")
    stock["org_id"] = resolved_org
    return page.url


async def visible_rows(page: Page) -> list[dict[str, str]]:
    links = await page.locator('a[href*="/new/disclosure/detail"]').all()
    result: list[dict[str, str]] = []
    for link in links:
        row = link.locator("xpath=ancestor::tr[1]")
        title = (await link.inner_text()).strip()
        href = await link.get_attribute("href")
        text = (await row.inner_text()).strip()
        date_match = re.search(r"20\d{2}-\d{2}-\d{2}", text)
        if title and href and date_match:
            result.append({"title": title, "url": href, "announcement_date": date_match.group(0)})
    return result


async def all_visible_rows(page: Page, max_pages: int, pause: float) -> list[dict[str, str]]:
    """Read the visible pagination pages without using hidden endpoints.

    CNINFO currently renders 30 rows per page. ``max_pages=0`` means continue
    until the visible next button is disabled; the detail limit remains a
    separate safety bound.
    """
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    page_no = 0
    while True:
        page_no += 1
        for item in await visible_rows(page):
            if item["url"] not in seen:
                seen.add(item["url"])
                rows.append(item)
        if max_pages and page_no >= max_pages:
            break
        next_button = page.locator(".el-pagination button.btn-next").first
        if not await next_button.count() or await next_button.is_disabled():
            break
        await asyncio.sleep(pause)
        await next_button.click(force=True)
        await page.wait_for_timeout(1200)
    return rows


def extract_pdf_text(payload: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(payload))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception:
        return ""


async def read_detail(
    page: Page,
    request: APIRequestContext,
    stock: dict[str, str],
    item: dict[str, str],
    pause: float,
    raw_dir: Path,
) -> dict:
    await asyncio.sleep(pause)
    url = item["url"] if item["url"].startswith("http") else f"https://www.cninfo.com.cn{item['url']}"
    await page.goto(url, wait_until="commit", timeout=15000)
    await page.wait_for_selector("body", state="attached", timeout=15000)
    await page.wait_for_timeout(300)
    await ensure_allowed(page)
    body = (await page.locator("body").inner_text()).strip()
    pdf_link = page.locator('a[href$=".PDF"], a[href$=".pdf"]').first
    pdf_url = await pdf_link.get_attribute("href") if await pdf_link.count() else None
    pdf_text = ""
    pdf_path = None
    pdf_error = None
    if pdf_url:
        if pdf_url.startswith("/"):
            pdf_url = f"https://www.cninfo.com.cn{pdf_url}"
        await asyncio.sleep(pause)
        try:
            response = await request.get(pdf_url, timeout=30000)
            if not response.ok:
                pdf_error = f"pdf download returned HTTP {response.status}"
            else:
                payload = await response.body()
                safe_name = f"{stock['stock_id']}_{item['announcement_date']}_{stable_key(item['url'])[:16]}.pdf"
                destination = raw_dir / safe_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
                pdf_path = str(destination)
                pdf_text = extract_pdf_text(payload)
                if not pdf_text:
                    pdf_error = "pdf downloaded but text extraction returned empty"
        except Exception as error:
            pdf_error = f"pdf download failed; no retry: {error}"
    return {
        "source": "cninfo",
        "content_type": "announcement",
        "stock_relation": "direct",
        **stock,
        **item,
        "url": url,
        "body": body[:50000],
        "pdf_url": pdf_url,
        "pdf_path": pdf_path,
        "pdf_text": pdf_text[:200000],
        "pdf_error": pdf_error,
        "published_at": f"{item['announcement_date']}T00:00:00+08:00",
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


async def collect_stock(page: Page, request: APIRequestContext, stock: dict[str, str], args: argparse.Namespace) -> list[dict]:
    if args.resolve_stock:
        url = await resolve_stock_by_visible_field(page, stock)
    else:
        url = f"https://www.cninfo.com.cn/new/disclosure/stock?stockCode={stock['stock_id']}&orgId={stock['org_id']}#latestAnnouncement"
        await open_stock_page(page, url)
    await page.wait_for_timeout(1500)
    await ensure_allowed(page)
    await apply_filter(page, args.start_date, args.end_date)
    items = await all_visible_rows(page, args.max_pages, args.page_pause_seconds)
    if args.focus_only:
        items = [item for item in items if FOCUS_TITLE_PATTERN.search(item["title"])]
    existing_urls = getattr(args, "existing_urls", set())
    items = [item for item in items if canonical_detail_url(item["url"]) not in existing_urls]
    if args.limit:
        remaining = max(0, args.limit - len(getattr(args, "existing_records", [])))
        items = items[:remaining]
    if args.index_only:
        collected_at = datetime.now(timezone.utc).isoformat()
        return [
            {
                "source": "cninfo",
                "content_type": "announcement_index",
                "stock_relation": "direct",
                **stock,
                **item,
                "url": item["url"] if item["url"].startswith("http") else f"https://www.cninfo.com.cn{item['url']}",
                "published_at": f"{item['announcement_date']}T00:00:00+08:00",
                "collected_at": collected_at,
            }
            for item in items
        ]
    records: list[dict] = []
    for item in items:
        try:
            records.append(await read_detail(page, request, stock, item, args.pause_seconds, Path(args.raw_dir)))
        except PlaywrightTimeoutError:
            records.append({"source": "collector", "content_type": "collection_error", **stock, **item, "error": "detail timeout; no retry"})
    return records


async def main(args: argparse.Namespace) -> None:
    if args.pause_seconds < 0.1:
        raise ValueError("pause-seconds must be at least 0.1")
    stocks = [parse_stock(value) for value in args.stock]
    output = Path(args.output)
    manifest_path = Path(args.manifest)
    existing_records: list[dict] = []
    if output.exists():
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("records"), list):
                existing_records = payload["records"]
        except (OSError, json.JSONDecodeError):
            existing_records = []
    args.existing_records = existing_records
    args.existing_urls = {
        canonical_detail_url(record["url"])
        for record in existing_records
        if record.get("url")
    }
    previous = set(json.loads(manifest_path.read_text(encoding="utf-8")).get("keys", [])) if manifest_path.exists() else set()
    collected_at = datetime.now(timezone.utc).isoformat()
    records: list[dict] = list(existing_records)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=not args.headed)
        page = await browser.new_page()
        request = await playwright.request.new_context(
            user_agent="llm-return-research/0.1 (academic prototype)",
            extra_http_headers={"Accept": "application/pdf"},
        )
        try:
            for index, stock in enumerate(stocks):
                if index:
                    await asyncio.sleep(args.stock_pause_seconds)
                records.extend(await collect_stock(page, request, stock, args))
        except AccessControlError as error:
            records.append({"source": "collector", "content_type": "collection_error", "error_type": "access_control", "error": str(error), "collected_at": collected_at})
        finally:
            await request.dispose()
            await browser.close()
    unique: list[dict] = list(existing_records)
    keys = set(previous)
    for record in records[len(existing_records):]:
        key = stable_key(record.get("url", "") or json.dumps(record, ensure_ascii=False, sort_keys=True))
        if key not in keys:
            keys.add(key)
            unique.append(record)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Escape non-ASCII characters in JSON on disk/stdout.  Some CNINFO pages
    # contain lone UTF-16 surrogates; ensure_ascii=False would make UTF-8
    # serialization fail and abort an otherwise valid stock collection.
    payload = build_output_payload(args, unique, collected_at)
    write_json_atomic(output, payload)
    manifest_path.write_text(json.dumps({"keys": sorted(keys)}, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "records": len(unique), "manifest": str(manifest_path),
    }, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", action="append", required=True, help="CODE:ORG_ID:NAME; repeat for multiple stocks")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD, exclusive")
    parser.add_argument("--limit", type=int, default=0, help="maximum detail pages per stock; 0 means unlimited")
    parser.add_argument("--index-only", action="store_true", help="collect visible announcement indexes without opening detail pages or downloading PDFs")
    parser.add_argument("--focus-only", action="store_true", help="download only research-priority announcement titles")
    parser.add_argument("--resolve-stock", action="store_true", help="resolve stock code through CNINFO's visible company field before filtering announcements")
    parser.add_argument("--pause-seconds", type=float, default=DEFAULT_PAUSE)
    parser.add_argument("--page-pause-seconds", type=float, default=3.0, help="pause between visible announcement list pages")
    parser.add_argument("--max-pages", type=int, default=0, help="maximum visible list pages; 0 means all pages")
    parser.add_argument("--stock-pause-seconds", type=float, default=20.0)
    parser.add_argument("--output", default="data/interim/cninfo_announcements.json")
    parser.add_argument("--manifest", default="data/interim/collector_manifest_cninfo.json")
    parser.add_argument("--raw-dir", default="data/raw/cninfo", help="保存页面可见 PDF 的原始文件")
    parser.add_argument("--headed", action="store_true")
    parsed = parser.parse_args()
    if parsed.limit < 0:
        parser.error("--limit must be non-negative; 0 means unlimited")
    if parsed.page_pause_seconds < 0.1 or parsed.max_pages < 0:
        parser.error("--page-pause-seconds must be at least 0.1 and --max-pages cannot be negative")
    asyncio.run(main(parsed))
