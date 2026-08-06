"""Low-rate, visible-content-only browser collector for permitted pages.

This script never logs in, exports cookies, calls hidden APIs, bypasses CAPTCHAs,
or retries access-control failures. The user must provide an authorized browser
storage state explicitly; otherwise pages are opened as an anonymous session.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from playwright.async_api import Page, async_playwright
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError, async_playwright


PAUSE_SECONDS = 3.0
CHINA_TZ = timezone(timedelta(hours=8))
BLOCK_MARKERS = ("验证码", "访问异常", "请求过于频繁", "安全验证", "captcha")
XUEQIU_LOGIN_MARKERS = ("登录", "注册")


class AccessControlError(RuntimeError):
    """A page exposed a recognizable access-control or CAPTCHA response."""

    def __init__(self, *, marker: str, url: str, title: str, excerpt: str) -> None:
        self.marker = marker
        self.url = url
        self.title = title
        self.excerpt = excerpt
        super().__init__(f"{marker} detected at {url}")


def stable_id(*parts: str) -> str:
    value = "|".join(part.strip() for part in parts if part)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def parse_cli_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=CHINA_TZ)
    except ValueError as error:
        raise ValueError(f"invalid date {value!r}; expected YYYY-MM-DD") from error


POSITIVE_EVENT_TERMS = ("业绩预增", "净利润增长", "扭亏", "中标", "订单", "回购", "增持", "分红", "获批", "扩产")
NEGATIVE_EVENT_TERMS = ("业绩预亏", "亏损", "下降", "减持", "立案", "处罚", "违规", "诉讼", "质押风险", "违约", "终止")
MANAGEMENT_TERMS = ("董事长", "总经理", "副总经理", "董事会秘书", "董秘", "高管", "管理层", "辞任", "聘任", "任职")


def event_hint(title: str, body: str = "") -> tuple[str, str]:
    text = f"{title} {body}"
    if any(term in text for term in MANAGEMENT_TERMS):
        event_type = "management_change"
    elif any(term in text for term in ("处罚", "立案", "违规", "诉讼", "违约", "质押")):
        event_type = "regulatory_or_legal_risk"
    elif any(term in text for term in ("业绩", "净利润", "扭亏", "预增", "预亏")):
        event_type = "performance"
    elif any(term in text for term in ("增持", "减持", "回购", "分红", "质押")):
        event_type = "shareholder_activity"
    elif any(term in text for term in ("中标", "订单", "合同", "获批", "扩产")):
        event_type = "contract_or_business"
    else:
        event_type = "other"
    positive = any(term in text for term in POSITIVE_EVENT_TERMS)
    negative = any(term in text for term in NEGATIVE_EVENT_TERMS)
    sentiment_hint = "positive" if positive and not negative else "negative" if negative and not positive else "mixed_or_neutral"
    return event_type, sentiment_hint


def normalize_display_date(value: str | None, collected_at: datetime) -> str | None:
    """Convert Eastmoney's month/day display into a timezone-aware ISO timestamp."""
    if not value:
        return None
    match = re.fullmatch(r"(\d{1,2})月(\d{1,2})日\s+(\d{1,2}):(\d{2})", value.strip())
    if not match:
        return None
    month, day, hour, minute = map(int, match.groups())
    year = collected_at.year
    if month > collected_at.month + 1:
        year -= 1
    return datetime(year, month, day, hour, minute, tzinfo=CHINA_TZ).isoformat()


def extract_relative_time(value: str) -> str | None:
    """Extract a plausible relative time without mistaking digits in usernames for it."""
    matches = re.findall(r"(?<!\d)(\d{1,3}(?:分钟前|小时前|天前))", value)
    for match in reversed(matches):
        number = int(re.match(r"\d+", match).group())
        if number <= 365:
            return match
    return next((item for item in ("今天", "昨天") if item in value), None)


def extract_modified_time(value: str) -> str | None:
    match = re.search(
        r"修改于\s*((?:今天|昨天)\s*\d{1,2}:\d{2}|\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}|20\d{2}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}|\d{1,3}(?:分钟前|小时前|天前))",
        value,
    )
    return match.group(1) if match else None


def record_key(record: dict) -> str:
    # Prefer stable identifiers. Dynamic page text must not make the same URL
    # appear new on every run.
    url = str(record.get("url") or "").strip()
    article_id = str(record.get("article_id") or "").strip()
    content_type = str(record.get("content_type") or "unknown")
    # Keep the stock-page link and its separately fetched detail record.
    # Both records share the same URL, but only the detail record contains
    # ``body``/``published_at_display``.
    if record.get("body") or record.get("published_at_display"):
        content_type = f"{content_type}_detail"
    if url:
        return stable_id(content_type, "url", url)
    if article_id:
        return stable_id(content_type, "article_id", article_id)
    return stable_id(
        content_type,
        "fallback",
        str(record.get("title") or ""),
        str(record.get("body") or record.get("visible_text") or ""),
    )


def load_manifest(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload.get("keys", []))


def save_manifest(path: Path, keys: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"keys": sorted(keys)}, ensure_ascii=False, indent=2), encoding="utf-8")


async def ensure_allowed(page: Page) -> None:
    text = await page.locator("body").inner_text()
    lowered = text.lower()
    marker = next((item for item in BLOCK_MARKERS if item.lower() in lowered), None)
    if marker:
        raise AccessControlError(
            marker=marker,
            url=page.url,
            title=await page.title(),
            excerpt=clean(text)[:500],
        )


async def eastmoney_focus(page: Page) -> list[dict]:
    await page.goto("https://finance.eastmoney.com/yaowen.html", wait_until="domcontentloaded")
    await page.wait_for_timeout(3000)
    await ensure_allowed(page)
    return await page.locator('a[href*="finance.eastmoney.com/a/20"]').evaluate_all(
        """links => { const seen = new Set(); return links.map(a => {
        const li = a.closest('li'); const title = (a.innerText || '').trim();
        const url = a.href; const text = (li?.innerText || '').trim();
        const m = text.match(/\\d{1,2}月\\d{1,2}日\\s+\\d{1,2}:\\d{2}/);
        return {title, url, published_at_display: m ? m[0] : null,
                summary: text.replace(title, '').replace(m ? m[0] : '', '').trim()};
        }).filter(x => x.title && !seen.has(x.url) && (seen.add(x.url), true)).slice(0, 10); }"""
    )


async def eastmoney_news_details(page: Page, listing: list[dict], limit: int = 3) -> list[dict]:
    """Read a small number of linked article pages sequentially."""
    details: list[dict] = []
    for item in listing[:limit]:
        await asyncio.sleep(PAUSE_SECONDS)
        await page.goto(item["url"], wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)
        await ensure_allowed(page)
        text = await page.locator("body").inner_text()
        lines = [clean(line) for line in text.splitlines() if clean(line)]
        title = item["title"]
        start = next((i for i, line in enumerate(lines) if line == title), 0)
        end_markers = ("文章来源：", "责任编辑：", "原标题：")
        body_lines = []
        for line in lines[start + 1 :]:
            if any(line.startswith(marker) for marker in end_markers):
                break
            if line not in {"网友评论", "登录 | 注册", "举报"}:
                body_lines.append(line)
        details.append({**item, "article_id": item["url"].rstrip("/").split("/")[-1].split(".")[0], "body": " ".join(body_lines)})
    return details


async def eastmoney_guba(page: Page, code: str) -> list[dict]:
    await page.goto(f"https://guba.eastmoney.com/list,{code}.html", wait_until="domcontentloaded")
    await page.wait_for_timeout(3000)
    await ensure_allowed(page)
    return await page.locator("table").first.evaluate(
        """table => [...table.querySelectorAll('tr')].slice(1, 11).map(tr => {
        const c = [...tr.querySelectorAll('td')].map(td => (td.innerText || '').trim());
        const a = tr.querySelector('a'); return {read_count: c[0] || null,
        comment_count: c[1] || null, title: c[2] || null, author_display: c[3] || null,
        last_updated: c[4] || null, url: a ? a.href : null}; })"""
    )


async def eastmoney_guba_details(page: Page, rows: list[dict], code: str, limit: int) -> list[dict]:
    """Read a bounded number of visible Guba post pages sequentially."""
    details = []
    for row in rows[:limit]:
        url = row.get("url")
        if not url:
            continue
        await asyncio.sleep(PAUSE_SECONDS)
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)
        await ensure_allowed(page)
        text = await page.locator("body").inner_text()
        lines = [clean(line) for line in text.splitlines() if clean(line)]
        date_match = re.search(r"20\d{2}[-年]\d{1,2}[-月]\d{1,2}(?:日)?\s+\d{1,2}:\d{2}", text)
        details.append({
            "source": "eastmoney",
            "content_type": "forum_post_detail",
            "stock_id": code,
            "title": row.get("title"),
            "url": url,
            "author_display": row.get("author_display"),
            "last_updated": row.get("last_updated"),
            "published_at_display": date_match.group(0) if date_match else row.get("last_updated"),
            "body": " ".join(lines[-80:])[:30000],
            "collected_at": datetime.now(timezone.utc).isoformat(),
        })
    return details


async def xueqiu_feed(page: Page, symbol: str, feed_type: str, max_records: int = 200) -> dict:
    """Collect visible雪球资讯/公告 cards without internal endpoints."""
    labels = {"news": "资讯", "announcement": "公告"}
    if feed_type not in labels:
        raise ValueError("feed_type must be 'news' or 'announcement'")
    await page.goto(f"https://xueqiu.com/S/{symbol}", wait_until="domcontentloaded")
    await page.wait_for_timeout(3000)
    await ensure_allowed(page)
    tab = page.locator("a").filter(has_text=labels[feed_type]).first
    if not await tab.count():
        return {"symbol": symbol, "feed_type": feed_type, "records": [], "url": page.url}
    await tab.evaluate("element => element.click()")
    await page.wait_for_timeout(1200)
    cards = await page.locator("article.timeline__item").evaluate_all(
        "els => els.map(article => ({visible_text: (article.innerText || '').trim(), links: [...article.querySelectorAll('a')].map(a => ({title: (a.innerText || '').trim(), url: a.href}))}))"
    )
    records = []
    seen = set()
    for card in cards[:max_records]:
        text = clean(card.get("visible_text", ""))
        article_url = next((x["url"] for x in card.get("links", []) if re.search(r"/\d+/\d+", x.get("url", ""))), None)
        if len(text) < 30 or not article_url or article_url in seen:
            continue
        seen.add(article_url)
        title = next((x["title"] for x in card.get("links", []) if x.get("title") and x["title"] not in {"转发", "讨论", "收藏", "点赞"}), None)
        records.append({"feed_type": feed_type, "title": title, "url": article_url, "visible_text": text[:12000]})
    return {"symbol": symbol, "feed_type": feed_type, "url": page.url, "record_count": len(records), "records": records}


async def xueqiu_symbol(
    page: Page,
    symbol: str,
    *,
    sort_type: str = "new",
    max_pages: int = 1,
    max_records: int = 100,
) -> dict:
    """Collect public discussion pages using only the visible stock page.

    ``sort_type`` is the visible ``新帖``/``热帖`` tab.  Pagination is driven
    by the rendered links; no internal endpoint or page API is used.
    """
    async def xueqiu_feed(page: Page, symbol: str, feed_type: str, max_records: int = 200) -> dict:
        """Collect visible雪球资讯/公告 cards, without using internal endpoints."""
        labels = {"news": "资讯", "announcement": "公告"}
        if feed_type not in labels:
            raise ValueError("feed_type must be 'news' or 'announcement'")
        await page.goto(f"https://xueqiu.com/S/{symbol}", wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)
        await ensure_allowed(page)
        tab = page.locator("a").filter(has_text=labels[feed_type]).first
        if not await tab.count():
            return {"symbol": symbol, "feed_type": feed_type, "records": [], "url": page.url}
        await tab.evaluate("element => element.click()")
        await page.wait_for_timeout(1200)
        cards = await page.locator("article.timeline__item").evaluate_all(
            """els => els.slice(0, arguments[0]).map(article => {
            const links = [...article.querySelectorAll('a')].map(a => ({title: (a.innerText || '').trim(), url: a.href}));
            return {visible_text: (article.innerText || '').trim(), links};
            })""",
            max_records,
        )
        records = []
        seen = set()
        for card in cards:
            text = clean(card.get("visible_text", ""))
            article_url = next((x["url"] for x in card.get("links", []) if re.search(r"/\d+/\d+", x.get("url", ""))), None)
            if len(text) < 30 or not article_url or article_url in seen:
                continue
            seen.add(article_url)
            title = next((x["title"] for x in card.get("links", []) if x.get("title") and x["title"] not in {"转发", "讨论", "收藏", "点赞"}), None)
            records.append({"feed_type": feed_type, "title": title, "url": article_url, "visible_text": text[:12000]})
        return {"symbol": symbol, "feed_type": feed_type, "url": page.url, "record_count": len(records), "records": records}
    if sort_type not in {"new", "hot"}:
        raise ValueError("sort_type must be 'new' or 'hot'")
    if max_pages < 1 or max_records < 1:
        raise ValueError("max_pages and max_records must be positive")
    await page.goto(f"https://xueqiu.com/S/{symbol}", wait_until="domcontentloaded")
    await page.wait_for_timeout(3000)
    await ensure_allowed(page)
    tab = page.locator("a").filter(has_text="热帖" if sort_type == "hot" else "新帖").first
    if await tab.count():
        await tab.evaluate("element => element.click()")
        await page.wait_for_timeout(1200)
    blocks: list[dict] = []
    seen: set[str] = set()
    page_numbers: list[int] = []
    for page_number in range(1, max_pages + 1):
        text = await page.locator("body").inner_text()
        lines = [clean(line) for line in text.splitlines() if clean(line)]
        discussion_start = next((i for i, line in enumerate(lines) if line == "全部讨论投资者关系交易资讯公告"), None)
        discussion_end = next((i for i, line in enumerate(lines[discussion_start + 1:], discussion_start + 1) if line == "简介"), len(lines)) if discussion_start is not None else 0
        discussion_lines = lines[discussion_start + 1:discussion_end] if discussion_start is not None else []
        current: list[str] = []
        page_blocks: list[str] = []
        for line in discussion_lines:
            if line == "" and current:
                body = " ".join(current).strip()
                if len(body) > 30:
                    page_blocks.append(body)
                current = []
            elif line not in {"新帖热帖", "转发", "讨论", "赞", "收藏", "下一页", "上一页"} and not re.fullmatch(r"\d+", line):
                current.append(line)
        if current:
            body = " ".join(current).strip()
            if len(body) > 30:
                page_blocks.append(body)
        if page_blocks:
            page_numbers.append(page_number)
        authors = await page.locator("article.timeline__item a.user-name").evaluate_all(
            "els => els.map(a => ({name: (a.innerText || '').trim(), url: a.href, href: a.getAttribute('href')}))"
        )
        for index, body in enumerate(page_blocks):
            key = stable_id(body)
            if key in seen:
                continue
            seen.add(key)
            author_match = re.search(r"(?:^|\s)([^\s]+?)(?=修改于|\d{1,2}-\d{1,2}\s|昨天|今天|\d{1,3}(?:分钟前|小时前|天前))", body)
            author = authors[index] if index < len(authors) else {}
            author_url = author.get("url")
            author_id = None
            if author_url:
                author_id_match = re.search(r"/u?/(\d+)$", author_url.rstrip("/"))
                author_id = author_id_match.group(1) if author_id_match else None
            time_display = extract_modified_time(body) or extract_relative_time(body)
            blocks.append({
                "page": page_number,
                "sort_type": sort_type,
                "visible_text": body[:12000],
                "author_name": author.get("name") or (author_match.group(1) if author_match else None),
                "author_id": author_id,
                "author_url": author_url,
                "published_at_display": None if extract_modified_time(body) else time_display,
                "modified_at_display": extract_modified_time(body),
            })
            if len(blocks) >= max_records:
                break
        if len(blocks) >= max_records or page_number == max_pages:
            break
        next_page = page.locator("div.pagination a").filter(has_text=str(page_number + 1)).first
        if not await next_page.count():
            break
        before = text
        await next_page.evaluate("element => element.click()")
        for _ in range(20):
            await page.wait_for_timeout(300)
            if (await page.locator("body").inner_text()) != before:
                break
    links = []
    for anchor in await page.locator("a").all():
        href = await anchor.get_attribute("href")
        label = clean(await anchor.inner_text())
        if href and label and ("/S/" in href or "/statuses/" in href):
            links.append({"title": label, "url": href})
    logged_in = "发帖" in lines and not any(line in XUEQIU_LOGIN_MARKERS for line in lines[:30])
    return {
        "symbol": symbol,
        "url": page.url,
        "logged_in": logged_in,
        "sort_type": sort_type,
        "pages_collected": page_numbers,
        "discussion_count": len(blocks),
        "discussions": blocks[:max_records],
        "visible_text": text[:12000],
        "visible_links": links[:50],
    }


async def eastmoney_stock_quote(page: Page, code: str) -> dict:
    market = "sh" if code.startswith(("5", "6", "688", "689")) else "sz"
    url = f"https://quote.eastmoney.com/{market}{code}.html"
    await page.goto(url, wait_until="domcontentloaded")
    await page.wait_for_timeout(3000)
    await ensure_allowed(page)
    return {"stock_id": code, "url": page.url, "visible_text": (await page.locator("body").inner_text())[:12000]}


async def eastmoney_stock_links(page: Page, code: str) -> list[dict]:
    """Collect stock-specific news, notices, and research links from its quote page."""
    market = "sh" if code.startswith(("5", "6", "688", "689")) else "sz"
    await page.goto(f"https://quote.eastmoney.com/{market}{code}.html", wait_until="domcontentloaded")
    await page.wait_for_timeout(3000)
    await ensure_allowed(page)
    links = await page.locator("a").all()
    result = []
    seen = set()
    for anchor in links:
        text = clean(await anchor.inner_text())
        href = await anchor.get_attribute("href")
        if not href or not text:
            continue
        if href.startswith("//"):
            href = "https:" + href
        if "notices/detail" in href:
            kind = "notice"
        elif "finance.eastmoney.com/a/" in href:
            kind = "stock_news"
        elif "report/singlestock" in href or "report/stock" in href:
            kind = "research"
        else:
            continue
        if href in seen:
            continue
        seen.add(href)
        event_type, sentiment_hint = event_hint(text)
        result.append({"stock_id": code, "stock_relation": "direct", "content_type": kind, "title": text, "url": href, "event_type": event_type, "sentiment_hint": sentiment_hint})
    return result


async def eastmoney_stock_detail(page: Page, item: dict) -> dict:
    await asyncio.sleep(PAUSE_SECONDS)
    url = item["url"].replace("http://", "https://", 1)
    await page.goto(url, wait_until="domcontentloaded")
    await page.wait_for_timeout(3000)
    await ensure_allowed(page)
    text = await page.locator("body").inner_text()
    date_match = re.search(r"(20\d{2}[-年]\d{1,2}[-月]\d{1,2}日?(?:\s+\d{1,2}:\d{2})?)", text)
    body = text[:30000]
    event_type, sentiment_hint = event_hint(item.get("title", ""), body)
    return {**item, "url": url, "published_at_display": date_match.group(1) if date_match else None, "body": body, "event_type": event_type, "sentiment_hint": sentiment_hint, "collected_at": datetime.now(timezone.utc).isoformat()}

async def eastmoney_stock_research(page: Page, code: str, limit: int) -> list[dict]:
    """Collect visible individual research-report links from a stock report page."""
    url = f"https://data.eastmoney.com/report/singlestock.jshtml?stockcode={code}"
    await page.goto(url, wait_until="domcontentloaded")
    await page.wait_for_timeout(3000)
    await ensure_allowed(page)
    links = await page.locator('a[href*="/report/info/"]').evaluate_all(
        """anchors => anchors.map(a => ({title: (a.innerText || '').trim(), url: a.href}))
        .filter(x => x.title && x.url)"""
    )
    result = []
    seen = set()
    for item in links:
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        result.append({"stock_id": code, "stock_relation": "direct", "content_type": "research", "title": item["title"], "url": item["url"]})
        if len(result) >= limit:
            break
    return result


async def eastmoney_research_detail(page: Page, item: dict) -> dict:
    """Read visible research text and the linked PDF URL."""
    await asyncio.sleep(PAUSE_SECONDS)
    await page.goto(item["url"].replace("http://", "https://", 1), wait_until="domcontentloaded")
    await page.wait_for_timeout(3000)
    await ensure_allowed(page)
    text = await page.locator("body").inner_text()
    date_match = re.search(r"(20\d{2}[-年]\d{1,2}[-月]\d{1,2}日?(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?)", text)
    pdf_links = await page.locator('a[href*="pdf.dfcfw.com"]').evaluate_all("anchors => anchors.map(a => a.href)")
    return {**item, "published_at_display": date_match.group(1) if date_match else None, "body": text[:30000], "pdf_url": pdf_links[0] if pdf_links else None, "collected_at": datetime.now(timezone.utc).isoformat()}
    return {**item, "url": url, "published_at_display": date_match.group(1) if date_match else None, "body": body, "event_type": event_type, "sentiment_hint": sentiment_hint, "collected_at": datetime.now(timezone.utc).isoformat()}


async def eastmoney_historical_search(
    page: Page,
    keyword: str,
    start_date: datetime,
    end_date: datetime,
    result_limit: int,
) -> list[dict]:
    """Probe one bounded historical search without enumerating page numbers."""
    query = f"{keyword} {start_date.year}"
    url = f"https://so.eastmoney.com/news/s?keyword={quote(query)}"
    await page.goto(url, wait_until="domcontentloaded")
    await page.wait_for_timeout(3000)
    await ensure_allowed(page)
    links = await page.locator('a[href*="/a/20"]').evaluate_all(
        """anchors => anchors.map(a => {
            const box = a.closest('div');
            const text = (box ? box.innerText : a.innerText || '').trim();
            const match = text.match(/20\\d{2}-\\d{1,2}-\\d{1,2}\\s+\\d{1,2}:\\d{2}:\\d{2}/);
            return {title: (a.innerText || '').trim(), url: a.href,
                    published_at_display: match ? match[0] : null,
                    summary: text.slice(0, 1200)};
        }).filter(x => x.title && x.url)"""
    )
    results: list[dict] = []
    seen: set[str] = set()
    for item in links:
        published = item.get("published_at_display")
        try:
            parsed = datetime.strptime(published or "", "%Y-%m-%d %H:%M:%S").replace(tzinfo=CHINA_TZ)
        except ValueError:
            continue
        if not (start_date <= parsed < end_date) or item["url"] in seen:
            continue
        seen.add(item["url"])
        results.append({"source": "eastmoney", "content_type": "historical_search_result", "stock_relation": "keyword_search", **item})
        if len(results) >= result_limit:
            break
    return results


async def eastmoney_historical_detail(page: Page, item: dict) -> dict:
    """Read one already-filtered visible historical result."""
    await asyncio.sleep(PAUSE_SECONDS)
    await page.goto(item["url"].replace("http://", "https://", 1), wait_until="domcontentloaded")
    await page.wait_for_timeout(3000)
    await ensure_allowed(page)
    text = await page.locator("body").inner_text()
    date_match = re.search(r"(20\d{2}[-年]\d{1,2}[-月]\d{1,2}日?(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?)", text)
    return {**item, "content_type": "historical_news", "published_at_display": date_match.group(1) if date_match else item.get("published_at_display"), "body": text[:30000], "collected_at": datetime.now(timezone.utc).isoformat()}


async def main(args: argparse.Namespace) -> None:
    global PAUSE_SECONDS
    if args.pause_seconds < 0:
        raise ValueError("pause-seconds must be non-negative")
    PAUSE_SECONDS = args.pause_seconds
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    collected_at = datetime.now(timezone.utc)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not args.headed)
        context = await browser.new_context(storage_state=args.storage_state or None)
        page = await context.new_page()
        records = []
        if args.historical_keyword:
            start_date = parse_cli_date(args.historical_start)
            end_date = parse_cli_date(args.historical_end)
            if start_date >= end_date:
                raise ValueError("historical-end must be after historical-start")
            results = await eastmoney_historical_search(
                page,
                args.historical_keyword,
                start_date,
                end_date,
                args.historical_limit,
            )
            for item in results:
                try:
                    records.append(await eastmoney_historical_detail(page, item))
                except PlaywrightTimeoutError:
                    records.append({
                        "source": "collector",
                        "content_type": "collection_error",
                        "url": item.get("url"),
                        "error": "historical detail timeout; no retry",
                        "collected_at": collected_at.isoformat(),
                    })
            codes = []
        else:
            codes = [code.strip() for code in args.guba_codes.split(",") if code.strip()]
        if len(codes) > args.max_stocks:
            raise ValueError(f"refusing to process {len(codes)} stocks; max is {args.max_stocks}")
        for code in codes:
            code = code.strip()
            if not code:
                continue
            try:
                stock_links = await eastmoney_stock_links(page, code)
                stock_name = args.stock_names.get(code)
                records += [{"source": "eastmoney", "stock_name": stock_name, **item} for item in stock_links[:args.stock_link_limit]]
                detail_items = [item for item in stock_links if item["content_type"] in {"stock_news", "notice"}][:args.stock_detail_limit]
                for item in detail_items:
                    try:
                        detail = await eastmoney_stock_detail(page, item)
                        records.append({"source": "eastmoney", "stock_name": stock_name, **detail})
                    except PlaywrightTimeoutError:
                        records.append({"source": "collector", "content_type": "collection_error", "stock_id": code, "url": item["url"], "error": "stock detail timeout; no retry", "collected_at": collected_at.isoformat()})
                if args.research_detail_limit > 0:
                    research_items = await eastmoney_stock_research(page, code, args.research_detail_limit)
                    for item in research_items:
                        try:
                            detail = await eastmoney_research_detail(page, item)
                            records.append({"source": "eastmoney", "stock_name": stock_name, **detail})
                        except PlaywrightTimeoutError:
                            records.append({"source": "collector", "content_type": "collection_error", "stock_id": code, "url": item["url"], "error": "research detail timeout; no retry", "collected_at": collected_at.isoformat()})
                if args.stock_links_only:
                    for feed_type in args.xueqiu_feeds:
                        symbol = args.xueqiu_symbols.get(code, args.xueqiu_symbol)
                        feed = await xueqiu_feed(page, symbol, feed_type, args.xueqiu_max_records)
                        records.append({"source": "xueqiu", "content_type": f"{feed_type}_visible", "stock_id": code, "stock_name": stock_name, **feed, "collected_at": collected_at.isoformat()})
                    await asyncio.sleep(PAUSE_SECONDS)
                    continue
                forum_rows = await eastmoney_guba(page, code)
                for row in forum_rows:
                    row["collected_at"] = collected_at.isoformat()
                records += [{"source": "eastmoney", "content_type": "forum_post", "stock_id": code, "stock_name": stock_name, **r} for r in forum_rows]
                for detail in await eastmoney_guba_details(page, forum_rows, code, args.forum_detail_limit):
                    detail["stock_name"] = stock_name
                    records.append(detail)
                await asyncio.sleep(PAUSE_SECONDS)
                records.append({"source": "eastmoney", "content_type": "stock_page", "stock_name": stock_name, **await eastmoney_stock_quote(page, code)})
                await asyncio.sleep(PAUSE_SECONDS)
                symbol = args.xueqiu_symbols.get(code, args.xueqiu_symbol)
                for sort_type in args.xueqiu_sort_types:
                    xueqiu_record = await xueqiu_symbol(
                        page,
                        symbol,
                        sort_type=sort_type,
                        max_pages=args.xueqiu_max_pages,
                        max_records=args.xueqiu_max_records,
                    )
                    records.append({"source": "xueqiu", "content_type": "market_discussion", "stock_id": code, "stock_name": stock_name, **xueqiu_record})
                    for number, discussion in enumerate(xueqiu_record.get("discussions", []), 1):
                        records.append({
                            "source": "xueqiu",
                            "content_type": "discussion_visible",
                            "stock_id": code,
                            "stock_name": stock_name,
                            "symbol": symbol,
                            "discussion_index": number,
                            **discussion,
                            "collected_at": collected_at.isoformat(),
                        })
            except PlaywrightTimeoutError:
                records.append({"source": "collector", "content_type": "collection_error", "stock_id": code, "error": "page navigation timeout; no retry", "collected_at": collected_at.isoformat()})
            except AccessControlError as error:
                records.append({
                    "source": "collector",
                    "content_type": "collection_error",
                    "stock_id": code,
                    "error_type": "access_control",
                    "error": str(error),
                    "marker": error.marker,
                    "page_url": error.url,
                    "page_title": error.title,
                    "page_excerpt": error.excerpt,
                    "collected_at": collected_at.isoformat(),
                })
                print(json.dumps({
                    "status": "stopped",
                    "reason": "access_control",
                    "stock_id": code,
                    "marker": error.marker,
                    "url": error.url,
                    "title": error.title,
                    "excerpt": error.excerpt,
                }, ensure_ascii=False), file=sys.stderr)
                break
        manifest = Path(args.manifest)
        previous = load_manifest(manifest)
        unique_records = []
        current = set(previous)
        for record in records:
            key = record_key(record)
            if key in current:
                continue
            current.add(key)
            unique_records.append(record)
        save_manifest(manifest, current)
        payload = {"collected_at": collected_at.isoformat(), "pause_seconds": PAUSE_SECONDS, "manifest": str(manifest), "records": unique_records}
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/interim/browser_collection.json")
    parser.add_argument("--guba-code", default="000001", help="兼容旧参数；优先使用 --guba-codes")
    parser.add_argument("--guba-codes", default=None, help="逗号分隔的股票代码，例如 000001,600000")
    parser.add_argument("--codes-file", help="每行一个股票代码；仍受 --max-stocks 限制")
    parser.add_argument("--xueqiu-symbol", default="SH000001")
    parser.add_argument("--xueqiu-symbols-json", default="{}", help="股票代码到雪球代码的 JSON 映射")
    parser.add_argument("--xueqiu-sort-types", nargs="+", choices=("new", "hot"), default=["hot"], help="雪球可见讨论排序")
    parser.add_argument("--xueqiu-max-pages", type=int, default=20, help="每种排序最多采集页数")
    parser.add_argument("--xueqiu-max-records", type=int, default=200, help="每种排序最多保存帖子数")
    parser.add_argument("--xueqiu-feeds", nargs="*", choices=("news", "announcement"), default=[], help="雪球股票页可见资讯/公告分类")
    parser.add_argument("--stock-names-json", default="{}", help="股票代码到股票名称的 JSON 映射")
    parser.add_argument("--max-stocks", type=int, default=10, help="单轮最大股票数，防止无界抓取")
    parser.add_argument("--news-detail-limit", type=int, default=6)
    parser.add_argument("--stock-link-limit", type=int, default=20)
    parser.add_argument("--stock-detail-limit", type=int, default=6)
    parser.add_argument("--research-detail-limit", type=int, default=0, help="每只股票抓取的研报正文数量")
    parser.add_argument("--stock-links-only", action="store_true", help="只采集股票页可见的公告、新闻和研报链接及限定详情")
    parser.add_argument("--forum-detail-limit", type=int, default=3)
    parser.add_argument("--pause-seconds", type=float, default=PAUSE_SECONDS, help="页面访问之间的最小间隔")
    parser.add_argument("--storage-state", help="optional authorized Playwright storage-state JSON")
    parser.add_argument("--headed", action="store_true", help="show browser for manual observation")
    parser.add_argument("--manifest", default="data/interim/collector_manifest.json")
    parser.add_argument("--historical-keyword", help="单个历史资讯检索关键词；启用后不执行股票页/股吧采集")
    parser.add_argument("--historical-start", default=None, help="历史检索起始日期，格式 YYYY-MM-DD")
    parser.add_argument("--historical-end", default=None, help="历史检索结束日期（不含），格式 YYYY-MM-DD")
    parser.add_argument("--historical-limit", type=int, default=10, help="历史结果和详情的最大数量")
    parsed = parser.parse_args()
    if parsed.codes_file:
        codes = []
        for line in Path(parsed.codes_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("stock_id,"):
                continue
            codes.append(line.split(",", 1)[0].strip())
        parsed.guba_codes = ",".join(codes)
    parsed.guba_codes = parsed.guba_codes or parsed.guba_code
    if parsed.historical_keyword and (not parsed.historical_start or not parsed.historical_end):
        parser.error("--historical-keyword requires --historical-start and --historical-end")
    if parsed.historical_limit < 1 or parsed.historical_limit > 20:
        parser.error("--historical-limit must be between 1 and 20")
    parsed.xueqiu_symbols = json.loads(parsed.xueqiu_symbols_json)
    parsed.stock_names = json.loads(parsed.stock_names_json)
    asyncio.run(main(parsed))
