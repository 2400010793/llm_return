"""Bounded breadth-first crawler for public Sina Finance/Technology pages."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import csv
from collections import deque
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

ALLOWED_HOSTS = {"finance.sina.com.cn", "cj.sina.com.cn"}
DEFAULT_ROOTS = ("https://finance.sina.com.cn/", "https://tech.sina.com.cn/")
BLOCK_MARKERS = ("验证码", "访问异常", "请求过于频繁", "安全验证", "captcha")
DATE_RE = re.compile(r"20\d{2}[-年/]\d{1,2}[-月/]\d{1,2}")
SKIP_EXTENSIONS = re.compile(r"\.(?:jpg|jpeg|png|gif|svg|css|js|ico|pdf|zip|mp4|mp3)(?:$|[?#])", re.I)


def normalize_url(base: str, href: str) -> str | None:
    if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
        return None
    absolute = urldefrag(urljoin(base, html.unescape(href)))[0]
    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in ALLOWED_HOSTS:
        return None
    if SKIP_EXTENSIONS.search(parsed.path):
        return None
    return absolute


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.meta: dict[str, str] = {}
        self.title_parts: list[str] = []
        self.body_parts: list[str] = []
        self.article_parts: list[str] = []
        self._title = False
        self._skip = 0
        self._body_depth = 0
        self._article_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "a" and attrs_dict.get("href"):
            self.links.append(attrs_dict["href"] or "")
        if tag == "meta":
            key = attrs_dict.get("name") or attrs_dict.get("property") or attrs_dict.get("itemprop")
            value = attrs_dict.get("content")
            if key and value:
                self.meta[key.lower()] = value.strip()
        if tag == "title":
            self._title = True
        if tag in {"script", "style", "noscript", "template"}:
            self._skip += 1
        if tag == "body":
            self._body_depth = 1
        elif self._body_depth:
            self._body_depth += 1
        node_id = attrs_dict.get("id", "")
        classes = attrs_dict.get("class", "") or ""
        if node_id == "article" or re.search(r"(?:article|artibody|article-content|main-content)", classes, re.I):
            self._article_depth += 1
        elif self._article_depth:
            self._article_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._title = False
        if tag in {"script", "style", "noscript", "template"} and self._skip:
            self._skip -= 1
        if self._article_depth:
            self._article_depth -= 1
        if self._body_depth:
            self._body_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        value = re.sub(r"\s+", " ", data).strip()
        if not value:
            return
        if self._title:
            self.title_parts.append(value)
        if self._body_depth:
            self.body_parts.append(value)
        if self._article_depth:
            self.article_parts.append(value)

    @property
    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    @property
    def body(self) -> str:
        return " ".join(self.article_parts or self.body_parts).strip()


def first_meta(meta: dict[str, str], keys: tuple[str, ...]) -> str | None:
    return next((meta[key] for key in keys if meta.get(key)), None)


def published_at(parser: PageParser) -> str | None:
    value = first_meta(parser.meta, ("article:published_time", "bytedance:published_time", "publishdate", "pubdate", "date"))
    if value:
        return value
    match = DATE_RE.search(parser.body)
    return match.group(0) if match else None


def article_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]


def load_stock_catalog(path: str | None) -> list[dict[str, str]]:
    if not path:
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return [
            {"stock_id": row["stock_id"].zfill(6), "stock_name": row["stock_name"]}
            for row in csv.DictReader(handle)
            if row.get("stock_id") and row.get("stock_name")
        ]


def match_stocks(title: str, body: str, catalog: list[dict[str, str]]) -> list[dict[str, str]]:
    text = f"{title} {body}"
    matches = []
    for stock in catalog:
        code, name = stock["stock_id"], stock["stock_name"]
        code_hit = re.search(rf"(?<!\d){re.escape(code)}(?!\d)", text) is not None
        name_hit = name in text
        if code_hit or name_hit:
            matches.append({
                **stock,
                "stock_match_method": "explicit_code" if code_hit else "company_name",
            })
    return matches


class Robots:
    def __init__(self, user_agent: str) -> None:
        self.user_agent = user_agent
        self.parsers: dict[str, RobotFileParser] = {}

    def allowed(self, url: str) -> bool:
        host = urlparse(url).netloc
        parser = self.parsers.get(host)
        if parser is None:
            parser = RobotFileParser(f"https://{host}/robots.txt")
            try:
                parser.read()
            except OSError:
                return False
            self.parsers[host] = parser
        return parser.can_fetch(self.user_agent, url)


def crawl(args: argparse.Namespace) -> dict:
    queue = deque((url, 0) for url in (args.root or DEFAULT_ROOTS))
    queued = {url for url, _ in queue}
    visited: set[str] = set()
    records: list[dict] = []
    errors: list[dict] = []
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    robots = Robots(args.user_agent)
    stock_catalog = load_stock_catalog(args.stock_catalog)
    started = time.monotonic()
    while queue and len(visited) < args.max_pages and time.monotonic() - started < args.max_seconds:
        url, depth = queue.popleft()
        if url in visited or not robots.allowed(url):
            continue
        visited.add(url)
        try:
            request = Request(url, headers={"User-Agent": args.user_agent, "Accept": "text/html,application/xhtml+xml"})
            with urlopen(request, timeout=args.timeout) as response:
                payload = response.read()
            parser = PageParser()
            parser.feed(payload.decode("utf-8", errors="replace"))
            if any(marker.lower() in parser.body.lower() for marker in BLOCK_MARKERS):
                errors.append({"url": url, "error": "access-control-marker"})
                continue
            raw_path = raw_dir / f"{article_id(url)}.html"
            raw_path.write_bytes(payload)
            published = published_at(parser)
            if published and len(parser.body) >= 120:
                title = first_meta(parser.meta, ("og:title",)) or parser.title
                records.append({"source": "sina_finance", "content_type": "financial_news", "article_id": article_id(url), "url": url, "depth": depth, "title": title, "published_at": published, "stock_matches": match_stocks(title, parser.body, stock_catalog), "body": parser.body[:args.max_body_chars], "raw_html": str(raw_path), "collected_at": datetime.now(timezone.utc).isoformat()})
            if depth < args.max_depth:
                for href in parser.links:
                    child = normalize_url(url, href)
                    if child and child not in queued:
                        queued.add(child)
                        queue.append((child, depth + 1))
        except Exception as exc:
            errors.append({"url": url, "depth": depth, "error": f"{type(exc).__name__}: {exc}"})
        if args.pause_seconds:
            time.sleep(args.pause_seconds)
    return {"source": "sina_finance", "roots": args.root or list(DEFAULT_ROOTS), "visited": len(visited), "queued": len(queue), "records": records, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description="有边界、遵守 robots.txt 的新浪 BFS 新闻爬虫")
    parser.add_argument("--root", action="append")
    parser.add_argument("--max-pages", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--pause-seconds", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-body-chars", type=int, default=50000)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--user-agent", default="llm-return-research/0.1 (academic prototype)")
    parser.add_argument("--stock-catalog", help="股票代码和名称 CSV，用于文章实体匹配")
    args = parser.parse_args()
    if args.max_pages < 1 or args.max_depth < 0 or args.max_seconds <= 0 or args.pause_seconds < 0:
        parser.error("参数范围无效")
    result = crawl(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "visited": result["visited"], "articles": len(result["records"]), "errors": len(result["errors"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
