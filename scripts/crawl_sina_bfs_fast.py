"""Fast, bounded BFS crawler for a small Sina HTML-method replication.

It keeps the paper's order of operations but downloads one BFS frontier in
parallel with a bounded thread pool: fetch HTML -> save snapshot -> parse
metadata/body/links -> enqueue the next frontier. It is intentionally not a
6.3-million-page runner and does not bypass robots.txt or access controls.
"""
from __future__ import annotations

import argparse
import csv
import hashlib

import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

ALLOWED_HOSTS = {"finance.sina.com.cn", "cj.sina.com.cn"}
# Keep both seeds inside the allowed domain set so both can contribute links.
DEFAULT_ROOTS = ("https://finance.sina.com.cn/", "https://cj.sina.com.cn/")
BLOCK_MARKERS = ("验证码", "访问异常", "请求过于频繁", "安全验证", "captcha")
DATE_RE = re.compile(r"20\d{2}[-年/]\d{1,2}[-月/]\d{1,2}(?:[^0-9]{1,10}\d{1,2}:\d{2})?")
SKIP_EXTENSIONS = re.compile(r"\.(?:jpg|jpeg|png|gif|svg|css|js|ico|pdf|zip|mp4|mp3)(?:$|[?#])", re.I)
STOCK_LINK_RE = re.compile(r"/realstock/company/(?:sh|sz)(\d{6})/", re.I)


def decode_html(payload: bytes, content_type: str) -> tuple[str, str]:
    """Decode Sina pages using HTTP/meta charset before Chinese fallbacks.

    The previous implementation forced UTF-8 and used ``errors='replace'``.
    Many historical Sina pages are GBK/GB18030, so that policy permanently
    converted valid Chinese bytes into U+FFFD before JSONL was written.
    """
    candidates: list[str] = []
    header = re.search(r"charset\s*=\s*[\"']?([\w-]+)", content_type or "", re.I)
    if header:
        candidates.append(header.group(1))
    probe = payload[:8192].decode("ascii", errors="ignore")
    meta = re.search(r"(?:charset\s*=|encoding\s*=)[\"']?\s*([\w-]+)", probe, re.I)
    if meta:
        candidates.append(meta.group(1))
    candidates.extend(["utf-8", "gb18030", "gbk", "big5"])
    seen: set[str] = set()
    for encoding in candidates:
        encoding = encoding.lower()
        if encoding in seen:
            continue
        seen.add(encoding)
        try:
            return payload.decode(encoding), encoding
        except (LookupError, UnicodeDecodeError):
            continue
    return payload.decode("utf-8", errors="replace"), "utf-8-replace"


def normalize_url(base: str, href: str) -> str | None:
    if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
        return None
    url = urldefrag(urljoin(base, html.unescape(href)))[0]
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in ALLOWED_HOSTS:
        return None
    if SKIP_EXTENSIONS.search(parsed.path):
        return None
    return url


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.stock_links: list[dict[str, str]] = []
        self.meta: dict[str, str] = {}
        self.title_parts: list[str] = []
        self.body_parts: list[str] = []
        self.article_parts: list[str] = []
        self._title = False
        self._skip = 0
        self._body = 0
        self._article = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "a" and values.get("href"):
            href = values["href"] or ""
            self.links.append(href)
            match = STOCK_LINK_RE.search(href)
            if match and self._article:
                self.stock_links.append({"stock_id": match.group(1), "stock_name": "", "stock_match_method": "sina_stock_link"})
        if tag == "meta":
            key = values.get("name") or values.get("property") or values.get("itemprop")
            value = values.get("content")
            if key and value:
                self.meta[key.lower()] = value.strip()
        if tag == "title":
            self._title = True
        if tag in {"script", "style", "noscript", "template"}:
            self._skip += 1
        if tag == "body":
            self._body = 1
        elif self._body:
            self._body += 1
        node_id = values.get("id", "")
        classes = values.get("class", "") or ""
        if node_id == "article" or re.search(r"article|artibody|article-content|main-content", classes, re.I):
            self._article += 1
        elif self._article:
            self._article += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._title = False
        if tag in {"script", "style", "noscript", "template"} and self._skip:
            self._skip -= 1
        if self._article:
            self._article -= 1
        if self._body:
            self._body -= 1

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = re.sub(r"\s+", " ", data).strip()
        if not text:
            return
        if self._title:
            self.title_parts.append(text)
        if self._body:
            self.body_parts.append(text)
        if self._article:
            self.article_parts.append(text)

    @property
    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    @property
    def body(self) -> str:
        return " ".join(self.article_parts or self.body_parts).strip()


def sha(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:24]


def first_meta(meta: dict[str, str], keys: tuple[str, ...]) -> str | None:
    return next((meta[key] for key in keys if meta.get(key)), None)


def load_catalog(path: str | None) -> list[dict[str, str]]:
    if not path:
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return [{"stock_id": row["stock_id"].zfill(6), "stock_name": row["stock_name"]} for row in csv.DictReader(stream) if row.get("stock_id") and row.get("stock_name")]


def matches(title: str, body: str, catalog: list[dict[str, str]]) -> list[dict[str, str]]:
    text = f"{title} {body}"
    result = []
    for stock in catalog:
        code_hit = re.search(rf"(?<!\d){re.escape(stock['stock_id'])}(?!\d)", text) is not None
        name_hit = stock["stock_name"] in text
        if code_hit or name_hit:
            result.append({**stock, "stock_match_method": "explicit_code" if code_hit else "company_name"})
    return result


def merge_stock_matches(link_matches: list[dict[str, str]], text_matches: list[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for item in link_matches + text_matches:
        existing = next((candidate for candidate in result if candidate["stock_id"] == item["stock_id"]), None)
        if existing is None:
            result.append(item)
        elif not existing.get("stock_name") and item.get("stock_name"):
            existing["stock_name"] = item["stock_name"]
    return result


def year_from_timestamp(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"(20\d{2})", value)
    return match.group(1) if match else None


def read_jsonl_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except (UnicodeError, json.JSONDecodeError):
                continue
    return records


def iter_jsonl_records(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.strip():
                try:
                    yield json.loads(line)
                except (UnicodeError, json.JSONDecodeError):
                    continue


def summarize_jsonl_records(path: Path) -> tuple[int, dict[str, int]]:
    record_count = 0
    year_counts: dict[str, int] = {}
    for record in iter_jsonl_records(path):
        record_count += 1
        year = year_from_timestamp(record.get("published_at"))
        if year:
            year_counts[year] = year_counts.get(year, 0) + 1
    return record_count, year_counts


def write_result_streaming(
    output: Path,
    roots: list[str],
    visited: int,
    records_path: Path,
    queued_remaining: int,
    elapsed_seconds: float,
    workers: int,
    coverage: dict[str, object],
) -> None:
    """Write the compatibility JSON summary without loading all records."""
    with output.open("w", encoding="utf-8") as stream:
        stream.write("{\n")
        stream.write(f'  "source": {json.dumps("sina_finance", ensure_ascii=False)},\n')
        stream.write(f'  "roots": {json.dumps(roots, ensure_ascii=False)},\n')
        stream.write(f'  "visited": {visited},\n')
        stream.write('  "records": [\n')
        first = True
        if records_path.exists():
            with records_path.open(encoding="utf-8", errors="replace") as records:
                for line in records:
                    if not line.strip():
                        continue
                    try:
                        json.loads(line)
                    except (UnicodeError, json.JSONDecodeError):
                        continue
                    if not first:
                        stream.write(",\n")
                    stream.write("    " + line.rstrip())
                    first = False
        stream.write("\n  ],\n")
        stream.write(f'  "queued_remaining": {queued_remaining},\n')
        stream.write(f'  "elapsed_seconds": {json.dumps(round(elapsed_seconds, 2))},\n')
        stream.write(f'  "workers": {workers},\n')
        stream.write(f'  "coverage": {json.dumps(coverage, ensure_ascii=False)}\n')
        stream.write("}\n")


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                json.loads(line)
            except (UnicodeError, json.JSONDecodeError):
                continue
            count += 1
    return count


def robots_allowed(url: str, user_agent: str, cache: dict[str, RobotFileParser], lock) -> bool:
    host = urlparse(url).netloc
    with lock:
        parser = cache.get(host)
        if parser is None:
            parser = RobotFileParser(f"https://{host}/robots.txt")
            try:
                parser.read()
            except OSError:
                return False
            cache[host] = parser
    return parser.can_fetch(user_agent, url)


def fetch_one(item: tuple[str, int], args, raw_dir: Path, catalog, robot_cache, robot_lock) -> dict:
    url, depth = item
    base = {"url": url, "depth": depth, "fetched_at": datetime.now(timezone.utc).isoformat()}
    if not robots_allowed(url, args.user_agent, robot_cache, robot_lock):
        return {**base, "status": "robots_denied", "links": [], "record": None}
    try:
        request = Request(url, headers={"User-Agent": args.user_agent, "Accept": "text/html,application/xhtml+xml"})
        with urlopen(request, timeout=args.timeout) as response:
            payload = response.read()
            content_type = response.headers.get("Content-Type", "")
        if content_type and "html" not in content_type.lower():
            return {**base, "status": "not_html", "links": [], "record": None}
        decoded, detected_encoding = decode_html(payload, content_type)
        parser = PageParser()
        parser.feed(decoded)
        if any(marker.lower() in parser.body.lower() for marker in BLOCK_MARKERS):
            return {**base, "status": "blocked_marker", "links": [], "record": None}
        raw_path = raw_dir / f"{sha(url)}.html"
        raw_path.write_bytes(payload)
        links = list(dict.fromkeys(filter(None, (normalize_url(url, href) for href in parser.links))))
        published = first_meta(parser.meta, ("article:published_time", "bytedance:published_time", "publishdate", "pubdate", "date")) or (DATE_RE.search(parser.body).group(0) if DATE_RE.search(parser.body) else None)
        title = first_meta(parser.meta, ("og:title",)) or parser.title
        page_stock_ids = list(dict.fromkeys(STOCK_LINK_RE.findall(decoded)))
        article_stock_ids = {item["stock_id"] for item in parser.stock_links}
        selected_stock_ids = page_stock_ids if args.paper_stock_matching else [stock_id for stock_id in page_stock_ids if stock_id in article_stock_ids]
        link_matches = [{"stock_id": stock_id, "stock_name": "", "stock_match_method": "sina_stock_link"} for stock_id in selected_stock_ids]
        record = None
        if published and len(parser.body) >= args.min_body_chars:
            stock_matches = merge_stock_matches(link_matches, matches(title, parser.body, catalog))
            if not args.exactly_one_stock or len(stock_matches) == 1:
                stored_body = parser.body[:args.max_body_chars]
                record = {"source": "sina_finance", "content_type": "financial_news", "article_id": sha(url), "url": url, "depth": depth, "title": title, "published_at": published, "stock_matches": stock_matches, "body": stored_body, "body_chars_full": len(parser.body), "body_chars_stored": len(stored_body), "body_truncated": len(stored_body) < len(parser.body), "encoding": detected_encoding, "raw_html": str(raw_path), "collected_at": base["fetched_at"]}
        return {**base, "status": "ok", "links": links, "record": record, "bytes": len(payload)}
    except Exception as exc:
        return {**base, "status": "error", "links": [], "record": None, "error": f"{type(exc).__name__}: {exc}"}


def crawl(args) -> dict:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(args.stock_catalog)
    state_path = Path(args.state_file) if args.state_file else output.with_suffix(".state.json")
    records_path = output.with_suffix(".records.jsonl")
    manifest_path = output.with_suffix(".manifest.jsonl")
    frontier = [(root, 0) for root in (args.root or DEFAULT_ROOTS)]
    queued = {url for url, _ in frontier}
    visited: set[str] = set()
    records_count = 0
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        frontier = [tuple(item) for item in state.get("frontier", [])]
        visited = set(state.get("visited", []))
        # The state file already stores the valid-record count.  Do not scan
        # the multi-gigabyte records JSONL on every restart.
        records_count = int(state.get("records_count", 0))
        if not visited:
            if manifest_path.exists():
                with manifest_path.open(encoding="utf-8", errors="replace") as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                        except (UnicodeError, json.JSONDecodeError):
                            continue
                        if item.get("url"):
                            visited.add(item["url"])
            # Older state files may not have stored the count.
            if "records_count" not in state:
                records_count = jsonl_count(records_path)
        queued = visited | {url for url, _ in frontier}
    elif records_path.exists():
        records_count = jsonl_count(records_path)
    robot_cache: dict[str, RobotFileParser] = {}
    from threading import Lock
    robot_lock = Lock()
    started = time.monotonic()
    with manifest_path.open("a", encoding="utf-8") as manifest:
        within_time = args.max_seconds is None or time.monotonic() - started < args.max_seconds
        while frontier and (args.max_pages is None or len(visited) < args.max_pages) and within_time:
            # Keep the number of in-flight futures bounded.  A large root file
            # must not turn into tens of thousands of simultaneous futures.
            batch_limit = max(args.workers * 20, 100)
            remaining = batch_limit if args.max_pages is None else min(batch_limit, args.max_pages - len(visited))
            batch = [(url, depth) for url, depth in frontier if url not in visited][:remaining]
            visited.update(url for url, _ in batch)
            next_frontier: list[tuple[str, int]] = []
            try:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = [pool.submit(fetch_one, item, args, raw_dir, catalog, robot_cache, robot_lock) for item in batch]
                    for future in as_completed(futures):
                        result = future.result()
                        manifest.write(json.dumps({key: value for key, value in result.items() if key != "links" and key != "record"}, ensure_ascii=False) + "\n")
                        if result.get("record"):
                            with records_path.open("a", encoding="utf-8") as record_stream:
                                record_stream.write(json.dumps(result["record"], ensure_ascii=False) + "\n")
                            records_count += 1
                        if result["depth"] < args.max_depth:
                            for child in result.get("links", []):
                                if child not in queued:
                                    queued.add(child)
                                    next_frontier.append((child, result["depth"] + 1))
            except (OSError, RuntimeError) as exc:
                manifest.write(json.dumps({"status": "batch_error", "error": f"{type(exc).__name__}: {exc}", "batch_size": len(batch)}, ensure_ascii=False) + "\n")
                manifest.flush()
                frontier = [(url, depth) for url, depth in frontier if url not in visited] + next_frontier
                continue
            manifest.flush()
            # Retain roots that were not included in this bounded batch.
            # Dropping them here would silently crawl only the first batch.
            remaining_frontier = [(url, depth) for url, depth in frontier if url not in visited]
            frontier = remaining_frontier + next_frontier
            state_path.write_text(json.dumps({"frontier": frontier, "visited_count": len(visited), "records_count": records_count}, ensure_ascii=False), encoding="utf-8")
            if args.pause_seconds:
                time.sleep(args.pause_seconds)
            within_time = args.max_seconds is None or time.monotonic() - started < args.max_seconds
    record_count, year_counts = summarize_jsonl_records(records_path)
    elapsed_seconds = time.monotonic() - started
    coverage = {"article_year_counts": dict(sorted(year_counts.items())), "earliest_article_year": min(year_counts) if year_counts else None, "latest_article_year": max(year_counts) if year_counts else None}
    write_result_streaming(output, args.root or list(DEFAULT_ROOTS), len(visited), records_path, len(frontier), elapsed_seconds, args.workers, coverage)
    # Keep the in-memory return value small; the complete records remain in
    # records.jsonl and are also streamed into the compatibility JSON above.
    result = {"source": "sina_finance", "roots": args.root or list(DEFAULT_ROOTS), "visited": len(visited), "records": [], "records_count": record_count, "queued_remaining": len(frontier), "elapsed_seconds": round(elapsed_seconds, 2), "workers": args.workers, "coverage": coverage}
    state_path.write_text(json.dumps({"frontier": frontier, "visited_count": len(visited), "records_count": record_count, "completed": not frontier}, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="小规模、并发受限的新浪 HTML BFS 复刻")
    parser.add_argument("--root", action="append")
    parser.add_argument("--root-file", help="每行一个根 URL；与 --root 等价")
    parser.add_argument("--workers", type=int, default=4, help="并发请求数，范围1-50")
    parser.add_argument("--max-pages", type=int, default=None, help="最大访问页面数；省略表示持续爬取至 frontier 为空")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=None, help="运行时间上限；省略表示直到 frontier 耗尽")
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--min-body-chars", type=int, default=120)
    parser.add_argument("--max-body-chars", type=int, default=12000, help="正文最多保存字符数；用于控制后续模型输入长度")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-file", help="断点状态文件；默认使用 output 对应的 .state.json")
    parser.add_argument("--resume", action="store_true", help="从 state-file 和 records.jsonl 继续")
    parser.add_argument("--stock-catalog")
    parser.add_argument("--exactly-one-stock", action="store_true", help="仅保留恰好匹配一只股票的文章")
    parser.add_argument("--paper-stock-matching", action="store_true", help="按论文描述扫描整个 HTML 的新浪股票标签")
    parser.add_argument("--user-agent", default="llm-return-research/0.1 (academic prototype)")
    args = parser.parse_args()
    if args.root_file:
        roots = [line.strip() for line in Path(args.root_file).read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
        args.root = (args.root or []) + roots
    if not 1 <= args.workers <= 50 or (args.max_pages is not None and args.max_pages < 1) or args.max_depth < 0 or args.pause_seconds < 0:
        parser.error("workers 应为 1-50，其他参数范围无效")
    result = crawl(args)
    print(json.dumps({"output": args.output, "visited": result["visited"], "articles": result["records_count"], "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
