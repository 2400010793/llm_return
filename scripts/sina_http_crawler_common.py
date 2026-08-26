"""Shared, conservative HTTP crawler for Sina Finance historical HTML pages.

Python 3.9 compatible.  This module deliberately uses urllib only: no browser
automation, proxy rotation, fingerprint spoofing, or access-control bypass.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import logging
import re
import time
import zlib
from collections import deque
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urldefrag, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.robotparser import RobotFileParser

ALLOWED_HOSTS = {"finance.sina.com.cn", "cj.sina.com.cn"}
DEFAULT_ROOTS = ("https://finance.sina.com.cn/", "https://cj.sina.com.cn/")
BLOCK_MARKERS = ("验证码", "访问异常", "请求过于频繁", "安全验证", "captcha", "security verification")
TRACKING_PARAMS = {"from", "source", "spm", "mod", "c", "cre", "loc", "vt", "pagetype"}
SKIP_EXTENSIONS = re.compile(r"\.(?:jpg|jpeg|png|gif|svg|webp|css|js|ico|woff2?|ttf|pdf|zip|gz|rar|7z|mp4|mp3|wav|avi|mov)(?:$|[?#])", re.I)
DATE_RE = re.compile(r"(?:20\d{2}年\s*\d{1,2}月\s*\d{1,2}日(?:[^0-9]{0,12}\d{1,2}:\d{2}(?::\d{2})?)?|20\d{2}[-/]\d{1,2}[-/]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:?\d{2})?)?)")
URL_DATE_RE = re.compile(r"/(20\d{2})[-_/](\d{1,2})[-_/](\d{1,2})")
MODERN_STOCK_RE = re.compile(r"/realstock/company/(sh|sz)(\d{6})/", re.I)
LEGACY_STOCK_RE = re.compile(r"/cgi-bin/stock/quote/quote\.cgi[^\"'<>]*[?&]symbol=([0-9A-Za-z]{2,6})", re.I)
HISTORICAL_PATHS = ("/t/", "/s/", "/e/", "/y/", "/o/", "/j/", "/view/", "/roll/", "/stock/t/", "/stock/s/", "/stock/e/", "/stock/y/", "/globe/")
HISTORICAL_ARTICLE_RE = re.compile(r"^/(?:t|s|e|y)/\d+\.html$", re.I)
LOW_PRIORITY_PATHS = ("/realstock/", "/zt_d/", "/fund/", "/stock/estate/", "/stock/newstock/", "/stock/quanshang/", "/video/", "/dav/", "/calc/", "/iframe/", "/api/", "/search/", "/photo/", "/comment/", "/corp/")
PARSER_VERSION = "sina-parser-v2"
NAVIGATION_TITLE_MARKERS = ("新浪财经_", "股票首页", "港股|港股行情", "美股|美股行情", "新浪期货_", "新浪外汇_", "财经面对面_", "ESG|", "股票博客", "意见领袖", "财经会议", "个人理财", "银行_", "保险频道", "新浪信托")
NAVIGATION_TEXT_MARKERS = ("财经首页", "新浪首页", "财经导航", "行情中心", "登录", "注册", "自选股", "基金", "外汇", "期货", "股票行情")


def article_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]


def normalize_url(base: str, href: str, include_http: bool = True, strip_tracking: bool = True) -> Optional[str]:
    if not href:
        return None
    href = html.unescape(href.strip())
    if href.lower().startswith(("javascript:", "mailto:", "tel:", "data:", "#")):
        return None
    absolute = urldefrag(urljoin(base, href))[0]
    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"} or (parsed.scheme == "http" and not include_http):
        return None
    if (parsed.hostname or "").lower() not in ALLOWED_HOSTS or SKIP_EXTENSIONS.search(parsed.path):
        return None
    query = parsed.query
    if strip_tracking and query:
        query = urlencode([(k, v) for k, v in parse_qsl(query, keep_blank_values=True) if k.lower() not in TRACKING_PARAMS], doseq=True)
    return urlunparse((parsed.scheme, (parsed.hostname or "").lower(), parsed.path or "/", parsed.params, query, ""))


def url_priority(url: str, include_navigation: bool = False) -> int:
    path = urlparse(url).path.lower()
    if any(path.startswith(prefix) for prefix in LOW_PRIORITY_PATHS):
        return 0 if include_navigation else -1
    return 2 if any(prefix in path for prefix in HISTORICAL_PATHS) else 1


def is_historical_article_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.hostname in ALLOWED_HOSTS and bool(HISTORICAL_ARTICLE_RE.fullmatch(parsed.path))


def uniform_frontier_key(item: Tuple[str, int, str, Optional[str], int], seed: int = 0) -> str:
    """Return a deterministic pseudo-random key for uniform frontier sampling."""
    return hashlib.sha256((str(seed) + "\0" + item[0]).encode("utf-8")).hexdigest()


def http_status_label(code: int) -> str:
    if code == 404: return "http_404"
    if code == 403: return "http_403"
    if code == 429: return "http_429"
    if 500 <= code <= 599: return "http_5xx"
    return "http_error"


class PageParser(HTMLParser):
    """Extract visible text and preserve body/sidebar regions for audit."""
    def __init__(self) -> None:
        HTMLParser.__init__(self, convert_charrefs=True)
        self.links = []  # type: List[str]
        self.meta = {}  # type: Dict[str, str]
        self.title_parts = []  # type: List[str]
        self.visible_parts = []  # type: List[str]
        self.body_parts = []  # type: List[str]
        self.article_parts = []  # type: List[str]
        self.body_link_matches = []  # type: List[Dict[str, Any]]
        self._skip = 0
        self._stack = []  # type: List[bool]
        self._title = False
        self._article_depth = 0
        self._body_depth = 0
        self._sidebar_depth = 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        values = dict(attrs)
        if tag == "a" and values.get("href"):
            href = values["href"] or ""
            self.links.append(href)
            hit = stock_href_match(href)
            if hit:
                hit["raw_href"] = href
                hit["is_body_region"] = bool(self._article_depth or (self._body_depth and not self._sidebar_depth))
                (self.body_link_matches if hit["is_body_region"] else self._all_link_matches).append(hit)
        if tag == "meta":
            key = values.get("name") or values.get("property") or values.get("itemprop")
            value = values.get("content")
            if key and value:
                self.meta[key.lower()] = value.strip()
        if tag == "title":
            self._title = True
        excluded = tag in {"script", "style", "noscript", "template"}
        if excluded:
            self._skip += 1
        node_id = values.get("id", "")
        classes = values.get("class", "") or ""
        sidebar = re.search(r"nav|comment|advert|recommend|footer|sidebar|login|toolbar|realstock|行情|广告|推荐", node_id + " " + classes, re.I)
        article = tag == "article" or node_id.lower() in {"article", "artibody", "article_content"} or re.search(r"article[-_]?content|artibody|main-content", classes, re.I)
        if tag == "body": self._body_depth = 1
        elif self._body_depth: self._body_depth += 1
        if sidebar: self._sidebar_depth += 1
        elif self._sidebar_depth: self._sidebar_depth += 1
        if article: self._article_depth = 1
        elif self._article_depth: self._article_depth += 1
        self._stack.append(bool(excluded))

    def handle_endtag(self, tag: str) -> None:
        if tag == "title": self._title = False
        if self._stack and self._stack.pop(): self._skip = max(0, self._skip - 1)
        if self._article_depth: self._article_depth -= 1
        if self._sidebar_depth: self._sidebar_depth -= 1
        if self._body_depth: self._body_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip: return
        value = re.sub(r"\s+", " ", data).strip()
        if not value: return
        self.visible_parts.append(value)
        if self._title: self.title_parts.append(value)
        if self._body_depth and not self._sidebar_depth: self.body_parts.append(value)
        if self._article_depth: self.article_parts.append(value)

    @property
    def title(self) -> str: return " ".join(self.title_parts).strip()
    @property
    def visible_text(self) -> str: return " ".join(self.visible_parts).strip()
    @property
    def body(self) -> str: return " ".join(self.article_parts or self.body_parts or self.visible_parts).strip()
    _all_link_matches = []


def stock_href_match(href: str) -> Optional[Dict[str, Any]]:
    modern = MODERN_STOCK_RE.search(href)
    if modern:
        return {"stock_id": modern.group(2), "stock_name": "", "raw_href": href, "match_method": "modern_realstock", "legacy_symbol": None}
    legacy = LEGACY_STOCK_RE.search(href)
    if legacy:
        symbol = legacy.group(1)
        return {"stock_id": None, "stock_name": "", "raw_href": href, "match_method": "legacy_quote", "legacy_symbol": symbol, "normalized_stock_id": None, "stock_id_mapping_status": "unmapped", "mapping_source": None}
    return None


def extract_dates(parser: PageParser, url: str) -> Dict[str, Any]:
    keys = ("article:published_time", "bytedance:published_time", "publishdate", "pubdate", "date", "datepublished")
    raw = next((parser.meta.get(k) for k in keys if parser.meta.get(k)), None)
    source = "meta" if raw else None
    confidence = 1.0 if raw else 0.0
    if not raw:
        for text, label, score in ((parser.title, "title", .9), (parser.body[:1200], "body_front", .85), (parser.visible_text, "visible_text", .75)):
            match = DATE_RE.search(text)
            if match:
                raw, source, confidence = match.group(0), label, score; break
    if not raw:
        match = URL_DATE_RE.search(url)
        if match:
            raw, source, confidence = "-".join(match.groups()), "url", .55
    normalized = None
    published_date = None
    year = None
    if raw:
        value = raw.strip().replace("年", "-").replace("月", "-").replace("日", "")
        value = re.sub(r"\s+", " ", value)
        match = re.search(r"(20\d{2})[-/]([0-9]{1,2})[-/]([0-9]{1,2})(?:[ T]+([0-9]{1,2}):([0-9]{2})(?::([0-9]{2}))?)?", value)
        if match:
            y, mo, d, hh, mm, ss = match.groups(); year = int(y); published_date = "%04d-%02d-%02d" % (int(y), int(mo), int(d))
            normalized = published_date + ("T%02d:%s:%s+08:00" % (int(hh), mm, ss or "00") if hh else "T00:00:00+08:00")
    candidates = sorted(set(int(y) for y in re.findall(r"20\d{2}", parser.body[:5000])))
    if year and year not in candidates: candidates.append(year); candidates.sort()
    return {"published_at_raw": raw, "published_at_normalized": normalized, "published_date": published_date, "published_year": year, "content_year_candidates": candidates, "date_source": source, "date_confidence": confidence}


def decode_payload(payload: bytes, content_type: str) -> Tuple[str, str]:
    charset = re.search(r"charset\s*=\s*[\"']?([\w-]+)", content_type, re.I)
    candidates = [charset.group(1)] if charset else []
    head = payload[:4096].decode("ascii", errors="ignore")
    meta = re.search(r"charset\s*=\s*[\"']?([\w-]+)", head, re.I)
    if meta: candidates.append(meta.group(1))
    candidates.extend(["utf-8", "gb18030", "gbk", "big5"])
    for encoding in candidates:
        try: return payload.decode(encoding), encoding
        except (LookupError, UnicodeDecodeError): pass
    raise UnicodeDecodeError("unknown", payload, 0, min(1, len(payload)), "no supported encoding")


def decompress_payload(payload: bytes, encoding: str) -> bytes:
    if "gzip" in encoding.lower(): return gzip.decompress(payload)
    if "deflate" in encoding.lower():
        try: return zlib.decompress(payload)
        except zlib.error: return zlib.decompress(payload, -zlib.MAX_WBITS)
    return payload


def load_catalog(path: Optional[str]) -> List[Dict[str, str]]:
    if not path: return []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return [{"stock_id": row["stock_id"].zfill(6), "stock_name": row["stock_name"]} for row in csv.DictReader(handle) if row.get("stock_id") and row.get("stock_name")]


def text_matches(title: str, body: str, catalog: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    text = title + " " + body; result = []
    for stock in catalog:
        code = re.search(r"(?<!\d)" + re.escape(stock["stock_id"]) + r"(?!\d)", text)
        name = stock["stock_name"] in text
        if code or name: result.append({**stock, "match_method": "explicit_code" if code else "company_name", "context": (code or name).group(0) if hasattr((code or name), "group") else stock["stock_name"]})
    return result


def classify_body_quality(url: str, title: str, body: str, parser: PageParser, published_year: Optional[int], min_body_chars: int) -> str:
    """Classify article versus navigation pages before writing article JSONL.

    Old Sina table pages often contain a long header/sidebar, so length alone
    is insufficient.  A title/path navigation marker or navigation-heavy text
    is retained in manifest but excluded from the article dataset.
    """
    if len(body) < min_body_chars:
        return "too_short"
    title_value = title.strip()
    path = urlparse(url).path.lower()
    title_navigation = any(marker.lower() in title_value.lower() for marker in NAVIGATION_TITLE_MARKERS)
    navigation_hits = sum(body.lower().count(marker.lower()) for marker in NAVIGATION_TEXT_MARKERS)
    has_article_container = bool(parser.article_parts)
    historical_article_path = any(path.startswith(prefix) for prefix in HISTORICAL_PATHS)
    if title_navigation or (not has_article_container and navigation_hits >= 4 and not historical_article_path):
        return "navigation_only"
    if not published_year:
        return "low_quality"
    return "valid"


class Robots:
    def __init__(self, user_agent: str, enabled: bool = True) -> None:
        self.user_agent, self.enabled = user_agent, enabled; self.parsers = {}  # type: Dict[str, RobotFileParser]
    def allowed(self, url: str) -> bool:
        if not self.enabled: return True
        parsed = urlparse(url); host = parsed.netloc
        if host not in self.parsers:
            parser = RobotFileParser()
            robots_url = urlunparse((parsed.scheme, host, "/robots.txt", "", "", ""))
            parser.set_url(robots_url)
            try:
                request = Request(robots_url, headers={"User-Agent": self.user_agent})
                with urlopen(request, timeout=10) as response:
                    if getattr(response, "status", 200) < 200 or getattr(response, "status", 200) >= 300:
                        return False
                    parser.parse(response.read().decode("utf-8", errors="replace").splitlines())
            except (OSError, IOError, URLError): return False
            self.parsers[host] = parser
        return self.parsers[host].can_fetch(self.user_agent, url)


def _write_jsonl(handle: Any, value: Dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"); handle.flush()


def add_arguments(parser: argparse.ArgumentParser, strategy_required: bool = False) -> None:
    if strategy_required: parser.add_argument("--strategy", choices=("bfs", "dfs"), required=True)
    else: parser.add_argument("--strategy", choices=("bfs", "dfs"), default="bfs")
    parser.add_argument("--frontier-policy", choices=("fifo", "uniform"), default="fifo", help="候选链接调度策略；uniform 对当前 frontier 做可复现的伪随机均匀抽样")
    parser.add_argument("--uniform-seed", type=int, default=0, help="uniform frontier 的稳定随机种子")
    parser.add_argument("--root", action="append"); parser.add_argument("--max-pages", type=int, default=20); parser.add_argument("--max-depth", type=int, default=2); parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--pause-seconds", type=float, default=2); parser.add_argument("--timeout", type=float, default=30); parser.add_argument("--connect-timeout", type=float, default=10); parser.add_argument("--max-retries", type=int, default=1); parser.add_argument("--max-response-bytes", type=int, default=10000000)
    parser.add_argument("--user-agent", default="llm-return-research/0.1 (academic prototype)"); parser.add_argument("--raw-dir", required=True); parser.add_argument("--articles-output", "--output", dest="articles_output", required=True); parser.add_argument("--manifest-output", "--manifest", dest="manifest_output"); parser.add_argument("--summary-output", dest="summary_output"); parser.add_argument("--state-output", "--state", dest="state_output"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--min-body-chars", type=int, default=120); parser.add_argument("--max-body-chars", type=int, default=50000); parser.add_argument("--stock-catalog"); parser.add_argument("--exactly-one-stock", action="store_true"); parser.add_argument("--paper-stock-matching", action="store_true"); parser.add_argument("--include-navigation-pages", action="store_true"); parser.add_argument("--historical-only-links", action="store_true", help="只扩展 /t|s|e|y/数字.html 历史文章链接"); parser.add_argument("--include-http", action="store_true"); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--verbose", action="store_true"); parser.add_argument("--no-robots", action="store_true")


def crawl(args: argparse.Namespace) -> Dict[str, Any]:
    # 老新浪页面大量使用 HTTP；允许 HTTP 是默认行为，--include-http 保留为兼容性显式开关。
    roots = [normalize_url("https://finance.sina.com.cn/", r, True) for r in (args.root or DEFAULT_ROOTS)]
    roots = [r for r in roots if r]
    manifest_path = Path(args.manifest_output or str(args.articles_output).replace(".jsonl", "_manifest.jsonl")); summary_path = Path(args.summary_output or str(args.articles_output).replace(".jsonl", "_summary.json")); state_path = Path(args.state_output or str(args.articles_output).replace(".jsonl", "_state.json")); raw_dir = Path(args.raw_dir)
    for p in (manifest_path, summary_path, state_path, Path(args.articles_output), raw_dir): p.parent.mkdir(parents=True, exist_ok=True) if p.suffix else p.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(args.stock_catalog); started = time.monotonic(); started_at = datetime.now(timezone.utc).isoformat(); visited = set(); queued = set(); discovery = 0; visit_order = 0; records_written = set(); statuses = {}
    articles_file = Path(args.articles_output)
    if articles_file.exists():
        with articles_file.open(encoding="utf-8") as existing_articles:
            for line in existing_articles:
                try:
                    existing_id = json.loads(line).get("article_id")
                    if existing_id: records_written.add(existing_id)
                except (ValueError, TypeError):
                    continue
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8")); expected = {"traversal": args.strategy, "frontier_policy": args.frontier_policy, "uniform_seed": args.uniform_seed, "roots": roots, "parser_version": PARSER_VERSION, "allowed_hosts": sorted(ALLOWED_HOSTS)}
        for key in expected:
            if key in state and state.get(key) != expected[key]: raise ValueError("resume state mismatch: " + key)
        visited = set(state.get("visited", [])); queued = set(state.get("queued", [])); discovery = int(state.get("discovery_count", 0)); visit_order = int(state.get("visit_count", len(visited)))
        frontier_items = [tuple(item) for item in state.get("frontier", [])]
    else:
        frontier_items = [(r, 0, r, None, i) for i, r in enumerate(roots)]; queued = set(roots)
    frontier = deque(frontier_items) if args.strategy == "bfs" and args.frontier_policy == "fifo" else (list(reversed(frontier_items)) if args.strategy == "dfs" else list(frontier_items)); robots = Robots(args.user_agent, not args.no_robots)
    def checkpoint() -> None:
        state = {"visited": sorted(visited), "queued": sorted(queued), "frontier": list(frontier), "visit_count": visit_order, "discovery_count": discovery, "last_success_at": datetime.now(timezone.utc).isoformat(), "traversal": args.strategy, "frontier_policy": args.frontier_policy, "uniform_seed": args.uniform_seed, "roots": roots, "allowed_hosts": sorted(ALLOWED_HOSTS), "parser_version": PARSER_VERSION, "parameters": vars(args)}
        tmp = state_path.with_suffix(state_path.suffix + ".tmp"); tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"); tmp.replace(state_path)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        with manifest_path.open("a", encoding="utf-8") as manifest, Path(args.articles_output).open("a", encoding="utf-8") as articles:
            while frontier and len(visited) < args.max_pages and time.monotonic() - started < args.max_seconds:
                if args.frontier_policy == "uniform":
                    best_index = min(range(len(frontier)), key=lambda index: uniform_frontier_key(frontier[index], args.uniform_seed))
                    item = frontier.pop(best_index)
                else:
                    item = frontier.popleft() if args.strategy == "bfs" else frontier.pop()
                url, depth, root_url, parent_url, order = item
                if url in visited: continue
                visited.add(url); visit_order += 1; retrieved = datetime.now(timezone.utc).isoformat(); base = {"requested_url": url, "final_url": url, "traversal": args.strategy, "root_url": root_url, "depth": depth, "parent_url": parent_url, "discovery_order": order, "visit_order": visit_order, "retrieved_at": retrieved}
                result = dict(base); links = []; raw_path = None
                if not robots.allowed(url): result.update(status="robots_denied", article_record_written=False)
                elif args.dry_run: result.update(status="ok", dry_run=True, article_record_written=False)
                else:
                    try:
                        if args.pause_seconds: time.sleep(args.pause_seconds)
                        request = Request(url, headers={"User-Agent": args.user_agent, "Accept": "text/html,application/xhtml+xml"})
                        attempt = 0; response = None; payload = b""
                        while attempt <= args.max_retries:
                            attempt += 1
                            try:
                                response = urlopen(request, timeout=args.timeout); payload = response.read(args.max_response_bytes + 1); break
                            except HTTPError as exc:
                                if exc.code in (403, 429, 503): raise
                                raise
                            except (TimeoutError, URLError):
                                if attempt > args.max_retries: raise
                                time.sleep(min(30.0, 2 ** (attempt - 1)))
                        if len(payload) > args.max_response_bytes: result.update(status="too_large", attempt=attempt); raise ValueError("response exceeds max-response-bytes")
                        ctype = response.headers.get("Content-Type", "") if response else ""
                        result.update(http_status=getattr(response, "status", 200), content_type=ctype.split(";", 1)[0].strip().lower(), attempt=attempt)
                        if ctype and "html" not in ctype.lower(): result.update(status="non_html", article_record_written=False); _write_jsonl(manifest, result); statuses[result["status"]] = statuses.get(result["status"], 0) + 1; continue
                        payload = decompress_payload(payload, response.headers.get("Content-Encoding", "") if response else "")
                        decoded, encoding = decode_payload(payload, ctype); result["encoding"] = encoding; result["content_length"] = len(payload)
                        parser = PageParser(); parser._all_link_matches = []; parser.feed(decoded)
                        if any(marker.lower() in (parser.visible_text + " " + decoded[:1000]).lower() for marker in BLOCK_MARKERS): result.update(status="blocked_marker", blocked_marker=True, article_record_written=False); _write_jsonl(manifest, result); statuses["blocked_marker"] = statuses.get("blocked_marker", 0) + 1; continue
                        raw_path = raw_dir / (article_id(url) + ".html"); raw_path.write_bytes(payload); links = list(dict.fromkeys(filter(None, (normalize_url(url, h, True) for h in parser.links))))
                        if args.historical_only_links:
                            links = [link for link in links if is_historical_article_url(link)]
                        links = [x for x in links if args.include_navigation_pages or url_priority(x, False) >= 0]
                        dates = extract_dates(parser, url); title = parser.meta.get("og:title") or parser.title; body = parser.body; quality = classify_body_quality(url, title, body, parser, dates["published_year"], args.min_body_chars)
                        all_stocks = parser._all_link_matches; body_stocks = parser.body_link_matches; catalog_hits = text_matches(title, body, catalog); final = []
                        for hit in all_stocks if args.paper_stock_matching else body_stocks:
                            if hit.get("stock_id") or hit.get("legacy_symbol"): final.append(hit)
                        final.extend(catalog_hits); unique = []; seen = set()
                        for hit in final:
                            key = hit.get("stock_id") or ("legacy:" + str(hit.get("legacy_symbol")));
                            if key not in seen: seen.add(key); unique.append(hit)
                        record_written = False
                        if quality == "valid" and (not args.exactly_one_stock or len(unique) == 1):
                            truncated = len(body) > args.max_body_chars; article = {"source":"sina_finance","content_type":"financial_news","article_id":article_id(url),"url":url,"canonical_url":url,"traversal":args.strategy,"root_url":root_url,"depth":depth,"parent_url":parent_url,"discovery_order":order,"visit_order":visit_order,"title":title,"title_raw":title,**dates,"body_extraction_method":"article_container" if parser.article_parts else "legacy_visible_text","body_quality":quality,"body_chars":len(body),"body":body[:args.max_body_chars],"body_sha256":hashlib.sha256(body.encode("utf-8")).hexdigest(),"paper_stock_tags":all_stocks,"body_stock_matches":body_stocks,"text_stock_matches":catalog_hits,"final_stock_matches":unique,"stock_match_count":len(unique),"raw_html_path":str(raw_path),"retrieved_at":retrieved,"http_status":result.get("http_status"),"content_type_header":ctype,"encoding":encoding,"parser_version":PARSER_VERSION,"body_truncated":truncated}
                            if article["article_id"] not in records_written:
                                _write_jsonl(articles, article); records_written.add(article["article_id"])
                            record_written = True
                        result.update(status="ok" if quality == "valid" else "low_quality", title=title, **dates, body_chars=len(body), body_quality=quality, article_record_written=record_written, links_found=len(links), links_enqueued=0, raw_html_path=str(raw_path), elapsed_seconds=round(time.monotonic() - started, 3))
                        for child in links:
                            if child not in queued and depth < args.max_depth:
                                queued.add(child); discovery += 1; child_item = (child, depth + 1, root_url, url, discovery)
                                if args.strategy == "bfs": frontier.append(child_item)
                                else: frontier.append(child_item)
                                result["links_enqueued"] += 1
                    except HTTPError as exc:
                        error_status = http_status_label(exc.code)
                        result.update(status=error_status, http_status=exc.code, error_type="http_error", error_message=str(exc), attempt=attempt)
                    except (TimeoutError, URLError) as exc: result.update(status="timeout" if isinstance(exc, TimeoutError) else "connection_error", error_type=type(exc).__name__, error_message=str(exc), attempt=locals().get("attempt", 1))
                    except UnicodeDecodeError as exc: result.update(status="decode_error", error_type="decode_error", error_message=str(exc))
                    except ValueError as exc: result.setdefault("status", "parse_error"); result.update(error_type="value_error", error_message=str(exc))
                    except Exception as exc: result.update(status="parse_error", error_type=type(exc).__name__, error_message=str(exc))
                _write_jsonl(manifest, result); statuses[result.get("status", "ok")] = statuses.get(result.get("status", "ok"), 0) + 1
                if visit_order % max(1, args.checkpoint_every) == 0: checkpoint()
    except KeyboardInterrupt:
        checkpoint(); raise
    checkpoint(); elapsed = round(time.monotonic() - started, 2)
    article_rows = []
    with articles_file.open(encoding="utf-8") as article_stream:
        for line in article_stream:
            try: article_rows.append(json.loads(line))
            except (ValueError, TypeError): continue
    manifest_rows = []
    with manifest_path.open(encoding="utf-8") as manifest_stream:
        for line in manifest_stream:
            try: manifest_rows.append(json.loads(line))
            except (ValueError, TypeError): continue
    year_counts = {}
    for row in article_rows:
        year = row.get("published_year")
        if year is not None: year_counts[str(year)] = year_counts.get(str(year), 0) + 1
    total_articles = len(article_rows)
    status_counts = {}
    quality_counts = {}
    for item in manifest_rows:
        status = item.get("status", "unknown")
        if status == "http_error" and isinstance(item.get("http_status"), int):
            status = http_status_label(item["http_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        quality = item.get("body_quality")
        if quality: quality_counts[quality] = quality_counts.get(quality, 0) + 1
    page_count = len(manifest_rows)
    unique_ids = len({row.get("article_id") for row in article_rows if row.get("article_id")})
    unique_body_hashes = len({row.get("body_sha256") for row in article_rows if row.get("body_sha256")})
    duplicate_body_rows = total_articles - unique_body_hashes
    root_counts = {}
    for row in manifest_rows:
        root = row.get("root_url", "unknown"); bucket = root_counts.setdefault(root, {"visited": 0, "valid_pages": 0, "article_records": 0, "http_404": 0})
        bucket["visited"] += 1
        bucket["valid_pages"] += int(row.get("body_quality") == "valid")
        bucket["article_records"] += int(row.get("article_record_written", False))
        bucket["http_404"] += int(row.get("status") == "http_404" or (row.get("status") == "http_error" and row.get("http_status") == 404))
    coverage = {"article_year_counts": dict(sorted(year_counts.items())), "earliest_published_year": min((int(y) for y in year_counts), default=None), "latest_published_year": max((int(y) for y in year_counts), default=None), "body_quality_rate": round(sum(row.get("body_quality") == "valid" for row in article_rows) / total_articles, 4) if total_articles else 0, "page_valid_rate": round(quality_counts.get("valid", 0) / page_count, 4) if page_count else 0, "article_record_rate": round(total_articles / page_count, 4) if page_count else 0, "page_quality_counts": quality_counts, "date_extraction_rate": round(sum(bool(row.get("published_date")) for row in article_rows) / total_articles, 4) if total_articles else 0, "stock_article_rate": round(sum(bool(row.get("final_stock_matches")) for row in article_rows) / total_articles, 4) if total_articles else 0, "unique_article_ids": unique_ids, "unique_body_hashes": unique_body_hashes, "duplicate_body_rows": duplicate_body_rows, "root_counts": root_counts}
    summary = {"source":"sina_finance","traversal":args.strategy,"started_at":started_at,"finished_at":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":elapsed,"roots":roots,"visited":len(visited),"successful_pages":status_counts.get("ok",0),"article_records":total_articles,"queued_remaining":len(frontier),"statuses":status_counts,"coverage":coverage,"parameters":vars(args),"files":{"articles_jsonl":str(args.articles_output),"manifest_jsonl":str(manifest_path),"raw_html_dir":str(raw_dir),"state_file":str(state_path)}}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"); return summary
