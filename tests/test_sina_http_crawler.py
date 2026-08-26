import gzip
import sys
from urllib.error import HTTPError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from sina_http_crawler_common import (  # noqa: E402
    PageParser,
    decode_payload,
    extract_dates,
    normalize_url,
    stock_href_match,
    http_status_label,
    is_historical_article_url,
    uniform_frontier_key,
)


def test_normalize_url_keeps_old_http_and_removes_fragment_tracking():
    value = normalize_url(
        "https://finance.sina.com.cn/t/1.html",
        "../view/2000-06-11/2.html?from=x&id=22#comments",
    )
    assert value == "https://finance.sina.com.cn/view/2000-06-11/2.html?id=22"
    assert normalize_url(value, "https://example.com/a.html") is None


def test_parser_extracts_legacy_date_body_and_separates_sidebar_stock():
    parser = PageParser()
    parser._all_link_matches = []
    parser.feed(
        """<html><head><title>旧闻 2001年02月05日</title></head><body>
        <table id='article'><tr><td>2001年02月05日 10:57 正文讨论 2000年股市。</td>
        <td><a href='/cgi-bin/stock/quote/quote.cgi?symbol=0008'>正文股票</a></td></tr></table>
        <div class='sidebar'><a href='/realstock/company/sh600519/'>推荐</a></div>
        </body></html>"""
    )
    dates = extract_dates(parser, "https://finance.sina.com.cn/t/1.html")
    assert dates["published_year"] == 2001
    assert dates["published_date"] == "2001-02-05"
    assert 2000 in dates["content_year_candidates"]
    assert parser.body
    assert any(item.get("legacy_symbol") == "0008" for item in parser.body_link_matches)
    assert any(item.get("stock_id") == "600519" for item in parser._all_link_matches)


def test_stock_link_formats_and_encoding_fallback():
    assert stock_href_match("/realstock/company/sz002280/")["stock_id"] == "002280"
    legacy = stock_href_match("/cgi-bin/stock/quote/quote.cgi?symbol=0021")
    assert legacy["legacy_symbol"] == "0021"
    assert legacy["stock_id_mapping_status"] == "unmapped"
    text, encoding = decode_payload("中文标题".encode("gb18030"), "text/html")
    assert text == "中文标题"
    assert encoding in {"utf-8", "gb18030", "gbk"}
    assert gzip.decompress(gzip.compress(b"html")) == b"html"


def test_uniform_frontier_key_is_reproducible_and_seeded():
    item = ("https://finance.sina.com.cn/t/34687.html", 0, "root", None, 0)
    other = ("https://finance.sina.com.cn/t/34688.html", 0, "root", None, 1)
    assert uniform_frontier_key(item, 7) == uniform_frontier_key(item, 7)
    assert uniform_frontier_key(item, 7) != uniform_frontier_key(item, 8)
    assert uniform_frontier_key(item, 7) != uniform_frontier_key(other, 7)


def test_http_error_status_categories_are_available():
    assert HTTPError("https://example.com", 404, "missing", {}, None).code == 404
    assert HTTPError("https://example.com", 429, "limited", {}, None).code == 429
    assert http_status_label(404) == "http_404"
    assert http_status_label(429) == "http_429"
    assert http_status_label(503) == "http_5xx"


def test_historical_only_link_filter_accepts_legacy_article_shapes():
    assert is_historical_article_url("https://finance.sina.com.cn/t/34687.html")
    assert is_historical_article_url("http://finance.sina.com.cn/e/32600.html")
    assert not is_historical_article_url("https://finance.sina.com.cn/t/2001")
    assert not is_historical_article_url("https://finance.sina.com.cn/stock/t/2026-08-06/doc-x.shtml")
