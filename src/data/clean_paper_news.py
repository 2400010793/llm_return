"""Deterministic, auditable news cleaning for the paper-style experiments."""

from __future__ import annotations

import hashlib
import html
import re
from typing import Any

import pandas as pd

from src.text.bow_features import tokenize_zh_words

_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)

# Common page chrome that must not become a text signal.
_BOILERPLATE = (
    "责任编辑", "文章来源", "原标题", "免责声明", "风险提示",
    "版权声明", "未经许可不得转载", "点击进入", "下载客户端",
)


def normalize_text(value: Any) -> str:
    """Remove markup, URLs and layout noise without deleting Chinese text."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = html.unescape(str(value)).replace("\ufeff", "").replace("\u3000", " ")
    text = _HTML.sub(" ", text)
    text = _URL.sub(" ", text)
    text = text.replace("\r", "\n")
    lines: list[str] = []
    for line in text.split("\n"):
        line = _WHITESPACE.sub(" ", line).strip()
        if not line or any(line.startswith(marker) for marker in _BOILERPLATE):
            continue
        lines.append(line)
    return _WHITESPACE.sub(" ", " ".join(lines)).strip()


def paper_token_text(text: str, stopwords: set[str] | None = None) -> str:
    """Create a stable word-level representation for Chinese BOW/TF-IDF."""
    stopwords = stopwords or set()
    tokens = [token for token in tokenize_zh_words(text) if token not in stopwords]
    return " ".join(tokens)


def clean_news_frame(
    frame: pd.DataFrame,
    *,
    min_chars: int = 50,
    max_chars: int = 50000,
    timezone: str = "Asia/Shanghai",
    stopwords: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean, validate and exact-deduplicate news without using future data."""
    aliases = {"headline": "title", "content": "body", "published": "published_at"}
    result = frame.rename(columns={k: v for k, v in aliases.items() if k in frame and v not in frame}).copy()
    # CNINFO pages contain a short HTML viewer body and a substantially richer
    # extracted PDF body.  Prefer the PDF text, but retain a fallback marker.
    if "pdf_text" in result:
        pdf = result["pdf_text"].fillna("").map(str).map(normalize_text)
        html_body = result.get("body", pd.Series("", index=result.index)).fillna("").map(str).map(normalize_text)
        result["body_source"] = pdf.map(lambda value: "pdf" if value else "html_body")
        result["body"] = pdf.where(pdf.str.len() > 0, html_body)
    required = {"title", "body", "published_at"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"news table is missing columns: {', '.join(sorted(missing))}")
    if "article_id" not in result:
        result["article_id"] = result.index.astype(str)
    if "stock_id" not in result:
        result["stock_id"] = ""
    result["published_at"] = pd.to_datetime(result["published_at"], errors="coerce")
    if result["published_at"].dt.tz is None:
        result["published_at"] = result["published_at"].dt.tz_localize(timezone)
    else:
        result["published_at"] = result["published_at"].dt.tz_convert(timezone)
    result["title_clean"] = result["title"].map(normalize_text)
    result["body_raw_chars"] = result["body"].fillna("").map(lambda x: len(str(x)))
    result["body_clean"] = result["body"].map(normalize_text)
    result["body_clean_chars"] = result["body_clean"].str.len()
    result["text"] = (result["title_clean"] + "。" + result["body_clean"]).str.strip("。 ")
    result["text_chars"] = result["text"].str.len()
    result["text_hash"] = result["text"].map(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())
    result["paper_tokens"] = result["text"].map(lambda x: paper_token_text(x, stopwords))
    result["paper_token_count"] = result["paper_tokens"].map(lambda x: len(x.split()) if x else 0)
    before = len(result)
    result = result[result["published_at"].notna()]
    invalid_time = before - len(result)
    before = len(result)
    result = result[result["text_chars"].between(min_chars, max_chars)]
    invalid_length = before - len(result)
    before = len(result)
    result = result.drop_duplicates(subset=["text_hash", "stock_id"], keep="first")
    duplicates = before - len(result)
    result = result.sort_values(["published_at", "stock_id", "article_id"]).reset_index(drop=True)
    stats = {"raw_rows": int(len(frame)), "invalid_time": int(invalid_time), "invalid_length": int(invalid_length), "duplicates": int(duplicates), "clean_rows": int(len(result)), "stocks": int(result["stock_id"].nunique()), "body_source_pdf": int((result.get("body_source", pd.Series("", index=result.index)) == "pdf").sum()), "body_source_html": int((result.get("body_source", pd.Series("", index=result.index)) == "html_body").sum()), "body_clean_chars_total": int(result["body_clean_chars"].sum())}
    return result, stats
