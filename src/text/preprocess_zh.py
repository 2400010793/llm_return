"""Conservative Chinese news preprocessing utilities."""

from __future__ import annotations

import hashlib
import re
from typing import Optional


_WHITESPACE = re.compile(r"\s+")
_HTML = re.compile(r"<[^>]+>")


def clean_text(text: Optional[str]) -> str:
    """Normalize a news field without deleting Chinese characters or numbers."""
    if not text:
        return ""
    value = _HTML.sub(" ", str(text))
    value = value.replace("\u3000", " ").replace("\ufeff", "")
    return _WHITESPACE.sub(" ", value).strip()


def combine_news_text(headline: Optional[str], body: Optional[str]) -> str:
    """Combine title and body while retaining an explicit title boundary."""
    title = clean_text(headline)
    content = clean_text(body)
    if title and content:
        return f"标题：{title}。正文：{content}"
    return title or content


def text_hash(text: str) -> str:
    """Return a stable hash for embedding-cache validation."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
