"""Conservative Chinese news preprocessing utilities."""

from __future__ import annotations

import hashlib
import html
import math
import re
import unicodedata
from typing import Optional


_WHITESPACE = re.compile(r"\s+")
_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")


def _normalize_compatibility_text(text: str) -> str:
    """Apply NFKC while retaining East Asian punctuation code points."""
    return "".join(
        char if unicodedata.category(char).startswith("P")
        else unicodedata.normalize("NFKC", char)
        for char in text
    )


def remove_invalid_unicode(text: str) -> str:
    """Remove lone UTF-16 surrogate code points from collector text.

    A few historical CNINFO JSON payloads contain escaped surrogate code
    points that are not valid Unicode scalar values. Keeping them makes both
    UTF-8 hashing and ``ensure_ascii=False`` JSON output fail. They carry no
    recoverable text content, so drop them before downstream processing.
    """
    result: list[str] = []
    index = 0
    while index < len(text):
        code_point = ord(text[index])
        if 0xD800 <= code_point <= 0xDBFF:
            if index + 1 < len(text):
                next_code_point = ord(text[index + 1])
                if 0xDC00 <= next_code_point <= 0xDFFF:
                    result.append(chr(
                        0x10000
                        + ((code_point - 0xD800) << 10)
                        + (next_code_point - 0xDC00)
                    ))
                    index += 2
                    continue
            index += 1
            continue
        if 0xDC00 <= code_point <= 0xDFFF:
            index += 1
            continue
        result.append(text[index])
        index += 1
    return "".join(result)


def clean_text(text: Optional[str]) -> str:
    """Normalize a news field without deleting Chinese characters or numbers."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    value = remove_invalid_unicode(str(text))
    value = _normalize_compatibility_text(value)
    value = html.unescape(value)
    value = _URL.sub(" ", value)
    value = _HTML.sub(" ", value)
    value = _INVISIBLE.sub("", value)
    value = _CONTROL.sub("", value)
    value = value.replace("\u3000", " ")
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
    return hashlib.sha256(remove_invalid_unicode(str(text)).encode("utf-8")).hexdigest()
