"""Deterministic deduplication helpers for browser-collected news records."""

from __future__ import annotations

import re
import pandas as pd

from src.text.preprocess_zh import clean_text, text_hash


def normalize_title(title: object) -> str:
    """Normalize title text for exact duplicate detection."""
    value = clean_text(str(title or "")).lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value)


def add_news_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Add stable URL, article-ID, title and text hashes without dropping rows."""
    result = frame.copy()
    empty = pd.Series("", index=result.index, dtype="string")
    result["article_id"] = result.get("article_id", empty).astype("string")
    url_values = result["url"] if "url" in result else result.get("source_url", empty)
    result["url_key"] = url_values.fillna("").astype(str).str.strip()
    result["article_id_key"] = result["article_id"].fillna("").astype(str).str.strip()
    title_values = result.get("title", empty)
    result["title_key"] = title_values.map(normalize_title)
    summary = result.get("summary", empty).fillna("").astype(str)
    result["text_key"] = (result["title_key"] + "|" + summary.map(normalize_title)).map(text_hash)
    return result


def deduplicate_news(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (unique_rows, duplicate_rows), preferring stable IDs then text keys."""
    keyed = add_news_keys(frame)
    duplicate = keyed["url_key"].ne("") & keyed["url_key"].duplicated(keep="first")
    duplicate |= keyed["url_key"].eq("") & keyed["article_id_key"].ne("") & keyed["article_id_key"].duplicated(keep="first")
    duplicate |= keyed["url_key"].eq("") & keyed["article_id_key"].eq("") & keyed["text_key"].duplicated(keep="first")
    return keyed.loc[~duplicate].reset_index(drop=True), keyed.loc[duplicate].reset_index(drop=True)
