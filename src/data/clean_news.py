"""Validate and normalize raw news records."""

from __future__ import annotations

import pandas as pd

from src.text.preprocess_zh import combine_news_text, text_hash


REQUIRED_COLUMNS = {"article_id", "stock_id", "published_at", "headline", "body"}


def clean_news(news: pd.DataFrame, timezone: str = "Asia/Shanghai") -> pd.DataFrame:
    """Normalize news timestamps, text, and duplicate records.

    The function does not infer missing stock IDs or timestamps; those records
    remain visible through validation errors rather than being silently dropped.
    """
    missing = REQUIRED_COLUMNS.difference(news.columns)
    if missing:
        raise ValueError(f"News table is missing columns: {', '.join(sorted(missing))}")
    result = news.copy()
    result["published_at"] = pd.to_datetime(result["published_at"], errors="coerce")
    if result["published_at"].isna().any():
        raise ValueError("News contains invalid published_at values")
    if result["published_at"].dt.tz is None:
        result["published_at"] = result["published_at"].dt.tz_localize(timezone)
    else:
        result["published_at"] = result["published_at"].dt.tz_convert(timezone)
    result["text"] = [
        combine_news_text(headline, body)
        for headline, body in zip(result["headline"], result["body"])
    ]
    result["text_hash"] = result["text"].map(text_hash)
    result = result.drop_duplicates(subset=["article_id", "stock_id"], keep="first")
    return result.sort_values(["published_at", "stock_id", "article_id"]).reset_index(drop=True)
