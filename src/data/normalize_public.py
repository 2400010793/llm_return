"""Normalize user-exported public news/forum records into one schema."""

from __future__ import annotations

import pandas as pd


NEWS_COLUMNS = {
    "article_id",
    "stock_id",
    "published_at",
    "headline",
    "body",
}

FORUM_COLUMNS = {
    "post_id",
    "stock_id",
    "published_at",
    "author_id_hash",
    "title",
    "content",
}


def normalize_news_export(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize a permitted Eastmoney/Snowball/news export."""
    missing = NEWS_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"News export missing columns: {', '.join(sorted(missing))}")
    result = frame.copy()
    result["source"] = source
    result["data_type"] = "news"
    return result


def normalize_forum_export(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize a permitted forum export without retaining raw author identity."""
    missing = FORUM_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Forum export missing columns: {', '.join(sorted(missing))}")
    result = frame.rename(columns={"post_id": "article_id", "title": "headline", "content": "body"}).copy()
    result["source"] = source
    result["data_type"] = "forum_post"
    return result
