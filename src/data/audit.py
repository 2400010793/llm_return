"""Data-audit summaries for raw news and price tables."""

from __future__ import annotations

from typing import Any

import pandas as pd


def audit_news(news: pd.DataFrame) -> dict[str, Any]:
    """Return a compact, JSON-serializable news quality report."""
    text = news.get("text", pd.Series(dtype="object")).fillna("").astype(str)
    report: dict[str, Any] = {
        "rows": int(len(news)),
        "columns": list(news.columns),
        "duplicate_article_ids": int(news.get("article_id", pd.Series(dtype="object")).duplicated().sum()),
        "missing_stock_id": int(news.get("stock_id", pd.Series(dtype="object")).isna().sum()),
        "missing_published_at": int(news.get("published_at", pd.Series(dtype="object")).isna().sum()),
        "empty_text": int((text.str.len() == 0).sum()),
        "unique_text_hashes": int(news.get("text_hash", text).nunique()),
        "text_length": {
            "min": int(text.str.len().min()) if len(text) else 0,
            "median": float(text.str.len().median()) if len(text) else 0.0,
            "max": int(text.str.len().max()) if len(text) else 0,
        },
    }
    if "published_at" in news and len(news):
        dates = pd.to_datetime(news["published_at"], errors="coerce")
        report["published_at_min"] = dates.min().isoformat()
        report["published_at_max"] = dates.max().isoformat()
    return report


def audit_prices(prices: pd.DataFrame) -> dict[str, Any]:
    """Return a compact, JSON-serializable price quality report."""
    report: dict[str, Any] = {
        "rows": int(len(prices)),
        "columns": list(prices.columns),
        "unique_stocks": int(prices["stock_id"].nunique()) if "stock_id" in prices else 0,
        "duplicate_stock_dates": int(prices.duplicated(["stock_id", "date"]).sum())
        if {"stock_id", "date"}.issubset(prices.columns)
        else None,
    }
    if "date" in prices and len(prices):
        dates = pd.to_datetime(prices["date"], errors="coerce")
        report["date_min"] = dates.min().isoformat()
        report["date_max"] = dates.max().isoformat()
    return report
