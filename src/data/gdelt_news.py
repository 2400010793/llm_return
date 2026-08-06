"""Download a small, reproducible GDELT DOC 2.0 news sample.

GDELT is used here for a public prototype only. The DOC API returns article
metadata and URLs, not a guaranteed historical full-text archive. Final
replication should replace this input with licensed RavenPack/Wind/CSMAR data
when available.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def build_query_url(
    query: str,
    *,
    max_records: int = 250,
    start_datetime: str | None = None,
    end_datetime: str | None = None,
) -> str:
    """Build a GDELT DOC 2.0 article-list URL."""
    params: dict[str, Any] = {
        "query": query,
        "mode": "artlist",
        "maxrecords": max_records,
        "format": "json",
        "sort": "datedesc",
    }
    if start_datetime:
        params["startdatetime"] = start_datetime
    if end_datetime:
        params["enddatetime"] = end_datetime
    return f"{API_URL}?{urllib.parse.urlencode(params)}"


def fetch_article_list(url: str, *, timeout: float = 30.0) -> list[dict[str, Any]]:
    """Fetch and validate the article list returned by GDELT."""
    request = urllib.request.Request(url, headers={"User-Agent": "llm-return-research/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        raise ValueError("Unexpected GDELT response: articles is not a list")
    return [item for item in articles if isinstance(item, dict)]


def normalize_articles(articles: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Normalize GDELT metadata into the project's raw-news schema."""
    rows = []
    for item in articles:
        url = str(item.get("url", ""))
        title = str(item.get("title", ""))
        seen_at = item.get("seendate")
        rows.append(
            {
                "article_id": url or title,
                "stock_id": pd.NA,
                "published_at": seen_at,
                "source": item.get("domain"),
                "headline": title,
                "body": "",  # DOC article-list mode does not guarantee full text.
                "url": url,
                "language": item.get("language"),
                "source_country": item.get("sourcecountry"),
                "data_source": "gdelt_doc_api",
            }
        )
    return pd.DataFrame(rows)


def download_sample(
    queries: Iterable[str],
    output_path: str | Path,
    *,
    max_records_per_query: int = 250,
    pause_seconds: float = 1.0,
) -> pd.DataFrame:
    """Download a small sample with a pause between public API requests."""
    records: list[dict[str, Any]] = []
    for index, query in enumerate(queries):
        if index:
            time.sleep(pause_seconds)
        records.extend(
            fetch_article_list(build_query_url(query, max_records=max_records_per_query))
        )
    result = normalize_articles(records).drop_duplicates(subset=["article_id"])
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".csv":
        result.to_csv(destination, index=False)
    else:
        result.to_parquet(destination, index=False)
    return result
