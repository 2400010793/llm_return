"""Merge stock-specific collection JSON files into a normalized table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.dedup_news import deduplicate_news
from src.text.preprocess_zh import combine_news_text, text_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", default="data/processed/stock_text_panel.parquet")
    args = parser.parse_args()
    rows = []
    for filename in args.inputs:
        payload = json.loads(Path(filename).read_text(encoding="utf-8"))
        rows.extend(payload.get("records", []))
    frame = pd.DataFrame(rows)
    # Focus-channel list records are intentionally excluded: the panel is
    # stock-specific only. Research links and quote pages remain as metadata,
    # while text modeling later uses non-empty article/post bodies.
    frame = frame[frame.get("content_type", "").ne("news_list")].copy()
    for column in ("stock_id", "stock_name", "content_type", "title", "body", "url", "published_at", "published_at_display", "collected_at", "source"):
        if column not in frame:
            frame[column] = None
    frame["text"] = [combine_news_text(a, b) for a, b in zip(frame["title"], frame["body"])]
    frame["text_hash"] = frame["text"].map(text_hash)
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce", utc=True)
    frame["collected_at"] = pd.to_datetime(frame["collected_at"], errors="coerce", utc=True)
    frame["stock_relation"] = frame.get("stock_relation", "direct")
    frame = frame.drop_duplicates(subset=["content_type", "url", "text_hash"], keep="first")
    frame, duplicates = deduplicate_news(frame)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".parquet":
        frame.to_parquet(destination, index=False)
    else:
        frame.to_csv(destination, index=False)
    duplicates.to_csv(destination.with_suffix(".duplicates.csv"), index=False)
    print(json.dumps({"rows": len(frame), "duplicates": len(duplicates), "output": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
