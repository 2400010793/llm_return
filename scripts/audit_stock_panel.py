"""Audit the normalized news-stock panel before modeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_frame(path: str) -> pd.DataFrame:
    source = Path(path)
    return pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="reports/stock_panel_audit.json")
    args = parser.parse_args()
    frame = load_frame(args.input)
    required = ["stock_id", "source", "content_type", "title", "body", "text", "published_at", "url"]
    missing = [column for column in required if column not in frame]
    date = pd.to_datetime(frame.get("published_at"), errors="coerce", utc=True) if "published_at" in frame else pd.Series(dtype="datetime64[ns, UTC]")
    body = frame.get("body", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    text = frame.get("text", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    duplicate_mask = frame.duplicated(subset=[c for c in ["content_type", "stock_id", "url", "text_hash"] if c in frame], keep=False)
    known_types = {"stock_news", "notice", "research", "forum_post", "forum_post_detail", "market_discussion", "stock_page"}
    relation_counts = frame.get("stock_relation", pd.Series("missing", index=frame.index)).fillna("missing").value_counts().to_dict()
    report = {
        "rows": int(len(frame)),
        "stocks": int(frame["stock_id"].nunique()) if "stock_id" in frame else 0,
        "sources": frame.get("source", pd.Series(dtype=str)).value_counts().to_dict(),
        "content_types": frame.get("content_type", pd.Series(dtype=str)).value_counts().to_dict(),
        "relation_counts": relation_counts,
        "missing_required_columns": missing,
        "empty_body_rows": int((body == "").sum()),
        "empty_text_rows": int((text == "").sum()),
        "invalid_published_at_rows": int(date.isna().sum()),
        "duplicate_candidate_rows": int(duplicate_mask.sum()),
        "unknown_content_type_rows": int((~frame.get("content_type", pd.Series(dtype=str)).isin(known_types)).sum()),
        "stock_mismatch_candidates": int((frame.get("stock_id", pd.Series("", index=frame.index)).fillna("").astype(str).str.len() != 6).sum()),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
