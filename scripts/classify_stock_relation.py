"""Rule-based relation labels for stock-specific text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd


def classify(row: pd.Series) -> str:
    text = f"{row.get('title') or ''} {row.get('body') or ''}"
    name = str(row.get("stock_name") or "")
    code = str(row.get("stock_id") or "")
    if name and name in text or code and code in text:
        return "direct"
    if row.get("content_type") in {"notice", "forum_post", "forum_post_detail", "stock_news"}:
        return "direct"
    if row.get("content_type") in {"market_discussion", "stock_page"}:
        return "market"
    return "unrelated"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="data/processed/stock_text_panel_labeled.parquet")
    args = parser.parse_args()
    frame = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    frame["stock_relation"] = frame.apply(classify, axis=1)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".parquet":
        frame.to_parquet(destination, index=False)
    else:
        frame.to_csv(destination, index=False)
    print(frame["stock_relation"].value_counts(dropna=False).to_json(force_ascii=False))


if __name__ == "__main__":
    main()
