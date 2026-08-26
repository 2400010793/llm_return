"""Build a return-labeled panel for the cleaned single-stock Sina corpus.

The target definitions intentionally match the repository's existing CNINFO
classification baseline: a three-trading-day event return is the weak training
target, while the first exchange day's return strictly after publication is
the out-of-sample evaluation target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_cninfo_100_classification_panel import attach_returns, code_with_exchange


REQUIRED_COLUMNS = {
    "article_id",
    "stock_id",
    "stock_name",
    "published_at",
    "text",
    "text_hash",
}


def build_panel(news: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(news.columns)
    if missing:
        raise ValueError(f"news input missing columns: {', '.join(sorted(missing))}")

    frame = news.copy()
    frame["article_id"] = frame["article_id"].astype(str).str.strip()
    if frame["article_id"].eq("").any() or frame["article_id"].duplicated().any():
        raise ValueError("article_id must be present and unique")
    frame["stock_id"] = (
        frame["stock_id"].astype(str).str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    )
    if frame["stock_id"].isna().any():
        raise ValueError("stock_id contains invalid values")

    # Parse in UTC first so all ISO timestamps, including explicit +08:00
    # offsets, resolve to the same local publication date.
    frame["published_at"] = (
        pd.to_datetime(frame["published_at"], errors="coerce", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    if frame["published_at"].isna().any():
        raise ValueError("published_at contains invalid values")
    frame["return_ticker"] = frame["stock_id"].map(code_with_exchange)
    if frame["return_ticker"].isna().any():
        raise ValueError("could not construct return ticker for every stock")

    result = attach_returns(frame, returns)
    result.insert(0, "row_index", range(1, len(result) + 1))
    result["label_available"] = result["next_day_return"].notna()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--news",
        type=Path,
        default=Path(
            "/home/gaozh/news_content_quality_20260812/cleaned/"
            "sina_single_stock_clean.parquet"
        ),
    )
    parser.add_argument("--returns", type=Path, default=Path("/home/gaozh/ret.parquet"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/gaozh/news_content_quality_20260812/classification/"
            "sina_single_stock_classification_panel.parquet"
        ),
    )
    args = parser.parse_args()

    returns = pd.read_parquet(args.returns)
    result = build_panel(pd.read_parquet(args.news), returns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)

    valid = result[result["next_day_return"].notna()].copy()
    year_summary = []
    for year, group in valid.groupby(valid["entry_date"].dt.year, sort=True):
        year_summary.append({
            "year": int(year),
            "rows": int(len(group)),
            "stocks": int(group["stock_id"].nunique()),
            "positive_rate": float(group["next_day_label"].mean()),
        })
    summary = {
        "news": str(args.news),
        "returns": str(args.returns),
        "output": str(args.output),
        "rows": int(len(result)),
        "stocks": int(result["stock_id"].nunique()),
        "article_id_unique": int(result["article_id"].nunique()),
        "event_target_valid": int(result["event_return_3d"].notna().sum()),
        "evaluation_target_valid": int(result["next_day_return"].notna().sum()),
        "evaluation_positive_rate": float(valid["next_day_label"].mean()),
        "returns_last_date": str(pd.to_datetime(returns.index, errors="coerce").max().date()),
        "year_summary": year_summary,
        "definitions": {
            "train_target": "event_return_3d: three return observations centered on publication date",
            "evaluation_target": "next_day_return: return on first exchange date strictly after publication date",
            "entry_rule": "first exchange date strictly greater than the Asia/Shanghai publication date",
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
