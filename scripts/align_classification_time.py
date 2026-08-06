"""Add author-style classification timing labels without changing row order.

This deliberately operates on the existing panel so precomputed embeddings
remain positionally aligned. Text cleaning and deduplication are out of scope
for this step.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def ticker_for(stock_id: object) -> str | None:
    code = str(stock_id or "").strip().zfill(6)
    if not re.fullmatch(r"\d{6}", code):
        return None
    return f"{code}.{'SH' if code.startswith(('5', '6')) else 'SZ'}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("panel")
    parser.add_argument("returns")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    frame = pd.read_parquet(args.panel).copy()
    original_order = frame["document_id"].astype(str).tolist()
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce")
    dates = pd.read_parquet(args.returns)
    dates.index = pd.to_datetime(dates.index, errors="coerce").tz_localize(None).normalize()
    dates = dates[~dates.index.isna()].sort_index()
    trading_dates = pd.DatetimeIndex(dates.index.drop_duplicates())

    frame["return_ticker"] = frame["stock_id"].map(ticker_for)
    frame["publication_date"] = frame["published_at"].dt.tz_localize(None).dt.normalize()
    frame["event_center_date"] = frame["publication_date"].map(
        lambda value: trading_dates[trading_dates >= value][0]
        if pd.notna(value) and (trading_dates >= value).any() else pd.NaT
    )
    # The first tradable day after publication is the next-period entry day.
    frame["trade_date"] = frame["publication_date"].map(
        lambda value: trading_dates[trading_dates > value][0]
        if pd.notna(value) and (trading_dates > value).any() else pd.NaT
    )

    event_returns: list[float] = []
    event_starts: list[pd.Timestamp] = []
    event_ends: list[pd.Timestamp] = []
    next_returns: list[float] = []
    for row in frame.itertuples(index=False):
        values = dates[row.return_ticker] if row.return_ticker in dates.columns else None
        center = row.event_center_date
        trade_date = row.trade_date
        event_value = np.nan
        event_start = pd.NaT
        event_end = pd.NaT
        next_value = np.nan
        if values is not None and pd.notna(center):
            center_pos = trading_dates.searchsorted(center, side="left")
            window_dates = trading_dates[center_pos - 1 : center_pos + 2]
            if len(window_dates) == 3:
                window = pd.to_numeric(values.reindex(window_dates), errors="coerce")
                if window.notna().all():
                    event_value = float((1.0 + window).prod() - 1.0)
                    event_start, event_end = window_dates[0], window_dates[-1]
        if values is not None and pd.notna(trade_date):
            next_value = pd.to_numeric(values.get(trade_date), errors="coerce")
            next_value = float(next_value) if pd.notna(next_value) else np.nan
        event_returns.append(event_value)
        event_starts.append(event_start)
        event_ends.append(event_end)
        next_returns.append(next_value)

    frame["event_start_date"] = event_starts
    frame["event_end_date"] = event_ends
    frame["event_return_3d"] = event_returns
    frame["event_label"] = np.where(
        pd.notna(frame["event_return_3d"]),
        (frame["event_return_3d"] > 0).astype("int8"),
        np.nan,
    )
    frame["next_day_return"] = next_returns
    frame["next_day_label"] = np.where(
        pd.notna(frame["next_day_return"]),
        (frame["next_day_return"] > 0).astype("int8"),
        np.nan,
    )

    # Preserve exact row order so existing embedding rows remain valid.
    if frame["document_id"].astype(str).tolist() != original_order:
        raise RuntimeError("time alignment changed panel row order")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    summary = {
        "input": args.panel,
        "output": str(output),
        "rows": int(len(frame)),
        "stocks": int(frame["stock_id"].nunique()),
        "event_label_valid": int(frame["event_label"].notna().sum()),
        "next_day_label_valid": int(frame["next_day_label"].notna().sum()),
        "event_positive_rate": float(frame["event_label"].mean()),
        "next_day_positive_rate": float(frame["next_day_label"].mean()),
        "row_order_preserved": True,
        "embedding_reuse_allowed": True,
    }
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()