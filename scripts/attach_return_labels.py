"""Attach forward return labels from a wide daily-return Parquet file."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_news_time(frame: pd.DataFrame) -> pd.Series:
    value = frame.get("published_at", pd.Series(pd.NaT, index=frame.index))
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    display = frame.get("published_at_display", pd.Series("", index=frame.index)).fillna("").astype(str)
    display_clean = display.str.replace("年", "-", regex=False).str.replace("月", "-", regex=False).str.replace("日", "", regex=False)
    display_parsed = pd.to_datetime(display_clean, errors="coerce", utc=True)
    body = frame.get("body", pd.Series("", index=frame.index)).fillna("").astype(str)
    extracted = body.str.extract(r"(20\d{2})年(\d{1,2})月(\d{1,2})日(?:\s+([0-9]{1,2}):([0-9]{2}))?", expand=True)
    body_clean = extracted[0] + "-" + extracted[1].str.zfill(2) + "-" + extracted[2].str.zfill(2) + " " + extracted[3].fillna("00").str.zfill(2) + ":" + extracted[4].fillna("00").str.zfill(2)
    body_parsed = pd.to_datetime(body_clean, errors="coerce", utc=True)
    result = parsed.fillna(display_parsed).fillna(body_parsed)
    # Guba list rows expose month-day and time in ``last_updated``. Resolve
    # the year from the collection timestamp without treating collection time
    # as publication time for rows that have no source timestamp.
    updated = frame.get("last_updated", pd.Series("", index=frame.index)).fillna("").astype(str)
    updated_match = updated.str.extract(r"(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})", expand=True)
    collected = pd.to_datetime(frame.get("collected_at", pd.Series(pd.NaT, index=frame.index)), errors="coerce", utc=True)
    updated_year = collected.dt.year.fillna(pd.Timestamp.utcnow().year).astype("Int64").astype(str)
    updated_clean = updated_year + "-" + updated_match[0].str.zfill(2) + "-" + updated_match[1].str.zfill(2) + " " + updated_match[2].str.zfill(2) + ":" + updated_match[3].str.zfill(2)
    updated_parsed = pd.to_datetime(updated_clean, errors="coerce", utc=True)
    return result.fillna(updated_parsed)


def code_with_exchange(code: object) -> str | None:
    raw = str(code or "").strip().zfill(6)
    if not re.fullmatch(r"\d{6}", raw):
        return None
    exchange = "SH" if raw.startswith(("5", "6")) else "SZ"
    return f"{raw}.{exchange}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("panel")
    parser.add_argument("--returns", default="/home/gaozh/ret.parquet")
    parser.add_argument("--output", default="data/processed/stock_text_panel_100_labeled.parquet")
    parser.add_argument("--horizons", default="1,5,20")
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel) if args.panel.endswith(".parquet") else pd.read_csv(args.panel)
    returns = pd.read_parquet(args.returns)
    returns.index = pd.to_datetime(returns.index).tz_localize(None).normalize()
    returns = returns.sort_index()
    trading_dates = pd.DatetimeIndex(returns.index.drop_duplicates())
    news_time = parse_news_time(panel)
    news_date = news_time.dt.tz_convert(None).dt.normalize()
    panel = panel.copy()
    panel["news_time"] = news_time
    panel["available_date"] = news_date + pd.Timedelta(days=1)
    panel["entry_date"] = panel["available_date"].map(
        lambda x: trading_dates[trading_dates >= x][0] if pd.notna(x) and (trading_dates >= x).any() else pd.NaT
    )
    panel["return_ticker"] = panel["stock_id"].map(code_with_exchange)
    horizons = tuple(int(x) for x in args.horizons.split(",") if x.strip())
    labels = []
    for _, row in panel.iterrows():
        ticker = row["return_ticker"]
        entry = row["entry_date"]
        values = returns[ticker] if ticker in returns.columns else None
        item = {}
        for horizon in horizons:
            label = float("nan")
            if values is not None and pd.notna(entry):
                dates = trading_dates[trading_dates >= entry]
                if len(dates) > horizon:
                    window = pd.to_numeric(values.reindex(dates[: horizon + 1]), errors="coerce").dropna()
                    if len(window) == horizon + 1:
                        label = float((1.0 + window.iloc[1:]).prod() - 1.0)
            item[f"return_{horizon}d"] = label
        labels.append(item)
    panel = pd.concat([panel.reset_index(drop=True), pd.DataFrame(labels)], axis=1)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(destination, index=False) if destination.suffix == ".parquet" else panel.to_csv(destination, index=False)
    summary = {"rows": len(panel), "stocks": int(panel["stock_id"].nunique()), "news_time_valid": int(panel["news_time"].notna().sum()), "labels": {f"return_{h}d": int(panel[f"return_{h}d"].notna().sum()) for h in horizons}, "output": str(destination)}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
