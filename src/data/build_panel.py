"""Build news observations with forward return labels without look-ahead."""

from __future__ import annotations

import pandas as pd


def _next_trading_date(date: pd.Timestamp, trading_dates: pd.DatetimeIndex) -> pd.Timestamp:
    position = trading_dates.searchsorted(date, side="left")
    if position >= len(trading_dates):
        return pd.NaT
    return trading_dates[position]


def attach_forward_returns(
    news: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 20),
    close_time: str = "15:00",
) -> pd.DataFrame:
    """Attach forward close-to-close returns based on news availability.

    News before the local market close maps to the next trading day; news at
    or after close maps to the following trading day. The target is the return
    from that entry day's close to the close `horizon` trading days later.
    """
    if "published_at" not in news or "stock_id" not in news:
        raise ValueError("News must contain published_at and stock_id")
    if not {"stock_id", "date", "close"}.issubset(prices.columns):
        raise ValueError("Prices must contain stock_id, date, and close")
    result = news.copy()
    # The baseline uses next-day close-to-close returns for every news item.
    # Keeping the close parameter explicit documents the market convention and
    # leaves room for an intraday variant without changing the public API.
    _ = pd.to_datetime(close_time).time()
    result["available_date"] = (
        result["published_at"].dt.normalize().dt.tz_localize(None)
        + pd.Timedelta(days=1)
    )
    trading_dates = pd.DatetimeIndex(sorted(prices["date"].drop_duplicates()))
    result["entry_date"] = result["available_date"].map(
        lambda value: _next_trading_date(value, trading_dates)
    )
    price_lookup = prices.set_index(["stock_id", "date"])["close"]
    for horizon in horizons:
        def forward_return(row: pd.Series) -> float:
            if pd.isna(row["entry_date"]):
                return float("nan")
            dates = trading_dates[trading_dates >= row["entry_date"]]
            if len(dates) <= horizon:
                return float("nan")
            start = price_lookup.get((row["stock_id"], dates[0]))
            end = price_lookup.get((row["stock_id"], dates[horizon]))
            if start is None or end is None:
                return float("nan")
            return float(end / start - 1.0)
        result[f"ret_{horizon}d"] = result.apply(forward_return, axis=1)
    return result
