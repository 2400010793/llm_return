"""Validate and normalize daily stock prices."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {"stock_id", "date", "close"}


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Normalize daily price records and reject duplicate stock-date rows."""
    missing = REQUIRED_COLUMNS.difference(prices.columns)
    if missing:
        raise ValueError(f"Price table is missing columns: {', '.join(sorted(missing))}")
    result = prices.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if result["date"].isna().any():
        raise ValueError("Prices contain invalid date values")
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    if result["close"].isna().any() or (result["close"] <= 0).any():
        raise ValueError("Prices contain missing or non-positive close values")
    if result.duplicated(["stock_id", "date"]).any():
        raise ValueError("Prices contain duplicate stock_id/date rows")
    return result.sort_values(["stock_id", "date"]).reset_index(drop=True)
