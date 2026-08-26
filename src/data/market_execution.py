"""Build executable daily return panels from adjusted OHLC observations."""

from __future__ import annotations

import numpy as np
import pandas as pd


OHLC_COLUMNS = {"date", "stock_id", "open", "close", "high", "low"}


def normalize_akshare_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize ``ak.stock_zh_a_hist`` output to stable English columns."""
    mapping = {
        "日期": "date",
        "股票代码": "stock_id",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover_pct",
    }
    result = frame.rename(columns=mapping).copy()
    missing = OHLC_COLUMNS.difference(result.columns)
    if missing:
        raise ValueError(f"OHLC data missing columns: {', '.join(sorted(missing))}")
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["stock_id"] = result["stock_id"].astype(str).str.extract(r"(\d{6})", expand=False)
    numeric = ["open", "close", "high", "low", "volume", "amount", "turnover_pct"]
    for column in numeric:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["date", "stock_id", "open", "close"])
    # Reject invalid adjusted prices before they can become return labels.
    price_columns = [column for column in ("open", "close", "high", "low") if column in result]
    invalid_prices = result[price_columns].le(0).any(axis=1)
    if invalid_prices.any():
        examples = result.loc[invalid_prices, ["stock_id", "date", *price_columns]].head(3)
        raise ValueError(
            "OHLC contains non-positive prices; choose a valid adjustment mode. "
            f"examples={examples.to_dict(orient='records')}"
        )
    return result.sort_values(["stock_id", "date"]).drop_duplicates(
        ["stock_id", "date"], keep="last"
    )


def price_limit_fraction(stock_id: object) -> float:
    """Return the standard board-level daily price-limit fraction.

    This is a conservative execution proxy and does not identify ST securities
    or temporary rule changes.  IPO initial trading days are handled separately.
    """
    code = str(stock_id).zfill(6)
    if code.startswith(("300", "301", "688")):
        return 0.20
    if code.startswith(("4", "8", "920")):
        return 0.30
    return 0.10


def build_execution_market_panel(
    ohlc: pd.DataFrame,
    *,
    start: object | None = None,
    end: object | None = None,
    minimum_listing_days: int = 5,
    limit_tolerance: float = 0.002,
) -> pd.DataFrame:
    """Create O2O/C2C/VWAP-proxy returns and conservative trade flags.

    Returns at date ``t`` run from the execution price on ``t`` to the same
    execution price on the next exchange trading date.  Missing observations
    inside a listing spell are treated as suspension days: prices are marked
    unchanged and both buy and sell flags are false.  The marked
    ``open_to_open_return`` is intended for portfolio accounting.  The
    ``tradable_open_to_open_return`` field is deliberately stricter: it is
    finite only when both consecutive exchange dates have observed opens, so
    it can be used as a supervised-learning target without silently turning a
    suspension into a zero-return label.
    """
    if minimum_listing_days < 0:
        raise ValueError("minimum_listing_days must be non-negative")
    if not 0 <= limit_tolerance < 0.05:
        raise ValueError("limit_tolerance must lie in [0, 0.05)")
    missing = OHLC_COLUMNS.difference(ohlc.columns)
    if missing:
        raise ValueError(f"OHLC data missing columns: {', '.join(sorted(missing))}")
    data = ohlc.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["stock_id"] = data["stock_id"].astype(str).str.zfill(6)
    data = data.dropna(subset=["date", "stock_id"]).sort_values(["stock_id", "date"])
    if data.duplicated(["stock_id", "date"]).any():
        raise ValueError("OHLC data must contain at most one row per stock and date")
    for column in ("open", "close", "high", "low", "volume", "amount"):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    observed_dates = pd.DatetimeIndex(sorted(data["date"].dropna().unique()))
    if end is not None:
        observed_dates = observed_dates[observed_dates <= pd.Timestamp(end)]
    if len(observed_dates) < 2:
        raise ValueError("execution panel needs at least two market dates")

    rows: list[pd.DataFrame] = []
    for stock_id, stock in data.groupby("stock_id", sort=False):
        stock = stock.set_index("date").reindex(observed_dates)
        stock["stock_id"] = stock_id
        observed = stock["open"].notna() & stock["close"].notna()
        listing_day = observed.cumsum() - 1
        first_observed = observed.cummax()
        last_date = stock.index[observed].max() if observed.any() else pd.NaT
        within_listing = first_observed & stock.index.to_series(index=stock.index).le(last_date)

        marked_open = stock["open"].where(observed).ffill().where(within_listing)
        marked_close = stock["close"].where(observed).ffill().where(within_listing)
        typical = stock[["open", "high", "low", "close"]].mean(axis=1).where(observed)
        marked_typical = typical.ffill().where(within_listing)
        previous_close = marked_close.shift(1)
        open_gap = marked_open / previous_close - 1.0
        limit = price_limit_fraction(stock_id)
        past_initial_window = listing_day.ge(minimum_listing_days)
        can_buy = observed & past_initial_window & open_gap.lt(limit - limit_tolerance)
        can_sell = observed & past_initial_window & open_gap.gt(-limit + limit_tolerance)
        next_observed = observed.shift(-1, fill_value=False)
        tradable_o2o = (stock["open"].shift(-1) / stock["open"] - 1.0).where(
            observed & next_observed
        )

        output = pd.DataFrame({
            "entry_date": stock.index,
            "stock_id": stock_id,
            "open": stock["open"].to_numpy(),
            "close": stock["close"].to_numpy(),
            "high": stock["high"].to_numpy(),
            "low": stock["low"].to_numpy(),
            "volume": stock.get("volume", pd.Series(np.nan, index=stock.index)).to_numpy(),
            "amount": stock.get("amount", pd.Series(np.nan, index=stock.index)).to_numpy(),
            "observed_market": observed.to_numpy(),
            "next_observed_market": next_observed.to_numpy(),
            "listing_trading_day": listing_day.to_numpy(dtype=int),
            "ipo_initial_window": (~past_initial_window & within_listing).to_numpy(),
            "open_gap_from_previous_close": open_gap.to_numpy(),
            "price_limit_fraction": limit,
            "can_buy": can_buy.to_numpy(),
            "can_sell": can_sell.to_numpy(),
            "eligible_signal": (observed & past_initial_window).to_numpy(),
            "tradable_open_to_open_return": tradable_o2o.to_numpy(),
            "open_to_open_return": (marked_open.shift(-1) / marked_open - 1.0).to_numpy(),
            "close_to_close_return": (marked_close.shift(-1) / marked_close - 1.0).to_numpy(),
            "vwap_proxy_return": (
                marked_typical.shift(-1) / marked_typical - 1.0
            ).to_numpy(),
            "open_to_close_return": (marked_close / marked_open - 1.0).to_numpy(),
        })
        if start is not None:
            output = output[output["entry_date"].ge(pd.Timestamp(start))]
        rows.append(output)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["entry_date", "stock_id"]
    ).reset_index(drop=True)


__all__ = [
    "build_execution_market_panel",
    "normalize_akshare_ohlc",
    "price_limit_fraction",
]
