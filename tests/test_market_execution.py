from __future__ import annotations

import pandas as pd
import pytest

from src.data.market_execution import build_execution_market_panel, price_limit_fraction


def _prices(stock_id: str, opens: list[float | None]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(opens))
    rows = []
    for date, value in zip(dates, opens):
        if value is None:
            continue
        rows.append({
            "date": date,
            "stock_id": stock_id,
            "open": value,
            "close": value,
            "high": value,
            "low": value,
            "volume": 100,
            "amount": value * 10_000,
        })
    return pd.DataFrame(rows)


def test_execution_returns_are_forward_o2o_and_exclude_ipo_window() -> None:
    prices = _prices("000001", [10, 10, 10, 10, 10, 11, 12])
    panel = build_execution_market_panel(
        prices, start="2024-01-05", minimum_listing_days=5
    )
    friday = panel[panel["entry_date"].eq(pd.Timestamp("2024-01-05"))].iloc[0]
    monday = panel[panel["entry_date"].eq(pd.Timestamp("2024-01-08"))].iloc[0]
    assert not bool(friday["eligible_signal"])
    assert bool(monday["eligible_signal"])
    assert monday["open_to_open_return"] == pytest.approx(12 / 11 - 1)
    assert monday["tradable_open_to_open_return"] == pytest.approx(12 / 11 - 1)


def test_suspension_day_is_untradable_and_marked_at_zero_return() -> None:
    prices = pd.concat([
        _prices("000001", [10, 10, 10, 10, 10, 10, None, 11]),
        _prices("000002", [10, 10, 10, 10, 10, 10, 10, 10]),
    ], ignore_index=True)
    panel = build_execution_market_panel(prices, minimum_listing_days=5)
    suspended = panel[panel["entry_date"].eq(pd.Timestamp("2024-01-09"))].iloc[0]
    assert not bool(suspended["observed_market"])
    assert not bool(suspended["can_buy"])
    assert not bool(suspended["can_sell"])
    assert suspended["open_to_open_return"] == pytest.approx(0.1)
    assert pd.isna(suspended["tradable_open_to_open_return"])

    before_suspension = panel[
        panel["entry_date"].eq(pd.Timestamp("2024-01-08"))
        & panel["stock_id"].eq("000001")
    ].iloc[0]
    assert not bool(before_suspension["next_observed_market"])
    assert pd.isna(before_suspension["tradable_open_to_open_return"])


def test_open_at_price_limit_blocks_the_corresponding_direction() -> None:
    prices = _prices("000001", [10, 10, 10, 10, 10, 11, 9.9])
    panel = build_execution_market_panel(prices, minimum_listing_days=5)
    limit_up = panel[panel["entry_date"].eq(pd.Timestamp("2024-01-08"))].iloc[0]
    limit_down = panel[panel["entry_date"].eq(pd.Timestamp("2024-01-09"))].iloc[0]
    assert not bool(limit_up["can_buy"])
    assert bool(limit_up["can_sell"])
    assert bool(limit_down["can_buy"])
    assert not bool(limit_down["can_sell"])


def test_board_price_limit_proxy() -> None:
    assert price_limit_fraction("000001") == pytest.approx(0.10)
    assert price_limit_fraction("300001") == pytest.approx(0.20)
    assert price_limit_fraction("688001") == pytest.approx(0.20)
    assert price_limit_fraction("830001") == pytest.approx(0.30)
