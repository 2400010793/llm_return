from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.build_sina_single_stock_classification_panel import build_panel


def test_build_panel_uses_strictly_later_exchange_date() -> None:
    news = pd.DataFrame({
        "article_id": ["a", "b"],
        "stock_id": ["1", "000001"],
        "stock_name": ["甲", "甲"],
        "published_at": [
            "2024-01-02T07:00:00+08:00",
            "2024-01-02T18:00:00+08:00",
        ],
        "text": ["文本一", "文本二"],
        "text_hash": ["h1", "h2"],
    })
    dates = pd.to_datetime([
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"
    ])
    returns = pd.DataFrame({"000001.SZ": [-0.01, 0.02, 0.03, -0.04]}, index=dates)

    result = build_panel(news, returns)

    assert result["row_index"].tolist() == [1, 2]
    assert result["entry_date"].tolist() == [pd.Timestamp("2024-01-03")] * 2
    np.testing.assert_allclose(result["next_day_return"], [0.03, 0.03])
    assert result["next_day_label"].tolist() == [1.0, 1.0]


def test_build_panel_preserves_rows_without_a_future_market_date() -> None:
    news = pd.DataFrame({
        "article_id": ["a"],
        "stock_id": ["000001"],
        "stock_name": ["甲"],
        "published_at": ["2024-01-04T12:00:00+08:00"],
        "text": ["文本"],
        "text_hash": ["h"],
    })
    returns = pd.DataFrame(
        {"000001.SZ": [0.01]}, index=pd.to_datetime(["2024-01-04"])
    )

    result = build_panel(news, returns)

    assert len(result) == 1
    assert pd.isna(result.loc[0, "entry_date"])
    assert pd.isna(result.loc[0, "next_day_return"])
    assert not bool(result.loc[0, "label_available"])
