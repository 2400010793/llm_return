import pandas as pd

from src.data.build_panel import attach_forward_returns
from src.data.clean_news import clean_news
from src.data.clean_prices import clean_prices


def test_news_price_pipeline_respects_market_close() -> None:
    prices = clean_prices(pd.DataFrame({
        "stock_id": ["A"] * 4,
        "date": pd.date_range("2024-01-02", periods=4, freq="B"),
        "close": [100, 110, 121, 133.1],
    }))
    news = clean_news(pd.DataFrame({
        "article_id": ["before", "after"],
        "stock_id": ["A", "A"],
        "published_at": ["2024-01-02 14:00", "2024-01-02 16:00"],
        "headline": ["盘中消息", "收盘后消息"],
        "body": ["公司订单增长", "公司发布公告"],
    }))
    panel = attach_forward_returns(news, prices, horizons=(1,))

    before = panel.set_index("article_id").loc["before"]
    after = panel.set_index("article_id").loc["after"]
    assert before["entry_date"] == pd.Timestamp("2024-01-03")
    assert after["entry_date"] == pd.Timestamp("2024-01-03")
    assert before["ret_1d"] == 121 / 110 - 1
