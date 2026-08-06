import pandas as pd

from src.data.normalize_public import normalize_forum_export, normalize_news_export


def test_normalize_public_news_and_forum_exports() -> None:
    news = normalize_news_export(pd.DataFrame({
        "article_id": ["n1"], "stock_id": ["000001.SZ"],
        "published_at": ["2024-01-01"], "headline": ["标题"], "body": ["正文"],
    }), "eastmoney")
    forum = normalize_forum_export(pd.DataFrame({
        "post_id": ["p1"], "stock_id": ["000001.SZ"],
        "published_at": ["2024-01-01"], "author_id_hash": ["hash"],
        "title": ["标题"], "content": ["内容"],
    }), "xueqiu")
    assert news.loc[0, "data_type"] == "news"
    assert forum.loc[0, "data_type"] == "forum_post"
    assert "author_id" not in forum.columns
