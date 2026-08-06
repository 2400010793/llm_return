import pandas as pd

from src.data.dedup_news import deduplicate_news


def test_deduplicates_by_url_then_article_id_then_text():
    frame = pd.DataFrame(
        [
            {"article_id": "a1", "title": "标题 A", "summary": "正文", "url": "https://x/a1"},
            {"article_id": "a1-copy", "title": "标题 A", "summary": "正文", "url": "https://x/a1"},
            {"article_id": "a2", "title": "标题 B", "summary": "正文", "url": ""},
            {"article_id": "a2", "title": "标题 B2", "summary": "正文2", "url": ""},
            {"article_id": "", "title": "同一 标题", "summary": "相同内容", "url": ""},
            {"article_id": "", "title": "同一标题", "summary": "相同内容", "url": ""},
        ]
    )
    unique, duplicate = deduplicate_news(frame)
    assert len(unique) == 3
    assert len(duplicate) == 3
