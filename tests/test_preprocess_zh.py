from src.text.preprocess_zh import clean_text, combine_news_text, text_hash


def test_clean_text_removes_markup_urls_and_invisible_characters() -> None:
    value = "\ufeffＡＢＣ\u200b 公司&nbsp;增长 <p>订单</p> https://example.com\x00"
    assert clean_text(value) == "ABC 公司 增长 订单"


def test_clean_text_handles_nan_and_preserves_chinese_punctuation() -> None:
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text("盈利增长，业绩改善！") == "盈利增长，业绩改善！"


def test_clean_text_and_hash_remove_invalid_surrogate_code_points() -> None:
    value = "公告\udbc0文本"
    assert clean_text(value) == "公告文本"
    assert len(text_hash(value)) == 64


def test_combine_news_text_keeps_title_boundary_after_cleaning() -> None:
    assert combine_news_text("标题\u200b", "正文&nbsp;内容") == "标题：标题。正文：正文 内容"