import numpy as np

from src.text.lmmd import lmmd_features
from src.text.paper_preprocess import expand_english_contractions, tokenize_paper_words
from src.text.sestm import fit_sestm


def test_paper_word_preprocessing_and_lmmd_features() -> None:
    assert "cannot" in expand_english_contractions("can't")
    assert tokenize_paper_words("公司上涨 123") == ["公司", "上涨"]
    features = lmmd_features("公司盈利增长，但存在风险")
    assert features["positive_count"] == 2
    assert features["negative_count"] == 1
    assert features["lmmd_score"] > 0


def test_sestm_fit_and_transform() -> None:
    texts = ["公司盈利增长", "业绩上涨利好", "公司亏损下降", "经营风险增加"]
    model = fit_sestm(texts, [1, 1, 0, 0], min_df=1, max_sentiment_words=4)
    scores = model.transform(["公司盈利", "公司亏损"])
    assert scores.shape == (2,)
    assert np.all((scores >= 0) & (scores <= 1))
