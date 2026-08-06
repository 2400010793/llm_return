from src.text.lexicon_features import lexicon_score


def test_lexicon_score_is_transparent():
    result = lexicon_score("业绩增长，股价上涨，但仍有风险")
    assert result["positive_count"] == 2
    assert result["negative_count"] == 1
    assert result["lexicon_score"] > 0
