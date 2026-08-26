from src.text.lexicon_features import lexicon_score


def test_lexicon_score_is_transparent():
    result = lexicon_score("业绩增长，股价上涨，但仍有风险")
    assert result["positive_count"] == 2
    assert result["negative_count"] == 1
    assert result["lexicon_score"] > 0


def test_finance_lexicon_exposes_risk_uncertainty_and_negation_features():
    result = lexicon_score("业绩大幅增长，但并未亏损，仍存在重大风险，可能受到监管处罚")
    assert result["positive_count"] >= 1
    assert result["negative_count"] >= 1
    assert result["risk_count"] >= 1
    assert result["uncertainty_count"] >= 1
    assert result["litigious_count"] >= 1
    assert result["negated_negative_count"] >= 1
    assert result["adjusted_negative_count"] < result["negative_count"]
