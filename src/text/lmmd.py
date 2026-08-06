"""Financial dictionary features with a replaceable LMMD-style vocabulary."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from src.text.paper_preprocess import tokenize_paper_words

DEFAULT_LMMD = {
    "positive": frozenset({"增长", "上涨", "盈利", "利好", "改善", "增持", "突破"}),
    "negative": frozenset({"下降", "下跌", "亏损", "利空", "风险", "减持", "违约"}),
    "uncertainty": frozenset({"可能", "或许", "不确定", "风险", "预计", "大概"}),
    "litigious": frozenset({"诉讼", "仲裁", "纠纷", "违法", "处罚"}),
    "constraining": frozenset({"限制", "禁止", "约束", "监管", "冻结"}),
}


def lmmd_features(
    text: str,
    lexicon: Mapping[str, Iterable[str]] = DEFAULT_LMMD,
    *,
    stop_words: Iterable[str] = (),
) -> dict[str, int | float]:
    """Return LMMD-style category counts and normalized rates."""
    tokens = tokenize_paper_words(text, stop_words=stop_words)
    total = max(len(tokens), 1)
    result: dict[str, int | float] = {"word_count": len(tokens)}
    for category, words in lexicon.items():
        count = sum(tokens.count(str(word).lower()) for word in words)
        result[f"{category}_count"] = count
        result[f"{category}_rate"] = count / total
    positive = int(result.get("positive_count", 0))
    negative = int(result.get("negative_count", 0))
    result["lmmd_score"] = (positive - negative) / total
    return result