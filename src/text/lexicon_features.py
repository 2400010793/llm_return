"""Auditable Chinese financial lexicon features.

The vocabulary is deliberately kept in code so that every experiment can
record exactly which version was used. It is a practical Chinese adaptation
of the project's financial-dictionary baseline.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

LEXICON_VERSION = "cn-finance-v2"

DEFAULT_POSITIVE = frozenset({
    "上涨", "增长", "利好", "超预期", "盈利", "突破", "增持", "回升",
    "改善", "上调", "高于预期", "业绩预增", "扭亏", "订单增加", "获奖",
    "回购", "分红", "扩张", "创新高", "景气", "复苏", "强劲", "领先",
})
DEFAULT_NEGATIVE = frozenset({
    "下跌", "下降", "亏损", "利空", "风险", "暴跌", "减持", "回落",
    "恶化", "下调", "低于预期", "业绩预亏", "亏损扩大", "订单减少", "减值",
    "违约", "退市", "暂停上市", "大幅下滑", "大幅下降", "承压", "疲软",
})
DEFAULT_RISK = frozenset({
    "风险", "风险提示", "重大风险", "经营风险", "市场风险", "信用风险",
    "流动性风险", "减值", "质押", "担保", "违约", "处罚", "监管措施",
    "退市风险", "商誉减值", "无法保证",
})
DEFAULT_UNCERTAINTY = frozenset({
    "可能", "或许", "不确定", "预计", "预期", "大概", "或将", "尚未确定",
    "存在不确定性", "不排除", "难以判断", "取决于", "有待观察",
})
DEFAULT_LITIGATION = frozenset({
    "诉讼", "仲裁", "纠纷", "违法", "违规", "处罚", "被调查", "立案",
    "问询函", "监管函", "行政处罚",
})
DEFAULT_CONSTRAINING = frozenset({
    "限制", "禁止", "约束", "监管", "冻结", "暂停", "终止", "延期",
    "无法", "不利影响", "受限",
})
DEFAULT_NEGATIONS = frozenset({"不", "未", "无", "没有", "并非", "并未", "否认", "尚无"})

_COUNT_TOKEN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z]+(?:\.[A-Za-z]+)?|\d+(?:\.\d+)?%?")


def _count_phrases(value: str, phrases: Iterable[str]) -> int:
    return sum(value.count(str(phrase)) for phrase in phrases)


def _negated_count(value: str, phrases: Iterable[str], negations: Iterable[str]) -> int:
    count = 0
    for phrase in phrases:
        for match in re.finditer(re.escape(str(phrase)), value):
            prefix = value[max(0, match.start() - 6):match.start()]
            if any(prefix.endswith(str(word)) for word in negations):
                count += 1
    return count


def lexicon_score(
    text: str,
    positive: Iterable[str] = DEFAULT_POSITIVE,
    negative: Iterable[str] = DEFAULT_NEGATIVE,
    *,
    risk: Iterable[str] = DEFAULT_RISK,
    uncertainty: Iterable[str] = DEFAULT_UNCERTAINTY,
    litigious: Iterable[str] = DEFAULT_LITIGATION,
    constraining: Iterable[str] = DEFAULT_CONSTRAINING,
    negations: Iterable[str] = DEFAULT_NEGATIONS,
) -> dict[str, int | float | str]:
    """Return raw, normalized, risk and negation-aware finance features."""
    value = str(text or "")
    positive = tuple(str(word) for word in positive)
    negative = tuple(str(word) for word in negative)
    categories = {
        "positive": positive,
        "negative": negative,
        "risk": tuple(str(word) for word in risk),
        "uncertainty": tuple(str(word) for word in uncertainty),
        "litigious": tuple(str(word) for word in litigious),
        "constraining": tuple(str(word) for word in constraining),
    }
    token_count = max(len(_COUNT_TOKEN.findall(value)), 1)
    result: dict[str, int | float | str] = {
        "word_count": token_count,
        "char_count": len(value),
        "lexicon_version": LEXICON_VERSION,
    }
    for category, words in categories.items():
        count = _count_phrases(value, words)
        result[f"{category}_count"] = count
        result[f"{category}_rate"] = count / token_count
    pos = int(result["positive_count"])
    neg = int(result["negative_count"])
    negated_pos = _negated_count(value, positive, negations)
    negated_neg = _negated_count(value, negative, negations)
    adjusted_pos = max(pos - negated_pos, 0)
    adjusted_neg = max(neg - negated_neg, 0)
    result.update({
        "negated_positive_count": negated_pos,
        "negated_negative_count": negated_neg,
        "adjusted_positive_count": adjusted_pos,
        "adjusted_negative_count": adjusted_neg,
        "adjusted_lexicon_score": (adjusted_pos - adjusted_neg) / token_count,
        "lexicon_score": (pos - neg) / token_count,
    })
    return result
