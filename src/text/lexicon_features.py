"""Simple, auditable Chinese financial lexicon features."""

from __future__ import annotations

import re
from collections.abc import Iterable

DEFAULT_POSITIVE = frozenset({"上涨", "增长", "利好", "超预期", "盈利", "突破", "增持", "回升"})
DEFAULT_NEGATIVE = frozenset({"下跌", "下降", "亏损", "利空", "风险", "暴跌", "减持", "回落"})


def lexicon_score(text: str, positive: Iterable[str] = DEFAULT_POSITIVE, negative: Iterable[str] = DEFAULT_NEGATIVE) -> dict[str, int | float]:
    """Count lexicon hits; this is a transparent baseline, not a classifier."""
    value = str(text or "")
    pos = sum(value.count(word) for word in positive)
    neg = sum(value.count(word) for word in negative)
    tokens = max(len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", value)), 1)
    return {"positive_count": pos, "negative_count": neg, "lexicon_score": (pos - neg) / tokens}
