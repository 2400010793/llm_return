"""A leakage-safe, practical implementation of the SESTM baseline.

SESTM (Ke, Kelly, and Xiu) screens sentiment words, estimates positive and
negative word distributions, and estimates an article sentiment probability by
penalized multinomial likelihood.  This implementation exposes each fitted
stage and uses a stable binary-logit approximation for the final probability;
it is intended for the Chinese research prototype and keeps all fitting on the
training window.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from src.text.paper_preprocess import tokenize_paper_words


@dataclass
class SESTM:
    vocabulary: list[str]
    vectorizer: CountVectorizer
    selected_words: list[str]
    model: LogisticRegression
    selected_indices: list[int]

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        matrix = self.vectorizer.transform([" ".join(tokenize_paper_words(t)) for t in texts])
        return self.model.predict_proba(matrix[:, self.selected_indices])[:, 1]


def fit_sestm(
    texts: Iterable[str],
    labels: Iterable[int],
    *,
    max_features: int = 20_000,
    min_df: int = 2,
    max_sentiment_words: int = 500,
    C: float = 1.0,
) -> SESTM:
    """Fit SESTM-like sentiment features using training texts only.

    Screening is based on absolute class-frequency differences.  The selected
    words then feed a penalized logistic likelihood, which is the robust
    operational equivalent for sparse Chinese pilot samples.
    """
    values = [" ".join(tokenize_paper_words(text)) for text in texts]
    y = np.asarray(list(labels), dtype=int)
    if len(values) != len(y) or len(values) == 0:
        raise ValueError("texts and labels must be non-empty and have equal length")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("labels must be binary 0/1")
    vectorizer = CountVectorizer(token_pattern=r"(?u)\S+", min_df=min_df, max_features=max_features)
    counts = vectorizer.fit_transform(values)
    vocabulary = list(vectorizer.get_feature_names_out())
    pos = np.asarray(counts[y == 1].sum(axis=0)).ravel()
    neg = np.asarray(counts[y == 0].sum(axis=0)).ravel()
    pos_rate = pos / max(int((y == 1).sum()), 1)
    neg_rate = neg / max(int((y == 0).sum()), 1)
    selected_idx = np.argsort(np.abs(pos_rate - neg_rate))[::-1][:max_sentiment_words]
    selected = [vocabulary[i] for i in selected_idx]
    if len(np.unique(y)) < 2:
        raise ValueError("SESTM requires both positive and negative labels")
    model = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
    model.fit(counts[:, selected_idx], y)
    return SESTM(vocabulary, vectorizer, selected, model, list(selected_idx))