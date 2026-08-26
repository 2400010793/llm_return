"""Leakage-safe pieces of the paper's text-mining and portfolio pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from src.evaluation.prediction_metrics import regression_metrics
from src.portfolio import portfolio_metrics, quantile_portfolio
from src.text.bow_features import fit_word_tfidf, transform_tfidf


@dataclass
class SentimentResult:
    probabilities: np.ndarray
    vectorizer: object
    model: LogisticRegression


def fit_predict_sentiment(
    train_texts: Iterable[str],
    train_returns: Iterable[float],
    test_texts: Iterable[str],
    *,
    min_df: int = 1,
    max_features: int | None = 100_000,
    C: float = 1.0,
) -> SentimentResult:
    """Fit a training-only TF-IDF logistic sentiment model.

    The weak labels are the sign of returns from the training window only.
    """
    texts = list(train_texts)
    returns = np.asarray(list(train_returns), dtype=float)
    if len(texts) != len(returns) or not texts:
        raise ValueError("training texts and returns must be non-empty and aligned")
    labels = (returns > 0).astype(int)
    if len(np.unique(labels)) < 2:
        raise ValueError("sentiment training window needs both return signs")
    vectorizer = fit_word_tfidf(texts, min_df=min_df, max_df=1.0, max_features=max_features)
    x_train = transform_tfidf(vectorizer, texts)
    x_test = transform_tfidf(vectorizer, list(test_texts))
    model = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
    model.fit(x_train, labels)
    return SentimentResult(model.predict_proba(x_test)[:, 1], vectorizer, model)


@dataclass
class ReturnResult:
    predictions: np.ndarray
    vectorizer: object
    model: Ridge
    metrics: dict[str, float]


def fit_predict_return_ridge(
    train_texts: Iterable[str],
    train_returns: Iterable[float],
    test_texts: Iterable[str],
    test_returns: Iterable[float],
    *,
    alpha: float = 100.0,
    min_df: int = 1,
    max_features: int | None = 100_000,
) -> ReturnResult:
    """Fit pooled panel TF-IDF + Ridge using only the training window."""
    train = list(train_texts)
    test = list(test_texts)
    y_train = np.asarray(list(train_returns), dtype=float)
    y_test = np.asarray(list(test_returns), dtype=float)
    if len(train) != len(y_train) or not train:
        raise ValueError("training texts and returns must be non-empty and aligned")
    vectorizer = fit_word_tfidf(train, min_df=min_df, max_df=1.0, max_features=max_features)
    x_train = transform_tfidf(vectorizer, train)
    x_test = transform_tfidf(vectorizer, test)
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return ReturnResult(predictions, vectorizer, model, regression_metrics(y_test, predictions))


__all__ = [
    "ReturnResult", "SentimentResult", "fit_predict_return_ridge",
    "fit_predict_sentiment", "portfolio_metrics", "quantile_portfolio",
]
