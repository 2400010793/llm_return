"""Leakage-safe pieces of the paper's text-mining and portfolio pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge

from src.evaluation.prediction_metrics import regression_metrics
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


def quantile_portfolio(
    frame: pd.DataFrame,
    *,
    prediction: str = "prediction",
    realized: str = "realized_return",
    date: str = "entry_date",
    quantiles: int = 5,
) -> pd.DataFrame:
    """Form daily equal-weighted low/high and long-short portfolios."""
    required = {prediction, realized, date}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing portfolio columns: {', '.join(sorted(missing))}")
    rows = []
    for day, group in frame.dropna(subset=list(required)).groupby(date):
        if len(group) < quantiles:
            continue
        ranks = group[prediction].rank(method="first")
        bucket = pd.qcut(ranks, q=quantiles, labels=False, duplicates="drop")
        low = group.loc[bucket == bucket.min(), realized].mean()
        high = group.loc[bucket == bucket.max(), realized].mean()
        rows.append({"date": day, "low": low, "high": high, "long_short": high - low, "n": len(group)})
    return pd.DataFrame(rows)


def portfolio_metrics(portfolio: pd.DataFrame, *, annualization: int = 252) -> dict[str, float]:
    """Summarize daily portfolio returns."""
    if portfolio.empty:
        return {"n_days": 0.0, "mean": float("nan"), "volatility": float("nan"), "sharpe": float("nan")}
    values = portfolio["long_short"].astype(float).to_numpy()
    mean = float(np.mean(values))
    vol = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    return {"n_days": float(len(values)), "mean": mean, "volatility": vol, "sharpe": mean / vol * np.sqrt(annualization) if vol and np.isfinite(vol) else float("nan")}
