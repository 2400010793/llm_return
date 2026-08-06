"""Leakage-safe sentiment classification and return prediction models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.prediction_metrics import regression_metrics
from src.text.bow_features import fit_word_tfidf, transform_tfidf


@dataclass
class SentimentModelResult:
    probabilities: np.ndarray
    labels: np.ndarray
    vectorizer: object
    model: LogisticRegression
    metrics: dict[str, float]


def make_sentiment_labels(forward_three_day_return: Iterable[float]) -> np.ndarray:
    """Make the paper's binary weak label: 1 iff the three-day return is > 0.

    The input must already be the chosen three-day return; this prevents silently
    substituting a one-day or five-day target.
    """
    values = np.asarray(list(forward_three_day_return), dtype=float)
    labels = np.full(values.shape, -1, dtype=np.int8)
    labels[np.isfinite(values)] = (values[np.isfinite(values)] > 0).astype(np.int8)
    return labels


def make_paper_event_three_day_return(
    prices: pd.DataFrame,
    *,
    stock_col: str = "stock_id",
    date_col: str = "date",
    close_col: str = "close",
) -> pd.Series:
    """Return the paper-style event window P[t+1]/P[t-2]-1.

    The PDF describes the label as the return from the day before publication
    through the day after publication (three daily intervals around the event).
    This function is separate from the forward t+1:t+3 label so the two cannot
    be confused.
    """
    required = {stock_col, date_col, close_col}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"missing price columns: {', '.join(sorted(missing))}")
    result = prices.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.sort_values([stock_col, date_col])
    grouped = result.groupby(stock_col, sort=False)[close_col]
    result["paper_event_3d_return"] = grouped.shift(-1) / grouped.shift(2) - 1.0
    return result.set_index([stock_col, date_col])["paper_event_3d_return"]


def make_forward_return(
    prices: pd.DataFrame,
    horizon: int,
    *,
    stock_col: str = "stock_id",
    date_col: str = "date",
    close_col: str = "close",
) -> pd.Series:
    """Create one independent forward close-to-close return label."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    required = {stock_col, date_col, close_col}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"missing price columns: {', '.join(sorted(missing))}")
    result = prices.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.sort_values([stock_col, date_col])
    grouped = result.groupby(stock_col, sort=False)[close_col]
    target = grouped.shift(-horizon) / result[close_col] - 1.0
    index = pd.MultiIndex.from_frame(result[[stock_col, date_col]])
    return pd.Series(target.to_numpy(), index=index, name=f"forward_{horizon}d_return")


def _make_features(train_texts: list[str], test_texts: list[str], max_features: int | None):
    vectorizer = fit_word_tfidf(train_texts, min_df=1, max_df=1.0, max_features=max_features)
    return vectorizer, transform_tfidf(vectorizer, train_texts), transform_tfidf(vectorizer, test_texts)


def fit_sentiment_model(
    train_texts: Iterable[str],
    train_three_day_returns: Iterable[float],
    predict_texts: Iterable[str],
    *,
    C: float = 1.0,
    max_features: int | None = 100_000,
) -> SentimentModelResult:
    """Fit TF-IDF + logistic regression using labels from training data only."""
    texts, returns, predictions_text = list(train_texts), np.asarray(list(train_three_day_returns), float), list(predict_texts)
    if len(texts) != len(returns) or not texts:
        raise ValueError("training texts and three-day returns must be aligned and non-empty")
    finite = np.isfinite(returns)
    labels = make_sentiment_labels(returns[finite])
    if len(np.unique(labels)) < 2:
        raise ValueError("training window must contain both positive and non-positive labels")
    clean_texts = [t for t, ok in zip(texts, finite) if ok]
    vectorizer, x_train, x_predict = _make_features(clean_texts, predictions_text, max_features)
    model = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
    model.fit(x_train, labels)
    probabilities = model.predict_proba(x_predict)[:, 1]
    return SentimentModelResult(probabilities, labels, vectorizer, model, {})


def sentiment_metrics(actual_three_day_returns: Iterable[float], probabilities: Iterable[float]) -> dict[str, float]:
    returns, probs = np.asarray(list(actual_three_day_returns), float), np.asarray(list(probabilities), float)
    mask = np.isfinite(returns) & np.isfinite(probs)
    actual = (returns[mask] > 0).astype(int)
    predicted = (probs[mask] >= 0.5).astype(int)
    return {"n": float(mask.sum()), "accuracy": float(accuracy_score(actual, predicted)), "f1": float(f1_score(actual, predicted, zero_division=0))}


@dataclass
class ReturnModelResult:
    predictions: np.ndarray
    vectorizer: object
    model: RegressorMixin
    metrics: dict[str, float]


def _regressor(name: str, alpha: float, random_state: int) -> RegressorMixin:
    name = name.lower()
    if name == "ols":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=alpha)
    if name == "lasso":
        return Lasso(alpha=alpha, max_iter=5000)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
    if name == "nn":
        return make_pipeline(StandardScaler(with_mean=False), MLPRegressor(hidden_layer_sizes=(128, 32), early_stopping=True, max_iter=300, random_state=random_state))
    raise ValueError("model must be one of: ols, ridge, lasso, random_forest, nn")


def fit_return_model(
    train_texts: Iterable[str], train_returns: Iterable[float], predict_texts: Iterable[str], actual_returns: Iterable[float],
    *, model_name: str = "ridge", alpha: float = 100.0, max_features: int | None = 100_000, random_state: int = 42,
) -> ReturnModelResult:
    """Fit one independent model for one horizon; vectorization is train-only."""
    train, y, predict, actual = list(train_texts), np.asarray(list(train_returns), float), list(predict_texts), np.asarray(list(actual_returns), float)
    finite = np.isfinite(y)
    if len(train) != len(y) or not finite.any():
        raise ValueError("training texts and returns must be aligned with finite targets")
    clean_train = [t for t, ok in zip(train, finite) if ok]
    vectorizer, x_train, x_predict = _make_features(clean_train, predict, max_features)
    model = _regressor(model_name, alpha, random_state)
    model.fit(x_train, y[finite])
    predictions = model.predict(x_predict)
    metrics = regression_metrics(actual, predictions) if len(actual) else {}
    return ReturnModelResult(predictions, vectorizer, model, metrics)
