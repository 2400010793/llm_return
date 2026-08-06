"""Leakage-safe text return prediction baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge

from src.text.bow_features import fit_tfidf, transform_tfidf


@dataclass
class TfidfRidgeResult:
    predictions: np.ndarray
    vectorizer: object
    model: Ridge


def fit_predict_tfidf_ridge(
    train_texts: list[str],
    train_returns: np.ndarray,
    test_texts: list[str],
    *,
    alpha: float = 100.0,
    min_df: int = 1,
    max_features: int | None = 100_000,
) -> TfidfRidgeResult:
    """Fit TF-IDF and Ridge on training observations and predict test data."""
    if len(train_texts) != len(train_returns):
        raise ValueError("train_texts and train_returns must have the same length")
    if not train_texts:
        raise ValueError("training data cannot be empty")
    vectorizer = fit_tfidf(
        train_texts,
        min_df=min_df,
        max_df=1.0,
        max_features=max_features,
    )
    x_train = transform_tfidf(vectorizer, train_texts)
    x_test = transform_tfidf(vectorizer, test_texts)
    model = Ridge(alpha=alpha)
    model.fit(x_train, np.asarray(train_returns, dtype=float))
    return TfidfRidgeResult(model.predict(x_test), vectorizer, model)
