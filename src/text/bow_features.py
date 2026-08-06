"""Time-safe bag-of-words and TF-IDF feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import joblib
import jieba
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer


def tokenize_zh_words(text: str) -> list[str]:
    """Segment Chinese text into words for the paper-aligned word BOW baseline."""
    tokens = []
    for token in jieba.lcut(str(text or ""), HMM=True):
        token = token.strip()
        if not token or not any("\u4e00" <= char <= "\u9fff" for char in token):
            continue
        tokens.append(token)
    return tokens


def fit_word_tfidf(
    train_texts: Iterable[str],
    *,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_df: float = 0.95,
    max_features: Optional[int] = 100_000,
    stop_words: Optional[list[str]] = None,
) -> TfidfVectorizer:
    """Fit a jieba word-level TF-IDF vectorizer on training texts only."""
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_zh_words,
        token_pattern=None,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        stop_words=stop_words,
        sublinear_tf=True,
    )
    vectorizer.fit(train_texts)
    return vectorizer


def fit_word_count(
    train_texts: Iterable[str],
    *,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_df: float = 0.95,
    max_features: Optional[int] = 100_000,
) -> CountVectorizer:
    """Fit a jieba word-count BOW model on training texts only."""
    vectorizer = CountVectorizer(
        tokenizer=tokenize_zh_words,
        token_pattern=None,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )
    vectorizer.fit(train_texts)
    return vectorizer


def make_hashing_bow(*, n_features: int = 2**18, ngram_range: tuple[int, int] = (1, 2)) -> HashingVectorizer:
    """Create a stateless word BOW model for streaming or large corpora."""
    return HashingVectorizer(
        tokenizer=tokenize_zh_words,
        token_pattern=None,
        ngram_range=ngram_range,
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
    )


def fit_tfidf(
    train_texts: Iterable[str],
    *,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 5,
    max_df: float = 0.95,
    max_features: Optional[int] = 100_000,
) -> TfidfVectorizer:
    """Fit TF-IDF on training texts only."""
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True,
    )
    vectorizer.fit(train_texts)
    return vectorizer


def transform_tfidf(vectorizer: TfidfVectorizer, texts: Iterable[str]) -> sparse.csr_matrix:
    """Transform texts with an already-fitted training-only vectorizer."""
    return vectorizer.transform(texts).tocsr()


def save_tfidf(vectorizer: TfidfVectorizer, matrix: sparse.spmatrix, path: str | Path) -> None:
    """Save a vectorizer and sparse matrix using a shared prefix."""
    prefix = Path(path)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, prefix.with_suffix(".vectorizer.joblib"))
    sparse.save_npz(prefix.with_suffix(".npz"), matrix)
