"""Unified training-only text representation adapters for the replication study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import sparse

from src.text.bow_features import fit_word_count, fit_word_tfidf, tokenize_zh_words, transform_tfidf
from src.text.embeddings import encode_local_transformer
from src.text.lexicon_features import lexicon_score
from src.text.lmmd import lmmd_features
from src.text.semantic_embeddings import _mean_word_vectors, encode_bge_m3, encode_sentence_transformer, encode_word2vec
from src.text.sestm import fit_sestm
from src.text.api_embeddings import encode_api_embeddings
from src.text.ollama_client import ollama_embed


@dataclass
class RepresentationResult:
    train: np.ndarray | sparse.spmatrix
    predict: np.ndarray | sparse.spmatrix
    fitted: object | None
    name: str


def _numeric_features(texts: list[str], kind: str) -> np.ndarray:
    rows = []
    for text in texts:
        values = lexicon_score(text) if kind == "lexicon" else lmmd_features(text)
        rows.append([float(v) for v in values.values()])
    return np.asarray(rows, dtype=np.float32)


def fit_representation(
    name: str,
    train_texts: Iterable[str],
    predict_texts: Iterable[str],
    *,
    sentiment_labels: Iterable[int] | None = None,
    max_features: int | None = 100_000,
    embedding_model: str | None = None,
    device: str | None = None,
) -> RepresentationResult:
    """Fit or encode one representation without fitting on prediction texts.

    ``embedding_model`` is used for ``qwen``, ``llama`` or ``sentence`` only
    when the caller supplies a local model backend; remote/API embeddings are
    intentionally passed in as precomputed arrays by ``fit_precomputed``.
    """
    train, predict = list(train_texts), list(predict_texts)
    key = name.lower().replace("-", "_")
    if key in {"lexicon", "lmmd"}:
        return RepresentationResult(_numeric_features(train, key), _numeric_features(predict, key), None, key)
    if key in {"bow", "count"}:
        vectorizer = fit_word_count(train, min_df=1, max_df=1.0, max_features=max_features)
        return RepresentationResult(vectorizer.transform(train), vectorizer.transform(predict), vectorizer, "bow")
    if key in {"tfidf", "word_tfidf"}:
        vectorizer = fit_word_tfidf(train, min_df=1, max_df=1.0, max_features=max_features)
        return RepresentationResult(transform_tfidf(vectorizer, train), transform_tfidf(vectorizer, predict), vectorizer, "tfidf")
    if key == "sestm":
        if sentiment_labels is None:
            raise ValueError("SESTM requires training sentiment labels")
        model = fit_sestm(train, sentiment_labels, min_df=1)
        return RepresentationResult(model.transform(train)[:, None], model.transform(predict)[:, None], model, key)
    if key == "word2vec":
        x_train, model = encode_word2vec(train, texts=None, min_count=1)
        x_predict = _mean_word_vectors([tokenize_zh_words(t) for t in predict], model)
        return RepresentationResult(x_train, x_predict, model, key)
    if key in {"roberta", "bert"}:
        defaults = {
            "roberta": "hfl/chinese-roberta-wwm-ext",
            "bert": "hfl/chinese-bert-wwm-ext",
        }
        model_name = embedding_model or defaults[key]
        return RepresentationResult(encode_local_transformer(train, model_name, device=device), encode_local_transformer(predict, model_name, device=device), model_name, key)
    if key in {"bge", "bge_m3"}:
        model_name = embedding_model or "BAAI/bge-m3"
        local_only = model_name.startswith("/")
        return RepresentationResult(
            encode_bge_m3(train, model_name=model_name, device=device, local_files_only=local_only),
            encode_bge_m3(predict, model_name=model_name, device=device, local_files_only=local_only),
            model_name,
            "bge_m3",
        )
    if key in {"sentence", "sentence_transformer"}:
        if not embedding_model:
            raise ValueError("sentence representation requires embedding_model")
        return RepresentationResult(encode_sentence_transformer(train, model_name=embedding_model, device=device), encode_sentence_transformer(predict, model_name=embedding_model, device=device), embedding_model, key)
    if key in {"qwen", "llama", "ollama"}:
        vectors = np.asarray(ollama_embed(train + predict, model=embedding_model or "qwen2.5:32b"), dtype=np.float32)
        return RepresentationResult(vectors[: len(train)], vectors[len(train) :], embedding_model or "qwen2.5:32b", key)
    if key in {"openai", "api", "openai_embedding"}:
        vectors = encode_api_embeddings(train + predict, model=embedding_model or "text-embedding-3-small")
        return RepresentationResult(vectors[: len(train)], vectors[len(train) :], embedding_model or "text-embedding-3-small", key)
    raise ValueError("unknown representation: lexicon, lmmd, sestm, bow, tfidf, word2vec, roberta, bge_m3, sentence")


def fit_precomputed(train_embeddings: np.ndarray, predict_embeddings: np.ndarray, name: str) -> RepresentationResult:
    """Wrap cached Qwen/LLaMA/OpenAI embeddings without refitting or API calls."""
    train = np.asarray(train_embeddings, dtype=np.float32)
    predict = np.asarray(predict_embeddings, dtype=np.float32)
    if train.ndim != 2 or predict.ndim != 2 or train.shape[1] != predict.shape[1]:
        raise ValueError("precomputed embeddings must be 2-D with matching dimensions")
    return RepresentationResult(train, predict, None, name)
