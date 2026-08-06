"""Optional Word2Vec and sentence-embedding backends.

All optional dependencies are imported lazily so the baseline package remains
usable without downloading models or installing heavyweight libraries.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from src.text.bow_features import tokenize_zh_words


def encode_word2vec(
    train_texts: Iterable[str],
    texts: Iterable[str] | None = None,
    *,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    workers: int = 1,
    epochs: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, object]:
    """Train Word2Vec on training texts and mean-pool document vectors.

    The vocabulary is learned from ``train_texts`` only. If ``texts`` is
    omitted, vectors for the training documents are returned.
    """
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise RuntimeError("Word2Vec requires gensim; install the optional text dependencies first.") from exc

    training = [tokenize_zh_words(text) for text in train_texts]
    model = Word2Vec(
        sentences=training,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )
    values = training if texts is None else [tokenize_zh_words(text) for text in texts]
    return _mean_word_vectors(values, model), model


def _mean_word_vectors(tokenized: list[list[str]], model: object) -> np.ndarray:
    size = int(model.wv.vector_size)
    result = np.zeros((len(tokenized), size), dtype=np.float32)
    for row, words in enumerate(tokenized):
        vectors = [model.wv[word] for word in words if word in model.wv]
        if vectors:
            result[row] = np.mean(vectors, axis=0)
    return result


def encode_sentence_transformer(
    texts: Iterable[str],
    *,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 32,
    max_length: int | None = None,
    device: str | None = None,
    local_files_only: bool = False,
) -> np.ndarray:
    """Encode text with a multilingual or Chinese sentence-transformer."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Sentence embeddings require sentence-transformers; install requirements-transformers.txt first."
        ) from exc
    values = list(texts)
    if not values:
        return np.empty((0, 0), dtype=np.float32)
    model = SentenceTransformer(model_name, device=device, local_files_only=local_files_only)
    if max_length is not None:
        model.max_seq_length = max_length
    return np.asarray(model.encode(values, batch_size=batch_size, normalize_embeddings=False), dtype=np.float32)


def encode_bge_m3(
    texts: Iterable[str],
    *,
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 8,
    max_length: int = 512,
    device: str | None = None,
    local_files_only: bool = False,
) -> np.ndarray:
    """Encode with the Chinese/multilingual BGE-M3 sentence embedding model."""
    return encode_sentence_transformer(
        texts,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        local_files_only=local_files_only,
    )
