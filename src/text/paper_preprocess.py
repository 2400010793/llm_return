"""Reproducible text preprocessing interfaces used by the paper baselines.

The original paper applies aggressive word-level cleaning for Word2Vec, SESTM,
and LMMD, while transformer models receive raw text.  This module keeps those
two paths separate and exposes the cleaning decisions explicitly.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import jieba

_HTML = re.compile(r"<[^>]+>")
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u4e00-\u9fff]+")
_SPACE = re.compile(r"\s+")


def expand_english_contractions(text: str) -> str:
    """Expand common contractions before word-level feature extraction."""
    replacements = {
        "can't": "cannot", "won't": "will not", "n't": " not",
        "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
        "'m": " am", "'s": " is",
    }
    value = text
    for old, new in replacements.items():
        value = re.sub(re.escape(old), new, value, flags=re.IGNORECASE)
    return value


def clean_for_word_models(text: str, *, lowercase: bool = True) -> str:
    """Clean text for word models without removing Chinese characters."""
    value = _HTML.sub(" ", str(text or "")).replace("\u3000", " ")
    value = expand_english_contractions(value)
    if lowercase:
        value = value.lower()
    return _SPACE.sub(" ", value).strip()


def tokenize_paper_words(
    text: str,
    *,
    stop_words: Iterable[str] = (),
    remove_numbers: bool = True,
    remove_punctuation: bool = True,
) -> list[str]:
    """Tokenize English/Chinese text for Word2Vec, SESTM, and dictionary models."""
    value = clean_for_word_models(text)
    stops = {str(word).lower() for word in stop_words}
    tokens: list[str] = []
    for chunk in _WORD.findall(value):
        if remove_numbers and chunk.isdigit():
            continue
        if chunk in stops:
            continue
        if "\u4e00" <= chunk[0] <= "\u9fff" and len(chunk) > 1:
            pieces = [x.strip() for x in jieba.lcut(chunk) if x.strip()]
        else:
            pieces = [chunk]
        tokens.extend(piece for piece in pieces if piece not in stops)
    return tokens


def truncate_tokens(tokens: list[str], max_tokens: int) -> list[str]:
    """Apply the paper's keep-the-leading-tokens rule."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    return tokens[:max_tokens]