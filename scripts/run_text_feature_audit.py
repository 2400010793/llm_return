"""Run TF-IDF/BOW and lexicon feature extraction without return labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.text.bow_features import fit_tfidf, fit_word_tfidf, transform_tfidf
from src.text.lexicon_features import lexicon_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="reports/text_feature_audit.json")
    args = parser.parse_args()
    frame = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    frame = frame[frame["text"].fillna("").astype(str).str.strip().ne("")].copy()
    if frame.empty:
        raise ValueError("no non-empty text rows")
    texts = frame["text"].astype(str).tolist()
    # With this small sample, min_df=1 is used for feature-audit only.
    char_vectorizer = fit_tfidf(texts, min_df=1, max_df=1.0)
    word_vectorizer = fit_word_tfidf(texts, min_df=1, max_df=1.0)
    char_matrix = transform_tfidf(char_vectorizer, texts)
    word_matrix = transform_tfidf(word_vectorizer, texts)
    lexicon = pd.DataFrame([lexicon_score(text) for text in texts])
    report = {
        "warning": "feature audit only; no return labels and no predictive performance is reported",
        "rows": len(frame),
        "char_tfidf": {"vocabulary_size": len(char_vectorizer.vocabulary_), "shape": list(char_matrix.shape), "nnz": int(char_matrix.nnz)},
        "word_tfidf_jieba": {"vocabulary_size": len(word_vectorizer.vocabulary_), "shape": list(word_matrix.shape), "nnz": int(word_matrix.nnz)},
        "lexicon": {"positive_total": int(lexicon["positive_count"].sum()), "negative_total": int(lexicon["negative_count"].sum())},
        "content_types": frame["content_type"].value_counts().to_dict(),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
