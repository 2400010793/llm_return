"""Validate the BOW/TF-IDF baseline on an exported news table.

Expected columns: title/headline, body/summary/content, and return_1d.
The split is chronological when a published_at/date column exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sklearn.linear_model import Ridge

from src.models.baselines import fit_predict_tfidf_ridge
from src.text.lexicon_features import lexicon_score
from src.text.bow_features import fit_word_tfidf, transform_tfidf
from src.text.preprocess_zh import combine_news_text


def load_frame(path: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    return pd.read_json(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--return-column", default="return_1d")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--output", default="reports/bow_validation.json")
    args = parser.parse_args()

    frame = load_frame(args.input).copy()
    date_column = next((c for c in ("published_at", "date", "collected_at") if c in frame), None)
    if date_column:
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
        frame = frame.sort_values(date_column, kind="stable")
    headline = frame.get("headline", frame.get("title", pd.Series("", index=frame.index)))
    body = frame.get("body", frame.get("content", frame.get("summary", pd.Series("", index=frame.index))))
    frame["text"] = [combine_news_text(a, b) for a, b in zip(headline, body)]
    lexicon = frame["text"].map(lexicon_score).apply(pd.Series)
    frame = pd.concat([frame, lexicon.add_prefix("lexicon_")], axis=1)
    frame = frame.dropna(subset=[args.return_column])
    split = max(1, int(len(frame) * (1 - args.test_fraction)))
    if split >= len(frame):
        raise ValueError("need at least one chronological test row; small samples are pipeline-only")
    train, test = frame.iloc[:split], frame.iloc[split:]
    result = fit_predict_tfidf_ridge(train.text.tolist(), train[args.return_column].to_numpy(), test.text.tolist())
    word_vectorizer = fit_word_tfidf(train.text.tolist())
    word_model = Ridge(alpha=1.0)
    word_model.fit(transform_tfidf(word_vectorizer, train["text"].tolist()), train[args.return_column].to_numpy())
    word_predictions = word_model.predict(transform_tfidf(word_vectorizer, test["text"].tolist()))
    output = {
        "warning": "pipeline validation only; do not interpret small-sample predictions as paper evidence",
        "n_train": len(train),
        "n_test": len(test),
        "chronological_split": True,
        "test_start": str(test[date_column].iloc[0]) if date_column else None,
        "test_end": str(test[date_column].iloc[-1]) if date_column else None,
        "character_tfidf_vocabulary_size": len(result.vectorizer.vocabulary_),
        "character_tfidf_predictions": result.predictions.tolist(),
        "word_tfidf_vocabulary_size": len(word_vectorizer.vocabulary_),
        "word_tfidf_predictions": word_predictions.tolist(),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
