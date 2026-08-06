"""Run paper-style weak sentiment classification on a chronological panel.

The label is 1 iff the paper/event three-day return is positive. Feature
vocabularies and classifiers are fitted on the training window only. Optional
hyperparameter search uses the tail of the training window as a chronological
validation set; the final selected model is refit on all training rows.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.representation_models import fit_sentiment_classifier
from src.evaluation.classification import summarize_classification_stability
from src.evaluation.mcs import classification_loss_matrix, model_confidence_set
from src.text.bow_features import (
    fit_tfidf,
    fit_word_count,
    fit_word_tfidf,
    make_hashing_bow,
    transform_tfidf,
)
from src.text.lexicon_features import lexicon_score


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _truncate_text(value: object, max_chars: int) -> str:
    text = "" if value is None else str(value)
    return text if max_chars <= 0 else text[:max_chars]


def _features(name: str, train_texts: list[str], test_texts: list[str], args: Any):
    if ":" in name and name.endswith((".npy",)):
        label, path = name.split(":", 1)
        matrix = np.load(path)
        if len(matrix) != len(train_texts) + len(test_texts):
            raise ValueError("precomputed .npy representation row count must equal train plus predict rows")
        return matrix[:len(train_texts)], matrix[len(train_texts):]
    if name in {"roberta", "bert"}:
        from src.text.embeddings import encode_local_transformer
        model_name = {
            "roberta": "hfl/chinese-roberta-wwm-ext",
            "bert": "hfl/chinese-bert-wwm-ext",
        }[name]
        return (
            encode_local_transformer(train_texts, model_name=model_name, batch_size=args.transformer_batch_size, max_length=args.transformer_max_length),
            encode_local_transformer(test_texts, model_name=model_name, batch_size=args.transformer_batch_size, max_length=args.transformer_max_length),
        )
    if name == "word_tfidf":
        vectorizer = fit_word_tfidf(train_texts, min_df=args.min_df, max_df=args.max_df, max_features=args.max_features)
        return transform_tfidf(vectorizer, train_texts), transform_tfidf(vectorizer, test_texts)
    if name == "char_tfidf":
        vectorizer = fit_tfidf(train_texts, min_df=args.min_df, max_df=args.max_df, max_features=args.max_features)
        return transform_tfidf(vectorizer, train_texts), transform_tfidf(vectorizer, test_texts)
    if name == "word_count":
        vectorizer = fit_word_count(train_texts, min_df=args.min_df, max_df=args.max_df, max_features=args.max_features)
        return vectorizer.transform(train_texts).tocsr(), vectorizer.transform(test_texts).tocsr()
    if name == "hashing_bow":
        vectorizer = make_hashing_bow(n_features=args.hashing_features)
        return vectorizer.transform(train_texts).tocsr(), vectorizer.transform(test_texts).tocsr()
    if name == "lexicon":
        columns = ["positive_count", "negative_count", "lexicon_score"]
        train = pd.DataFrame([lexicon_score(text) for text in train_texts], columns=columns).to_numpy(dtype=np.float32)
        test = pd.DataFrame([lexicon_score(text) for text in test_texts], columns=columns).to_numpy(dtype=np.float32)
        return train, test
    if name == "word2vec":
        from src.text.semantic_embeddings import encode_word2vec
        train, model = encode_word2vec(train_texts, vector_size=args.word2vec_size, epochs=args.word2vec_epochs, seed=args.seed)
        # Reuse the training vocabulary/model rather than fitting on test text.
        from src.text.semantic_embeddings import _mean_word_vectors
        from src.text.bow_features import tokenize_zh_words
        test = _mean_word_vectors([tokenize_zh_words(text) for text in test_texts], model)
        return train, test
    raise ValueError(f"unsupported representation: {name}")


def _grid(model: str, enabled: bool, stage: str) -> dict[str, list[Any]] | None:
    if not enabled:
        return None
    if model in {"logistic", "logistic_regression"}:
        values = [0.1, 1.0, 10.0] if stage == "coarse" else [0.01, 0.1, 1.0, 10.0, 100.0]
        return {"C": values, "solver": ["liblinear"], "class_weight": ["balanced"]}
    if model in {"random_forest", "rf"}:
        return {"n_estimators": [200] if stage == "coarse" else [200, 500], "max_depth": [6, 12] if stage == "coarse" else [6, 12, 24, None], "min_samples_leaf": [1, 2] if stage == "coarse" else [1, 2, 5, 10], "max_features": ["sqrt"]}
    if model in {"mlp", "nn"}:
        if stage == "coarse":
            return {"hidden_layer_sizes": [(64,), (128, 32)], "alpha": [1e-3], "learning_rate_init": [3e-4, 1e-3], "batch_size": [128], "max_iter": [200, 400], "solver": ["adam"], "early_stopping": [False]}
        return {"hidden_layer_sizes": [(32,), (64,), (128,), (128, 32)], "alpha": [1e-4, 1e-3, 1e-2], "learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3], "batch_size": [64, 128, 256], "max_iter": [200, 400, 800], "solver": ["adam"], "early_stopping": [False]}
    raise ValueError(f"unsupported classifier: {model}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("panel")
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--predict-start", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--date-column", default="entry_date")
    parser.add_argument("--target-column", default="return_3d_event")
    parser.add_argument("--representations", default="word_tfidf,char_tfidf,word_count,hashing_bow,lexicon,word2vec")
    parser.add_argument("--classifiers", default="logistic,random_forest,mlp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="", help="Comma-separated seeds for stability runs; overrides --seed")
    parser.add_argument("--tune", action="store_true", help="Search a chronological validation tail of training data")
    parser.add_argument("--search-stage", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--validation-start", default="", help="Optional explicit validation start; otherwise use the training tail")
    parser.add_argument("--pooling", choices=["pooled"], default="pooled", help="All stocks share one representation and classifier")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-input-chars", type=int, default=12000,
                        help="Leading characters passed to text representations; 0 keeps the full text")
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-df", type=float, default=1.0)
    parser.add_argument("--max-features", type=int, default=100000)
    parser.add_argument("--hashing-features", type=int, default=2**18)
    parser.add_argument("--word2vec-size", type=int, default=100)
    parser.add_argument("--word2vec-epochs", type=int, default=20)
    parser.add_argument("--transformer-batch-size", type=int, default=8)
    parser.add_argument("--transformer-max-length", type=int, default=256)
    parser.add_argument("--output", default="reports/classification/results.json")
    parser.add_argument("--qwen-npy", default="", help="Precomputed Qwen/Ollama matrix, ordered like the sorted panel")
    parser.add_argument("--roberta-npy", default="", help="Precomputed Chinese RoBERTa matrix")
    parser.add_argument("--bert-npy", default="", help="Precomputed Chinese BERT matrix")
    parser.add_argument("--bge_m3-npy", default="", help="Precomputed BGE-M3 matrix")
    parser.add_argument("--mcs", action="store_true", help="Run block-bootstrap MCS on test-set losses")
    parser.add_argument("--mcs-loss", choices=["log_loss", "brier"], default="log_loss")
    parser.add_argument("--mcs-alpha", type=float, default=0.10)
    parser.add_argument("--mcs-reps", type=int, default=1000)
    parser.add_argument("--mcs-block-length", type=int, default=1)
    parser.add_argument("--mcs-seed", type=int, default=42)
    args = parser.parse_args()
    _seed_everything(args.seed)

    frame = pd.read_parquet(args.panel).copy()
    required = {args.text_column, args.date_column, args.target_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    frame[args.date_column] = pd.to_datetime(frame[args.date_column], errors="coerce")
    frame[args.text_column] = frame[args.text_column].map(
        lambda value: _truncate_text(value, args.max_input_chars)
    )
    frame = frame.dropna(subset=[args.date_column]).sort_values(args.date_column).reset_index(drop=True)
    train = frame[frame[args.date_column] < args.train_end].copy()
    test = frame[frame[args.date_column] >= args.predict_start].copy()
    if train.empty or test.empty:
        raise ValueError("training and prediction windows must both be non-empty")

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()] or [args.seed]
    output: dict[str, Any] = {
        "train_rows": len(train),
        "predict_rows": len(test),
        "pooling": args.pooling,
        "train_start": str(train[args.date_column].min()),
        "train_end_exclusive": str(pd.Timestamp(args.train_end)),
        "predict_start": str(pd.Timestamp(args.predict_start)),
        "predict_end": str(test[args.date_column].max()),
        "max_input_chars": args.max_input_chars,
        "results": [],
    }
    prediction_store: dict[str, np.ndarray] = {}
    representations = [x.strip() for x in args.representations.split(",") if x.strip()]
    if args.qwen_npy:
        representations.append("npy:" + args.qwen_npy)
    if args.roberta_npy:
        representations.append("npy:" + args.roberta_npy)
    if args.bert_npy:
        representations.append("npy:" + args.bert_npy)
    if args.bge_m3_npy:
        representations.append("bge_m3:" + args.bge_m3_npy)
    for representation in representations:
        x_train, x_test = _features(representation, train[args.text_column].tolist(), test[args.text_column].tolist(), args)
        for classifier in [x.strip() for x in args.classifiers.split(",") if x.strip()]:
            for seed in seeds:
                _seed_everything(seed)
                result = fit_sentiment_classifier(
                    x_train, train[args.target_column], x_test, test[args.target_column],
                    model_name=classifier, random_state=seed,
                    validation_fraction=args.validation_fraction,
                    param_grid=_grid(classifier, args.tune, args.search_stage),
                )
                output["results"].append({
                    "representation": representation,
                    "classifier": classifier,
                    "seed": seed,
                    "tuned": args.tune,
                    "best_params": result.best_params,
                    **result.metrics,
                })
                key = f"{representation}|{classifier}|seed={seed}"
                prediction_store[key] = result.probabilities
    output["stability_summary"] = summarize_classification_stability(output["results"])
    if args.mcs:
        output["mcs"] = model_confidence_set(
            classification_loss_matrix(test[args.target_column], prediction_store, loss=args.mcs_loss),
            alpha=args.mcs_alpha,
            bootstrap_reps=args.mcs_reps,
            block_length=args.mcs_block_length,
            seed=args.mcs_seed,
        )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
