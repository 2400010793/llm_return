"""Run the available text representation baselines on a text column.

This script only fits train-dependent representations on the supplied data;
use a time-split panel or separate train/test files for strict forecasting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow direct execution via ``python scripts/run_text_representations.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ingest import read_table
from src.text.bow_features import (
    fit_tfidf,
    fit_word_count,
    fit_word_tfidf,
    make_hashing_bow,
    transform_tfidf,
)
from src.text.lexicon_features import lexicon_score


def _read_input(path: str, text_column: str) -> pd.DataFrame:
    source = Path(path)
    if source.is_dir():
        files = sorted(source.glob("part-*.jsonl"))
        if not files:
            raise ValueError(f"no part-*.jsonl files found in {source}")
        columns = [text_column, "row_index", "document_id", "stock_id_alignment", "announcement_date_alignment"]
        frames = []
        for file in files:
            rows = []
            with file.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        rows.append({column: record.get(column) for column in columns if column in record})
            frames.append(pd.DataFrame(rows))
        return pd.concat(frames, ignore_index=True)
    return read_table(source)


def _truncate_text(value: object, max_chars: int) -> str:
    text = "" if value is None else str(value)
    return text if max_chars <= 0 else text[:max_chars]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output-dir", default="data/processed/text_representations")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-input-chars", type=int, default=12000,
                        help="Leading characters passed to encoders; 0 keeps the full text")
    parser.add_argument("--word-tfidf", action="store_true")
    parser.add_argument("--word-count", action="store_true")
    parser.add_argument("--hashing-bow", action="store_true")
    parser.add_argument("--char-tfidf", action="store_true")
    parser.add_argument("--lexicon", action="store_true")
    parser.add_argument("--word2vec", action="store_true")
    parser.add_argument("--sentence-transformer", action="store_true")
    parser.add_argument("--bge-m3", action="store_true")
    parser.add_argument("--bge-m3-model", default="BAAI/bge-m3")
    parser.add_argument("--roberta", action="store_true")
    parser.add_argument("--roberta-model", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--bert", action="store_true")
    parser.add_argument("--bert-model", default="hfl/chinese-bert-wwm-ext")
    parser.add_argument("--transformer-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum Transformer tokens; RoBERTa must not exceed 512")
    parser.add_argument("--ollama", action="store_true")
    parser.add_argument("--ollama-model", default="qwen3-embedding:8b")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11435")
    parser.add_argument("--api-embeddings", action="store_true")
    parser.add_argument("--api-model", default="text-embedding-3-small")
    parser.add_argument("--api-base-url", default="https://api.openai.com/v1")
    args = parser.parse_args()
    frame = _read_input(args.input, args.text_column)
    if args.text_column not in frame:
        raise ValueError(f"missing text column: {args.text_column}")
    texts = [
        _truncate_text(value, args.max_input_chars)
        for value in frame[args.text_column].tolist()
    ]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata_columns = [c for c in ("row_index", "document_id", "stock_id_alignment", "announcement_date_alignment") if c in frame]
    metadata = frame[metadata_columns].copy() if metadata_columns else pd.DataFrame({"row_index": np.arange(len(frame))})
    metadata["text_sha256"] = [__import__("hashlib").sha256(t.encode("utf-8")).hexdigest() for t in texts]
    metadata.to_parquet(output / "metadata.parquet", index=False)
    summary: dict[str, object] = {
        "rows": len(texts),
        "max_input_chars": args.max_input_chars,
        "representations": [],
    }
    if args.word_tfidf:
        vectorizer = fit_word_tfidf(texts, min_df=1, max_df=1.0)
        matrix = transform_tfidf(vectorizer, texts)
        from src.text.bow_features import save_tfidf
        save_tfidf(vectorizer, matrix, output / "word_tfidf")
        summary["representations"].append({"name": "word_tfidf", "shape": list(matrix.shape)})
    if args.word_count:
        vectorizer = fit_word_count(texts, min_df=1, max_df=1.0)
        matrix = vectorizer.transform(texts).tocsr()
        import joblib
        joblib.dump(vectorizer, output / "word_count.vectorizer.joblib")
        from scipy import sparse
        sparse.save_npz(output / "word_count.npz", matrix)
        summary["representations"].append({"name": "word_count", "shape": list(matrix.shape)})
    if args.hashing_bow:
        vectorizer = make_hashing_bow()
        matrix = vectorizer.transform(texts).tocsr()
        from scipy import sparse
        sparse.save_npz(output / "hashing_bow.npz", matrix)
        summary["representations"].append({"name": "hashing_bow", "shape": list(matrix.shape)})
    if args.char_tfidf:
        vectorizer = fit_tfidf(texts, min_df=1, max_df=1.0)
        matrix = transform_tfidf(vectorizer, texts)
        from src.text.bow_features import save_tfidf
        save_tfidf(vectorizer, matrix, output / "char_tfidf")
        summary["representations"].append({"name": "char_tfidf", "shape": list(matrix.shape)})
    if args.lexicon:
        features = pd.DataFrame([lexicon_score(text) for text in texts])
        features.to_parquet(output / "lexicon_features.parquet", index=False)
        summary["representations"].append({"name": "lexicon", "shape": list(features.shape)})
    if args.word2vec:
        from src.text.semantic_embeddings import encode_word2vec
        vectors, model = encode_word2vec(texts, vector_size=100, epochs=20)
        np.save(output / "word2vec.npy", vectors)
        model.save(str(output / "word2vec.model"))
        summary["representations"].append({"name": "word2vec", "shape": list(vectors.shape)})
    if args.sentence_transformer or args.bge_m3:
        from src.text.semantic_embeddings import encode_bge_m3, encode_sentence_transformer
        if args.bge_m3:
            vectors = encode_bge_m3(
                texts,
                model_name=args.bge_m3_model,
                batch_size=args.transformer_batch_size,
                max_length=args.max_length,
                local_files_only=Path(args.bge_m3_model).is_absolute(),
            )
        else:
            vectors = encode_sentence_transformer(texts, batch_size=args.transformer_batch_size)
        name = "bge_m3" if args.bge_m3 else "sentence_transformer"
        np.save(output / f"{name}.npy", vectors)
        summary["representations"].append({"name": name, "shape": list(vectors.shape)})
    if args.roberta:
        from src.text.embeddings import encode_local_transformer
        vectors = encode_local_transformer(texts, model_name=args.roberta_model, batch_size=args.transformer_batch_size, max_length=args.max_length, local_files_only=Path(args.roberta_model).is_absolute())
        np.save(output / "chinese_roberta.npy", vectors)
        summary["representations"].append({"name": "chinese_roberta", "shape": list(vectors.shape)})
    if args.bert:
        from src.text.embeddings import encode_local_transformer
        vectors = encode_local_transformer(texts, model_name=args.bert_model, batch_size=args.transformer_batch_size, max_length=args.max_length)
        np.save(output / "chinese_bert.npy", vectors)
        summary["representations"].append({"name": "chinese_bert", "model": args.bert_model, "shape": list(vectors.shape)})
    if args.ollama:
        from src.text.ollama_client import ollama_embed
        vectors = []
        for start in range(0, len(texts), 8):
            vectors.extend(ollama_embed(texts[start : start + 8], model=args.ollama_model, base_url=args.ollama_base_url))
        matrix = np.asarray(vectors, dtype=np.float32)
        np.save(output / "ollama.npy", matrix)
        summary["representations"].append({"name": "ollama", "model": args.ollama_model, "shape": list(matrix.shape)})
    if args.api_embeddings:
        from src.text.api_embeddings import encode_api_embeddings
        vectors = encode_api_embeddings(texts, model=args.api_model, base_url=args.api_base_url)
        np.save(output / "api_embeddings.npy", vectors)
        summary["representations"].append({"name": "api_embeddings", "model": args.api_model, "shape": list(vectors.shape)})
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
