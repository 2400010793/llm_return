"""Create a position-aligned mean/CLS/max pooling ablation for Sina news."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text.embeddings import encode_local_transformer_poolings


PAPER_MODEL_ROOT = Path(os.environ.get(
    "PAPER_HK_MODEL_ROOT", "/home/team/.cache/huggingface/paper_hk_models"
))
MODELS = {
    "chinese_bert": PAPER_MODEL_ROOT / "ckiplab-bert-base-chinese",
    "chinese_roberta": Path("/home/team/llm_return/models/chinese-roberta-wwm-ext"),
    "bge_m3": Path("/home/team/llm_return/models/bge-m3"),
    "xlm_roberta_large": PAPER_MODEL_ROOT / "xlm-roberta-large",
}
DEFAULT_BATCH_SIZES = {
    "chinese_bert": 8,
    "chinese_roberta": 8,
    "bge_m3": 4,
    "xlm_roberta_large": 2,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--text-column", default="text_plain")
    parser.add_argument("--row-column", default="article_id")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rows", type=int, default=None, help="Optional leading-row probe limit")
    args = parser.parse_args()

    model_path = MODELS[args.model]
    if not model_path.is_dir():
        raise FileNotFoundError(f"local model is missing: {model_path}")
    frame = pd.read_parquet(args.input)
    if args.rows is not None:
        if args.rows < 1:
            raise ValueError("--rows must be positive")
        frame = frame.iloc[:args.rows].copy()
    required = {args.text_column, args.row_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"input is missing columns: {sorted(missing)}")
    if frame[args.row_column].isna().any() or frame[args.row_column].duplicated().any():
        raise ValueError(f"{args.row_column} must be non-null and unique")
    texts = frame[args.text_column].fillna("").astype(str).tolist()
    if any(not text.strip() for text in texts):
        raise ValueError(f"{args.text_column} contains empty text")

    matrices = encode_local_transformer_poolings(
        texts,
        str(model_path),
        batch_size=args.batch_size or DEFAULT_BATCH_SIZES[args.model],
        max_length=args.max_length,
        device=args.device,
        local_files_only=True,
    )
    dimensions = {matrix.shape[1] for matrix in matrices.values()}
    if len(dimensions) != 1 or any(matrix.shape[0] != len(frame) for matrix in matrices.values()):
        raise ValueError("pooling outputs are not row/dimension aligned")
    if any(not np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("pooling outputs contain non-finite values")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for pooling, matrix in matrices.items():
        np.save(args.output_dir / f"{pooling}.npy", matrix)
    metadata = frame[[args.row_column]].copy()
    metadata["row_index"] = np.arange(len(frame), dtype=np.int64)
    metadata["text_sha256"] = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    metadata.to_parquet(args.output_dir / "metadata.parquet", index=False)
    summary = {
        "model": args.model,
        "model_path": str(model_path),
        "input": str(args.input),
        "text_column": args.text_column,
        "row_column": args.row_column,
        "rows": len(frame),
        "max_length": args.max_length,
        "batch_size": args.batch_size or DEFAULT_BATCH_SIZES[args.model],
        "pooling": {
            "mean": "masked token mean excluding padding and tokenizer special tokens",
            "cls": "first-token hidden state",
            "max": "masked token max excluding padding and tokenizer special tokens",
        },
        "outputs": {name: list(matrix.shape) for name, matrix in matrices.items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
