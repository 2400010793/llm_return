"""Generate and cache embeddings with a locally running Ollama model.

The model, endpoint, text hash and row count are stored beside the .npy file
so results remain auditable and can be regenerated deterministically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ingest import read_table
from src.text.ollama_client import ollama_embed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True, help="输出 .npy 路径")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--model", default="qwen3-embedding:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11435")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    frame = read_table(args.input)
    if args.text_column not in frame:
        raise ValueError(f"missing text column: {args.text_column}")
    texts = frame[args.text_column].fillna("").astype(str).tolist()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), args.batch_size):
        vectors.extend(ollama_embed(texts[start : start + args.batch_size], model=args.model, base_url=args.base_url, timeout=args.timeout))
    matrix = np.asarray(vectors, dtype=np.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, matrix)
    metadata = {
        "model": args.model,
        "base_url": args.base_url,
        "text_column": args.text_column,
        "rows": len(texts),
        "dimension": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "text_sha256": hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
