"""Small local Qwen3-Embedding GGUF token-embedding probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from llama_cpp import Llama
from llama_cpp.llama_cpp import LLAMA_POOLING_TYPE_NONE


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--text", default="公司业绩增长。\n\n分析股票收益。")
    p.add_argument("--max-length", type=int, default=500)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    tokens_text = args.text
    model = Llama(
        model_path=str(args.model), n_ctx=args.max_length, n_batch=args.max_length,
        n_gpu_layers=-1, main_gpu=0, embedding=True,
        pooling_type=LLAMA_POOLING_TYPE_NONE, verbose=False,
    )
    token_ids = model.tokenize(tokens_text.encode("utf-8"), add_bos=False, special=False)
    if len(token_ids) > args.max_length:
        token_ids = token_ids[:args.max_length]
    values = model.embed(tokens_text, normalize=False, truncate=True)
    array = np.asarray(values, dtype=np.float32)
    result = {
        "model": str(args.model),
        "text": tokens_text,
        "token_count": len(token_ids),
        "embedding_shape": list(array.shape),
        "embedding_dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "hidden_size": int(array.shape[-1]) if array.ndim == 2 else None,
        "pooling": "none",
        "max_length": args.max_length,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()