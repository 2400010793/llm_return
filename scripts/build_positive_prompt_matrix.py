"""Assemble one independent positive-prompt representation from shard outputs.

The command deliberately accepts exactly one prompt and one model per run.  It
never concatenates prompt experiments.  Shards are copied in numeric order and
aligned by their persisted ``row_index`` metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROMPTS = {
    "profit": "分析股票盈利",
    "excess_return": "分析股票超额收益",
    "return": "分析股票收益",
    "loss": "分析股票亏损",
}


def _semantic_indices(tokens: list[str], representation: str) -> list[int]:
    if representation in {"prompt_mean", "full_mean"}:
        return list(range(len(tokens)))
    if representation == "stock_span":
        indices = [i for i, token in enumerate(tokens) if "股票" in token]
        if not indices:
            indices = [i for i, token in enumerate(tokens) if token in {"股", "票"}]
    elif representation == "return_span":
        indices = [
            i for i, token in enumerate(tokens)
            if any(word in token for word in ("收益", "盈利", "亏损", "超", "额", "收", "益", "盈", "利", "亏", "损"))
        ]
    elif representation == "excess_span":
        # Keep only the direction-specific word.  Chinese tokenizers may emit
        # ``超额`` as one token or as ``超`` + ``额``; both forms are accepted,
        # but never include ``收益`` or the common instruction prefix.
        indices = [i for i, token in enumerate(tokens) if token in {"超额", "超", "额"}]
    elif representation == "profit_span":
        indices = [i for i, token in enumerate(tokens) if token in {"盈利", "盈", "利"}]
    elif representation == "plain_return_span":
        indices = [i for i, token in enumerate(tokens) if token in {"收益", "收", "益"}]
    elif representation == "loss_span":
        indices = [i for i, token in enumerate(tokens) if token in {"亏损", "亏", "损"}]
    else:
        raise ValueError(f"unsupported representation: {representation}")
    if not indices:
        raise ValueError(f"no token matches {representation}: {tokens}")
    return indices


def _metadata(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    frame = pd.DataFrame(rows)
    if "row_index" not in frame or frame["row_index"].isna().any():
        raise ValueError(f"invalid metadata row_index: {path}")
    frame["row_index"] = pd.to_numeric(frame["row_index"], errors="raise").astype(np.int64)
    if frame["row_index"].duplicated().any():
        raise ValueError(f"duplicate row_index in {path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("sina", "cninfo"), required=True)
    parser.add_argument(
        "--model", choices=("roberta", "bge_m3", "qwen3_embedding_8b"), required=True,
    )
    parser.add_argument("--prompt", choices=tuple(PROMPTS), required=True)
    parser.add_argument("--variant", choices=("short", "masked_short"), required=True)
    parser.add_argument("--representation", choices=("prompt_mean", "full_mean", "article_mean", "return_token", "return_span", "stock_span", "excess_span", "profit_span", "plain_return_span", "loss_span"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()

    qwen_layout = args.model == "qwen3_embedding_8b"
    shard_root = args.embeddings_root / args.dataset / args.model
    shard_frames: list[pd.DataFrame] = []
    shard_arrays: list[np.ndarray] = []
    prompt_text = PROMPTS[args.prompt] + ("。" if qwen_layout else "")
    for shard_id in range(args.shards):
        leaf = (
            args.embeddings_root / args.dataset / args.prompt / args.variant / f"shard-{shard_id}"
            if qwen_layout else
            shard_root / f"shard-{shard_id}" / args.prompt / args.variant
        )
        metadata = _metadata(leaf / "metadata.jsonl")
        token_info = json.loads((leaf / "prompt_tokens.json").read_text(encoding="utf-8"))
        if token_info.get("text") != prompt_text:
            raise ValueError(f"prompt hash/text mismatch in {leaf}")
        if qwen_layout and args.representation == "article_mean":
            matrix = np.asarray(
                np.load(leaf / "article_mean_embeddings.npy", mmap_mode="r"),
                dtype=np.float16,
            )
        elif qwen_layout and args.representation in {"prompt_mean", "return_token"}:
            tokens = list(token_info["tokens"])
            indices = (
                list(range(len(tokens))) if args.representation == "prompt_mean" else
                [i for i, token in enumerate(tokens) if token == "收益"]
            )
            if not indices:
                raise ValueError(f"no 收益 token in {leaf}: {tokens}")
            token_matrix = np.load(leaf / "prompt_token_embeddings.npy", mmap_mode="r")
            matrix = np.asarray(token_matrix[:, indices, :].mean(axis=1), dtype=np.float16)
        elif qwen_layout:
            raise ValueError(
                f"Qwen layout does not support representation {args.representation}"
            )
        else:
            pooling = np.load(leaf / "short_pooling.npz", mmap_mode="r")
            if args.representation == "full_mean":
                matrix = np.asarray(pooling["full_mean"], dtype=np.float32)
            elif args.representation == "prompt_mean":
                matrix = np.asarray(pooling["prompt_mean"], dtype=np.float32)
            else:
                tokens = list(token_info["tokens"])
                indices = _semantic_indices(tokens, args.representation)
                token_matrix = np.load(leaf / "prompt_token_embeddings.npy", mmap_mode="r")
                matrix = np.asarray(token_matrix[:, indices, :].mean(axis=1), dtype=np.float32)
        if len(metadata) != len(matrix):
            raise ValueError(f"metadata/matrix mismatch in {leaf}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"non-finite embedding values in {leaf}")
        shard_frames.append(metadata[["row_index"]].copy())
        shard_arrays.append(matrix)

    metadata = pd.concat(shard_frames, ignore_index=True)
    matrix = np.concatenate(shard_arrays, axis=0)
    # CNINFO embedding shards persist zero-based source positions, while the
    # canonical 350,577-row panel uses one-based row_index values.
    if args.dataset == "cninfo":
        metadata["row_index"] += 1
    order = np.argsort(metadata["row_index"].to_numpy(), kind="stable")
    metadata = metadata.iloc[order].reset_index(drop=True)
    matrix = matrix[order]
    if metadata["row_index"].duplicated().any():
        raise ValueError("duplicate row_index after shard assembly")
    destination = args.output_root / args.dataset / args.model / args.prompt / args.variant / args.representation
    destination.mkdir(parents=True, exist_ok=True)
    matrix_path = destination / "matrix.npy"
    temporary_matrix = destination / ".matrix.npy.partial"
    temporary_matrix.unlink(missing_ok=True)
    persisted = np.lib.format.open_memmap(
        temporary_matrix, mode="w+", dtype=matrix.dtype, shape=matrix.shape,
    )
    for start in range(0, len(matrix), 1024):
        persisted[start:start + 1024] = matrix[start:start + 1024]
    persisted.flush()
    del persisted
    temporary_matrix.replace(matrix_path)
    metadata.to_parquet(destination / "metadata.parquet", index=False)
    manifest = {
        "format_version": "positive_prompt_matrix_v1",
        "dataset": args.dataset,
        "model": args.model,
        "prompt": args.prompt,
        "prompt_text": prompt_text,
        "variant": args.variant,
        "representation": args.representation,
        "rows": int(len(matrix)),
        "dimension": int(matrix.shape[1]),
        "row_index_min": int(metadata.row_index.min()),
        "row_index_max": int(metadata.row_index.max()),
        "shards": args.shards,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
