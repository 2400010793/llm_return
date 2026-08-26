"""Build one matrix on the frozen CNINFO Qwen strict intersection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


TARGETS = {
    "future_return": ("未来", "收益"),
    "return": ("收益",),
    "loss": ("亏损",),
    "excess_return": ("超额", "收益"),
    "profit": ("盈利",),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt", choices=tuple(TARGETS), required=True)
    parser.add_argument("--variant", choices=("short", "masked_short"), required=True)
    parser.add_argument("--kind", choices=("article_mean", "target_token"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    shard_ids = [int(value) for value in manifest["common_shards"]]
    rows = int(manifest["rows"])
    first = args.root / args.prompt / args.variant / f"shard-{shard_ids[0]}"
    article = np.load(first / "article_mean_embeddings.npy", mmap_mode="r")
    hidden = int(article.shape[1])
    token_indices: list[int] = []
    if args.kind == "target_token":
        info = json.loads((first / "prompt_tokens.json").read_text(encoding="utf-8"))
        token_indices = [
            index for index, token in enumerate(info["tokens"])
            if token in TARGETS[args.prompt]
        ]
        if not token_indices:
            raise ValueError(f"target tokens missing: {info['tokens']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    incoming = args.output.with_name(args.output.name + f".incoming.{os.getpid()}")
    if args.output.exists():
        matrix = np.load(args.output, mmap_mode="r")
        if matrix.shape == (rows, hidden) and np.isfinite(matrix).all():
            print(json.dumps({"output": str(args.output), "resumed": True}))
            return
        raise FileExistsError(f"refusing invalid existing output: {args.output}")
    matrix = np.lib.format.open_memmap(
        incoming, mode="w+", dtype=np.float16, shape=(rows, hidden)
    )
    offset = 0
    try:
        for shard_id in shard_ids:
            shard = args.root / args.prompt / args.variant / f"shard-{shard_id}"
            if args.kind == "article_mean":
                values = np.asarray(
                    np.load(shard / "article_mean_embeddings.npy", mmap_mode="r"),
                    dtype=np.float32,
                )
            else:
                info = json.loads(
                    (shard / "prompt_tokens.json").read_text(encoding="utf-8")
                )
                current = [
                    index for index, token in enumerate(info["tokens"])
                    if token in TARGETS[args.prompt]
                ]
                if current != token_indices:
                    raise ValueError(f"token positions changed in shard {shard_id}")
                tokens = np.load(shard / "prompt_token_embeddings.npy", mmap_mode="r")
                values = np.asarray(tokens[:, current, :], dtype=np.float32).mean(axis=1)
            if values.ndim != 2 or values.shape[1] != hidden:
                raise ValueError(f"bad matrix shape in shard {shard_id}: {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite values in shard {shard_id}")
            matrix[offset:offset + len(values)] = values.astype(np.float16)
            offset += len(values)
        if offset != rows:
            raise ValueError(f"wrote {offset} rows, expected {rows}")
        matrix.flush()
        del matrix
        incoming.replace(args.output)
    except Exception:
        try:
            incoming.unlink()
        except FileNotFoundError:
            pass
        raise
    sidecar = {
        "format_version": "qwen_cninfo_strict_matrix_v1",
        "prompt": args.prompt, "variant": args.variant, "kind": args.kind,
        "rows": rows, "hidden_size": hidden, "dtype": "float16",
        "intersection_manifest": str(args.manifest),
        "target_token_indices": token_indices,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(sidecar, ensure_ascii=False))


if __name__ == "__main__":
    main()
