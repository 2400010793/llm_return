"""Assemble one aligned masked-short pooled representation from shard outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_json(path, lines=True)
    if "row_index" not in frame:
        raise ValueError(f"row_index missing from {path}")
    frame["row_index"] = pd.to_numeric(frame["row_index"], errors="raise").astype(np.int64)
    if frame["row_index"].duplicated().any():
        raise ValueError(f"duplicate row_index in {path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-root", type=Path, required=True)
    parser.add_argument("--model", choices=("roberta",), default="roberta")
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--representation", choices=("body_mean", "full_mean", "prompt_mean"), default="body_mean")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    arrays: list[np.ndarray] = []
    prompt_audit = None
    for shard in range(args.shards):
        candidates = sorted((args.embeddings_root / f"shard-{shard}").glob(
            f"group-*/{args.prompt_id}/masked_short"
        ))
        if len(candidates) != 1:
            raise ValueError(
                f"expected one completed group for shard={shard}, prompt={args.prompt_id}; "
                f"found {len(candidates)}"
            )
        leaf = candidates[0]
        # COMPLETED is written at the shard/group level, alongside the
        # prompt directories, by the Slurm staging script.
        if not (leaf.parents[1] / "COMPLETED").exists():
            raise FileNotFoundError(f"incomplete aligned embedding shard: {leaf}")
        metadata = _metadata(leaf / "metadata.jsonl")
        spec = json.loads((leaf / "prompt_spec.json").read_text(encoding="utf-8"))
        current_audit = spec["token_audit"]
        if prompt_audit is None:
            prompt_audit = current_audit
        elif current_audit["prompt_sha256"] != prompt_audit["prompt_sha256"]:
            raise ValueError(f"prompt differs between shards: {leaf}")
        pooling = np.load(leaf / "short_pooling.npz", mmap_mode="r")
        if args.representation not in pooling.files:
            raise ValueError(f"representation {args.representation} missing from {leaf}")
        matrix = np.asarray(pooling[args.representation], dtype=np.float32)
        if len(metadata) != len(matrix) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid matrix/metadata in {leaf}")
        frames.append(metadata[["row_index"]].copy())
        arrays.append(matrix)

    metadata = pd.concat(frames, ignore_index=True)
    matrix = np.concatenate(arrays, axis=0)
    order = np.argsort(metadata["row_index"].to_numpy(), kind="stable")
    metadata = metadata.iloc[order].reset_index(drop=True)
    matrix = matrix[order]
    if metadata["row_index"].duplicated().any():
        raise ValueError("duplicate row_index after assembly")
    destination = args.output_root / args.model / args.prompt_id / "masked_short" / args.representation
    destination.mkdir(parents=True, exist_ok=True)
    temp = destination / ".matrix.npy.partial"
    temp.unlink(missing_ok=True)
    persisted = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32, shape=matrix.shape)
    persisted[:] = matrix
    persisted.flush()
    del persisted
    temp.replace(destination / "matrix.npy")
    metadata.to_parquet(destination / "metadata.parquet", index=False)
    manifest = {
        "format_version": "aligned_prompt_matrix_v1",
        "model": args.model,
        "prompt_id": args.prompt_id,
        "variant": "masked_short",
        "representation": args.representation,
        "rows": int(len(matrix)),
        "dimension": int(matrix.shape[1]),
        "row_index_min": int(metadata.row_index.min()),
        "row_index_max": int(metadata.row_index.max()),
        "shards": args.shards,
        "prompt_audit": prompt_audit,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
