"""Audit completeness and paired Prompt alignment for Sina CNINFO-style outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ("roberta", "bge_m3", "ckip_bert", "xlm_roberta_large")
VARIANTS = ("plain", "short", "masked_short", "long", "masked_long")
PAIRS = (("short", "masked_short"), ("long", "masked_long"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=4)
    parser.add_argument("--expected-rows", type=int, default=4928)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    combinations = []
    errors: list[str] = []
    for model in MODELS:
        model_tokens: dict[tuple[int, str], np.ndarray] = {}
        for variant in VARIANTS:
            row_indexes: list[np.ndarray] = []
            shapes = set()
            for shard in range(args.expected_shards):
                directory = args.root / f"shard-{shard}" / model / variant
                required = ["summary.json", "metadata.jsonl", "short_pooling.npz"]
                if variant != "plain":
                    required += [
                        "prompt_token_embeddings.npy", "prompt_input_ids.npy", "prompt_tokens.json"
                    ]
                missing = [name for name in required if not (directory / name).is_file()]
                if missing:
                    errors.append(f"{directory}: missing {missing}")
                    continue
                summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
                metadata = pd.read_json(directory / "metadata.jsonl", lines=True)
                indexes = pd.to_numeric(metadata["row_index"], errors="raise").to_numpy(np.int64)
                row_indexes.append(indexes)
                with np.load(directory / "short_pooling.npz") as archive:
                    for name in summary["outputs"]:
                        if name == "prompt_token_embeddings":
                            continue
                        if name not in archive:
                            errors.append(f"{directory}: {name} absent from pooled archive")
                            continue
                        if name in archive:
                            matrix = archive[name]
                            if len(matrix) != len(metadata) or not np.isfinite(matrix).all():
                                errors.append(f"{directory}: invalid {name} shape/values")
                            shapes.add((name, tuple(matrix.shape[1:])))
                if variant != "plain":
                    prompt = np.load(directory / "prompt_token_embeddings.npy", mmap_mode="r")
                    prompt_ids = np.load(directory / "prompt_input_ids.npy")
                    if len(prompt) != len(metadata) or prompt.shape[1] != len(prompt_ids):
                        errors.append(f"{directory}: prompt row/token mismatch")
                    if not np.isfinite(prompt).all():
                        errors.append(f"{directory}: non-finite prompt embeddings")
                    model_tokens[(shard, variant)] = prompt_ids
            combined = np.concatenate(row_indexes) if row_indexes else np.array([], dtype=np.int64)
            if len(combined) != args.expected_rows:
                errors.append(f"{model}/{variant}: rows {len(combined)} != {args.expected_rows}")
            if len(np.unique(combined)) != len(combined):
                errors.append(f"{model}/{variant}: duplicate row_index")
            if len(combined) and not np.array_equal(
                np.sort(combined), np.arange(1, args.expected_rows + 1, dtype=np.int64)
            ):
                errors.append(f"{model}/{variant}: row_index does not cover 1..N")
            combinations.append({
                "model": model, "variant": variant, "rows": len(combined),
                "features": sorted(name for name, _ in shapes),
            })
        for shard in range(args.expected_shards):
            for raw, masked in PAIRS:
                left = model_tokens.get((shard, raw))
                right = model_tokens.get((shard, masked))
                if left is None or right is None or not np.array_equal(left, right):
                    errors.append(f"{model}/shard-{shard}: Prompt IDs differ for {raw}/{masked}")
    report = {
        "root": str(args.root), "expected_rows": args.expected_rows,
        "expected_shards": args.expected_shards, "combinations": combinations,
        "errors": errors, "complete": not errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
