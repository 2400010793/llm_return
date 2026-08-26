"""Audit pairwise similarity and separation of aligned prompt representations."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def _find_prompt(root: Path, shard: int, prompt_id: str) -> Path | None:
    candidates = sorted((root / f"shard-{shard}").glob(f"group-*/{prompt_id}/masked_short"))
    if len(candidates) != 1:
        return None
    group_root = candidates[0].parents[1]
    return candidates[0] if (group_root / "COMPLETED").exists() else None


def _stats(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    norms = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.divide(np.sum(left * right, axis=1), norms, out=np.zeros(len(left)), where=norms > 0)
    distance = np.linalg.norm(left - right, axis=1)
    return {
        "rows": int(len(left)),
        "cosine_mean": float(np.mean(cosine)),
        "cosine_median": float(np.median(cosine)),
        "cosine_p05": float(np.quantile(cosine, 0.05)),
        "cosine_p95": float(np.quantile(cosine, 0.95)),
        "l2_mean": float(np.mean(distance)),
        "l2_median": float(np.median(distance)),
        "l2_p95": float(np.quantile(distance, 0.95)),
    }


def _load(leaf: Path, representation: str) -> tuple[np.ndarray, np.ndarray]:
    pooling = np.load(leaf / "short_pooling.npz", allow_pickle=False)
    if representation not in pooling.files:
        raise KeyError(f"{representation} missing from {leaf}")
    metadata = pd.read_json(leaf / "metadata.jsonl", lines=True)
    return np.asarray(pooling[representation], dtype=np.float32), metadata["row_index"].to_numpy(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--axes-json", type=Path, required=True, help="JSON mapping axis -> prompt IDs")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--representation", choices=("body_mean", "full_mean", "prompt_mean"), default="body_mean")
    parser.add_argument("--anchor-root", type=Path, default=None)
    parser.add_argument("--anchor-json", type=Path, default=None, help="JSON mapping axis -> anchor prompt ID")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    axes = json.loads(args.axes_json.read_text(encoding="utf-8"))
    anchor_ids = json.loads(args.anchor_json.read_text(encoding="utf-8")) if args.anchor_json else {}
    observations: list[dict] = []
    shard_counts: dict[str, int] = {}
    for axis, prompt_ids in axes.items():
        if len(prompt_ids) < 2:
            continue
        for left_id, right_id in combinations(prompt_ids, 2):
            per_shard = []
            for shard in range(args.shards):
                left_leaf = _find_prompt(args.root, shard, left_id)
                right_leaf = _find_prompt(args.root, shard, right_id)
                if left_leaf is None or right_leaf is None:
                    continue
                left, left_rows = _load(left_leaf, args.representation)
                right, right_rows = _load(right_leaf, args.representation)
                if not np.array_equal(left_rows, right_rows):
                    raise ValueError(f"row_index mismatch in axis={axis}, shard={shard}")
                per_shard.append(_stats(left, right))
            if per_shard:
                merged = {key: float(np.average([row[key] for row in per_shard], weights=[row["rows"] for row in per_shard])) for key in per_shard[0]}
                merged.update({"axis": axis, "left": left_id, "right": right_id, "representation": args.representation, "shards": len(per_shard)})
                observations.append(merged)
                shard_counts[f"{axis}:{left_id}:{right_id}"] = len(per_shard)
        anchor_id = anchor_ids.get(axis)
        neutral_id = prompt_ids[1] if len(prompt_ids) == 3 else None
        if args.anchor_root and anchor_id and neutral_id:
            per_shard = []
            for shard in range(args.shards):
                left_leaf = _find_prompt(args.root, shard, neutral_id)
                right_leaf = _find_prompt(args.anchor_root, shard, anchor_id)
                if left_leaf is None or right_leaf is None:
                    continue
                left, left_rows = _load(left_leaf, args.representation)
                right, right_rows = _load(right_leaf, args.representation)
                if not np.array_equal(left_rows, right_rows):
                    raise ValueError(f"row_index mismatch for anchor axis={axis}, shard={shard}")
                per_shard.append(_stats(left, right))
            if per_shard:
                merged = {key: float(np.average([row[key] for row in per_shard], weights=[row["rows"] for row in per_shard])) for key in per_shard[0]}
                merged.update({"axis": axis, "left": neutral_id, "right": anchor_id, "comparison": "neutral_vs_anchor", "representation": args.representation, "shards": len(per_shard)})
                observations.append(merged)
                shard_counts[f"{axis}:neutral_vs_anchor"] = len(per_shard)
    result = {
        "format_version": "aligned_embedding_similarity_audit_v1",
        "representation": args.representation,
        "comparisons": observations,
        "shard_counts": shard_counts,
        "note": "Raw cosine is descriptive; predictive comparisons use leakage-safe rolling evaluation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"comparisons": len(observations), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
