"""Compare prompt_token_embeddings at aligned positions and target spans."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def _leaf(root: Path, shard: int, prompt_id: str) -> Path | None:
    candidates = sorted((root / f"shard-{shard}").glob(f"group-*/{prompt_id}/masked_short"))
    if len(candidates) != 1 or not (candidates[0].parents[1] / "COMPLETED").exists():
        return None
    return candidates[0]


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(np.sum(left * right, axis=1), denom, out=np.zeros(len(left)), where=denom > 0)


def _stats(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    values = _cosine(left, right)
    distance = np.linalg.norm(left - right, axis=1)
    return {
        "rows": int(len(values)),
        "cosine_mean": float(values.mean()),
        "cosine_median": float(np.median(values)),
        "cosine_p05": float(np.quantile(values, 0.05)),
        "cosine_p95": float(np.quantile(values, 0.95)),
        "l2_mean": float(distance.mean()),
    }


def _load(leaf: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    tokens = np.load(leaf / "prompt_token_embeddings.npy", mmap_mode="r")
    metadata = pd.read_json(leaf / "metadata.jsonl", lines=True)
    spec = json.loads((leaf / "prompt_spec.json").read_text(encoding="utf-8"))
    return np.asarray(tokens, dtype=np.float32), metadata["row_index"].to_numpy(np.int64), spec["token_audit"]


def _pair(left_root: Path, right_root: Path, shards: int, left_id: str, right_id: str, *, target_span: bool = False) -> dict:
    position_values: list[list[float]] | None = None
    target_values: list[dict[str, float]] = []
    prompt_mean_values: list[dict[str, float]] = []
    left_start = right_start = None
    for shard in range(shards):
        left_leaf, right_leaf = _leaf(left_root, shard, left_id), _leaf(right_root, shard, right_id)
        if left_leaf is None or right_leaf is None:
            continue
        left, left_rows, left_audit = _load(left_leaf)
        right, right_rows, right_audit = _load(right_leaf)
        if not np.array_equal(left_rows, right_rows):
            raise ValueError(f"row_index mismatch for {left_id}/{right_id}, shard={shard}")
        if left_start is None:
            left_start = left_audit["target_start_token"]
            right_start = right_audit["target_start_token"]
        width = min(left.shape[1], right.shape[1])
        if position_values is None:
            position_values = [[] for _ in range(width)]
        for pos in range(width):
            position_values[pos].extend(_cosine(left[:, pos, :], right[:, pos, :]).tolist())
        if target_span:
            left_target = left[:, left_audit["target_token_indices"], :].mean(axis=1)
            right_target = right[:, right_audit["target_token_indices"], :].mean(axis=1)
            target_values.append(_stats(left_target, right_target))
        prompt_mean_values.append(_stats(left.mean(axis=1), right.mean(axis=1)))
    if not prompt_mean_values:
        return {"shards": 0}
    position_summary = []
    for index, values in enumerate(position_values or []):
        array = np.asarray(values, dtype=float)
        position_summary.append({"position": index, "cosine_mean": float(array.mean()), "cosine_median": float(np.median(array))})
    result = {
        "shards": len(prompt_mean_values),
        "aligned_position_cosine": position_summary,
        "prompt_token_mean": {
            key: float(np.average([item[key] for item in prompt_mean_values], weights=[item["rows"] for item in prompt_mean_values]))
            for key in prompt_mean_values[0] if key != "rows"
        },
        "target_start_tokens": {"left": left_start, "right": right_start},
    }
    if target_values:
        result["target_span_mean"] = {
            key: float(np.average([item[key] for item in target_values], weights=[item["rows"] for item in target_values]))
            for key in target_values[0] if key != "rows"
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--axes-json", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--anchor-root", type=Path, default=None)
    parser.add_argument("--anchor-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    axes = json.loads(args.axes_json.read_text(encoding="utf-8"))
    anchors = json.loads(args.anchor_json.read_text(encoding="utf-8")) if args.anchor_json else {}
    comparisons = []
    for axis, prompt_ids in axes.items():
        for left_id, right_id in combinations(prompt_ids, 2):
            comparisons.append({"axis": axis, "left": left_id, "right": right_id, **_pair(args.root, args.root, args.shards, left_id, right_id, target_span=True)})
        if args.anchor_root and axis in anchors and len(prompt_ids) == 3:
            comparisons.append({"axis": axis, "left": prompt_ids[1], "right": anchors[axis], "comparison": "neutral_vs_anchor", **_pair(args.root, args.anchor_root, args.shards, prompt_ids[1], anchors[axis], target_span=True)})
    result = {"format_version": "prompt_token_embedding_similarity_v1", "comparisons": comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"comparisons": len(comparisons), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
