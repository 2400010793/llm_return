"""Build aligned three-point factors directly from prompt token embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.aligned_prompts import three_point_factors


def _leaf(root: Path, shard: int, prompt_id: str) -> Path | None:
    candidates = sorted((root / f"shard-{shard}").glob(f"group-*/{prompt_id}/masked_short"))
    if len(candidates) != 1 or not (candidates[0].parents[1] / "COMPLETED").exists():
        return None
    return candidates[0]


def _load_prompt(root: Path, prompt_id: str, shards: int) -> tuple[np.ndarray, np.ndarray, dict]:
    values: list[np.ndarray] = []
    row_ids: list[np.ndarray] = []
    audit = None
    for shard in range(shards):
        leaf = _leaf(root, shard, prompt_id)
        if leaf is None:
            continue
        hidden = np.load(leaf / "prompt_token_embeddings.npy", mmap_mode="r")
        metadata = pd.read_json(leaf / "metadata.jsonl", lines=True)
        spec = json.loads((leaf / "prompt_spec.json").read_text(encoding="utf-8"))
        audit = spec["token_audit"]
        values.append(np.asarray(hidden[:, audit["target_token_indices"], :].mean(axis=1), dtype=np.float32))
        row_ids.append(metadata["row_index"].to_numpy(np.int64))
    if not values or audit is None:
        raise FileNotFoundError(f"no completed shards for {prompt_id}")
    matrix = np.concatenate(values, axis=0)
    rows = np.concatenate(row_ids, axis=0)
    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    matrix = matrix[order]
    if len(np.unique(rows)) != len(rows) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid token span matrix for {prompt_id}")
    return matrix, rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--axes-json", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    axes = json.loads(args.axes_json.read_text(encoding="utf-8"))
    manifests = []
    for axis, prompt_ids in axes.items():
        if len(prompt_ids) != 3:
            raise ValueError(f"axis {axis} must have low, neutral, high prompt IDs")
        matrices = []
        row_reference = None
        audits = {}
        for prompt_id in prompt_ids:
            matrix, rows, audit = _load_prompt(args.root, prompt_id, args.shards)
            if row_reference is None:
                row_reference = rows
            elif not np.array_equal(row_reference, rows):
                raise ValueError(f"row_index mismatch in axis {axis}")
            matrices.append(matrix)
            audits[prompt_id] = audit
        factors = three_point_factors(matrices[0], matrices[1], matrices[2])
        axis_root = args.output_root / "roberta" / axis / "masked_short" / "target_span_mean"
        axis_root.mkdir(parents=True, exist_ok=True)
        for name, matrix in factors.items():
            np.save(axis_root / f"{name}.npy", matrix)
        pd.DataFrame({"row_index": row_reference}).to_parquet(axis_root / "metadata.parquet", index=False)
        manifest = {
            "format_version": "aligned_three_point_token_span_factors_v1",
            "axis": axis,
            "representation": "target_span_mean",
            "prompt_ids": prompt_ids,
            "rows": int(len(row_reference)),
            "dimensions": {name: list(matrix.shape) for name, matrix in factors.items()},
            "finite": all(bool(np.isfinite(matrix).all()) for matrix in factors.values()),
            "token_audits": audits,
        }
        (axis_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        manifests.append(manifest)
    summary = {"format_version": "aligned_three_point_token_span_factor_summary_v1", "axes": manifests}
    summary_path = args.output_root / "roberta" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"axes": list(axes), "output_root": str(args.output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
