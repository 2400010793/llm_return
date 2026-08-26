"""Build three-point vector/scalar factors from aligned prompt matrices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.aligned_prompts import LEVELS, load_aligned_prompt_config, three_point_factors


def _load_matrix(root: Path, model: str, prompt_id: str, representation: str):
    leaf = root / model / prompt_id / "masked_short" / representation
    matrix = np.load(leaf / "matrix.npy", mmap_mode="r")
    metadata = pd.read_parquet(leaf / "metadata.parquet")
    if len(metadata) != len(matrix):
        raise ValueError(f"matrix/metadata mismatch: {leaf}")
    rows = metadata["row_index"].to_numpy(dtype=np.int64)
    if len(np.unique(rows)) != len(rows):
        raise ValueError(f"duplicate row_index: {leaf}")
    return np.asarray(matrix, dtype=np.float32), rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--model", choices=("roberta",), default="roberta")
    parser.add_argument("--representation", choices=("body_mean", "full_mean", "prompt_mean"), default="body_mean")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_aligned_prompt_config(args.config)
    outputs = []
    for axis, spec in config["axes"].items():
        matrices = {}
        row_reference = None
        for level in LEVELS:
            prompt_id = spec["levels"][level]["prompt_id"]
            matrix, rows = _load_matrix(args.matrix_root, args.model, prompt_id, args.representation)
            if row_reference is None:
                row_reference = rows
            elif not np.array_equal(rows, row_reference):
                raise ValueError(f"row_index mismatch inside axis {axis}")
            matrices[level] = matrix
        factors = three_point_factors(matrices["low"], matrices["neutral"], matrices["high"])
        axis_root = args.output_root / args.model / axis / "masked_short" / args.representation
        axis_root.mkdir(parents=True, exist_ok=True)
        for name, matrix in factors.items():
            np.save(axis_root / f"{name}.npy", matrix)
        pd.DataFrame({"row_index": row_reference}).to_parquet(axis_root / "metadata.parquet", index=False)
        manifest = {
            "format_version": "aligned_three_point_factors_v1",
            "model": args.model,
            "axis": axis,
            "variant": "masked_short",
            "representation": args.representation,
            "levels": {level: spec["levels"][level]["prompt_id"] for level in LEVELS},
            "rows": int(len(row_reference)),
            "dimensions": {name: list(matrix.shape) for name, matrix in factors.items()},
            "finite": all(bool(np.isfinite(matrix).all()) for matrix in factors.values()),
        }
        (axis_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs.append(manifest)
    summary = {"format_version": "aligned_three_point_factor_summary_v1", "axes": outputs}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / args.model / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (args.output_root / args.model / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"axes": [item["axis"] for item in outputs]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
