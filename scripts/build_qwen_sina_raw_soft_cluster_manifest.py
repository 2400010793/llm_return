"""Build raw 4096-dimensional Qwen return-token soft-cluster manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--config-manifest", type=Path, required=True)
    args = parser.parse_args()
    folds, configs = [], []
    for variant in ("short", "masked_short"):
        base = (
            args.regression_root / "matrices" / "sina" / "qwen3_embedding_8b"
            / "return" / variant / "return_token"
        )
        configs.append({
            "task_id": len(configs), "prompt": "return", "model": "qwen3_embedding_8b",
            "variant": variant, "representation": "return_token", "projection_mode": "raw",
        })
        for year in range(2018, 2027):
            folds.append({
                "task_id": len(folds), "prompt": "return", "span": "return_token",
                "model": "qwen3_embedding_8b", "variant": variant, "test_year": year,
                "panel": str(args.panel), "matrix": str(base / "matrix.npy"),
                "metadata": str(base / "metadata.parquet"),
                "representation": "return_token", "projection_mode": "raw",
            })
    args.fold_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(folds).to_csv(args.fold_manifest, sep="\t", index=False)
    pd.DataFrame(configs).to_csv(args.config_manifest, sep="\t", index=False)
    print(f"folds={len(folds)} configs={len(configs)}")


if __name__ == "__main__":
    main()
