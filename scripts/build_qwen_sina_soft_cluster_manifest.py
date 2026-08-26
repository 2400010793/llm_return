"""Build Qwen Sina soft-cluster fold and evaluation manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPRESENTATIONS = ("article_mean", "prompt_mean", "return_token")
VARIANTS = ("short", "masked_short")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--config-manifest", type=Path, required=True)
    args = parser.parse_args()

    folds: list[dict[str, object]] = []
    configs: list[dict[str, object]] = []
    for variant in VARIANTS:
        for representation in REPRESENTATIONS:
            base = (
                args.regression_root / "matrices" / "sina" / "qwen3_embedding_8b"
                / "return" / variant / representation
            )
            matrix = base / "matrix.npy"
            metadata = base / "metadata.parquet"
            if not matrix.is_file() or not metadata.is_file():
                raise FileNotFoundError(f"missing Qwen matrix artifacts: {base}")
            configs.append({
                "task_id": len(configs), "prompt": "return",
                "model": "qwen3_embedding_8b", "variant": variant,
                "representation": representation,
            })
            for test_year in range(2018, 2027):
                folds.append({
                    "task_id": len(folds), "prompt": "return", "span": representation,
                    "model": "qwen3_embedding_8b", "variant": variant,
                    "test_year": test_year, "panel": str(args.panel),
                    "matrix": str(matrix), "metadata": str(metadata),
                    "representation": representation,
                })

    args.fold_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(folds).to_csv(args.fold_manifest, sep="\t", index=False)
    pd.DataFrame(configs).to_csv(args.config_manifest, sep="\t", index=False)
    print(f"folds={len(folds)} configs={len(configs)}")


if __name__ == "__main__":
    main()
