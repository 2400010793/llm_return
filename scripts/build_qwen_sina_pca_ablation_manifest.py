"""Build the Qwen Sina Ridge PCA ablation manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    reducers = [("none", 0), *(("pca", value) for value in (32, 64, 128, 256))]
    for variant in ("short", "masked_short"):
        for representation in ("article_mean", "prompt_mean", "return_token"):
            base = (
                args.regression_root / "matrices" / "sina" / "qwen3_embedding_8b"
                / "return" / variant / representation
            )
            for reducer, components in reducers:
                name = reducer if reducer == "none" else f"pca{components}"
                rows.append({
                    "task_id": len(rows), "variant": variant,
                    "representation": representation, "reducer": reducer,
                    "components": components, "panel": str(args.panel),
                    "matrix": str(base / "matrix.npy"),
                    "metadata": str(base / "metadata.parquet"),
                    "output": str(
                        args.regression_root / "pca_ablation_v1" / "results"
                        / f"{variant}_{representation}_{name}.json"
                    ),
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"tasks={len(rows)}")


if __name__ == "__main__":
    main()
