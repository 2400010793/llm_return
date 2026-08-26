"""Build the Sina direction-token soft-cluster fold manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SPANS = {
    "profit": "profit_span",
    "excess_return": "excess_span",
    "return": "plain_return_span",
    "loss": "loss_span",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("sina", "cninfo"), default="sina")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    rows = []
    for prompt, span in SPANS.items():
        for model in ("roberta", "bge_m3"):
            for variant in ("short", "masked_short"):
                base = args.root / "matrices" / args.dataset / model / prompt / variant / span
                matrix, metadata = base / "matrix.npy", base / "metadata.parquet"
                if not args.allow_missing and (not matrix.is_file() or not metadata.is_file()):
                    raise FileNotFoundError(f"missing direction-token matrix: {base}")
                for year in range(2018, 2027):
                    rows.append({
                        "task_id": len(rows), "prompt": prompt, "span": span,
                        "model": model, "variant": variant, "test_year": year,
                        "panel": str(args.panel), "matrix": str(matrix),
                        "metadata": str(metadata),
                    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"tasks={len(rows)} configs={len(rows) // 9}")


if __name__ == "__main__":
    main()
