"""Create one independent matrix-assembly task per prompt configuration."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", required=True, choices=("sina", "cninfo"))
    p.add_argument("--shards", type=int, required=True)
    p.add_argument(
        "--direction-only",
        action="store_true",
        help="Build only each prompt's own direction-word span.",
    )
    args = p.parse_args()
    rows = []
    direction_spans = {
        "profit": "profit_span",
        "excess_return": "excess_span",
        "return": "plain_return_span",
        "loss": "loss_span",
    }
    for prompt in ("profit", "excess_return", "return", "loss"):
        for model in ("roberta", "bge_m3"):
            for variant in ("short", "masked_short"):
                representations = (
                    (direction_spans[prompt],)
                    if args.direction_only
                    else ("prompt_mean", "full_mean", "return_span", "stock_span")
                )
                for representation in representations:
                    rows.append({
                        "dataset": args.dataset, "prompt": prompt, "model": model,
                        "variant": variant, "representation": representation,
                        "shards": args.shards,
                    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"wrote {len(rows)} tasks to {args.output}")


if __name__ == "__main__":
    main()
