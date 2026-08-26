"""Build deterministic cache and rolling-fold manifests for minimal prompt v2."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd


MODELS = ("roberta", "bge_m3")
LENGTHS = ("short", "long")
TARGETS = ("event_return_3d", "next_day_return")
YEARS = tuple(range(2018, 2027))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument(
        "--prompt-length", choices=LENGTHS, action="append", dest="prompt_lengths",
        help="Prompt length to include; repeat as needed (default: short and long)",
    )
    args = parser.parse_args()
    lengths = tuple(dict.fromkeys(args.prompt_lengths or LENGTHS))
    cache = [
        {"task_id": index, "model": model, "prompt_length": length}
        for index, (model, length) in enumerate(product(MODELS, lengths))
    ]
    folds = []
    for index, (model, length, masked, target, year) in enumerate(
        product(MODELS, lengths, (False, True), TARGETS, YEARS)
    ):
        variant = f"masked_{length}" if masked else length
        folds.append({
            "task_id": index, "model": model, "prompt_length": length,
            "variant": variant, "target": target, "test_year": year,
        })
    args.cache_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cache).to_csv(args.cache_manifest, sep="\t", index=False)
    pd.DataFrame(folds).to_csv(args.fold_manifest, sep="\t", index=False)
    print(
        f"cache_tasks={len(cache)} fold_tasks={len(folds)} "
        f"prompt_lengths={','.join(lengths)}"
    )


if __name__ == "__main__":
    main()
