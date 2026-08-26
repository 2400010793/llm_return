"""Build the BGE-M3 soft return-family rolling manifest."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        {
            "task_id": index, "feature_mode": mode,
            "target": target, "test_year": year,
        }
        for index, (mode, target, year) in enumerate(product(
            ("unmasked", "masked", "fusion"),
            ("event_return_3d", "next_day_return"), range(2018, 2027),
        ))
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"tasks={len(rows)}")


if __name__ == "__main__":
    main()
