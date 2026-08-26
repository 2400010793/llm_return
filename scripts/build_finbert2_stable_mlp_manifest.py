"""Build the focused FinBERT2 stable-MLP classification manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


CANDIDATES = (
    ("short", "body_mean"),
    ("plain", "full_mean"),
    ("masked_short", "body_mean"),
)
SEEDS = (42, 43, 44, 45, 46)
SCHEDULES = ("constant", "cosine")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("reports/classification/pooled_embeddings/finbert2_stable_mlp"),
    )
    args = parser.parse_args()

    rows = []
    for variant, feature in CANDIDATES:
        for schedule in SCHEDULES:
            for seed in SEEDS:
                stem = (
                    f"finbert2_base_{variant}_{feature}_stable_simple_mlp_"
                    f"accuracy_burnin10_{schedule}_seed{seed}"
                )
                rows.append({
                    "task_id": len(rows),
                    "model": "finbert2_base",
                    "variant": variant,
                    "feature": feature,
                    "seed": seed,
                    "learning_rate_schedule": schedule,
                    "early_stopping_min_epoch": 10,
                    "output": str(args.report_root / f"{stem}.json"),
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} tasks to {args.output}")


if __name__ == "__main__":
    main()
