"""Summarize five-seed title/body MLP classification stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def weighted(rows: list[dict], key: str) -> float:
    counts = np.asarray([row["n"] for row in rows], dtype=float)
    values = np.asarray([row[key] for row in rows], dtype=float)
    return float(np.average(values, weights=counts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.root.glob("*.json"))
    paths.extend(sorted((args.root / "seed_stability").glob("*.json")))
    per_seed: list[dict] = []
    per_year: list[dict] = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = report.get("results", [])
        if len(rows) != 9:
            continue
        years = [int(row["test_year"]) for row in rows]
        if years != list(range(2018, 2027)):
            raise ValueError(f"unexpected rolling years in {path}: {years}")
        metric = str(rows[0]["validation_selection_metric"])
        if metric not in {"accuracy", "balanced_accuracy"}:
            continue
        seed = int(report["design"]["seed"])
        overall_accuracy = weighted(rows, "accuracy")
        overall_majority = weighted(rows, "majority_accuracy")
        epochs = np.asarray(
            [row["best_params"]["selected_epoch"] for row in rows], dtype=float
        )
        per_seed.append({
            "selection_metric": metric,
            "seed": seed,
            "weighted_accuracy": overall_accuracy,
            "weighted_majority_accuracy": overall_majority,
            "accuracy_lift_pp": 100.0 * (overall_accuracy - overall_majority),
            "equal_year_accuracy": float(np.mean([row["accuracy"] for row in rows])),
            "years_above_majority": int(sum(
                row["accuracy"] > row["majority_accuracy"] for row in rows
            )),
            "selected_epoch_mean": float(epochs.mean()),
            "selected_epoch_median": float(np.median(epochs)),
            "source": str(path),
        })
        for row in rows:
            per_year.append({
                "selection_metric": metric,
                "seed": seed,
                "test_year": int(row["test_year"]),
                "accuracy": float(row["accuracy"]),
                "majority_accuracy": float(row["majority_accuracy"]),
                "accuracy_lift_pp": 100.0 * (
                    row["accuracy"] - row["majority_accuracy"]
                ),
                "selected_epoch": int(row["best_params"]["selected_epoch"]),
                "predicted_positive_rate": float(row["predicted_positive_rate"]),
            })

    seed_frame = pd.DataFrame(per_seed).sort_values(
        ["selection_metric", "seed"]
    )
    year_frame = pd.DataFrame(per_year).sort_values(
        ["selection_metric", "test_year", "seed"]
    )
    expected = {
        metric: list(group["seed"])
        for metric, group in seed_frame.groupby("selection_metric")
    }
    for metric in ("accuracy", "balanced_accuracy"):
        if expected.get(metric) != [42, 43, 44, 45, 46]:
            raise ValueError(
                f"{metric} requires seeds 42-46, found {expected.get(metric)}"
            )

    summaries = []
    for metric, group in seed_frame.groupby("selection_metric", sort=True):
        values = group["weighted_accuracy"]
        summaries.append({
            "selection_metric": metric,
            "seed_count": int(len(group)),
            "weighted_accuracy_mean": float(values.mean()),
            "weighted_accuracy_std": float(values.std(ddof=1)),
            "weighted_accuracy_min": float(values.min()),
            "weighted_accuracy_max": float(values.max()),
            "accuracy_lift_pp_mean": float(group["accuracy_lift_pp"].mean()),
            "years_above_majority_mean": float(group["years_above_majority"].mean()),
            "years_above_majority_min": int(group["years_above_majority"].min()),
            "years_above_majority_max": int(group["years_above_majority"].max()),
        })
    summary_frame = pd.DataFrame(summaries)
    year_summary = year_frame.groupby(
        ["selection_metric", "test_year"], as_index=False
    ).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        accuracy_min=("accuracy", "min"),
        accuracy_max=("accuracy", "max"),
        majority_accuracy=("majority_accuracy", "first"),
        accuracy_lift_pp_mean=("accuracy_lift_pp", "mean"),
        selected_epoch_mean=("selected_epoch", "mean"),
        selected_epoch_std=("selected_epoch", "std"),
        selected_epoch_min=("selected_epoch", "min"),
        selected_epoch_max=("selected_epoch", "max"),
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    seed_frame.to_csv(args.output_prefix.with_suffix(".per_seed.csv"), index=False)
    summary_frame.to_csv(args.output_prefix.with_suffix(".summary.csv"), index=False)
    year_summary.to_csv(args.output_prefix.with_suffix(".per_year.csv"), index=False)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps({
            "root": str(args.root),
            "seed_summary": summaries,
            "per_seed": per_seed,
            "per_year": year_summary.to_dict(orient="records"),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
