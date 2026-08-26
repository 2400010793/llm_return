"""Summarize completed parameter-matched Prompt-token gate reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.artifacts import atomic_json, completed_bundle_matches
from src.evaluation.prediction_metrics import daily_rank_ic, return_prediction_metrics


def report_row(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    bundle = Path(report["artifacts"])
    if not completed_bundle_matches(bundle, report["experiment_id"]):
        raise ValueError(f"artifact bundle does not match report: {path}")
    result = report["result"]
    metrics = result["validation_metrics"]
    weights = result["validation_token_weight_summary"]
    row = {
        "report": str(path),
        "experiment_id": report["experiment_id"],
        "task": result["task"], "gate_mode": result["gate_mode"],
        "seed": int(report["spec"]["optimization"]["seed"]),
        "best_epoch": int(result["best_epoch"]),
        "rank_ic_mean": metrics.get("rank_ic_mean"),
        "rank_ic_std": metrics.get("rank_ic_std"),
        "rank_ic_positive_rate": metrics.get("rank_ic_positive_rate"),
        "correlation": metrics.get("correlation"),
        "oos_r2_vs_historical_mean": metrics.get("oos_r2_vs_historical_mean"),
        "direction_accuracy": metrics.get("direction_accuracy"),
        "auc": metrics.get("auc"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "mcc": metrics.get("mcc"),
        "normalized_entropy_mean": weights["normalized_entropy_mean"],
        "mean_position_std_across_announcements": weights[
            "mean_position_std_across_announcements"
        ],
    }
    if result["task"] == "regression":
        stock_day = pd.read_parquet(
            bundle / "validation" / "stock_day_predictions.parquet"
        )
        trimmed = stock_day.loc[stock_day["actual_return"].abs() <= 0.2].copy()
        robust = return_prediction_metrics(
            trimmed["actual_return"].to_numpy(),
            trimmed["prediction"].to_numpy(),
            historical_mean=float(trimmed["actual_return"].mean()),
        )
        robust.update(daily_rank_ic(trimmed, min_stocks=5))
        row.update({
            "abs_return_le_20pct_n": int(len(trimmed)),
            "abs_return_le_20pct_correlation": robust["correlation"],
            "abs_return_le_20pct_oos_r2_vs_historical_mean": robust[
                "oos_r2_vs_historical_mean"
            ],
            "abs_return_le_20pct_rank_ic_mean": robust["rank_ic_mean"],
        })
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [report_row(path) for path in args.reports]
    identities = {(row["task"], row["gate_mode"], row["seed"]) for row in rows}
    if len(identities) != len(rows):
        raise ValueError("duplicate task/gate_mode reports")
    task_values = {row["task"] for row in rows}
    if len(task_values) != 1:
        raise ValueError("one comparison cannot mix tasks")
    task = task_values.pop()
    primary = "rank_ic_mean" if task == "regression" else "auc"
    ranked = sorted(rows, key=lambda row: float(row[primary]), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["validation_rank"] = rank
    frame = pd.DataFrame(rows)
    aggregate = []
    for gate_mode, group in frame.groupby("gate_mode", sort=True):
        values = pd.to_numeric(group[primary], errors="raise")
        aggregate.append({
            "gate_mode": gate_mode, "n_seeds": int(len(group)),
            f"{primary}_mean": float(values.mean()),
            f"{primary}_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            f"{primary}_min": float(values.min()),
            f"{primary}_max": float(values.max()),
        })
    aggregate.sort(key=lambda row: row[f"{primary}_mean"], reverse=True)
    by_identity = {(row["seed"], row["gate_mode"]): row for row in rows}
    paired_deltas = []
    for seed in sorted({row["seed"] for row in rows}):
        if all((seed, mode) in by_identity for mode in ("uniform", "static", "dynamic")):
            dynamic = float(by_identity[(seed, "dynamic")][primary])
            paired_deltas.append({
                "seed": seed,
                "dynamic_minus_uniform": dynamic - float(by_identity[(seed, "uniform")][primary]),
                "dynamic_minus_static": dynamic - float(by_identity[(seed, "static")][primary]),
            })
    output = {
        "format_version": "dynamic_prompt_gate_comparison_v1",
        "selection_scope": "validation_only",
        "task": task, "primary_metric": primary,
        "selected_individual_run": ranked[0],
        "selected_gate_by_seed_mean": aggregate[0],
        "aggregate_by_gate": aggregate,
        "paired_deltas": paired_deltas,
        "ranking": ranked,
    }
    atomic_json(args.output, output)
    pd.DataFrame(ranked).to_csv(args.output.with_suffix(".csv"), index=False)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
