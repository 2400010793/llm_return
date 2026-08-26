"""Aggregate completed return-token Logistic folds into reusable predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-folds", type=int, default=9)
    parser.add_argument("--factor-prefix", default="return_span_logistic")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    metrics = sorted(args.fold_root.glob("*/*/*/metrics.json"))
    if args.strict and len(metrics) != args.expected_folds:
        raise ValueError(
            f"expected {args.expected_folds} folds; found {len(metrics)}"
        )
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in metrics]
    yearly = pd.DataFrame([{
        "model": report.get("model", "bge_m3"),
        "prompt_length": report.get("prompt_length", "short"),
        "feature_mode": report["feature_mode"],
        "target": report["target"],
        "test_year": report["test_year"],
        **report["selected"],
        **report["test_metrics"],
    } for report in reports])
    news_parts = []
    stock_day_parts = []
    for path, report in zip(metrics, reports):
        identifiers = {
            "model": report.get("model", "bge_m3"),
            "prompt_length": report.get("prompt_length", "short"),
            "feature_mode": report["feature_mode"],
            "target": report["target"],
            "test_year": int(report["test_year"]),
        }
        news_part = pd.read_parquet(path.parent / "news_predictions.parquet")
        stock_day_part = pd.read_parquet(path.parent / "stock_day_predictions.parquet")
        for key, value in identifiers.items():
            news_part[key] = value
            stock_day_part[key] = value
        news_parts.append(news_part)
        stock_day_parts.append(stock_day_part)
    news = pd.concat(news_parts, ignore_index=True)
    stock_days = pd.concat(stock_day_parts, ignore_index=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(args.output_root / "yearly_metrics.csv", index=False)
    news.to_parquet(args.output_root / "news_predictions_all.parquet", index=False)
    stock_days.to_parquet(args.output_root / "stock_day_predictions_all.parquet", index=False)
    overall_rows = []
    for keys, part in yearly.groupby(
        ["model", "prompt_length", "feature_mode", "target"], sort=True,
    ):
        weights = part["stock_days"].to_numpy(dtype=float)
        overall_rows.append({
            "model": keys[0], "prompt_length": keys[1],
            "feature_mode": keys[2], "target": keys[3],
            "weighted_accuracy": float(
                (part["accuracy"] * weights).sum() / weights.sum()
            ),
            "weighted_majority_accuracy": float(
                (part["majority_accuracy"] * weights).sum() / weights.sum()
            ),
            "mean_auc": float(part["auc"].mean()),
            "stock_days": int(weights.sum()),
            "years": int(part["test_year"].nunique()),
        })
    overall = pd.DataFrame(overall_rows)
    overall["accuracy_delta"] = (
        overall["weighted_accuracy"] - overall["weighted_majority_accuracy"]
    )
    overall.to_csv(args.output_root / "overall.csv", index=False)
    factor_root = args.output_root / "simple_states" / "predictions"
    result_root = args.output_root / "simple_states" / "results"
    factor_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    next_day = stock_days[stock_days["target"] == "next_day_return"]
    for keys, part in next_day.groupby(
        ["model", "prompt_length", "feature_mode"], sort=True,
    ):
        factor_id = "_".join((args.factor_prefix, *map(str, keys)))
        prediction_path = factor_root / f"{factor_id}.parquet"
        part[["stock_id", "entry_date", "probability"]].to_parquet(
            prediction_path, index=False,
        )
        manifest_rows.append({
            "task_id": len(manifest_rows),
            "source_task_id": "return_span_logistic",
            "predictions": str(prediction_path.resolve()),
            "factor_id": factor_id,
            "output_dir": str((result_root / factor_id).resolve()),
            "prediction_column": "probability",
            "duplicate_policy": "error",
        })
    pd.DataFrame(manifest_rows).to_csv(
        args.output_root / "simple_states_manifest.tsv", sep="\t", index=False,
    )
    summary = {
        "format_version": "return_span_logistic_summary_v1",
        "folds": len(metrics),
        "expected_folds": args.expected_folds,
        "complete": len(metrics) == args.expected_folds,
        "configurations": overall.to_dict(orient="records"),
        "simple_states_tasks": len(manifest_rows),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
