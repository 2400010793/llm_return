"""Aggregate cross-model rolling predictions and prepare simple_states factors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected", type=int, default=648)
    args = p.parse_args()
    metrics = sorted(args.root.glob("**/metrics.json"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in metrics]
    if len(reports) != args.expected:
        raise ValueError(f"expected {args.expected} completed jobs, found {len(reports)}")
    rows, predictions = [], []
    for path, report in zip(metrics, reports):
        rows.append({k: v for k, v in report.items() if k != "rows"} | report["rows"])
        part = pd.read_parquet(path.parent / "stock_day_predictions.parquet")
        part["variant"] = report["variant"]
        part["target"] = report["target"]
        part["semantics"] = ",".join(report["semantics"])
        part["components"] = report["components_per_model"]
        part["cluster"] = report["cluster"]
        predictions.append(part)
    metrics_frame = pd.DataFrame(rows)
    all_predictions = pd.concat(predictions, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(args.output / "metrics.csv", index=False)
    all_predictions.to_parquet(args.output / "stock_day_predictions_all.parquet", index=False)
    factors = []
    for keys, part in all_predictions[all_predictions["target"].eq("next_day_return")].groupby(
        ["variant", "semantics", "components", "cluster"], sort=True,
    ):
        name = "xmodel_" + "_".join(str(value).replace(",", "_") for value in keys)
        path = args.output / "simple_states" / "predictions" / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        part[["stock_id", "entry_date", "prediction"]].to_parquet(path, index=False)
        factors.append({"factor_id": name, "predictions": str(path.resolve()), "prediction_column": "prediction"})
    (args.output / "simple_states_manifest.json").write_text(json.dumps(factors, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{
        "task_id": i, "predictions": row["predictions"],
        "factor_id": row["factor_id"], "prediction_column": row["prediction_column"],
    } for i, row in enumerate(factors)]).to_csv(
        args.output / "simple_states_manifest.tsv", sep="\t", index=False,
    )
    (args.output / "summary.json").write_text(json.dumps({"format_version": "cross_model_span_summary_v1", "folds": len(reports), "factors": len(factors)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"folds": len(reports), "factors": len(factors), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
