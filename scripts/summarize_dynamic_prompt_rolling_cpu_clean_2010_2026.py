"""Summarize CPU-only rolling clean prompt-token gate classification reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.artifacts import completed_bundle_matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=("roberta", "bge_m3"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for gate in ("uniform", "static", "dynamic"):
        for year in range(2018, 2027):
            path = args.root / f"{args.model}_masked_short_{gate}_test{year}_seed42.json"
            if not path.is_file():
                raise ValueError(f"missing rolling gate report: {path}")
            report = json.loads(path.read_text(encoding="utf-8"))
            bundle = Path(report["artifacts"])
            if not completed_bundle_matches(bundle, report["experiment_id"]):
                raise ValueError(f"artifact bundle does not match: {path}")
            result = report["result"]
            if result.get("test_years") != [year]:
                raise ValueError(f"test year mismatch in {path}")
            metrics = result["test_metrics"]
            rows.append({
                "model": args.model, "variant": "masked_short", "gate_mode": gate,
                "test_year": year, "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "auc": float(metrics["auc"]), "mcc": float(metrics["mcc"]),
                "best_epoch": int(result["best_epoch"]), "report": str(path),
            })
    frame = pd.DataFrame(rows)
    aggregate = []
    for gate, group in frame.groupby("gate_mode", sort=True):
        aggregate.append({
            "gate_mode": gate, "years": int(len(group)),
            "mean_accuracy": float(group["accuracy"].mean()),
            "min_accuracy": float(group["accuracy"].min()),
            "max_accuracy": float(group["accuracy"].max()),
            "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
            "mean_auc": float(group["auc"].mean()),
            "mean_mcc": float(group["mcc"].mean()),
        })
    aggregate.sort(key=lambda item: item["mean_accuracy"], reverse=True)
    output = {
        "format_version": "dynamic_prompt_cpu_rolling_summary_v1",
        "device": "cpu", "model": args.model, "variant": "masked_short",
        "primary_metric": "accuracy", "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    frame.to_csv(args.output.with_name(args.output.stem + "_by_year.csv"), index=False)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
