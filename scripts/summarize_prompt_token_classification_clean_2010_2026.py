"""Validate and summarize clean 2010--2026 rolling prompt-token classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=("roberta", "bge_m3"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.root.glob(f"{args.model}_*.json"))
    if len(paths) != 24:
        raise ValueError(f"expected 24 {args.model} reports; found {len(paths)}")
    rows: list[dict[str, object]] = []
    years: list[dict[str, object]] = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        folds = report.get("results", [])
        if [int(item["test_year"]) for item in folds] != list(range(2018, 2027)):
            raise ValueError(f"rolling year coverage mismatch: {path}")
        prediction = Path(report["predictions"])
        if not prediction.is_file():
            raise ValueError(f"prediction file is missing: {prediction}")
        accuracies = [float(item["accuracy"]) for item in folds]
        row = {
            "model": report["model"], "variant": report["variant"],
            "feature": report["feature"], "classifier": report["classifier"],
            "reducer": report["reducer"], "components": report["components"],
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "min_accuracy": min(accuracies), "max_accuracy": max(accuracies),
            "report": str(path),
        }
        preprocessing = report.get("preprocessing") or {}
        gate = preprocessing.get("token_gate") or {}
        row.update({
            "gate_method": gate.get("method", "none"),
            "gate_keep": gate.get("keep_tokens"),
        })
        rows.append(row)
        for item in folds:
            years.append({
                "report": str(path), "model": report["model"],
                "variant": report["variant"], "classifier": report["classifier"],
                "gate_method": row["gate_method"], "gate_keep": row["gate_keep"],
                "test_year": int(item["test_year"]),
                "accuracy": float(item["accuracy"]),
            })
    ranking = sorted(rows, key=lambda item: float(item["mean_accuracy"]), reverse=True)
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    output = {
        "format_version": "clean_prompt_token_rolling_classification_summary_v1",
        "model": args.model, "reports": len(paths), "folds_per_report": 9,
        "primary_metric": "accuracy", "ranking": ranking,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(ranking).to_csv(args.output.with_suffix(".csv"), index=False)
    pd.DataFrame(years).to_csv(args.output.with_name(args.output.stem + "_by_year.csv"), index=False)
    print(json.dumps({
        "output": str(args.output), "reports": len(paths), "top": ranking[:5],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
