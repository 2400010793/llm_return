"""Summarize validation-only pooled regression screens and lock candidates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.artifacts import atomic_json


DEFAULT_MANIFESTS = (
    Path("configs/generated/pooled_regression_screen_roberta_masked.tsv"),
    Path("configs/generated/pooled_regression_screen_bge_m3_masked.tsv"),
)


def selection_key(row: dict[str, Any]) -> tuple[float, float, float]:
    def finite(name: str, fallback: float) -> float:
        value = float(row.get(name, float("nan")))
        return value if np.isfinite(value) else fallback
    return (
        finite("rank_ic_mean", -np.inf),
        finite("oos_r2_vs_historical_mean", -np.inf),
        -finite("mse", np.inf),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="*", type=Path, default=list(DEFAULT_MANIFESTS))
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--output-prefix", type=Path,
        default=Path("reports/regression/pooled_embeddings/stock_day/validation_screening_summary"),
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for manifest_path in args.manifests:
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            tasks = list(csv.DictReader(handle, delimiter="\t"))
        for task in tasks:
            output = Path(task["output"])
            if not output.is_file():
                missing.append({"manifest": str(manifest_path), "task_id": int(task["task_id"]), "output": str(output)})
                continue
            report = json.loads(output.read_text(encoding="utf-8"))
            result = report["results"][0]
            metrics = result["validation_metrics"]
            rows.append({
                "manifest": str(manifest_path), "task_id": int(task["task_id"]),
                "output": str(output), "model": task["model"],
                "variant": task["variant"], "feature": task["feature"],
                "regressor": task["regressor"], "reducer": task["reducer"],
                "components": int(task["components"]),
                **{key: float(value) for key, value in metrics.items()},
            })
    if missing and not args.allow_incomplete:
        raise SystemExit(f"screening is incomplete: {len(missing)} tasks missing")
    if not rows:
        raise SystemExit("no completed regression screens")

    ranked = sorted(rows, key=selection_key, reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["validation_rank"] = rank
    selected_by_model = {}
    for model in sorted({row["model"] for row in rows}):
        selected_by_model[model] = max(
            (row for row in rows if row["model"] == model), key=selection_key,
        )
    report = {
        "format_version": "pooled_regression_validation_screening_summary_v1",
        "selection_rule": [
            "maximize stock-day validation rank_ic_mean",
            "tie-break maximize oos_r2_vs_historical_mean",
            "tie-break minimize mse",
        ],
        "uses_test_metrics": False,
        "complete": not missing,
        "completed_tasks": len(rows), "missing_tasks": missing,
        "selected_overall": ranked[0],
        "selected_by_model": selected_by_model,
        "ranking": ranked,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_prefix.with_suffix(".json"), report)
    pd.DataFrame(ranked).to_csv(args.output_prefix.with_suffix(".csv"), index=False)
    print(json.dumps({
        "complete": not missing, "completed": len(rows), "missing": len(missing),
        "selected": ranked[0],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
