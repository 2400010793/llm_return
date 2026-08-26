"""Summarize validation-only overnight embedding/model matrices."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.artifacts import atomic_json


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def pooled_results(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows, missing = [], []
    for task in read_manifest(path):
        output = Path(task["output"])
        if not output.is_file():
            missing.append(str(output))
            continue
        report = json.loads(output.read_text(encoding="utf-8"))
        result = report["results"][0]
        metrics = result["validation_metrics"]
        rows.append({
            **{key: task[key] for key in (
                "family", "model", "variant", "feature", "regressor",
                "reducer", "components", "search_stage", "output",
            )},
            "rank_ic_mean": metrics["rank_ic_mean"],
            "rank_ic_positive_rate": metrics["rank_ic_positive_rate"],
            "correlation": metrics["correlation"],
            "oos_r2_vs_historical_mean": metrics["oos_r2_vs_historical_mean"],
            "direction_accuracy": metrics["direction_accuracy"],
            "best_params": result["best_params"],
        })
    return rows, missing


def dynamic_results(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    rows, missing = [], []
    for manifest in paths:
        for task in read_manifest(manifest):
            output = Path(task["output"])
            if not output.is_file():
                missing.append(str(output))
                continue
            report = json.loads(output.read_text(encoding="utf-8"))
            result = report["result"]
            metrics = result["validation_metrics"]
            weights = result["validation_token_weight_summary"]
            rows.append({
                **{key: task[key] for key in (
                    "model", "variant", "window", "config_tag", "gate_mode",
                    "representation_size", "gate_hidden_size", "dropout",
                    "learning_rate", "weight_decay", "output",
                )},
                "best_epoch": result["best_epoch"],
                "rank_ic_mean": metrics["rank_ic_mean"],
                "rank_ic_positive_rate": metrics["rank_ic_positive_rate"],
                "oos_r2_vs_historical_mean": metrics["oos_r2_vs_historical_mean"],
                "normalized_entropy_mean": weights["normalized_entropy_mean"],
                "mean_position_std_across_announcements": weights[
                    "mean_position_std_across_announcements"
                ],
            })
    return rows, missing


def select_dynamic_confirmation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["model"], row["variant"], row["gate_mode"], row["config_tag"], row["window"]): row
        for row in rows
    }
    selected = []
    identities = sorted({
        (row["model"], row["variant"], row["gate_mode"])
        for row in rows if row["window"] == "tune2024"
    })
    for model, variant, gate_mode in identities:
        candidates = [
            row for row in rows
            if row["model"] == model and row["variant"] == variant
            and row["gate_mode"] == gate_mode and row["window"] == "tune2024"
        ]
        tuned = max(candidates, key=lambda row: float(row["rank_ic_mean"]))
        confirmation = by_key.get((
            model, variant, gate_mode, tuned["config_tag"], "confirm2025",
        ))
        selected.append({
            "model": model, "variant": variant, "gate_mode": gate_mode,
            "selected_config_tag": tuned["config_tag"],
            "tune2024_rank_ic_mean": tuned["rank_ic_mean"],
            "confirm2025_rank_ic_mean": (
                confirmation["rank_ic_mean"] if confirmation is not None else None
            ),
            "tune_report": tuned["output"],
            "confirmation_report": confirmation["output"] if confirmation is not None else None,
        })
    return selected


def main() -> None:
    pooled, pooled_missing = pooled_results(
        Path("configs/generated/overnight_pooled_v1.tsv")
    )
    dynamic, dynamic_missing = dynamic_results([
        Path("configs/generated/overnight_dynamic_v1_a.tsv"),
        Path("configs/generated/overnight_dynamic_v1_b.tsv"),
    ])
    pooled_ranked = sorted(
        pooled, key=lambda row: float(row["rank_ic_mean"]), reverse=True
    )
    selected_pooled = []
    for model, variant in sorted({(row["model"], row["variant"]) for row in pooled}):
        candidates = [
            row for row in pooled
            if row["model"] == model and row["variant"] == variant
        ]
        selected_pooled.append(max(candidates, key=lambda row: float(row["rank_ic_mean"])))
    report = {
        "format_version": "overnight_embedding_matrix_summary_v1",
        "selection_scope": "validation_only",
        "test_year_accessed": False,
        "complete": not pooled_missing and not dynamic_missing,
        "missing": {"pooled": pooled_missing, "dynamic": dynamic_missing},
        "pooled": {
            "completed": len(pooled), "selected_by_embedding": selected_pooled,
            "ranking": pooled_ranked,
        },
        "dynamic": {
            "completed": len(dynamic),
            "selection_and_confirmation": select_dynamic_confirmation(dynamic),
            "all_results": dynamic,
        },
    }
    output = Path("reports/overnight_v1_summary.json")
    atomic_json(output, report)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pooled_ranked).to_csv(
        output.with_name("overnight_v1_pooled_ranking.csv"), index=False
    )
    pd.DataFrame(report["dynamic"]["selection_and_confirmation"]).to_csv(
        output.with_name("overnight_v1_dynamic_confirmation.csv"), index=False
    )
    print(json.dumps({
        "complete": report["complete"], "pooled_completed": len(pooled),
        "dynamic_completed": len(dynamic),
        "missing": len(pooled_missing) + len(dynamic_missing),
    }, ensure_ascii=False, indent=2))
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

