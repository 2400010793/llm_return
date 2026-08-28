"""Summarize the completed Sina CNINFO pooled-classifier matrix.

The matrix is selected with validation metrics only. Test metrics are emitted
as out-of-sample diagnostics and are never used to choose a winner.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


MODELS = ("roberta", "bge_m3", "ckip_bert", "xlm_roberta_large")
VARIANTS = ("plain", "short", "masked_short", "long", "masked_long")
FEATURES = ("body_mean", "title_body_mean", "full_mean", "cls", "full_max")
CLASSIFIERS = (
    "logistic",
    "linear_svm",
    "sgd",
    "simple_mlp",
    "hist_gradient_boosting",
)
EXPECTED_TEST_YEARS = (2023, 2024, 2025, 2026)
EXPECTED_PANEL_ROWS = 4928


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := _finite(value)) is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _std(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := _finite(value)) is not None]
    if not numbers:
        return None
    average = sum(numbers) / len(numbers)
    return math.sqrt(sum((number - average) ** 2 for number in numbers) / len(numbers))


def _metric(row: dict[str, Any], name: str) -> Any:
    return row.get(name)


def _validation_metric(row: dict[str, Any], name: str) -> Any:
    return row.get("validation_metrics", {}).get(name)


def _target_names(design: dict[str, Any]) -> tuple[Any, Any]:
    return (
        design.get("train_target", design.get("train_target_column")),
        design.get("evaluation_target", design.get("evaluation_target_column")),
    )


def _cell_key(payload: dict[str, Any]) -> tuple[str, str, str, str]:
    experiment = payload["experiment"]
    return (
        str(experiment["model"]),
        str(experiment["variant"]),
        str(experiment["feature"]),
        str(experiment["classifier"]),
    )


def _validate_report(payload: dict[str, Any], path: Path) -> None:
    experiment = payload.get("experiment", {})
    design = payload.get("design", {})
    input_info = payload.get("input", {})
    alignment = input_info.get("embedding_alignment", input_info.get("alignment", {}))
    results = payload.get("results", [])
    train_target, evaluation_target = _target_names(design)
    required = {
        "model": experiment.get("model"),
        "variant": experiment.get("variant"),
        "feature": experiment.get("feature"),
        "classifier": experiment.get("classifier"),
    }
    if (
        required["model"] not in MODELS
        or required["variant"] not in VARIANTS
        or required["feature"] not in FEATURES
        or required["classifier"] not in CLASSIFIERS
    ):
        raise ValueError(f"unexpected matrix cell in {path}: {required}")
    if experiment.get("reducer") != "none":
        raise ValueError(f"non-none reducer in {path}")
    if (
        train_target != "event_return_3d"
        or evaluation_target != "next_day_return"
        or design.get("run_mode") != "final-test"
        or design.get("seed") != 42
    ):
        raise ValueError(f"protocol mismatch in {path}: {design}")
    if len(results) != len(EXPECTED_TEST_YEARS):
        raise ValueError(f"expected four result folds in {path}")
    years = tuple(sorted(int(row["test_year"]) for row in results))
    if years != EXPECTED_TEST_YEARS:
        raise ValueError(f"unexpected test years in {path}: {years}")
    if (
        input_info.get("embedding_rows") != EXPECTED_PANEL_ROWS
        or alignment.get("panel_rows") != EXPECTED_PANEL_ROWS
        or alignment.get("embedding_rows") != EXPECTED_PANEL_ROWS
        or alignment.get("matched_rows") != EXPECTED_PANEL_ROWS
        or alignment.get("unmatched_panel_rows") != 0
    ):
        raise ValueError(f"row alignment mismatch in {path}: {alignment}")
    artifact_bundle = Path(payload.get("artifact_bundle", ""))
    if not (artifact_bundle / "COMPLETED").is_file():
        raise ValueError(f"artifact bundle is not complete in {path}: {artifact_bundle}")
    prediction_path = payload.get("predictions")
    if not prediction_path or not Path(prediction_path).is_file():
        raise ValueError(f"prediction file is missing in {path}: {prediction_path}")


def summarize_report(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    _validate_report(payload, path)
    experiment = payload["experiment"]
    design = payload["design"]
    input_info = payload["input"]
    alignment = input_info.get("embedding_alignment", input_info.get("alignment", {}))
    results = payload["results"]
    row: dict[str, Any] = {
        "model": experiment["model"],
        "variant": experiment["variant"],
        "feature": experiment["feature"],
        "classifier": experiment["classifier"],
        "reducer": experiment.get("reducer", "none"),
        "embedding_dimension": input_info.get("embedding_dimension"),
        "folds": len(results),
        "panel_rows": alignment.get("panel_rows"),
        "matched_rows": alignment.get("matched_rows"),
        "report": str(path.resolve()),
        "artifact_bundle": str(Path(payload["artifact_bundle"]).resolve()),
    }
    for metric in (
        "accuracy",
        "auc",
        "balanced_accuracy",
        "mcc",
        "accuracy_lift_vs_majority",
    ):
        row[f"mean_validation_{metric}"] = _mean(
            _validation_metric(fold, metric) for fold in results
        )
        row[f"std_validation_{metric}"] = _std(
            _validation_metric(fold, metric) for fold in results
        )
        row[f"mean_test_{metric}"] = _mean(_metric(fold, metric) for fold in results)
        row[f"std_test_{metric}"] = _std(_metric(fold, metric) for fold in results)
    return row


def load_matrix(result_roots: Iterable[Path]) -> list[dict[str, Any]]:
    reports: dict[tuple[str, str, str, str], tuple[Path, dict[str, Any]]] = {}
    for root in result_roots:
        for path in sorted(root.glob("*.json")):
            if path.name.endswith(".summary.json"):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            experiment = payload.get("experiment", {})
            if (
                experiment.get("reducer") != "none"
                or experiment.get("model") not in MODELS
                or experiment.get("variant") not in VARIANTS
                or experiment.get("feature") not in FEATURES
                or experiment.get("classifier") not in CLASSIFIERS
            ):
                # The canonical directory also contains PCA and other
                # historical ablations; they are not part of this matrix.
                continue
            key = _cell_key(payload)
            if key in reports:
                raise ValueError(f"duplicate matrix cell {key}: {reports[key][0]} and {path}")
            reports[key] = (path, payload)

    expected = set(itertools.product(MODELS, VARIANTS, FEATURES, CLASSIFIERS))
    observed = set(reports)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(f"matrix coverage mismatch; missing={missing}, extra={extra}")
    rows = [summarize_report(payload, path) for path, payload in reports.values()]
    rows.sort(key=lambda row: (row["model"], row["variant"], row["feature"], row["classifier"]))
    return rows


def _rank(rows: list[dict[str, Any]], metric: str, output_name: str) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get(metric) is not None,
            row.get(metric) if row.get(metric) is not None else float("-inf"),
            row["model"], row["variant"], row["feature"], row["classifier"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(ordered, start=1):
        row[output_name] = rank


def aggregate(rows: list[dict[str, Any]], dimension: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[dimension])].append(row)
    output: list[dict[str, Any]] = []
    for level, members in sorted(groups.items()):
        item: dict[str, Any] = {"dimension": dimension, "level": level, "cells": len(members)}
        for metric in (
            "mean_validation_accuracy",
            "mean_validation_auc",
            "mean_validation_balanced_accuracy",
            "mean_test_accuracy",
            "mean_test_auc",
            "mean_test_balanced_accuracy",
            "mean_test_accuracy_lift_vs_majority",
        ):
            item[metric] = _mean(member.get(metric) for member in members)
        item["best_validation_accuracy"] = max(
            (member["mean_validation_accuracy"] for member in members),
            default=None,
        )
        item["best_validation_auc"] = max(
            (member["mean_validation_auc"] for member in members),
            default=None,
        )
        output.append(item)
    output.sort(key=lambda item: (-(item["mean_validation_accuracy"] or float("-inf")), item["level"]))
    return output


def _csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    number = _finite(value)
    return f"{number:.6f}" if number is not None else "NA"


def _markdown_table(rows: list[dict[str, Any]], fields: list[tuple[str, str]]) -> list[str]:
    lines = ["| " + " | ".join(label for _, label in fields) + " |"]
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        values = []
        for key, _ in fields:
            value = row.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                values.append(str(value))
            elif isinstance(value, float):
                values.append(_fmt(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    best = summary["selection"]
    rows = summary["matrix"]
    lines = [
        "# Sina CNINFO embedding/classifier matrix summary",
        "",
        "## Protocol",
        "",
        "- Matrix: 4 models × 5 Prompt variants × 5 pooled representations × 5 classifiers = 500 cells.",
        "- Rolling design: 6-year fit, 2-year validation, 1-year test; test folds 2023–2026.",
        "- Training target: `event_return_3d`; evaluation target: `next_day_return`.",
        "- Selection: validation accuracy/AUC only. Test metrics are OOS diagnostics.",
        "- All cells use reducer `none`, seed 42, and the aligned 4,928-row panel.",
        "",
        "## Coverage audit",
        "",
        f"- Completed and audited cells: **{summary['audit']['included_cells']}/{summary['audit']['expected_cells']}**.",
        f"- Four test folds: **{summary['audit']['four_fold_cells']}/{summary['audit']['expected_cells']}**.",
        f"- Exact row alignment: **{summary['audit']['aligned_cells']}/{summary['audit']['expected_cells']}**.",
        "",
        "## Validation-first winners",
        "",
    ]
    for name, row in best.items():
        cell = row["cell"]
        lines.append(
            f"- {name}: `{cell['model']} / {cell['variant']} / {cell['feature']} / "
            f"{cell['classifier']}`; value = **{_fmt(row['value'])}**."
        )
    lines.extend(["", "## Top 20 by mean validation accuracy", ""])
    lines.extend(_markdown_table(
        sorted(rows, key=lambda row: row["rank_mean_validation_accuracy"])[:20],
        [
            ("rank_mean_validation_accuracy", "Rank"),
            ("model", "Model"),
            ("variant", "Prompt"),
            ("feature", "Feature"),
            ("classifier", "Classifier"),
            ("mean_validation_accuracy", "Val accuracy"),
            ("mean_validation_auc", "Val AUC"),
            ("mean_test_accuracy", "Test accuracy"),
            ("mean_test_auc", "Test AUC"),
        ],
    ))
    for dimension in ("model", "variant", "feature", "classifier"):
        lines.extend(["", f"## Aggregate by {dimension}", ""])
        lines.extend(_markdown_table(
            summary["aggregates"][dimension],
            [
                ("level", dimension),
                ("cells", "Cells"),
                ("mean_validation_accuracy", "Mean val accuracy"),
                ("mean_validation_auc", "Mean val AUC"),
                ("best_validation_accuracy", "Best val accuracy"),
                ("mean_test_accuracy", "Mean test accuracy"),
                ("mean_test_auc", "Mean test AUC"),
            ],
        ))
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "该矩阵同时比较了 500 个 cells，validation 最优值可能包含模型选择偏差；不能把 validation winner 直接当作独立 OOS alpha。最终结论应结合按年结果、paired prediction tests 以及 Linear SVM 的收敛警告审计。",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_summary(result_roots: Iterable[Path]) -> dict[str, Any]:
    rows = load_matrix(result_roots)
    _rank(rows, "mean_validation_accuracy", "rank_mean_validation_accuracy")
    _rank(rows, "mean_validation_auc", "rank_mean_validation_auc")
    _rank(rows, "mean_test_accuracy", "rank_mean_test_accuracy")
    _rank(rows, "mean_test_auc", "rank_mean_test_auc")
    best_accuracy = max(rows, key=lambda row: row["mean_validation_accuracy"])
    best_auc = max(rows, key=lambda row: row["mean_validation_auc"])
    best_test_accuracy = max(rows, key=lambda row: row["mean_test_accuracy"])
    best_test_auc = max(rows, key=lambda row: row["mean_test_auc"])
    audit = {
        "expected_cells": len(MODELS) * len(VARIANTS) * len(FEATURES) * len(CLASSIFIERS),
        "included_cells": len(rows),
        "four_fold_cells": sum(row["folds"] == 4 for row in rows),
        "aligned_cells": sum(
            row["panel_rows"] == EXPECTED_PANEL_ROWS and row["matched_rows"] == EXPECTED_PANEL_ROWS
            for row in rows
        ),
        "models": list(MODELS),
        "variants": list(VARIANTS),
        "features": list(FEATURES),
        "classifiers": list(CLASSIFIERS),
    }
    if audit["included_cells"] != audit["expected_cells"]:
        raise ValueError(f"incomplete matrix audit: {audit}")
    if audit["four_fold_cells"] != audit["expected_cells"] or audit["aligned_cells"] != audit["expected_cells"]:
        raise ValueError(f"fold/alignment audit failed: {audit}")
    return {
        "protocol": {
            "fit_years": 6,
            "validation_years": 2,
            "test_years": 1,
            "test_folds": list(EXPECTED_TEST_YEARS),
            "train_target": "event_return_3d",
            "evaluation_target": "next_day_return",
            "reducer": "none",
            "seed": 42,
            "selection_rule": "validation-first; test metrics are OOS diagnostics only",
        },
        "audit": audit,
        "selection": {
            "mean_validation_accuracy": {
                "cell": {key: best_accuracy[key] for key in ("model", "variant", "feature", "classifier")},
                "value": best_accuracy["mean_validation_accuracy"],
            },
            "mean_validation_auc": {
                "cell": {key: best_auc[key] for key in ("model", "variant", "feature", "classifier")},
                "value": best_auc["mean_validation_auc"],
            },
            "mean_test_accuracy_diagnostic": {
                "cell": {key: best_test_accuracy[key] for key in ("model", "variant", "feature", "classifier")},
                "value": best_test_accuracy["mean_test_accuracy"],
            },
            "mean_test_auc_diagnostic": {
                "cell": {key: best_test_auc[key] for key in ("model", "variant", "feature", "classifier")},
                "value": best_test_auc["mean_test_auc"],
            },
        },
        "aggregates": {
            dimension: aggregate(rows, dimension)
            for dimension in ("model", "variant", "feature", "classifier")
        },
        "matrix": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        action="append",
        default=[],
        help="Result directory; repeat to combine reused and new outputs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/classification/"
            "sina_cninfo_method_embedding_classifier_matrix/matrix.json"
        ),
    )
    args = parser.parse_args()
    roots = args.results_root or [
        Path("/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cninfo_method_v1/classification"),
        Path("/data/alpha_team2/shares/llm_return/datasets/news_content_quality_20260812/cninfo_method_v1/prompt_embedding_classification/classification"),
    ]
    summary = build_summary(roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _csv_write(args.output.with_suffix(".csv"), summary["matrix"])
    for dimension, rows in summary["aggregates"].items():
        _csv_write(args.output.with_name(f"aggregates_{dimension}.csv"), rows)
    write_markdown(args.output.with_suffix(".md"), summary)
    print(json.dumps({"audit": summary["audit"], "selection": summary["selection"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
