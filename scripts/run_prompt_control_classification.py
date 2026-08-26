"""Evaluate one selected encoder control under both leakage-safe rolling protocols."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_prompt_mechanism_fold import _fit_logistic, _indices
from src.analysis.prompt_token_mechanisms import rolling_windows
from src.data.pooled_embeddings import load_pooled_embeddings
from src.evaluation.classification import (
    evaluate_binary_classification,
    paired_classification_comparison,
)


def _load(root: Path, model: str, variant: str, feature: str, rows: int) -> np.ndarray:
    values = load_pooled_embeddings(
        root, model=model, variant=variant, feature=feature,
        require_complete_rows=rows, max_matrix_gib=4,
    )
    expected = np.arange(1, rows + 1, dtype=np.int64)
    if not np.array_equal(values.metadata["row_index"].to_numpy(dtype=np.int64), expected):
        raise ValueError("control embedding rows are not contiguous")
    return values.matrix


def run(args: argparse.Namespace) -> dict[str, object]:
    selection = json.loads(args.control_selection.read_text(encoding="utf-8"))
    spec = selection["control_specs"][args.spec_index]
    if spec["model"] != args.model or spec["source_variant"] != args.source_variant:
        raise ValueError("classification task does not match control specification")
    panel = pd.read_parquet(args.panel).sort_values("row_index", kind="stable").reset_index(drop=True)
    if len(panel) != args.expected_rows or not np.array_equal(
        panel["row_index"].to_numpy(dtype=np.int64), np.arange(1, args.expected_rows + 1),
    ):
        raise ValueError("classification panel must exactly cover contiguous row_index")
    years = pd.to_datetime(panel["entry_date"], errors="coerce").dt.year.dropna().astype(int)
    windows = rolling_windows(years)
    features = ["body_mean", "full_mean"]
    if spec["mode"] != "no_prompt_position_matched":
        features.insert(0, "prompt_mean")
    metric_rows = []
    prediction_rows = []
    for feature in features:
        original = _load(
            args.source_embedding_root, args.model, args.source_variant,
            feature, args.expected_rows,
        )
        controlled = _load(
            args.control_embedding_root, args.model, args.output_variant,
            feature, args.expected_rows,
        )
        for window in windows:
            positions = _indices(panel, window)
            original_validation, original_test = _fit_logistic(
                original[positions["fit"]], panel.iloc[positions["fit"]][args.target],
                original[positions["validation"]], original[positions["all_train"]],
                panel.iloc[positions["all_train"]][args.target], original[positions["test"]],
            )
            control_validation, control_test = _fit_logistic(
                controlled[positions["fit"]], panel.iloc[positions["fit"]][args.target],
                controlled[positions["validation"]], controlled[positions["all_train"]],
                panel.iloc[positions["all_train"]][args.target], controlled[positions["test"]],
            )
            test = panel.iloc[positions["test"]].reset_index(drop=True)
            original_metrics = evaluate_binary_classification(test[args.target], original_test)
            control_metrics = evaluate_binary_classification(test[args.target], control_test)
            comparison = paired_classification_comparison(
                test[args.target], original_test, control_test, test["entry_date"],
                n_bootstrap=args.bootstrap, seed=42, metrics=("accuracy",),
            )
            metric_rows.append({
                "model": args.model, "source_variant": args.source_variant,
                "output_variant": args.output_variant, "control_id": spec["control_id"],
                "semantic_group": spec["semantic_group"], "is_placebo": spec["is_placebo"],
                "target": args.target, "feature": feature,
                "test_year": int(window["test_year"]),
                "original_accuracy": original_metrics["accuracy"],
                "control_accuracy": control_metrics["accuracy"],
                "control_minus_original": comparison["delta"]["accuracy"],
                "delta_ci_low": comparison["clustered_95_ci"]["accuracy"][0],
                "delta_ci_high": comparison["clustered_95_ci"]["accuracy"][1],
                "test_n": control_metrics["n"],
            })
            prediction_rows.append(pd.DataFrame({
                "row_index": test["row_index"], "entry_date": test["entry_date"],
                "stock_id": test["stock_id"].astype(str), "actual_return": test[args.target],
                "target": args.target, "feature": feature,
                "original_probability": original_test, "control_probability": control_test,
            }))
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite control classification: {args.output}")
    stage = args.output.with_name(f".{args.output.name}.partial")
    stage.mkdir(parents=True)
    try:
        pd.DataFrame(metric_rows).to_parquet(stage / "metrics.parquet", index=False)
        pd.concat(prediction_rows, ignore_index=True).to_parquet(
            stage / "predictions.parquet", index=False,
        )
        report = {
            "format_version": "prompt_control_classification_v1", "spec": spec,
            "target": args.target, "output_variant": args.output_variant,
            "features": features, "years": [int(row["test_year"]) for row in windows],
            "bootstrap": args.bootstrap,
        }
        (stage / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        (stage / "COMPLETED").write_text("prompt_control_classification_v1\n", encoding="utf-8")
        args.output.parent.mkdir(parents=True, exist_ok=True); stage.replace(args.output)
        return report
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--source-embedding-root", type=Path, required=True)
    parser.add_argument("--control-embedding-root", type=Path, required=True)
    parser.add_argument("--control-selection", type=Path, required=True)
    parser.add_argument("--spec-index", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-variant", required=True)
    parser.add_argument("--output-variant", required=True)
    parser.add_argument("--target", choices=("event_return_3d", "next_day_return"), required=True)
    parser.add_argument("--expected-rows", type=int, default=75894)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
