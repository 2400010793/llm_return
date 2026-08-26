"""Run rolling classifiers on a training-only reduced prompt-token matrix.

The primary protocol uses the one-day ``next_day_return`` target for both
fitting and evaluation. The three-day event target can be supplied explicitly
for a legacy ablation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_pooled_embedding_classification import evaluate, fit_one, scale_windows, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--classifier",
        choices=(
            "logistic", "linear_svm", "sgd", "simple_mlp", "mlp",
            "random_forest", "knn",
        ),
        required=True,
    )
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--matrix-summary", type=Path, default=None)
    parser.add_argument(
        "--train-target-column", default="next_day_return",
        help="One-day classification target; event_return_3d is legacy-only.",
    )
    parser.add_argument(
        "--evaluation-target-column", default="next_day_return",
        help="One-day evaluation target.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument(
        "--learning-rate-schedule",
        choices=("constant", "cosine", "inverse_time"),
        default="constant",
    )
    parser.add_argument(
        "--early-stopping-metric",
        choices=("accuracy", "balanced_accuracy", "auc"),
        default="balanced_accuracy",
    )
    parser.add_argument("--early-stopping-min-epoch", type=int, default=1)
    parser.add_argument("--final-epoch-multiplier", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.final_epoch_multiplier <= 0:
        raise ValueError("--final-epoch-multiplier must be positive")
    if args.early_stopping_min_epoch < 1:
        raise ValueError("--early-stopping-min-epoch must be positive")
    seed_everything(args.seed)
    summary_path = args.matrix_summary
    if summary_path is None:
        candidate = args.matrix.with_suffix(".summary.json")
        fallback = args.matrix.parent / "summary.json"
        summary_path = candidate if candidate.is_file() else fallback
    preprocessing = None
    if summary_path.is_file():
        preprocessing = json.loads(summary_path.read_text(encoding="utf-8"))
        if preprocessing.get("model") != args.model or preprocessing.get("variant") != args.variant:
            raise ValueError(
                f"matrix summary identity mismatch: "
                f"{preprocessing.get('model')}/{preprocessing.get('variant')} != "
                f"{args.model}/{args.variant}"
            )
        summary_components = preprocessing.get(
            "components", preprocessing.get("prompt_components", -1)
        )
        if int(summary_components) != args.components:
            raise ValueError("matrix summary components do not match --components")
    frame = pd.read_parquet(args.panel)
    required = {
        "row_index", "entry_date", args.train_target_column,
        args.evaluation_target_column,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"panel missing columns: {', '.join(sorted(missing))}")
    matrix = np.load(args.matrix, mmap_mode="r")
    if len(matrix) != len(frame):
        raise ValueError(f"matrix/panel rows differ: {len(matrix)} != {len(frame)}")
    if preprocessing is not None and int(preprocessing.get("rows", -1)) != len(matrix):
        raise ValueError("matrix summary rows do not match matrix")
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    valid = frame["entry_date"].notna().to_numpy()
    source_positions = frame.loc[valid, "row_index"].to_numpy(dtype=np.int64) - 1
    frame = frame.loc[valid].copy()
    order = np.argsort(frame["entry_date"].to_numpy(), kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    source_positions = source_positions[order]
    frame["year"] = frame["entry_date"].dt.year
    years = sorted(int(value) for value in frame["year"].unique())
    helper_args = SimpleNamespace(
        classifier=args.classifier, search_stage="coarse", seed=args.seed,
        learning_rate_schedule=args.learning_rate_schedule,
        early_stopping_metric=args.early_stopping_metric,
        early_stopping_min_epoch=args.early_stopping_min_epoch,
        max_iter_override=args.max_iter,
        final_epoch_multiplier=args.final_epoch_multiplier,
    )
    results, prediction_frames = [], []
    positions = list(range(8, len(years)))
    if args.max_folds is not None:
        if args.max_folds < 1:
            raise ValueError("--max-folds must be positive")
        positions = positions[:args.max_folds]
    for position in positions:
        test_year = years[position]
        in_years = years[position - 8:position]
        fit_years, validation_years = in_years[:6], in_years[6:]
        fit_idx = np.flatnonzero(frame["year"].isin(fit_years).to_numpy())
        val_idx = np.flatnonzero(frame["year"].isin(validation_years).to_numpy())
        all_idx = np.flatnonzero(frame["year"].isin(in_years).to_numpy())
        test_idx = np.flatnonzero(frame["year"].eq(test_year).to_numpy())
        x_fit = np.asarray(matrix[source_positions[fit_idx]], dtype=np.float32)
        x_val = np.asarray(matrix[source_positions[val_idx]], dtype=np.float32)
        x_all = np.asarray(matrix[source_positions[all_idx]], dtype=np.float32)
        x_test = np.asarray(matrix[source_positions[test_idx]], dtype=np.float32)
        scaled = args.classifier not in ("mlp", "simple_mlp")
        x_fit, x_val, x_all, x_test = scale_windows(x_fit, x_val, x_all, x_test, scaled)
        (
            probabilities, params, selected_validation_score, validation_probabilities,
            validation_model, final_model, early_stopping_history,
            hyperparameter_search_history,
        ) = fit_one(
            x_fit, frame.iloc[fit_idx][args.train_target_column],
            x_val, frame.iloc[val_idx][args.evaluation_target_column],
            x_all, frame.iloc[all_idx][args.train_target_column], x_test, helper_args,
        )
        metrics = evaluate(
            frame.iloc[test_idx][args.evaluation_target_column], probabilities
        )
        validation_metrics = evaluate(
            frame.iloc[val_idx][args.evaluation_target_column],
            validation_probabilities,
        )
        results.append({
            "test_year": test_year, "fit_years": fit_years,
            "validation_years": validation_years, "model": args.model,
            "variant": args.variant, "feature": "prompt_tokens_flat",
            "classifier": args.classifier, "reducer": "sampled_training_only_pca",
            "reducer_components": args.components, "scaled": scaled,
            "best_params": params,
            "validation_selection_metric": args.early_stopping_metric,
            "validation_selection_score": selected_validation_score,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_metrics": validation_metrics,
            **metrics,
            "n_test_rows": len(test_idx),
            "early_stopping_history": early_stopping_history,
            "hyperparameter_search_history": hyperparameter_search_history,
            "train_target": args.train_target_column,
            "evaluation_target": args.evaluation_target_column,
        })
        fold = frame.iloc[test_idx][
            ["row_index", "entry_date", args.evaluation_target_column]
        ].copy()
        fold["test_year"], fold["probability"] = test_year, probabilities
        prediction_frames.append(fold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output.with_suffix(".predictions.parquet")
    pd.concat(prediction_frames, ignore_index=True).to_parquet(prediction_path, index=False)
    report = {
        "model": args.model, "variant": args.variant,
        "feature": "prompt_tokens_flat", "classifier": args.classifier,
        "reducer": "sampled_training_only_pca", "components": args.components,
        "matrix": str(args.matrix), "matrix_summary": str(summary_path) if preprocessing is not None else None,
        "train_target_column": args.train_target_column,
        "evaluation_target_column": args.evaluation_target_column,
        "learning_rate_schedule": args.learning_rate_schedule,
        "early_stopping_metric": args.early_stopping_metric,
        "early_stopping_min_epoch": args.early_stopping_min_epoch,
        "preprocessing": preprocessing,
        "predictions": str(prediction_path), "results": results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "folds": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
