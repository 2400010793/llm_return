"""Fast, fixed-feature learning-rate schedule diagnostic on a sampled fold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_pooled_embedding_classification import (
    evaluate,
    fit_mlp_classifier_with_validation,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fit-sample", type=int, default=60000)
    parser.add_argument("--validation-sample", type=int, default=20000)
    parser.add_argument(
        "--early-stopping-metric",
        choices=("accuracy", "balanced_accuracy", "auc"),
        default="balanced_accuracy",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--schedules", nargs="+",
        choices=("constant", "cosine", "inverse_time"),
        default=("constant", "cosine", "inverse_time"),
    )
    args = parser.parse_args()
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    frame = pd.read_parquet(args.panel)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame = frame.loc[frame["entry_date"].notna()].copy()
    frame["year"] = frame["entry_date"].dt.year
    years = sorted(frame["year"].unique())
    fit_years, validation_years = years[:6], years[6:8]
    fit_idx = np.flatnonzero(frame["year"].isin(fit_years).to_numpy())
    val_idx = np.flatnonzero(frame["year"].isin(validation_years).to_numpy())
    if len(fit_idx) > args.fit_sample:
        fit_idx = rng.choice(fit_idx, size=args.fit_sample, replace=False)
    if len(val_idx) > args.validation_sample:
        val_idx = rng.choice(val_idx, size=args.validation_sample, replace=False)

    matrix = np.load(args.matrix, mmap_mode="r")
    # The matrix is row_index ordered; the panel row_index is one-based.
    x_fit = np.asarray(matrix[frame.iloc[fit_idx]["row_index"].to_numpy() - 1], dtype=np.float32)
    x_val = np.asarray(matrix[frame.iloc[val_idx]["row_index"].to_numpy() - 1], dtype=np.float32)
    y_fit = pd.to_numeric(frame.iloc[fit_idx]["next_day_return"], errors="coerce").to_numpy()
    labels = (y_fit > 0).astype(np.int8)
    y_validation = frame.iloc[val_idx]["next_day_return"]
    args.output_root.mkdir(parents=True, exist_ok=True)

    candidate = {
        "hidden_layer_sizes": (32,), "alpha": 1e-3, "max_iter": 40,
        "batch_size": 128, "learning_rate_init": 1e-3,
        "n_iter_no_change": 10,
    }
    for schedule in args.schedules:
        fitted, probabilities, score, selected_epoch, history = fit_mlp_classifier_with_validation(
            x_fit, labels, x_val, y_validation,
            candidate={**candidate, "learning_rate_schedule": schedule}, seed=args.seed,
            selection_metric=args.early_stopping_metric,
        )
        metrics = evaluate(y_validation, probabilities)
        payload = {
            "learning_rate_schedule": schedule,
            "diagnostic": {
                "fit_years": fit_years, "validation_years": validation_years,
                "fit_sample": len(fit_idx), "validation_sample": len(val_idx),
                "feature": "roberta:masked_short:prompt_pca128_body",
                "classifier": "simple_mlp", "seed": args.seed,
                "early_stopping_metric": args.early_stopping_metric,
            },
            "results": [{
                "test_year": validation_years[-1],
                "accuracy": metrics["accuracy"],
                "majority_accuracy": metrics["majority_accuracy"],
                "n_test_rows": len(val_idx),
                "validation_selection_score": score,
                "validation_accuracy": metrics["accuracy"],
                "validation_balanced_accuracy": metrics["balanced_accuracy"],
                "validation_auc": metrics["auc"],
                "best_params": {**candidate, "learning_rate_schedule": schedule,
                                "selected_epoch": selected_epoch},
                "early_stopping_history": history,
            }],
        }
        (args.output_root / f"{schedule}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    print(json.dumps({"output_root": str(args.output_root), "fit_sample": len(fit_idx), "validation_sample": len(val_idx)}))


if __name__ == "__main__":
    main()
