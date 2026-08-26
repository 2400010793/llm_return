"""Compare final-fit epoch budgets without changing validation selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_pooled_embedding_classification import evaluate, fit_one, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-sample", type=int, default=60000)
    parser.add_argument("--validation-sample", type=int, default=20000)
    parser.add_argument("--test-sample", type=int, default=20000)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    frame = pd.read_parquet(args.panel)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame = frame.loc[frame["entry_date"].notna()].copy()
    frame["year"] = frame["entry_date"].dt.year
    years = sorted(frame["year"].unique())
    fit_years, validation_years, test_year = years[:6], years[6:8], years[8]
    indices = {}
    for name, mask, limit in (
        ("fit", frame["year"].isin(fit_years), args.fit_sample),
        ("validation", frame["year"].isin(validation_years), args.validation_sample),
        ("test", frame["year"].eq(test_year), args.test_sample),
    ):
        values = np.flatnonzero(mask.to_numpy())
        indices[name] = rng.choice(values, size=min(limit, len(values)), replace=False)
    matrix = np.load(args.matrix, mmap_mode="r")
    def get_x(index):
        rows = frame.iloc[index]["row_index"].to_numpy(dtype=np.int64) - 1
        return np.asarray(matrix[rows], dtype=np.float32)
    x_fit, x_val, x_test = get_x(indices["fit"]), get_x(indices["validation"]), get_x(indices["test"])
    y_fit = frame.iloc[indices["fit"]]["next_day_return"]
    y_val = frame.iloc[indices["validation"]]["next_day_return"]
    y_all = pd.concat([y_fit, y_val], ignore_index=True)
    x_all = np.concatenate([x_fit, x_val], axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for multiplier in (1.0, 1.25, 1.5):
        helper_args = SimpleNamespace(
            classifier="simple_mlp", search_stage="coarse", seed=args.seed,
            learning_rate_schedule="constant", max_iter_override=args.max_iter,
            patience_override=args.max_iter, final_epoch_multiplier=multiplier,
            early_stopping_metric="balanced_accuracy",
        )
        (
            probabilities, params, validation_accuracy, validation_probabilities,
            validation_model, final_model, history, search_history,
        ) = fit_one(x_fit, y_fit, x_val, y_val, x_all, y_all, x_test, helper_args)
        metrics = evaluate(
            frame.iloc[indices["test"]]["next_day_return"], probabilities
        )
        results.append({
            "test_year": int(test_year),
            "multiplier": multiplier,
            "validation_accuracy": validation_accuracy,
            "test_accuracy": metrics["accuracy"],
            "test_majority_accuracy": metrics["majority_accuracy"],
            "n_test_rows": len(indices["test"]),
            "selected_validation_epoch": params["selected_epoch"],
            "final_fit_epoch": params["final_fit_epoch"],
            "history": history,
        })
    payload = {
        "feature": "roberta:masked_short:prompt_pca128_body",
        "classifier": "simple_mlp",
        "learning_rate_schedule": "constant",
        "protocol": {
            "fit_years": fit_years, "validation_years": validation_years,
            "test_year": int(test_year), "fit_sample": len(indices["fit"]),
            "validation_sample": len(indices["validation"]), "test_sample": len(indices["test"]),
            "max_iter": args.max_iter, "seed": args.seed,
        },
        "results": results,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
