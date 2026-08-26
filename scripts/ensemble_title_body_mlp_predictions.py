"""Build an auditable five-seed probability ensemble for title/body MLPs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.classification import evaluate_binary_classification


KEYS = ["row_index", "entry_date", "test_year"]


def load_aligned(paths: list[Path]) -> tuple[pd.DataFrame, np.ndarray]:
    reference = None
    probabilities = []
    for path in paths:
        frame = pd.read_parquet(path).sort_values(KEYS).reset_index(drop=True)
        required = {*KEYS, "actual_label", "probability"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{path} contains duplicated prediction keys")
        if reference is None:
            reference = frame[[*KEYS, "actual_label"]].copy()
        elif not frame[[*KEYS, "actual_label"]].equals(reference):
            raise ValueError(f"prediction rows do not align: {path}")
        probabilities.append(frame["probability"].to_numpy(dtype=float))
    if reference is None:
        raise ValueError("at least one prediction file is required")
    return reference, np.column_stack(probabilities)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.prediction) != len(args.seed):
        raise ValueError("each prediction file requires one seed")
    if len(set(args.seed)) != len(args.seed):
        raise ValueError("seeds must be unique")

    frame, probabilities = load_aligned(args.prediction)
    frame["probability"] = probabilities.mean(axis=1).astype(np.float32)
    frame["seed_probability_std"] = probabilities.std(axis=1).astype(np.float32)
    valid = frame["actual_label"].notna() & np.isfinite(frame["probability"])
    yearly = []
    for year, group in frame.loc[valid].groupby("test_year", sort=True):
        metrics = evaluate_binary_classification(
            group["actual_label"].to_numpy(dtype=float),
            group["probability"].to_numpy(dtype=float),
        )
        yearly.append({"test_year": int(year), **metrics})
    overall = evaluate_binary_classification(
        frame.loc[valid, "actual_label"].to_numpy(dtype=float),
        frame.loc[valid, "probability"].to_numpy(dtype=float),
    )
    report = {
        "format_version": "title_body_mlp_seed_ensemble_v1",
        "method": "arithmetic_mean_probability",
        "threshold": 0.5,
        "seeds": args.seed,
        "inputs": [str(path) for path in args.prediction],
        "prediction_rows": int(len(frame)),
        "valid_evaluation_rows": int(valid.sum()),
        "overall": overall,
        "results": yearly,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output.with_suffix(".predictions.parquet")
    frame.to_parquet(prediction_path, index=False)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "predictions": str(prediction_path),
        "accuracy": overall["accuracy"],
        "majority_accuracy": overall["majority_accuracy"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
