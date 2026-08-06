"""Run a simple time-split TF-IDF + Ridge baseline on a panel table.

The input must contain `text`, `ret_1d`, and a date column named `entry_date`.
This script is intentionally explicit: it fits TF-IDF only on rows before the
test start date.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.data.ingest import read_table
from src.evaluation.prediction_metrics import regression_metrics
from src.models.baselines import fit_predict_tfidf_ridge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("panel")
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--target", default="ret_1d")
    args = parser.parse_args()
    frame = read_table(args.panel)
    if not {"text", "entry_date", args.target}.issubset(frame.columns):
        raise ValueError("Panel must contain text, entry_date, and target columns")
    frame["entry_date"] = frame["entry_date"].astype("datetime64[ns]")
    frame = frame.dropna(subset=["text", args.target, "entry_date"])
    train = frame[frame["entry_date"] < args.test_start]
    test = frame[frame["entry_date"] >= args.test_start]
    if train.empty or test.empty:
        raise ValueError("Both training and test windows must contain rows")
    result = fit_predict_tfidf_ridge(
        train["text"].astype(str).tolist(),
        train[args.target].to_numpy(dtype=float),
        test["text"].astype(str).tolist(),
    )
    metrics = regression_metrics(test[args.target].to_numpy(dtype=float), result.predictions)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
