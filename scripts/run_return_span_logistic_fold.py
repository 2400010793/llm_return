"""Fit one rolling Logistic baseline on the exact return-token fusion features."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_soft_return_family_fold import (
    FEATURE_MODES,
    _fit_projection,
    _positions,
    _project,
)
from src.evaluation.artifacts import atomic_json


C_VALUES = (0.01, 0.1, 1.0, 10.0)
CLASS_WEIGHTS = (None, "balanced")
THRESHOLDS = tuple(np.linspace(0.1, 0.9, 161))


def _stock_days(
    frame: pd.DataFrame, positions: np.ndarray, probabilities: np.ndarray, target: str,
) -> pd.DataFrame:
    selected = frame.iloc[positions][["stock_id", "entry_date", target]].copy()
    selected["probability"] = probabilities
    selected["entry_date"] = pd.to_datetime(selected["entry_date"], errors="coerce")
    return selected.groupby(["stock_id", "entry_date"], as_index=False, observed=True).agg(
        probability=("probability", "mean"), actual_return=(target, "mean"),
        news=("probability", "size"),
    )


def _metrics(
    stock_days: pd.DataFrame, *, majority_probability: float, threshold: float = 0.5,
) -> dict[str, float | int]:
    actual = stock_days["actual_return"].gt(0).astype(np.int8).to_numpy()
    probabilities = stock_days["probability"].to_numpy(dtype=float)
    return {
        "stock_days": int(len(stock_days)),
        "dates": int(stock_days["entry_date"].nunique()),
        "accuracy": float(accuracy_score(actual, probabilities >= threshold)),
        "log_loss": float(log_loss(actual, probabilities, labels=[0, 1])),
        "auc": float(roc_auc_score(actual, probabilities)),
        "majority_accuracy": float(np.mean(
            (majority_probability >= 0.5) == actual
        )),
        "majority_probability": float(majority_probability),
        "threshold": float(threshold),
    }


def _select_accuracy_threshold(stock_days: pd.DataFrame) -> tuple[float, float]:
    actual = stock_days["actual_return"].gt(0).to_numpy()
    probabilities = stock_days["probability"].to_numpy(dtype=float)
    scored = [
        (
            float(np.mean((probabilities >= threshold) == actual)),
            -abs(float(threshold) - 0.5),
            float(threshold),
        )
        for threshold in THRESHOLDS
    ]
    accuracy, _, threshold = max(scored)
    return threshold, accuracy


def _model(c_value: float, class_weight: str | None) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    mechanism_cache = args.mechanism_cache_root / args.model / args.prompt_length
    manifest = json.loads((mechanism_cache / "manifest.json").read_text(encoding="utf-8"))
    window = next(
        row for row in manifest["windows"] if int(row["test_year"]) == args.test_year
    )
    frame = pd.read_parquet(mechanism_cache / "rows.parquet")
    scopes = _positions(frame, window, args.target)
    span_cache = args.span_cache_root / args.model / args.prompt_length
    span_manifest = json.loads((span_cache / "manifest.json").read_text(encoding="utf-8"))
    unmasked = np.load(span_cache / "short_return_span.npy", mmap_mode="r")
    masked = np.load(span_cache / "masked_short_return_span.npy", mmap_mode="r")
    returns = pd.to_numeric(frame[args.target], errors="coerce").to_numpy(dtype=float)

    fit_pca, fit_scaler = _fit_projection(
        unmasked, masked, scopes["fit"], args.feature_mode,
        components=args.components, sample_rows=args.pca_sample_rows,
    )
    x_fit = _project(
        unmasked, masked, scopes["fit"], args.feature_mode, fit_pca, fit_scaler,
    )
    x_validation = _project(
        unmasked, masked, scopes["validation"], args.feature_mode, fit_pca, fit_scaler,
    )
    y_fit = (returns[scopes["fit"]] > 0).astype(np.int8)
    fit_stock_days = _stock_days(
        frame, scopes["fit"], np.zeros(len(scopes["fit"]), dtype=float), args.target,
    )
    fit_majority_probability = float(fit_stock_days["actual_return"].gt(0).mean())

    search_rows = []
    for c_value in C_VALUES:
        for class_weight in CLASS_WEIGHTS:
            model = _model(c_value, class_weight).fit(x_fit, y_fit)
            probabilities = model.predict_proba(x_validation)[:, 1]
            validation_stock_days = _stock_days(
                frame, scopes["validation"], probabilities, args.target,
            )
            threshold, _ = _select_accuracy_threshold(validation_stock_days)
            metrics = _metrics(
                validation_stock_days,
                majority_probability=fit_majority_probability,
                threshold=threshold,
            )
            search_rows.append({
                "C": c_value,
                "class_weight": class_weight or "none",
                **metrics,
            })
    search = pd.DataFrame(search_rows).sort_values(
        ["accuracy", "log_loss", "C"], ascending=[False, True, True],
    ).reset_index(drop=True)
    winner = search.iloc[0]

    all_pca, all_scaler = _fit_projection(
        unmasked, masked, scopes["all_train"], args.feature_mode,
        components=args.components, sample_rows=args.pca_sample_rows,
    )
    x_all = _project(
        unmasked, masked, scopes["all_train"], args.feature_mode, all_pca, all_scaler,
    )
    x_test = _project(
        unmasked, masked, scopes["test"], args.feature_mode, all_pca, all_scaler,
    )
    final = _model(
        float(winner["C"]),
        None if winner["class_weight"] == "none" else str(winner["class_weight"]),
    ).fit(x_all, (returns[scopes["all_train"]] > 0).astype(np.int8))
    test_probabilities = final.predict_proba(x_test)[:, 1]
    stock_days = _stock_days(frame, scopes["test"], test_probabilities, args.target)
    all_train_stock_days = _stock_days(
        frame, scopes["all_train"],
        np.zeros(len(scopes["all_train"]), dtype=float), args.target,
    )
    all_train_majority_probability = float(
        all_train_stock_days["actual_return"].gt(0).mean()
    )
    test_metrics = _metrics(
        stock_days, majority_probability=all_train_majority_probability,
        threshold=float(winner["threshold"]),
    )

    destination = args.output_root / args.feature_mode / args.target / str(args.test_year)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Logistic fold: {destination}")
    stage = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    stage.mkdir(parents=True)
    try:
        search.to_csv(stage / "validation_search.csv", index=False)
        prediction = frame.iloc[scopes["test"]][
            ["row_index", "stock_id", "entry_date", args.target]
        ].copy()
        prediction["probability"] = test_probabilities
        prediction.to_parquet(stage / "news_predictions.parquet", index=False)
        stock_days.to_parquet(stage / "stock_day_predictions.parquet", index=False)
        report = {
            "format_version": "return_span_logistic_fold_v1",
            "model": args.model,
            "prompt_length": args.prompt_length,
            "phrase": span_manifest["phrase"],
            "tokens": span_manifest["tokens"],
            "feature_mode": args.feature_mode,
            "target": args.target,
            "test_year": args.test_year,
            "window": window,
            "selected": {
                "C": float(winner["C"]),
                "class_weight": str(winner["class_weight"]),
                "validation_log_loss": float(winner["log_loss"]),
                "validation_accuracy": float(winner["accuracy"]),
                "validation_auc": float(winner["auc"]),
                "threshold": float(winner["threshold"]),
            },
            "test_metrics": test_metrics,
            "rows": {name: int(len(values)) for name, values in scopes.items()},
        }
        atomic_json(stage / "metrics.json", report)
        (stage / "COMPLETED").write_text(
            "return_span_logistic_fold_v1\n", encoding="utf-8"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(destination)
        return {**report, "output": str(destination)}
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--span-cache-root", type=Path, required=True)
    parser.add_argument("--mechanism-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", choices=("roberta", "bge_m3"), default="bge_m3")
    parser.add_argument("--prompt-length", choices=("short", "long"), default="short")
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="fusion")
    parser.add_argument(
        "--target", choices=("event_return_3d", "next_day_return"),
        default="next_day_return",
    )
    parser.add_argument("--test-year", type=int, choices=range(2018, 2027), required=True)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--pca-sample-rows", type=int, default=10000)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
