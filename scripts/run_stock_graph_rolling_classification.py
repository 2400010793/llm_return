"""Run one rolling stock-day GraphSAGE classification fold on CPU."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.artifacts import (
    acquire_bundle_lock,
    atomic_joblib,
    atomic_json,
    completed_bundle_matches,
    experiment_id,
    file_fingerprint,
    release_bundle_lock,
    write_completed,
)
from src.evaluation.classification import evaluate_binary_classification
from src.models.stock_graph_sage import StockGraphSAGEClassifier


def repeated_metrics(
    returns: np.ndarray, probabilities: np.ndarray, counts: np.ndarray
) -> dict[str, float]:
    repetitions = np.asarray(counts, dtype=np.int64)
    if (repetitions < 1).any():
        raise ValueError("announcement counts must be positive")
    return evaluate_binary_classification(
        np.repeat(np.asarray(returns, dtype=float), repetitions),
        np.repeat(np.asarray(probabilities, dtype=float), repetitions),
    )


def make_features(
    own: np.ndarray,
    neighbor: np.ndarray | None,
    indices: np.ndarray,
    scaler: StandardScaler,
    *,
    fit: bool,
    neighbor_present: np.ndarray | None = None,
) -> np.ndarray:
    own_values = np.asarray(own[indices], dtype=np.float32)
    if fit:
        scaler.fit(own_values)
    own_values = scaler.transform(own_values).astype(np.float32, copy=False)
    if neighbor is None:
        return np.ascontiguousarray(own_values)
    if neighbor_present is None or neighbor_present.shape != (len(indices),):
        raise ValueError("graph features require one neighbor-presence flag per row")
    neighbor_values = scaler.transform(
        np.asarray(neighbor[indices], dtype=np.float32)
    ).astype(np.float32, copy=False)
    neighbor_values[~neighbor_present] = 0.0
    return np.ascontiguousarray(
        np.concatenate((own_values, neighbor_values), axis=1), dtype=np.float32
    )


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("self", "industry", "random"), required=True)
    parser.add_argument("--test-year", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    args = parser.parse_args()
    if not 2018 <= args.test_year <= 2026:
        raise ValueError("test-year must be between 2018 and 2026")

    metadata_path = args.dataset_root / "metadata.parquet"
    self_path = args.dataset_root / "self_pca256.npy"
    summary_path = args.dataset_root / "summary.json"
    neighbor_path = (
        None if args.mode == "self"
        else args.dataset_root / f"{args.mode}_neighbor.npy"
    )
    required = [metadata_path, self_path, summary_path, args.dataset_root / "COMPLETED"]
    if neighbor_path is not None:
        required.append(neighbor_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"graph dataset is incomplete: {missing}")
    spec = {
        "format_version": "stock_graph_rolling_classification_v1",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "inputs": {
            "metadata": file_fingerprint(metadata_path, hash_content=True),
            "self": file_fingerprint(self_path),
            "neighbor": file_fingerprint(neighbor_path) if neighbor_path else None,
            "dataset_summary": file_fingerprint(summary_path, hash_content=True),
        },
        "design": {
            "fit_years": list(range(args.test_year - 8, args.test_year - 2)),
            "validation_years": [args.test_year - 2, args.test_year - 1],
            "test_year": args.test_year,
            "target": "next_day_return direction",
            "prediction_unit": "stock-day node",
            "primary_accuracy": "announcement-count-weighted stock-day accuracy",
        },
        "model": {
            "type": "self_mlp" if args.mode == "self" else "one_hop_mean_graphsage",
            "graph_mode": args.mode,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "gradient_clip_norm": args.gradient_clip_norm,
            "seed": args.seed,
            "device": "cpu",
        },
    }
    spec_id = experiment_id(spec)
    bundle = args.artifact_dir or args.output.with_suffix(".artifacts")
    if completed_bundle_matches(bundle, spec_id):
        print(json.dumps({"output": str(args.output), "resumed": True}, ensure_ascii=False))
        return
    writer_lock = acquire_bundle_lock(bundle)
    atexit.register(release_bundle_lock, writer_lock)
    atomic_json(bundle / "spec.json", {**spec, "experiment_id": spec_id})

    metadata = pd.read_parquet(metadata_path)
    own = np.load(self_path, mmap_mode="r")
    neighbor = np.load(neighbor_path, mmap_mode="r") if neighbor_path else None
    if own.ndim != 2 or len(own) != len(metadata):
        raise ValueError("self feature/metadata shape mismatch")
    if neighbor is not None and neighbor.shape != own.shape:
        raise ValueError("neighbor features must match self features")
    dates = pd.to_datetime(metadata["entry_date"], errors="coerce")
    targets = pd.to_numeric(metadata["next_day_return"], errors="coerce").to_numpy(dtype=float)
    counts = pd.to_numeric(metadata["announcement_count"], errors="raise").to_numpy(dtype=np.int64)
    neighbor_counts = (
        None if neighbor is None
        else pd.to_numeric(
            metadata[f"{args.mode}_neighbor_count"], errors="raise"
        ).to_numpy(dtype=np.int64)
    )
    years = dates.dt.year.to_numpy(dtype=np.int64)
    finite = np.isfinite(targets)
    fit_years = np.arange(args.test_year - 8, args.test_year - 2)
    validation_years = np.array([args.test_year - 2, args.test_year - 1])
    fit_idx = np.flatnonzero(finite & np.isin(years, fit_years))
    validation_idx = np.flatnonzero(finite & np.isin(years, validation_years))
    all_idx = np.flatnonzero(finite & np.isin(years, np.r_[fit_years, validation_years]))
    test_idx = np.flatnonzero(finite & (years == args.test_year))
    if not all((len(fit_idx), len(validation_idx), len(all_idx), len(test_idx))):
        raise ValueError("rolling fold contains an empty split")

    fit_scaler = StandardScaler()
    x_fit = make_features(
        own, neighbor, fit_idx, fit_scaler, fit=True,
        neighbor_present=(neighbor_counts[fit_idx] > 0 if neighbor_counts is not None else None),
    )
    x_validation = make_features(
        own, neighbor, validation_idx, fit_scaler, fit=False,
        neighbor_present=(
            neighbor_counts[validation_idx] > 0 if neighbor_counts is not None else None
        ),
    )
    labels = (targets > 0).astype(np.float32)
    model_kwargs = {
        "input_dimension": int(own.shape[1]),
        "use_neighbors": neighbor is not None,
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "gradient_clip_norm": args.gradient_clip_norm,
        "random_state": args.seed,
        "num_threads": args.num_threads,
    }
    validation_model = StockGraphSAGEClassifier(**model_kwargs).fit(
        x_fit,
        labels[fit_idx],
        sample_weight=counts[fit_idx],
        validation_data=(
            x_validation,
            labels[validation_idx],
            counts[validation_idx],
        ),
    )
    validation_probabilities = validation_model.predict_proba(x_validation)[:, 1]
    selected_epoch = validation_model.best_epoch_
    del x_fit, x_validation

    all_scaler = StandardScaler()
    x_all = make_features(
        own, neighbor, all_idx, all_scaler, fit=True,
        neighbor_present=(neighbor_counts[all_idx] > 0 if neighbor_counts is not None else None),
    )
    x_test = make_features(
        own, neighbor, test_idx, all_scaler, fit=False,
        neighbor_present=(neighbor_counts[test_idx] > 0 if neighbor_counts is not None else None),
    )
    final_model = StockGraphSAGEClassifier(
        **{**model_kwargs, "max_epochs": selected_epoch, "patience": selected_epoch + 1}
    ).fit(x_all, labels[all_idx], sample_weight=counts[all_idx])
    probabilities = final_model.predict_proba(x_test)[:, 1]

    stock_day_metrics = evaluate_binary_classification(targets[test_idx], probabilities)
    announcement_metrics = repeated_metrics(
        targets[test_idx], probabilities, counts[test_idx]
    )
    validation_stock_day_metrics = evaluate_binary_classification(
        targets[validation_idx], validation_probabilities
    )
    validation_announcement_metrics = repeated_metrics(
        targets[validation_idx], validation_probabilities, counts[validation_idx]
    )
    result = {
        "test_year": args.test_year,
        "fit_years": fit_years.tolist(),
        "validation_years": validation_years.tolist(),
        "mode": args.mode,
        "selected_epoch": selected_epoch,
        "validation_selected_weighted_accuracy": validation_model.best_validation_weighted_accuracy_,
        "n_fit_nodes": int(len(fit_idx)),
        "n_validation_nodes": int(len(validation_idx)),
        "n_all_train_nodes": int(len(all_idx)),
        "n_test_nodes": int(len(test_idx)),
        "n_test_announcements": int(counts[test_idx].sum()),
        "validation_stock_day_metrics": validation_stock_day_metrics,
        "validation_announcement_weighted_metrics": validation_announcement_metrics,
        "stock_day_metrics": stock_day_metrics,
        "announcement_weighted_metrics": announcement_metrics,
        **announcement_metrics,
    }
    predictions = metadata.iloc[test_idx][[
        "node_row_index", "entry_date", "stock_id", "next_day_return",
        "announcement_count", "industry",
    ]].copy()
    predictions["probability"] = probabilities
    prediction_path = args.output.with_suffix(".predictions.parquet")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(prediction_path, index=False)

    atomic_joblib(bundle / "fit_scaler.joblib", fit_scaler)
    atomic_joblib(bundle / "validation_model.joblib", validation_model)
    atomic_joblib(bundle / "all_train_scaler.joblib", all_scaler)
    atomic_joblib(bundle / "final_model.joblib", final_model)
    restored = joblib.load(bundle / "final_model.joblib").predict_proba(x_test)[:, 1]
    if not np.allclose(restored, probabilities, rtol=0.0, atol=1e-7):
        raise RuntimeError("reloaded GraphSAGE model changed test probabilities")
    report = {
        "format_version": "stock_graph_rolling_classification_report_v1",
        "experiment_id": spec_id,
        "spec": spec,
        "result": result,
        "predictions": str(prediction_path),
        "artifact_bundle": str(bundle),
        "runtime": {
            "total_seconds": time.perf_counter() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "provenance": {
            "task_record_id": os.environ.get("TASK_RECORD_ID"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    atomic_json(args.output, report)
    atomic_json(bundle / "report.json", report)
    artifact_files = sorted(
        path for path in bundle.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "COMPLETED", ".writer.lock"}
    )
    atomic_json(bundle / "manifest.json", {
        "experiment_id": spec_id,
        "files": {
            str(path.relative_to(bundle)): file_fingerprint(path, hash_content=True)
            for path in artifact_files
        },
        "external_outputs": {
            "report": file_fingerprint(args.output, hash_content=True),
            "predictions": file_fingerprint(prediction_path, hash_content=True),
        },
    })
    write_completed(bundle, spec_id)
    release_bundle_lock(writer_lock)
    atexit.unregister(release_bundle_lock)
    print(json.dumps({
        "output": str(args.output),
        "mode": args.mode,
        "test_year": args.test_year,
        "accuracy": announcement_metrics["accuracy"],
        "stock_day_accuracy": stock_day_metrics["accuracy"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
