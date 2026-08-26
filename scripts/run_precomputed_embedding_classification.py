"""Train reusable rolling classifiers on an existing full-panel NPY embedding.

The matrix is aligned through its persisted metadata key. Metadata row indices
may be explicitly offset (the all-input embeddings are 0-based while the
classification panel is 1-based). No Transformer encoding is performed.
The primary protocol uses one-day ``next_day_return`` for both fitting and
evaluation; the legacy three-day event target is an explicit override only.
"""

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
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_pooled_embedding_classification import (
    SUPPORTED_CLASSIFIERS,
    evaluate,
    finite_target,
    fit_one,
    preprocess_windows,
    seed_everything,
)
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


def align_precomputed_embedding(
    panel: pd.DataFrame,
    matrix_path: Path,
    metadata_path: Path,
    *,
    panel_row_index_column: str,
    metadata_row_index_column: str,
    metadata_row_index_offset: int,
    expected_rows: int | None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Align an NPY matrix to panel order using a unique persisted row key."""
    matrix = np.load(matrix_path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"embedding matrix must be 2-D; found {matrix.shape}")
    metadata = pd.read_parquet(metadata_path)
    if len(metadata) != len(matrix):
        raise ValueError(f"metadata/matrix row mismatch: {len(metadata)} != {len(matrix)}")
    if expected_rows is not None and len(matrix) != expected_rows:
        raise ValueError(f"expected {expected_rows} embedding rows; found {len(matrix)}")
    if panel_row_index_column not in panel:
        raise ValueError(f"panel missing {panel_row_index_column}")
    if metadata_row_index_column not in metadata:
        raise ValueError(f"metadata missing {metadata_row_index_column}")
    if panel[panel_row_index_column].isna().any():
        raise ValueError(f"panel key {panel_row_index_column} contains missing values")
    if metadata[metadata_row_index_column].isna().any():
        raise ValueError(
            f"metadata key {metadata_row_index_column} contains missing values"
        )

    if metadata_row_index_offset:
        panel_keys = pd.to_numeric(
            panel[panel_row_index_column], errors="raise"
        ).astype(np.int64)
        metadata_keys_raw = pd.to_numeric(
            metadata[metadata_row_index_column], errors="raise"
        ).astype(np.int64)
        metadata_keys = metadata_keys_raw + metadata_row_index_offset
        key_mode = "numeric_with_offset"
    else:
        panel_keys = panel[panel_row_index_column].astype(str)
        metadata_keys_raw = metadata[metadata_row_index_column].astype(str)
        metadata_keys = metadata_keys_raw
        key_mode = "exact_string"
    if panel_keys.duplicated().any() or metadata_keys.duplicated().any():
        raise ValueError("panel and metadata row keys must be unique")
    lookup = pd.Series(np.arange(len(metadata), dtype=np.int64), index=metadata_keys.to_numpy())
    positions = lookup.reindex(panel_keys.to_numpy())
    if positions.isna().any():
        examples = panel_keys[positions.isna().to_numpy()].head(10).tolist()
        raise ValueError(f"metadata does not cover panel row keys: {examples}")
    positions_array = positions.to_numpy(dtype=np.int64)
    aligned = np.asarray(matrix[positions_array], dtype=np.float32)
    if not np.isfinite(aligned).all():
        bad = int(aligned.size - np.isfinite(aligned).sum())
        raise ValueError(f"embedding matrix contains {bad} non-finite values")
    audit = {
        "method": "metadata_key",
        "key_mode": key_mode,
        "panel_key": panel_row_index_column,
        "metadata_key": metadata_row_index_column,
        "metadata_row_index_offset": metadata_row_index_offset,
        "panel_rows": int(len(panel)),
        "metadata_rows": int(len(metadata)),
        "matrix_rows": int(len(matrix)),
        "matrix_dimension": int(matrix.shape[1]),
        "matched_rows": int(len(positions_array)),
        "positionally_identical_after_offset": bool(
            np.array_equal(positions_array, np.arange(len(panel), dtype=np.int64))
        ),
    }
    if key_mode == "numeric_with_offset":
        audit.update({
            "panel_key_min": int(panel_keys.min()),
            "panel_key_max": int(panel_keys.max()),
            "metadata_key_raw_min": int(metadata_keys_raw.min()),
            "metadata_key_raw_max": int(metadata_keys_raw.max()),
        })
    return panel.copy(), aligned, audit


def build_spec(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "format_version": "precomputed_embedding_classification_bundle_v1",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "inputs": {
            "panel": file_fingerprint(args.panel, hash_content=True),
            "matrix": file_fingerprint(args.matrix),
            "metadata": file_fingerprint(args.metadata, hash_content=True),
        },
        "design": {
            "fit_years": 6,
            "validation_years": 2,
            "test_years": 1,
            "train_target": args.train_target_column,
            "evaluation_target": args.evaluation_target_column,
            "date_column": args.date_column,
            "panel_row_index_column": args.panel_row_index_column,
            "metadata_row_index_column": args.metadata_row_index_column,
            "metadata_row_index_offset": args.metadata_row_index_offset,
        },
        "experiment": {
            "embedding_model": args.embedding_model,
            "input_name": args.input_name,
            "classifier": args.classifier,
            "reducer": args.reducer,
            "reducer_components": args.reducer_components,
            "scaler": args.scaler,
            "search_stage": args.search_stage,
            "seed": args.seed,
        },
    }


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--embedding-model",
        choices=(
            "word2vec", "bge_m3", "chinese_bert", "chinese_roberta",
            "qwen3_embedding_8b",
        ),
        required=True,
    )
    parser.add_argument("--input-name", required=True)
    parser.add_argument("--classifier", choices=SUPPORTED_CLASSIFIERS, required=True)
    parser.add_argument("--reducer", choices=("none", "pca"), default="none")
    parser.add_argument("--reducer-components", type=int, default=128)
    parser.add_argument("--scaler", choices=("auto", "none", "standard"), default="auto")
    parser.add_argument(
        "--train-target-column", default="next_day_return",
        help="One-day classification target; event_return_3d is legacy-only.",
    )
    parser.add_argument(
        "--evaluation-target-column", default="next_day_return",
        help="One-day evaluation target.",
    )
    parser.add_argument("--date-column", default="entry_date")
    parser.add_argument("--panel-row-index-column", default="row_index")
    parser.add_argument("--metadata-row-index-column", default="row_index")
    parser.add_argument("--metadata-row-index-offset", type=int, default=0)
    parser.add_argument("--expected-rows", type=int, default=350577)
    parser.add_argument("--search-stage", choices=("coarse", "fine"), default="coarse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()
    if args.reducer == "pca" and args.reducer_components < 1:
        raise ValueError("reducer-components must be positive")
    seed_everything(args.seed)

    spec = build_spec(args)
    spec_id = experiment_id(spec)
    bundle = args.artifact_dir or args.output.with_suffix(".artifacts")
    if not args.force_recompute and completed_bundle_matches(bundle, spec_id):
        print(json.dumps({
            "output": str(args.output), "artifact_dir": str(bundle),
            "experiment_id": spec_id, "resumed": True,
        }, ensure_ascii=False))
        return
    writer_lock = acquire_bundle_lock(bundle)
    atexit.register(release_bundle_lock, writer_lock)
    atomic_json(bundle / "spec.json", {**spec, "experiment_id": spec_id})

    panel = pd.read_parquet(args.panel)
    required = {
        args.panel_row_index_column, args.date_column,
        args.train_target_column, args.evaluation_target_column,
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {', '.join(sorted(missing))}")
    frame, matrix, alignment_audit = align_precomputed_embedding(
        panel, args.matrix, args.metadata,
        panel_row_index_column=args.panel_row_index_column,
        metadata_row_index_column=args.metadata_row_index_column,
        metadata_row_index_offset=args.metadata_row_index_offset,
        expected_rows=args.expected_rows or None,
    )
    frame[args.date_column] = pd.to_datetime(frame[args.date_column], errors="coerce")
    valid_date = frame[args.date_column].notna().to_numpy()
    frame, matrix = frame.loc[valid_date].copy(), matrix[valid_date]
    order = np.argsort(frame[args.date_column].to_numpy(), kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    matrix = matrix[order]
    frame["year"] = frame[args.date_column].dt.year
    years = sorted(int(year) for year in frame["year"].dropna().unique())
    if len(years) < 9:
        raise ValueError(f"rolling design needs at least 9 calendar years; found {years}")

    helper_args = SimpleNamespace(
        reducer=args.reducer,
        reducer_components=args.reducer_components,
        seed=args.seed,
        token_gate_keep=0,
        token_gate_method="fisher",
        scaler=args.scaler,
        classifier=args.classifier,
        search_stage=args.search_stage,
    )
    results: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for test_position in range(8, len(years)):
        test_year = years[test_position]
        in_years = years[test_position - 8:test_position]
        fit_years, validation_years = in_years[:6], in_years[6:]
        fit_idx = np.flatnonzero(frame["year"].isin(fit_years).to_numpy())
        val_idx = np.flatnonzero(frame["year"].isin(validation_years).to_numpy())
        all_idx = np.flatnonzero(frame["year"].isin(in_years).to_numpy())
        test_idx = np.flatnonzero(frame["year"].eq(test_year).to_numpy())
        if not all((len(fit_idx), len(val_idx), len(all_idx), len(test_idx))):
            continue

        preprocess_started = time.perf_counter()
        transformed = preprocess_windows(
            matrix[fit_idx], matrix[val_idx], matrix[all_idx], matrix[test_idx], helper_args,
            y_fit=frame.iloc[fit_idx][args.train_target_column],
            y_all=frame.iloc[all_idx][args.train_target_column],
        )
        x_fit, x_val, x_all, x_test = transformed[:4]
        fit_reducer, all_reducer, scaled, preprocessors = transformed[4:8]
        preprocess_seconds = time.perf_counter() - preprocess_started

        fit_started = time.perf_counter()
        (
            probabilities, params, validation_accuracy, validation_probabilities,
            validation_model, final_model, best_history, search_history,
        ) = fit_one(
            x_fit, frame.iloc[fit_idx][args.train_target_column],
            x_val, frame.iloc[val_idx][args.evaluation_target_column],
            x_all, frame.iloc[all_idx][args.train_target_column],
            x_test, helper_args,
        )
        fit_seconds = time.perf_counter() - fit_started
        fold_bundle = bundle / f"test_year_{test_year}"
        atomic_joblib(fold_bundle / "fit_preprocessor.joblib", preprocessors["fit"])
        atomic_joblib(fold_bundle / "all_train_preprocessor.joblib", preprocessors["all_train"])
        atomic_joblib(fold_bundle / "validation_model.joblib", validation_model)
        atomic_joblib(fold_bundle / "final_model.joblib", final_model)
        atomic_json(fold_bundle / "classification_search_history.json", {
            "best_params": params,
            "validation_score": float(validation_accuracy),
            "best_epoch_history": best_history,
            "search_history": search_history,
        })
        restored = joblib.load(fold_bundle / "final_model.joblib").predict_proba(x_test)[:, 1]
        if not np.allclose(restored, probabilities, rtol=0.0, atol=1e-7):
            raise RuntimeError(f"reloaded final model changed probabilities for {test_year}")

        validation_metrics = evaluate(
            frame.iloc[val_idx][args.evaluation_target_column], validation_probabilities
        )
        metrics = evaluate(frame.iloc[test_idx][args.evaluation_target_column], probabilities)
        _, fit_label_mask = finite_target(frame.iloc[fit_idx][args.train_target_column])
        _, validation_label_mask = finite_target(
            frame.iloc[val_idx][args.evaluation_target_column]
        )
        _, all_train_label_mask = finite_target(
            frame.iloc[all_idx][args.train_target_column]
        )
        _, test_label_mask = finite_target(
            frame.iloc[test_idx][args.evaluation_target_column]
        )
        results.append({
            "test_year": test_year,
            "fit_years": fit_years,
            "validation_years": validation_years,
            "embedding_model": args.embedding_model,
            "input_name": args.input_name,
            "representation": f"{args.embedding_model}:{args.input_name}:global_mean",
            "feature": "global_mean",
            "classifier": args.classifier,
            "reducer": args.reducer,
            "reducer_components": args.reducer_components if args.reducer == "pca" else None,
            "scaled": scaled,
            "best_params": params,
            "validation_accuracy": validation_accuracy,
            "validation_metrics": validation_metrics,
            "n_fit": len(fit_idx),
            "n_fit_labeled": int(fit_label_mask.sum()),
            "n_validation": len(val_idx),
            "n_validation_labeled": int(validation_label_mask.sum()),
            "n_all_train": len(all_idx),
            "n_all_train_labeled": int(all_train_label_mask.sum()),
            "n_test_rows": len(test_idx),
            "n_test_labeled": int(test_label_mask.sum()),
            "preprocess_seconds": preprocess_seconds,
            "model_selection_and_final_fit_seconds": fit_seconds,
            "fit_explained_variance": float(fit_reducer.explained_variance_ratio_.sum()) if fit_reducer is not None else None,
            "all_train_explained_variance": float(all_reducer.explained_variance_ratio_.sum()) if all_reducer is not None else None,
            **metrics,
        })
        predictions = frame.iloc[test_idx][
            [args.panel_row_index_column, args.date_column, args.evaluation_target_column]
        ].copy()
        predictions["test_year"] = test_year
        predictions["probability"] = probabilities
        actual = pd.to_numeric(predictions[args.evaluation_target_column], errors="coerce").to_numpy(dtype=float)
        predictions["actual_label"] = np.where(
            np.isfinite(actual), (actual > 0).astype(np.int8), np.nan
        )
        prediction_frames.append(predictions)
        validation_predictions = frame.iloc[val_idx][
            [args.panel_row_index_column, args.date_column, args.evaluation_target_column]
        ].copy()
        validation_predictions["test_year"] = test_year
        validation_predictions["probability"] = validation_probabilities
        validation_predictions.to_parquet(
            fold_bundle / "validation_predictions.parquet", index=False
        )

    if not results:
        raise ValueError("no rolling test folds were produced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output.with_suffix(".predictions.parquet")
    pd.concat(prediction_frames, ignore_index=True).to_parquet(prediction_path, index=False)
    report = {
        "design": spec["design"],
        "input": {
            "panel": str(args.panel),
            "matrix": str(args.matrix),
            "metadata": str(args.metadata),
            "embedding_dimension": int(matrix.shape[1]),
            "alignment": alignment_audit,
        },
        "experiment": spec["experiment"],
        "years": years,
        "predictions": str(prediction_path),
        "artifact_bundle": str(bundle),
        "experiment_id": spec_id,
        "results": results,
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
        "task_record_id": os.environ.get("TASK_RECORD_ID"),
    })
    write_completed(bundle, spec_id)
    release_bundle_lock(writer_lock)
    atexit.unregister(release_bundle_lock)
    print(json.dumps({
        "output": str(args.output), "artifact_dir": str(bundle),
        "experiment_id": spec_id, "folds": len(results), "resumed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
