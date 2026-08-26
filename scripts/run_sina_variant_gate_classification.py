"""Evaluate leakage-safe gates across the five Sina document variants.

The embedding models produce one document vector per article, so the old
prompt-token gate is not applicable.  This runner instead gates the five
variants of one embedding model.  Each variant is standardized using the
current fit window, the gate is learned from fit-window labels only, and the
frozen representation is then passed to a rolling classifier.

Gate modes:

* ``uniform``: equal-weight average of the five standardized variants;
* ``variance``: training-only unsupervised variance-weighted average;
* ``fisher``: training-only class-separation weighted average;
* ``logistic_l1``: training-only sparse logistic group weights;
* ``dynamic``: a training-only Fisher direction creates a per-article soft
  variant weight, with no test labels used in the gate.

No PCA is applied in this first gate experiment.
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
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_precomputed_embedding_classification import (
    SUPPORTED_CLASSIFIERS,
    align_precomputed_embedding,
    evaluate,
    finite_target,
    fit_one,
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


VARIANTS = ("plain", "prompt_short", "prompt_long", "masked_short", "masked_long")
MODEL_DIRECTORIES = {
    "chinese_bert": "chinese_bert_prompt",
    "chinese_roberta": "chinese_roberta_prompt",
    "bge_m3": "bge_m3_prompt",
    "word2vec": "word2vec_prompt",
}
MATRIX_NAMES = {
    "chinese_bert": "chinese_bert.npy",
    "chinese_roberta": "chinese_roberta.npy",
    "bge_m3": "bge_m3.npy",
    "word2vec": "word2vec.npy",
}
GATE_MODES = ("uniform", "variance", "fisher", "logistic_l1", "dynamic")


def _softmax(values: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64) / temperature
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exponent = np.exp(np.clip(shifted, -50.0, 50.0))
    return (exponent / exponent.sum(axis=1, keepdims=True)).astype(np.float32)


def _normalise_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    values = np.where(np.isfinite(values) & (values >= 0), values, 0.0)
    total = float(values.sum())
    if total <= 0:
        return np.full(len(values), 1.0 / len(values), dtype=np.float32)
    return (values / total).astype(np.float32)


def _fisher_score(matrix: np.ndarray, labels: np.ndarray) -> float:
    finite = np.isfinite(labels)
    positive = finite & (labels > 0)
    negative = finite & (labels <= 0)
    if not positive.any() or not negative.any():
        raise ValueError("Fisher gate requires both target classes")
    values = np.asarray(matrix, dtype=np.float32)
    positive_values = values[positive]
    negative_values = values[negative]
    positive_mean = positive_values.mean(axis=0, dtype=np.float64)
    negative_mean = negative_values.mean(axis=0, dtype=np.float64)
    positive_var = positive_values.var(axis=0, dtype=np.float64)
    negative_var = negative_values.var(axis=0, dtype=np.float64)
    score = np.square(positive_mean - negative_mean) / (
        positive_var + negative_var + 1e-6
    )
    return float(np.mean(score))


def _fisher_direction(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    finite = np.isfinite(labels)
    positive = finite & (labels > 0)
    negative = finite & (labels <= 0)
    if not positive.any() or not negative.any():
        raise ValueError("dynamic gate requires both target classes")
    values = np.asarray(matrix, dtype=np.float32)
    direction = (
        values[positive].mean(axis=0, dtype=np.float64)
        - values[negative].mean(axis=0, dtype=np.float64)
    )
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0:
        return np.zeros(values.shape[1], dtype=np.float32)
    return (direction / norm).astype(np.float32)


def fit_variant_gate(
    raw_matrices: list[np.ndarray],
    scaled_matrices: list[np.ndarray],
    targets: np.ndarray,
    *,
    mode: str,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    """Fit one variant gate using only the supplied chronological window."""
    if len(raw_matrices) != len(VARIANTS) or len(scaled_matrices) != len(VARIANTS):
        raise ValueError(f"expected {len(VARIANTS)} variant matrices")
    if any(len(matrix) != len(targets) for matrix in raw_matrices):
        raise ValueError("variant matrices and targets must have equal rows")
    key = mode.lower().replace("-", "_")
    if key not in GATE_MODES:
        raise ValueError(f"gate mode must be one of: {', '.join(GATE_MODES)}")
    finite = np.isfinite(np.asarray(targets, dtype=float))
    if not finite.any():
        raise ValueError("variant gate received no finite training targets")

    scores: np.ndarray | None = None
    directions: list[np.ndarray] | None = None
    selector = None
    if key == "uniform":
        scores = np.ones(len(VARIANTS), dtype=np.float64)
    elif key == "variance":
        scores = np.asarray(
            [
                np.var(matrix[finite], axis=0, dtype=np.float64).mean()
                for matrix in raw_matrices
            ],
            dtype=np.float64,
        )
    elif key == "fisher":
        scores = np.asarray(
            [_fisher_score(matrix, targets) for matrix in scaled_matrices],
            dtype=np.float64,
        )
    elif key == "logistic_l1":
        design = np.hstack([matrix[finite] for matrix in scaled_matrices])
        labels = (np.asarray(targets, dtype=float)[finite] > 0).astype(np.int8)
        if len(np.unique(labels)) < 2:
            raise ValueError("L1 logistic gate requires both target classes")
        selector = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )
        selector.fit(design, labels)
        dimension = scaled_matrices[0].shape[1]
        coefficients = np.abs(selector.coef_[0]).reshape(len(VARIANTS), dimension)
        scores = np.linalg.norm(coefficients, axis=1)
    elif key == "dynamic":
        directions = [_fisher_direction(matrix, targets) for matrix in scaled_matrices]
        scores = np.ones(len(VARIANTS), dtype=np.float64)
    assert scores is not None
    if not np.isfinite(scores).all():
        raise ValueError("variant gate scores are non-finite")
    return {
        "mode": key,
        "weights": _normalise_scores(scores),
        "scores": scores.astype(np.float64),
        "directions": directions,
        "temperature": float(temperature),
        "selector": selector,
        "rows_used": int(finite.sum()),
    }


def transform_variant_gate(
    gate: dict[str, Any], scaled_matrices: list[np.ndarray]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a frozen gate and return its representation plus weight audit."""
    if len(scaled_matrices) != len(VARIANTS):
        raise ValueError(f"expected {len(VARIANTS)} variant matrices")
    rows = len(scaled_matrices[0])
    dimension = scaled_matrices[0].shape[1]
    if any(matrix.shape != (rows, dimension) for matrix in scaled_matrices):
        raise ValueError("variant matrices must have matching shapes")
    if gate["mode"] == "dynamic":
        directions = gate["directions"]
        scores = np.column_stack([
            np.abs(matrix @ direction) / np.sqrt(max(dimension, 1))
            for matrix, direction in zip(scaled_matrices, directions)
        ])
        weights = _softmax(scores, temperature=float(gate["temperature"]))
    else:
        weights = np.tile(
            np.asarray(gate["weights"], dtype=np.float32), (rows, 1)
        )
    combined = np.zeros((rows, dimension), dtype=np.float32)
    for index, matrix in enumerate(scaled_matrices):
        combined += weights[:, index, None] * np.asarray(matrix, dtype=np.float32)
    audit = {
        "mean_weights": weights.mean(axis=0).astype(float).tolist(),
        "std_weights": weights.std(axis=0, dtype=np.float64).astype(float).tolist(),
        "min_weights": weights.min(axis=0).astype(float).tolist(),
        "max_weights": weights.max(axis=0).astype(float).tolist(),
        "rows": int(rows),
        "dimension": int(dimension),
    }
    return combined, audit


def _transform_with_scalers(
    matrices: list[np.ndarray], scalers: list[StandardScaler]
) -> list[np.ndarray]:
    return [
        scaler.transform(matrix).astype(np.float32, copy=False)
        for matrix, scaler in zip(matrices, scalers)
    ]


def model_paths(root: Path, model: str) -> list[tuple[str, Path, Path]]:
    directory = root / MODEL_DIRECTORIES[model]
    matrix_name = MATRIX_NAMES[model]
    return [
        (variant, directory / variant / matrix_name, directory / variant / "metadata.parquet")
        for variant in VARIANTS
    ]


def build_spec(args: argparse.Namespace, paths: list[tuple[str, Path, Path]]) -> dict[str, Any]:
    return {
        "format_version": "sina_variant_gate_classification_bundle_v1",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "inputs": {
            "panel": file_fingerprint(args.panel, hash_content=True),
            "variants": {
                variant: {
                    "matrix": file_fingerprint(matrix),
                    "metadata": file_fingerprint(metadata, hash_content=True),
                }
                for variant, matrix, metadata in paths
            },
        },
        "design": {
            "fit_years": 6,
            "validation_years": 2,
            "test_years": 1,
            "train_target": args.target_column,
            "evaluation_target": args.target_column,
            "date_column": args.date_column,
            "stock_column": args.stock_column,
            "alignment_key": args.panel_row_index_column,
            "reducer": "none",
            "gate_fit_scope": "fit_years_only_then_all_train_years",
        },
        "experiment": {
            "embedding_model": args.model,
            "variants": list(VARIANTS),
            "gate_mode": args.gate_mode,
            "classifier": args.classifier,
            "scaler": "per_variant_standard_fit_only",
            "temperature": args.temperature,
            "search_stage": args.search_stage,
            "seed": args.seed,
        },
    }


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument(
        "--model", choices=tuple(MODEL_DIRECTORIES), required=True
    )
    parser.add_argument("--gate-mode", choices=GATE_MODES, required=True)
    parser.add_argument("--classifier", choices=SUPPORTED_CLASSIFIERS, default="logistic")
    parser.add_argument("--target-column", default="next_day_return")
    parser.add_argument("--stock-column", default="stock_id")
    parser.add_argument("--date-column", default="entry_date")
    parser.add_argument("--panel-row-index-column", default="article_id")
    parser.add_argument("--metadata-row-index-column", default="article_id")
    parser.add_argument("--metadata-row-index-offset", type=int, default=0)
    parser.add_argument("--expected-rows", type=int, default=4928)
    parser.add_argument("--search-stage", choices=("coarse", "fine"), default="coarse")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    seed_everything(args.seed)

    paths = model_paths(args.embedding_root, args.model)
    spec = build_spec(args, paths)
    spec_id = experiment_id(spec)
    bundle = args.artifact_dir or args.output.with_suffix(".artifacts")
    if not args.force_recompute and completed_bundle_matches(bundle, spec_id):
        print(
            json.dumps(
                {"output": str(args.output), "artifact_dir": str(bundle),
                 "experiment_id": spec_id, "resumed": True},
                ensure_ascii=False,
            )
        )
        return
    writer_lock = acquire_bundle_lock(bundle)
    atexit.register(release_bundle_lock, writer_lock)
    atomic_json(bundle / "spec.json", {**spec, "experiment_id": spec_id})

    panel = pd.read_parquet(args.panel)
    required = {
        args.panel_row_index_column,
        args.stock_column,
        args.date_column,
        args.target_column,
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {', '.join(sorted(missing))}")

    matrices: list[np.ndarray] = []
    audits: dict[str, Any] = {}
    aligned_frame: pd.DataFrame | None = None
    for variant, matrix_path, metadata_path in paths:
        current_frame, matrix, audit = align_precomputed_embedding(
            panel,
            matrix_path,
            metadata_path,
            panel_row_index_column=args.panel_row_index_column,
            metadata_row_index_column=args.metadata_row_index_column,
            metadata_row_index_offset=args.metadata_row_index_offset,
            expected_rows=args.expected_rows or None,
        )
        if aligned_frame is None:
            aligned_frame = current_frame
        else:
            expected_keys = aligned_frame[args.panel_row_index_column].astype(str).to_numpy()
            current_keys = current_frame[args.panel_row_index_column].astype(str).to_numpy()
            if not np.array_equal(expected_keys, current_keys):
                raise ValueError(f"variant {variant} changed article_id alignment order")
            if matrix.shape != matrices[0].shape:
                raise ValueError(
                    f"variant {variant} shape {matrix.shape} differs from {matrices[0].shape}"
                )
        matrices.append(matrix)
        audits[variant] = audit
    assert aligned_frame is not None
    frame = aligned_frame
    frame[args.date_column] = pd.to_datetime(frame[args.date_column], errors="coerce")
    valid_date = frame[args.date_column].notna().to_numpy()
    frame = frame.loc[valid_date].copy()
    matrices = [matrix[valid_date] for matrix in matrices]
    order = np.argsort(frame[args.date_column].to_numpy(), kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    matrices = [matrix[order] for matrix in matrices]
    frame["year"] = frame[args.date_column].dt.year
    years = sorted(int(year) for year in frame["year"].dropna().unique())
    if len(years) < 9:
        raise ValueError(f"rolling design needs at least 9 calendar years; found {years}")

    helper_args = type(
        "GateClassifierArgs",
        (),
        {
            "reducer": "none",
            "reducer_components": 0,
            "seed": args.seed,
            "token_gate_keep": 0,
            "token_gate_method": "fisher",
            "scaler": "none",
            "classifier": args.classifier,
            "search_stage": args.search_stage,
        },
    )()
    results: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for test_position in range(8, len(years)):
        test_year = years[test_position]
        in_years = years[test_position - 8:test_position]
        fit_years, validation_years = in_years[:6], in_years[6:8]
        fit_idx = np.flatnonzero(frame["year"].isin(fit_years).to_numpy())
        val_idx = np.flatnonzero(frame["year"].isin(validation_years).to_numpy())
        all_idx = np.flatnonzero(frame["year"].isin(in_years).to_numpy())
        test_idx = np.flatnonzero(frame["year"].eq(test_year).to_numpy())
        if not all((len(fit_idx), len(val_idx), len(all_idx), len(test_idx))):
            continue

        fit_raw = [matrix[fit_idx] for matrix in matrices]
        val_raw = [matrix[val_idx] for matrix in matrices]
        all_raw = [matrix[all_idx] for matrix in matrices]
        test_raw = [matrix[test_idx] for matrix in matrices]
        fit_scalers = [StandardScaler().fit(matrix) for matrix in fit_raw]
        all_scalers = [StandardScaler().fit(matrix) for matrix in all_raw]
        fit_scaled = _transform_with_scalers(fit_raw, fit_scalers)
        val_scaled = _transform_with_scalers(val_raw, fit_scalers)
        all_scaled = _transform_with_scalers(all_raw, all_scalers)
        test_scaled = _transform_with_scalers(test_raw, all_scalers)
        y_fit = pd.to_numeric(
            frame.iloc[fit_idx][args.target_column], errors="coerce"
        ).to_numpy(dtype=float)
        y_all = pd.to_numeric(
            frame.iloc[all_idx][args.target_column], errors="coerce"
        ).to_numpy(dtype=float)
        fit_gate = fit_variant_gate(
            fit_raw,
            fit_scaled,
            y_fit,
            mode=args.gate_mode,
            temperature=args.temperature,
            seed=args.seed,
        )
        all_gate = fit_variant_gate(
            all_raw,
            all_scaled,
            y_all,
            mode=args.gate_mode,
            temperature=args.temperature,
            seed=args.seed,
        )
        x_fit, fit_weight_audit = transform_variant_gate(fit_gate, fit_scaled)
        x_val, validation_weight_audit = transform_variant_gate(fit_gate, val_scaled)
        x_all, all_weight_audit = transform_variant_gate(all_gate, all_scaled)
        x_test, test_weight_audit = transform_variant_gate(all_gate, test_scaled)

        fit_started = time.perf_counter()
        (
            probabilities,
            params,
            validation_accuracy,
            validation_probabilities,
            validation_model,
            final_model,
        ) = fit_one(
            x_fit,
            frame.iloc[fit_idx][args.target_column],
            x_val,
            frame.iloc[val_idx][args.target_column],
            x_all,
            frame.iloc[all_idx][args.target_column],
            x_test,
            helper_args,
        )
        fit_seconds = time.perf_counter() - fit_started
        fold_bundle = bundle / f"test_year_{test_year}"
        atomic_joblib(
            fold_bundle / "fit_gate.joblib",
            {"gate": fit_gate, "scalers": fit_scalers},
        )
        atomic_joblib(
            fold_bundle / "all_train_gate.joblib",
            {"gate": all_gate, "scalers": all_scalers},
        )
        atomic_joblib(fold_bundle / "validation_model.joblib", validation_model)
        atomic_joblib(fold_bundle / "final_model.joblib", final_model)
        atomic_json(fold_bundle / "fit_gate_weights.json", fit_weight_audit)
        atomic_json(fold_bundle / "validation_gate_weights.json", validation_weight_audit)
        atomic_json(fold_bundle / "all_train_gate_weights.json", all_weight_audit)
        atomic_json(fold_bundle / "test_gate_weights.json", test_weight_audit)
        restored = joblib.load(fold_bundle / "final_model.joblib").predict_proba(x_test)[:, 1]
        if not np.allclose(restored, probabilities, rtol=0.0, atol=1e-7):
            raise RuntimeError(f"reloaded final model changed probabilities for {test_year}")

        validation_metrics = evaluate(
            frame.iloc[val_idx][args.target_column], validation_probabilities
        )
        metrics = evaluate(frame.iloc[test_idx][args.target_column], probabilities)
        _, fit_label_mask = finite_target(frame.iloc[fit_idx][args.target_column])
        _, validation_label_mask = finite_target(
            frame.iloc[val_idx][args.target_column]
        )
        _, all_train_label_mask = finite_target(
            frame.iloc[all_idx][args.target_column]
        )
        _, test_label_mask = finite_target(frame.iloc[test_idx][args.target_column])
        results.append({
            "test_year": test_year,
            "fit_years": fit_years,
            "validation_years": validation_years,
            "embedding_model": args.model,
            "variants": list(VARIANTS),
            "gate_mode": args.gate_mode,
            "classifier": args.classifier,
            "representation": f"{args.model}:variant_gate:{args.gate_mode}",
            "reducer": "none",
            "scaled": True,
            "best_params": params,
            "validation_accuracy": validation_accuracy,
            "validation_metrics": validation_metrics,
            "fit_gate": {
                "weights": np.asarray(fit_gate["weights"]).astype(float).tolist(),
                "scores": np.asarray(fit_gate["scores"]).astype(float).tolist(),
                "rows_used": fit_gate["rows_used"],
            },
            "all_train_gate": {
                "weights": np.asarray(all_gate["weights"]).astype(float).tolist(),
                "scores": np.asarray(all_gate["scores"]).astype(float).tolist(),
                "rows_used": all_gate["rows_used"],
            },
            "n_fit": len(fit_idx),
            "n_fit_labeled": int(fit_label_mask.sum()),
            "n_validation": len(val_idx),
            "n_validation_labeled": int(validation_label_mask.sum()),
            "n_all_train": len(all_idx),
            "n_all_train_labeled": int(all_train_label_mask.sum()),
            "n_test_rows": len(test_idx),
            "n_test_labeled": int(test_label_mask.sum()),
            "fit_weight_audit": fit_weight_audit,
            "test_weight_audit": test_weight_audit,
            "fit_seconds": fit_seconds,
            **metrics,
        })
        predictions = frame.iloc[test_idx][
            [
                args.panel_row_index_column,
                args.stock_column,
                args.date_column,
                args.target_column,
            ]
        ].copy()
        predictions["test_year"] = test_year
        predictions["probability"] = probabilities
        actual = pd.to_numeric(
            predictions[args.target_column], errors="coerce"
        ).to_numpy(dtype=float)
        predictions["actual_label"] = np.where(
            np.isfinite(actual), (actual > 0).astype(np.int8), np.nan
        )
        prediction_frames.append(predictions)
        validation_predictions = frame.iloc[val_idx][
            [args.panel_row_index_column, args.date_column, args.target_column]
        ].copy()
        validation_predictions["test_year"] = test_year
        validation_predictions["probability"] = validation_probabilities
        validation_predictions.to_parquet(
            fold_bundle / "validation_predictions.parquet", index=False
        )

    if not results:
        raise ValueError("no rolling gate-classification folds were produced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output.with_suffix(".predictions.parquet")
    pd.concat(prediction_frames, ignore_index=True).to_parquet(prediction_path, index=False)
    report = {
        "design": spec["design"],
        "input": {
            "panel": str(args.panel),
            "embedding_root": str(args.embedding_root),
            "embedding_dimension": int(matrices[0].shape[1]),
            "alignment": audits,
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
        path
        for path in bundle.rglob("*")
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
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_dir": str(bundle),
                "experiment_id": spec_id,
                "folds": len(results),
                "resumed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
