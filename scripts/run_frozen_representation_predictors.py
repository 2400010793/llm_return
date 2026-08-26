"""Fit fine-grained predictors after a frozen gate, with optional PCA first."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.artifacts import atomic_json, experiment_id, file_fingerprint
from src.models.return_prediction import (
    fit_preprocessor,
    return_regressor_candidates,
    select_return_regressor,
    select_ridge_alpha,
)


RIDGE_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0)


def parse_components(value: str) -> list[int]:
    components = sorted(set(int(item) for item in value.split(",") if item.strip()))
    if any(item < 1 for item in components):
        raise argparse.ArgumentTypeError("PCA components must be positive")
    return components


def normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize_json(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [normalize_json(item) for item in value]
    if isinstance(value, list):
        return [normalize_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def config_key(representation: dict[str, Any], reducer: str, components: int | None,
               regressor: str, params: dict[str, Any]) -> str:
    value = {
        "embedding": {
            "model": representation["embedding"]["model"],
            "variant": representation["embedding"]["variant"],
        },
        "gate": representation["gate"],
        "reducer": reducer, "components": components,
        "regressor": regressor, "params": normalize_json(params),
    }
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--representation-report", type=Path, required=True)
    parser.add_argument("--pca-components", type=parse_components, default=parse_components("16,32"))
    parser.add_argument("--row-index-column", default="row_index")
    parser.add_argument("--stock-column", default="stock_id")
    parser.add_argument("--date-column", default="entry_date")
    parser.add_argument("--target-column", default="next_day_return")
    parser.add_argument("--min-stocks-per-day", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    representation_report = json.loads(args.representation_report.read_text(encoding="utf-8"))
    representation_spec = representation_report["spec"]
    representation_dir = Path(representation_report["artifact_dir"]) / "representations"
    paths = {
        "fit_rows": representation_dir / "predictor_fit_rows.npy",
        "fit": representation_dir / "predictor_fit.npy",
        "validation_rows": representation_dir / "predictor_validation_rows.npy",
        "validation": representation_dir / "predictor_validation.npy",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"frozen representation artifacts are missing: {missing}")
    spec = {
        "format_version": "frozen_representation_predictor_screen_v1",
        "panel": file_fingerprint(args.panel, hash_content=True),
        "representation_report": file_fingerprint(args.representation_report, hash_content=True),
        "representation_experiment_id": representation_report["experiment_id"],
        "pca_components": args.pca_components,
        "regressors": {
            "ridge_alphas": list(RIDGE_ALPHAS),
            "huber_stage": "fine", "small_mlp_stage": "fine",
        },
        "seed": args.seed,
    }
    spec_id = experiment_id(spec)
    if args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("experiment_id") == spec_id:
            print(json.dumps({"output": str(args.output), "resumed": True, "experiment_id": spec_id}))
            return

    fit_rows = np.load(paths["fit_rows"], mmap_mode="r")
    x_fit_raw = np.load(paths["fit"], mmap_mode="r")
    validation_rows = np.load(paths["validation_rows"], mmap_mode="r")
    x_validation_raw = np.load(paths["validation"], mmap_mode="r")
    if len(fit_rows) != len(x_fit_raw) or len(validation_rows) != len(x_validation_raw):
        raise ValueError("row and representation arrays have inconsistent lengths")
    panel = pd.read_parquet(args.panel, columns=[
        args.row_index_column, args.stock_column, args.date_column, args.target_column,
    ])
    if panel[args.row_index_column].duplicated().any():
        raise ValueError("panel row_index must be unique")
    panel_by_row = panel.set_index(args.row_index_column, drop=False)
    fit_frame = panel_by_row.loc[np.asarray(fit_rows)].reset_index(drop=True)
    validation_frame = panel_by_row.loc[np.asarray(validation_rows)].reset_index(drop=True)
    y_fit = pd.to_numeric(fit_frame[args.target_column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y_fit).all():
        raise ValueError("exported predictor-fit rows contain non-finite targets")

    dimension = int(x_fit_raw.shape[1])
    reductions: list[tuple[str, int | None]] = [("none", None)]
    reductions.extend(("pca", item) for item in args.pca_components if item < dimension)
    candidates: list[dict[str, Any]] = []
    best_models: list[dict[str, Any]] = []
    for reducer, components in reductions:
        x_fit, x_validation, _ = fit_preprocessor(
            np.asarray(x_fit_raw), np.asarray(x_validation_raw),
            reducer=reducer, components=components or dimension, seed=args.seed,
        )
        ridge = select_ridge_alpha(
            x_fit, y_fit, x_validation, validation_frame, RIDGE_ALPHAS,
            stock_column=args.stock_column, date_column=args.date_column,
            target_column=args.target_column,
            min_stocks_per_day=args.min_stocks_per_day, max_expansions=0,
        )
        best_models.append({
            "reducer": reducer, "components": components, "regressor": "ridge",
            "params": {"alpha": ridge.alpha, "seed": args.seed},
            "validation_metrics": ridge.metrics,
        })
        for audit in ridge.audit:
            params = {"alpha": audit["alpha"], "seed": args.seed}
            metrics = {key: value for key, value in audit.items() if key not in {"alpha", "expansion_round"}}
            candidates.append({
                "config_key": config_key(representation_spec, reducer, components, "ridge", params),
                "reducer": reducer, "components": components,
                "regressor": "ridge", "params": params,
                "validation_metrics": metrics,
            })
        for regressor in ("huber_sgd", "small_mlp"):
            selection = select_return_regressor(
                regressor, x_fit, y_fit, x_validation, validation_frame,
                stock_column=args.stock_column, date_column=args.date_column,
                target_column=args.target_column,
                min_stocks_per_day=args.min_stocks_per_day, seed=args.seed,
                candidates=return_regressor_candidates(regressor, seed=args.seed, stage="fine"),
            )
            best_models.append({
                "reducer": reducer, "components": components, "regressor": regressor,
                "params": normalize_json(selection.params),
                "validation_metrics": selection.metrics,
            })
            for audit in selection.audit:
                params = normalize_json(audit["params"])
                metrics = {key: value for key, value in audit.items() if key != "params"}
                candidates.append({
                    "config_key": config_key(representation_spec, reducer, components, regressor, params),
                    "reducer": reducer, "components": components,
                    "regressor": regressor, "params": params,
                    "validation_metrics": metrics,
                })
    report = {
        "format_version": spec["format_version"], "experiment_id": spec_id,
        "spec": spec, "representation": {
            "experiment_id": representation_report["experiment_id"],
            "model": representation_spec["embedding"]["model"],
            "variant": representation_spec["embedding"]["variant"],
            "gate": representation_spec["gate"],
            "windows": representation_spec["windows"],
            "dimension": dimension,
        },
        "candidate_count": len(candidates), "candidates": candidates,
        "best_within_regressor": best_models,
        "runtime": {"seconds": time.perf_counter() - started},
        "provenance": {
            "task_record_id": os.environ.get("TASK_RECORD_ID"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        },
    }
    atomic_json(args.output, report)
    print(json.dumps({
        "output": str(args.output), "experiment_id": spec_id,
        "candidate_count": len(candidates), "resumed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
