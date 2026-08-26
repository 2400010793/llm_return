"""Run leakage-safe rolling regression on one Sina precomputed embedding.

This runner is intentionally separate from the old 350k-row pooled runner.  It
aligns the five Sina embedding variants through the exact ``article_id`` key,
uses the 6-year fit / 2-year validation / 1-year test protocol, and fits all
preprocessing on the corresponding training window only.  ``reducer=none`` is
the direct no-PCA experiment requested for the completed document embeddings.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_precomputed_embedding_classification import align_precomputed_embedding
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
from src.models.return_prediction import (
    aggregate_stock_day_predictions,
    evaluate_stock_day_predictions,
    finite_target,
    fit_preprocessor,
    make_return_regressor,
    return_regressor_candidates,
    select_return_regressor,
    select_ridge_alpha,
)
from src.portfolio import portfolio_metrics, quantile_portfolio


EMBEDDING_MODELS = (
    "word2vec", "bge_m3", "chinese_bert", "chinese_roberta", "qwen3_embedding_8b",
)
VARIANTS = ("plain", "prompt_short", "prompt_long", "masked_short", "masked_long")
RISK_TARGETS = {
    "post_realized_volatility_5d", "post_realized_volatility_20d",
    "volatility_jump", "volatility_jump_ratio_v2", "volume_shock", "range_shock", "event_abs_return_3d",
    "volatility_jump_log_v2", "next_intraday_rvol_log_change", "next_intraday_rvol_close",
    "next_intraday_spread_log_change", "next_intraday_spread_bps_mean",
    "next_pe_log_dev", "next_pb_log_dev", "next_ps_log_dev", "next_evtoebitda_log_dev",
}
VOLATILITY_TARGETS = {
    "post_realized_volatility_5d", "post_realized_volatility_20d", "volatility_jump",
    "volatility_jump_ratio_v2", "next_intraday_rvol_close",
}


def _qlike(actual: np.ndarray, predicted: np.ndarray) -> float:
    """QLIKE for strictly positive volatility-like targets."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0)
    if not mask.any():
        return float("nan")
    y = actual[mask]
    p = np.maximum(predicted[mask], 1e-8)
    return float(np.mean(y / p - np.log(y / p) - 1.0))


def build_spec(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "format_version": "sina_precomputed_embedding_regression_bundle_v1",
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
            "run_mode": args.run_mode,
            "target": args.target_column,
            "stock_column": args.stock_column,
            "date_column": args.date_column,
            "prediction_unit": "stock_day_mean_of_announcement_predictions",
            "alignment_key": "article_id",
            "preprocessing_fit_scope": "fit_years_only_then_all_train_years",
            "target_kind": "risk" if args.target_column in RISK_TARGETS else "return",
        },
        "experiment": {
            "embedding_model": args.embedding_model,
            "variant": args.variant,
            "regressor": args.regressor,
            "reducer": args.reducer,
            "reducer_components": (
                args.reducer_components if args.reducer == "pca" else None
            ),
            "alphas": sorted(set(args.alpha_values)),
            "max_alpha_expansions": args.max_alpha_expansions,
            "search_stage": args.search_stage,
            "min_stocks_per_day": args.min_stocks_per_day,
            "seed": args.seed,
        },
    }


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--embedding-model", choices=EMBEDDING_MODELS, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--target-column", default="next_day_return")
    parser.add_argument("--stock-column", default="stock_id")
    parser.add_argument("--date-column", default="entry_date")
    parser.add_argument("--panel-row-index-column", default="article_id")
    parser.add_argument("--metadata-row-index-column", default="article_id")
    parser.add_argument("--metadata-row-index-offset", type=int, default=0)
    parser.add_argument("--reducer", choices=("none", "pca"), default="none")
    parser.add_argument("--reducer-components", type=int, default=128)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--max-alpha-expansions", type=int, default=2)
    parser.add_argument(
        "--regressor",
        choices=("ridge", "elasticnet_sgd", "huber_sgd", "small_mlp"),
        default="ridge",
    )
    parser.add_argument("--run-mode", choices=("screen", "final-test"), default="final-test")
    parser.add_argument("--min-stocks-per-day", type=int, default=5)
    parser.add_argument("--expected-rows", type=int, default=4928)
    parser.add_argument("--search-stage", choices=("coarse", "fine"), default="coarse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()
    args.alpha_values = [
        float(value) for value in args.alphas.split(",") if value.strip()
    ]
    if not args.alpha_values or any(value <= 0 for value in args.alpha_values):
        raise ValueError("alphas must contain positive values")
    if args.reducer == "pca" and args.reducer_components < 1:
        raise ValueError("reducer-components must be positive")
    if args.min_stocks_per_day < 1:
        raise ValueError("min-stocks-per-day must be positive")

    spec = build_spec(args)
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
    frame, matrix, alignment_audit = align_precomputed_embedding(
        panel,
        args.matrix,
        args.metadata,
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
    required_years = 8 if args.run_mode == "screen" else 9
    if len(years) < required_years:
        raise ValueError(
            f"{args.run_mode} design needs at least {required_years} years; found {years}"
        )

    results: list[dict[str, Any]] = []
    announcement_predictions: list[pd.DataFrame] = []
    stock_day_predictions: list[pd.DataFrame] = []
    positions: list[int | None] | range = (
        [None] if args.run_mode == "screen" else range(8, len(years))
    )
    for position in positions:
        if position is None:
            test_year = None
            in_years = years[:8]
        else:
            test_year = years[position]
            in_years = years[position - 8:position]
        fit_years, validation_years = in_years[:6], in_years[6:8]
        fit_idx = np.flatnonzero(frame["year"].isin(fit_years).to_numpy())
        val_idx = np.flatnonzero(frame["year"].isin(validation_years).to_numpy())
        all_idx = np.flatnonzero(frame["year"].isin(in_years).to_numpy())
        test_idx = (
            np.array([], dtype=np.int64)
            if test_year is None
            else np.flatnonzero(frame["year"].eq(test_year).to_numpy())
        )
        y_fit, fit_mask = finite_target(frame.iloc[fit_idx][args.target_column])
        if not fit_mask.any():
            continue

        preprocess_started = time.perf_counter()
        x_fit, x_val, fit_transformer = fit_preprocessor(
            matrix[fit_idx][fit_mask],
            matrix[val_idx],
            reducer=args.reducer,
            components=args.reducer_components,
            seed=args.seed,
        )
        preprocess_seconds = time.perf_counter() - preprocess_started

        fit_started = time.perf_counter()
        if args.regressor == "ridge":
            ridge_selection = select_ridge_alpha(
                x_fit,
                y_fit[fit_mask],
                x_val,
                frame.iloc[val_idx],
                args.alpha_values,
                stock_column=args.stock_column,
                date_column=args.date_column,
                target_column=args.target_column,
                min_stocks_per_day=args.min_stocks_per_day,
                max_expansions=args.max_alpha_expansions,
            )
            selected_params = {"alpha": ridge_selection.alpha, "seed": args.seed}
            selection_model = ridge_selection.model
            validation_predictions = ridge_selection.predictions
            validation_metrics = ridge_selection.metrics
            selection_audit = ridge_selection.audit
        else:
            model_selection = select_return_regressor(
                args.regressor,
                x_fit,
                y_fit[fit_mask],
                x_val,
                frame.iloc[val_idx],
                stock_column=args.stock_column,
                date_column=args.date_column,
                target_column=args.target_column,
                min_stocks_per_day=args.min_stocks_per_day,
                seed=args.seed,
                candidates=return_regressor_candidates(
                    args.regressor, seed=args.seed, stage=args.search_stage
                ),
            )
            selected_params = model_selection.params
            selection_model = model_selection.model
            validation_predictions = model_selection.predictions
            validation_metrics = model_selection.metrics
            selection_audit = model_selection.audit
        fit_seconds = time.perf_counter() - fit_started

        fold_name = "screen" if args.run_mode == "screen" else f"test_year_{test_year}"
        fold_bundle = bundle / fold_name
        atomic_joblib(fold_bundle / "fit_preprocessor.joblib", fit_transformer)
        atomic_joblib(fold_bundle / "validation_model.joblib", selection_model)
        val_stock_day = aggregate_stock_day_predictions(
            frame.iloc[val_idx],
            validation_predictions,
            stock_column=args.stock_column,
            date_column=args.date_column,
            target_column=args.target_column,
        )
        val_stock_day.to_parquet(
            fold_bundle / "validation_stock_day_predictions.parquet", index=False
        )
        if args.run_mode == "screen":
            results.append({
                "fit_years": fit_years,
                "validation_years": validation_years,
                "embedding_model": args.embedding_model,
                "variant": args.variant,
                "target": args.target_column,
                "prediction_unit": "stock_day",
                "run_mode": args.run_mode,
                "regressor": args.regressor,
                "search_stage": args.search_stage,
                "reducer": args.reducer,
                "reducer_components": (
                    args.reducer_components if args.reducer == "pca" else None
                ),
                "best_params": selected_params,
                "selection_audit": selection_audit,
                "validation_metrics": validation_metrics,
                "n_fit_announcements": int(fit_mask.sum()),
                "n_validation_announcements": len(val_idx),
                "preprocess_seconds": preprocess_seconds,
                "model_selection_seconds": fit_seconds,
            })
            continue

        y_all, all_mask = finite_target(frame.iloc[all_idx][args.target_column])
        if not all_mask.any():
            continue
        preprocess_started = time.perf_counter()
        x_all, x_test, all_transformer = fit_preprocessor(
            matrix[all_idx][all_mask],
            matrix[test_idx],
            reducer=args.reducer,
            components=args.reducer_components,
            seed=args.seed,
        )
        preprocess_seconds += time.perf_counter() - preprocess_started
        final_started = time.perf_counter()
        final_model = make_return_regressor(args.regressor, selected_params)
        final_model.fit(x_all, y_all[all_mask])
        test_pred = np.asarray(final_model.predict(x_test), dtype=float)
        final_fit_seconds = time.perf_counter() - final_started
        atomic_joblib(fold_bundle / "all_train_preprocessor.joblib", all_transformer)
        atomic_joblib(fold_bundle / "final_model.joblib", final_model)
        restored = joblib.load(fold_bundle / "final_model.joblib").predict(x_test)
        if not np.allclose(restored, test_pred, rtol=0.0, atol=1e-7):
            raise RuntimeError(f"reloaded final model changed predictions for {test_year}")

        historical_mean = float(y_all[all_mask].mean())
        test_metrics, test_stock_day = evaluate_stock_day_predictions(
            frame.iloc[test_idx],
            test_pred,
            historical_mean=historical_mean,
            stock_column=args.stock_column,
            date_column=args.date_column,
            target_column=args.target_column,
            min_stocks_per_day=args.min_stocks_per_day,
        )
        is_risk_target = args.target_column in RISK_TARGETS
        portfolio = None if is_risk_target else quantile_portfolio(test_stock_day, realized="actual_return")
        test_stock_day["test_year"] = test_year
        stock_day_predictions.append(test_stock_day)
        announcement = frame.iloc[test_idx][
            [args.panel_row_index_column, args.stock_column, args.date_column, args.target_column]
        ].copy()
        announcement["test_year"] = test_year
        announcement["prediction"] = test_pred
        announcement_predictions.append(announcement)
        if portfolio is not None:
            portfolio.to_parquet(fold_bundle / "test_quintile_portfolio.parquet", index=False)
        if is_risk_target:
            test_metrics["direction_accuracy"] = float("nan")
            test_metrics["qlike"] = _qlike(
                test_stock_day["actual_return"].to_numpy(),
                test_stock_day["prediction"].to_numpy(),
            ) if args.target_column in VOLATILITY_TARGETS else float("nan")
        results.append({
            "test_year": test_year,
            "fit_years": fit_years,
            "validation_years": validation_years,
            "embedding_model": args.embedding_model,
            "variant": args.variant,
            "target": args.target_column,
            "prediction_unit": "stock_day",
            "run_mode": args.run_mode,
            "regressor": args.regressor,
            "search_stage": args.search_stage,
            "reducer": args.reducer,
            "reducer_components": args.reducer_components if args.reducer == "pca" else None,
            "best_params": selected_params,
            "selection_audit": selection_audit,
            "validation_metrics": validation_metrics,
            "historical_mean_benchmark": historical_mean,
            "n_fit_announcements": int(fit_mask.sum()),
            "n_validation_announcements": len(val_idx),
            "n_all_train_announcements": int(all_mask.sum()),
            "n_test_announcements": len(test_idx),
            "preprocess_seconds": preprocess_seconds,
            "model_selection_seconds": fit_seconds,
            "final_fit_seconds": final_fit_seconds,
            "target_kind": "risk" if is_risk_target else "return",
            "portfolio": portfolio_metrics(portfolio) if portfolio is not None else None,
            **test_metrics,
        })

    if not results:
        raise ValueError("no rolling regression folds were produced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    announcement_path = args.output.with_suffix(".announcement_predictions.parquet")
    stock_day_path = args.output.with_suffix(".stock_day_predictions.parquet")
    if args.run_mode == "final-test":
        pd.concat(announcement_predictions, ignore_index=True).to_parquet(
            announcement_path, index=False
        )
        pd.concat(stock_day_predictions, ignore_index=True).to_parquet(
            stock_day_path, index=False
        )
    report = {
        "design": spec["design"],
        "input": {
            "panel": str(args.panel),
            "matrix": str(args.matrix),
            "metadata": str(args.metadata),
            "embedding_rows": int(alignment_audit["matrix_rows"]),
            "embedding_dimension": int(matrix.shape[1]),
            "alignment": alignment_audit,
        },
        "experiment": spec["experiment"],
        "years": years,
        "announcement_predictions": (
            str(announcement_path) if args.run_mode == "final-test" else None
        ),
        "stock_day_predictions": (
            str(stock_day_path) if args.run_mode == "final-test" else None
        ),
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
            **(
                {
                    "announcement_predictions": file_fingerprint(
                        announcement_path, hash_content=True
                    ),
                    "stock_day_predictions": file_fingerprint(
                        stock_day_path, hash_content=True
                    ),
                }
                if args.run_mode == "final-test"
                else {}
            ),
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
