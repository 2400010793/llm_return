"""Run strict stock-day return prediction on one existing pooled embedding.

Models are fitted on announcement embeddings, while validation selection and
test evaluation use one equal-weighted prediction per stock and trading day.
The paper-faithful default target is the next open-to-open return: enter at the
first tradable open after publication and exit at the following tradable open.
The close-to-close ``next_day_return`` target remains available explicitly. No
Transformer inference is performed by this runner.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pooled_embeddings import (
    FEATURE_GROUPS,
    align_embeddings_to_panel,
    discover_pooled_parts,
    load_pooled_embeddings,
)
from src.evaluation.artifacts import (
    acquire_bundle_lock,
    atomic_joblib,
    atomic_json,
    build_input_fingerprints,
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


PAPER_RIDGE_ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 50.0, 100.0)


def build_spec(args: argparse.Namespace, parts: list[Path]) -> dict[str, Any]:
    spec = {
        "format_version": "pooled_stock_day_return_regression_bundle_v2",
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "inputs": build_input_fingerprints(args.panel, parts),
        "design": {
            "fit_years": args.fit_window_years,
            "validation_years": args.validation_window_years,
            "test_years": 1,
            "run_mode": args.run_mode,
            "target": args.target_column, "stock_column": args.stock_column,
            "date_column": args.date_column,
            "prediction_unit": "stock_day_mean_of_announcement_predictions",
            "portfolio": "daily_equal_weight_quintiles_no_cost_assumed_shortable",
        },
        "experiment": {
            "model": args.model, "variant": args.variant, "feature": args.feature,
            "regressor": args.regressor, "reducer": args.reducer,
            "reducer_components": args.reducer_components,
            "alphas": sorted(set(args.alpha_values)), "seed": args.seed,
            "max_alpha_expansions": args.max_alpha_expansions,
            "search_stage": args.search_stage,
            "min_stocks_per_day": args.min_stocks_per_day,
        },
    }
    if args.selection_correlation != "spearman":
        spec["experiment"]["selection_correlation"] = args.selection_correlation
    return spec


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument(
        "--model", choices=(
            "roberta", "bge_m3", "ckip_bert", "xlm_roberta_large",
            "qwen3_embedding_8b", "finbert2_base",
        ),
        required=True,
    )
    parser.add_argument(
        "--variant", choices=("short", "masked_short", "long", "masked_long", "plain"),
        required=True,
    )
    parser.add_argument("--feature", choices=tuple(FEATURE_GROUPS), required=True)
    parser.add_argument(
        "--target-column", default="next_day_open_to_open_return",
        help="Continuous one-day target; default is paper-style open-to-open.",
    )
    parser.add_argument("--stock-column", default="stock_id")
    parser.add_argument("--date-column", default="entry_date")
    parser.add_argument("--panel-row-index-column", default="row_index")
    parser.add_argument("--reducer", choices=("none", "pca"), default="none")
    parser.add_argument("--reducer-components", type=int, default=128)
    parser.add_argument(
        "--alphas",
        default=",".join(str(value) for value in PAPER_RIDGE_ALPHAS),
        help="Paper grid: 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 50, 100.",
    )
    parser.add_argument(
        "--max-alpha-expansions", type=int, default=0,
        help="Extra upper-bound expansions; 0 matches the paper's fixed grid.",
    )
    parser.add_argument(
        "--regressor",
        choices=("ridge", "elasticnet_sgd", "huber_sgd", "small_mlp"),
        default="ridge",
    )
    parser.add_argument(
        "--run-mode", choices=("screen", "final-test"), default="final-test",
        help=(
            "screen evaluates the final fit/validation window; final-test "
            "walks all eligible years, refits on fit+validation, and evaluates test folds."
        ),
    )
    parser.add_argument("--fit-window-years", type=int, default=6)
    parser.add_argument("--validation-window-years", type=int, default=2)
    parser.add_argument("--min-stocks-per-day", type=int, default=5)
    parser.add_argument(
        "--selection-correlation", choices=("spearman", "pearson"),
        default="spearman",
        help="Cross-sectional validation IC; simple_states uses Pearson.",
    )
    parser.add_argument("--expected-embedding-rows", type=int, default=350577)
    parser.add_argument("--max-embedding-matrix-gib", type=float, default=64.0)
    parser.add_argument("--search-stage", choices=("coarse", "fine"), default="coarse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()
    args.alpha_values = [float(value) for value in args.alphas.split(",") if value.strip()]
    if not args.alpha_values or any(value <= 0 for value in args.alpha_values):
        raise ValueError("alphas must contain positive values")
    if args.reducer == "pca" and args.reducer_components < 1:
        raise ValueError("reducer-components must be positive")
    if args.fit_window_years < 1 or args.validation_window_years < 1:
        raise ValueError("fit and validation windows must be positive")

    parts = discover_pooled_parts(args.embedding_root, args.model, args.variant)
    if not parts:
        raise ValueError("no complete pooled embedding parts")
    spec = build_spec(args, parts)
    spec_id = experiment_id(spec)
    bundle = args.artifact_dir or args.output.with_suffix(".artifacts")
    if not args.force_recompute and completed_bundle_matches(bundle, spec_id):
        print(json.dumps({"output": str(args.output), "experiment_id": spec_id, "resumed": True}))
        return
    writer_lock = acquire_bundle_lock(bundle)
    atexit.register(release_bundle_lock, writer_lock)
    atomic_json(bundle / "spec.json", {**spec, "experiment_id": spec_id})

    panel = pd.read_parquet(args.panel)
    required = {
        args.panel_row_index_column, args.stock_column, args.date_column, args.target_column,
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {', '.join(sorted(missing))}")
    embeddings = load_pooled_embeddings(
        args.embedding_root, model=args.model, variant=args.variant, feature=args.feature,
        require_complete_rows=args.expected_embedding_rows or None,
        max_matrix_gib=args.max_embedding_matrix_gib,
    )
    frame, matrix = align_embeddings_to_panel(
        panel, embeddings, panel_row_index_column=args.panel_row_index_column,
    )
    if len(frame) != len(panel):
        raise ValueError(f"only {len(frame)}/{len(panel)} panel rows have embeddings")
    frame[args.date_column] = pd.to_datetime(frame[args.date_column], errors="coerce")
    valid = frame[args.date_column].notna().to_numpy()
    frame, matrix = frame.loc[valid].copy(), matrix[valid]
    order = np.argsort(frame[args.date_column].to_numpy(), kind="stable")
    frame, matrix = frame.iloc[order].reset_index(drop=True), matrix[order]
    frame["year"] = frame[args.date_column].dt.year
    years = sorted(int(value) for value in frame["year"].dropna().unique())
    history_years = args.fit_window_years + args.validation_window_years
    required_years = history_years if args.run_mode == "screen" else history_years + 1
    if len(years) < required_years:
        raise ValueError(
            f"{args.run_mode} design needs at least {required_years} years; found {years}"
        )

    results: list[dict[str, Any]] = []
    announcement_predictions: list[pd.DataFrame] = []
    stock_day_predictions: list[pd.DataFrame] = []
    positions: list[int | None] | range = (
        [None] if args.run_mode == "screen" else range(history_years, len(years))
    )
    for position in positions:
        if position is None:
            test_year = None
            in_years = years[-history_years:]
        else:
            test_year = years[position]
            in_years = years[position - history_years:position]
        fit_years = in_years[:args.fit_window_years]
        validation_years = in_years[args.fit_window_years:]
        fit_idx = np.flatnonzero(frame["year"].isin(fit_years).to_numpy())
        val_idx = np.flatnonzero(frame["year"].isin(validation_years).to_numpy())
        all_idx = np.flatnonzero(frame["year"].isin(in_years).to_numpy())
        test_idx = (
            np.array([], dtype=np.int64) if test_year is None
            else np.flatnonzero(frame["year"].eq(test_year).to_numpy())
        )
        y_fit, fit_mask = finite_target(frame.iloc[fit_idx][args.target_column])
        if not fit_mask.any():
            continue

        preprocess_started = time.perf_counter()
        x_fit, x_val, fit_transformer = fit_preprocessor(
            matrix[fit_idx][fit_mask], matrix[val_idx], reducer=args.reducer,
            components=args.reducer_components, seed=args.seed,
        )
        preprocess_seconds = time.perf_counter() - preprocess_started

        fit_started = time.perf_counter()
        if args.regressor == "ridge":
            ridge_selection = select_ridge_alpha(
                x_fit, y_fit[fit_mask], x_val, frame.iloc[val_idx], args.alpha_values,
                stock_column=args.stock_column, date_column=args.date_column,
                target_column=args.target_column,
                min_stocks_per_day=args.min_stocks_per_day,
                max_expansions=args.max_alpha_expansions,
                selection_correlation=args.selection_correlation,
            )
            selected_params = {"alpha": ridge_selection.alpha, "seed": args.seed}
            selection_model = ridge_selection.model
            validation_predictions = ridge_selection.predictions
            validation_metrics = ridge_selection.metrics
            selection_audit = ridge_selection.audit
        else:
            model_selection = select_return_regressor(
                args.regressor, x_fit, y_fit[fit_mask], x_val, frame.iloc[val_idx],
                stock_column=args.stock_column, date_column=args.date_column,
                target_column=args.target_column,
                min_stocks_per_day=args.min_stocks_per_day, seed=args.seed,
                selection_correlation=args.selection_correlation,
                candidates=return_regressor_candidates(
                    args.regressor, seed=args.seed, stage=args.search_stage,
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
            frame.iloc[val_idx], validation_predictions, stock_column=args.stock_column,
            date_column=args.date_column, target_column=args.target_column,
        )
        val_stock_day.to_parquet(
            fold_bundle / "validation_stock_day_predictions.parquet", index=False
        )
        if args.run_mode == "screen":
            results.append({
                "fit_years": fit_years, "validation_years": validation_years,
                "model": args.model, "variant": args.variant, "feature": args.feature,
                "target": args.target_column, "prediction_unit": "stock_day",
                "run_mode": args.run_mode, "regressor": args.regressor,
                "search_stage": args.search_stage,
                **({"selection_correlation": args.selection_correlation}
                   if args.selection_correlation != "spearman" else {}),
                "reducer": args.reducer,
                "reducer_components": (
                    args.reducer_components if args.reducer == "pca" else None
                ),
                "best_params": selected_params, "selection_audit": selection_audit,
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
            matrix[all_idx][all_mask], matrix[test_idx], reducer=args.reducer,
            components=args.reducer_components, seed=args.seed,
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
            raise RuntimeError("reloaded final model changed predictions")

        historical_mean = float(y_all[all_mask].mean())
        test_metrics, test_stock_day = evaluate_stock_day_predictions(
            frame.iloc[test_idx], test_pred, historical_mean=historical_mean,
            stock_column=args.stock_column, date_column=args.date_column,
            target_column=args.target_column, min_stocks_per_day=args.min_stocks_per_day,
            correlation_method=args.selection_correlation,
        )
        portfolio = quantile_portfolio(test_stock_day, realized="actual_return")
        test_stock_day["test_year"] = test_year
        stock_day_predictions.append(test_stock_day)
        announcement = frame.iloc[test_idx][[
            args.panel_row_index_column, args.stock_column, args.date_column, args.target_column,
        ]].copy()
        announcement["test_year"] = test_year
        announcement["prediction"] = test_pred
        announcement_predictions.append(announcement)
        portfolio.to_parquet(fold_bundle / "test_quintile_portfolio.parquet", index=False)
        results.append({
            "test_year": test_year, "fit_years": fit_years,
            "validation_years": validation_years, "model": args.model,
            "variant": args.variant, "feature": args.feature,
            "target": args.target_column, "prediction_unit": "stock_day",
            "run_mode": args.run_mode, "regressor": args.regressor,
            "search_stage": args.search_stage,
            **({"selection_correlation": args.selection_correlation}
               if args.selection_correlation != "spearman" else {}),
            "reducer": args.reducer,
            "reducer_components": args.reducer_components if args.reducer == "pca" else None,
            "best_params": selected_params, "selection_audit": selection_audit,
            "validation_metrics": validation_metrics,
            "historical_mean_benchmark": historical_mean,
            "n_fit_announcements": int(fit_mask.sum()),
            "n_validation_announcements": len(val_idx),
            "n_all_train_announcements": int(all_mask.sum()),
            "n_test_announcements": len(test_idx),
            "preprocess_seconds": preprocess_seconds,
            "model_selection_seconds": fit_seconds,
            "final_fit_seconds": final_fit_seconds,
            "portfolio": portfolio_metrics(portfolio), **test_metrics,
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
            "panel": str(args.panel), "embedding_root": str(args.embedding_root),
            "embedding_rows": len(embeddings.metadata), "dimension": int(matrix.shape[1]),
            "alignment": {"method": "exact_row_index", "matched_rows": len(frame)},
        },
        "experiment": spec["experiment"], "years": years,
        "announcement_predictions": (
            str(announcement_path) if args.run_mode == "final-test" else None
        ),
        "stock_day_predictions": (
            str(stock_day_path) if args.run_mode == "final-test" else None
        ),
        "artifact_bundle": str(bundle), "experiment_id": spec_id, "results": results,
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
            **({
                "announcement_predictions": file_fingerprint(
                    announcement_path, hash_content=True
                ),
                "stock_day_predictions": file_fingerprint(
                    stock_day_path, hash_content=True
                ),
            } if args.run_mode == "final-test" else {}),
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
