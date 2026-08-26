"""Run daily Diebold-Mariano and Model Confidence Set comparisons.

Examples
--------
Classification prediction files produced by the pooled runner do not contain a
stock identifier, so ``--panel`` is required to recover it from ``row_index``::

    python scripts/run_forecast_comparison.py --task classification \
      --panel data/processed/cninfo_full_classification_panel.parquet \
      --prediction logistic=path/to/logistic.predictions.parquet \
      --prediction mlp=path/to/mlp.predictions.parquet \
      --output-dir reports/forecast_comparison/classification_next1

Regression stock-day files already contain stock and date identifiers::

    python scripts/run_forecast_comparison.py --task regression \
      --prediction ridge=path/to/ridge.stock_day_predictions.parquet \
      --prediction huber=path/to/huber.stock_day_predictions.parquet \
      --output-dir reports/forecast_comparison/regression_next1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.forecast_comparison import (
    classification_losses,
    daily_average_loss,
    model_confidence_set,
    pairwise_dm_tests,
    regression_losses,
)


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"prediction must use MODEL=PATH syntax: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise ValueError(f"model names must be non-empty and unique: {name!r}")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = path
    if len(result) < 2:
        raise ValueError("at least two --prediction values are required")
    return result


def _consistent_group_actuals(frame: pd.DataFrame, keys: list[str], actual_column: str) -> None:
    spread = frame.groupby(keys, sort=False, observed=True)[actual_column].agg(
        lambda values: float(np.nanmax(values) - np.nanmin(values))
    )
    bad = spread[spread > 1e-12]
    if not bad.empty:
        raise ValueError(f"inconsistent actual values within stock-day; examples={bad.index[:5].tolist()}")


def load_classification_predictions(
    predictions: dict[str, Path],
    panel_path: Path,
    *,
    row_column: str,
    stock_column: str,
    date_column: str,
    actual_column: str,
    prediction_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel = pd.read_parquet(panel_path, columns=[row_column, stock_column])
    if panel[row_column].duplicated().any():
        raise ValueError(f"panel {row_column} must be unique")
    aligned: pd.DataFrame | None = None
    audit: dict[str, Any] = {}
    for name, path in predictions.items():
        raw = pd.read_parquet(path)
        required = {row_column, date_column, actual_column, prediction_column}
        missing = required.difference(raw.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
        values = raw[[row_column, date_column, actual_column, prediction_column]].copy()
        values = values.merge(panel, on=row_column, how="inner", validate="many_to_one")
        values[date_column] = pd.to_datetime(values[date_column], errors="coerce")
        values[actual_column] = pd.to_numeric(values[actual_column], errors="coerce")
        values[prediction_column] = pd.to_numeric(values[prediction_column], errors="coerce")
        values = values.dropna(subset=[stock_column, date_column, actual_column, prediction_column])
        _consistent_group_actuals(values, [stock_column, date_column], actual_column)
        stock_day = values.groupby([stock_column, date_column], sort=True, observed=True).agg(
            actual=(actual_column, "first"), prediction=(prediction_column, "mean"),
            announcements=(prediction_column, "size"),
        ).reset_index()
        stock_day = stock_day.rename(columns={"prediction": name, "announcements": f"{name}__announcements"})
        audit[name] = {
            "path": str(path), "input_rows": int(len(raw)),
            "matched_rows": int(len(values)), "stock_days": int(len(stock_day)),
        }
        if aligned is None:
            aligned = stock_day.rename(columns={"actual": "actual_return"})
        else:
            candidate = stock_day.rename(columns={"actual": f"actual__{name}"})
            aligned = aligned.merge(candidate, on=[stock_column, date_column], how="inner", validate="one_to_one")
            other = aligned.pop(f"actual__{name}").to_numpy(dtype=float)
            reference = aligned["actual_return"].to_numpy(dtype=float)
            if not np.allclose(reference, other, rtol=0.0, atol=1e-12):
                raise ValueError(f"actual returns disagree after alignment for model {name}")
    assert aligned is not None
    return aligned.sort_values([date_column, stock_column], kind="stable").reset_index(drop=True), audit


def load_regression_predictions(
    predictions: dict[str, Path],
    *,
    stock_column: str,
    date_column: str,
    actual_column: str,
    prediction_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    aligned: pd.DataFrame | None = None
    audit: dict[str, Any] = {}
    for name, path in predictions.items():
        raw = pd.read_parquet(path)
        required = {stock_column, date_column, actual_column, prediction_column}
        missing = required.difference(raw.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
        values = raw[list(required)].copy()
        values[date_column] = pd.to_datetime(values[date_column], errors="coerce")
        values[actual_column] = pd.to_numeric(values[actual_column], errors="coerce")
        values[prediction_column] = pd.to_numeric(values[prediction_column], errors="coerce")
        values = values.dropna(subset=[stock_column, date_column, actual_column, prediction_column])
        _consistent_group_actuals(values, [stock_column, date_column], actual_column)
        stock_day = values.groupby([stock_column, date_column], sort=True, observed=True).agg(
            actual=(actual_column, "first"), prediction=(prediction_column, "mean"),
        ).reset_index().rename(columns={"prediction": name})
        audit[name] = {"path": str(path), "input_rows": int(len(raw)), "stock_days": int(len(stock_day))}
        if aligned is None:
            aligned = stock_day.rename(columns={"actual": "actual_return"})
        else:
            candidate = stock_day.rename(columns={"actual": f"actual__{name}"})
            aligned = aligned.merge(candidate, on=[stock_column, date_column], how="inner", validate="one_to_one")
            other = aligned.pop(f"actual__{name}").to_numpy(dtype=float)
            reference = aligned["actual_return"].to_numpy(dtype=float)
            if not np.allclose(reference, other, rtol=0.0, atol=1e-12):
                raise ValueError(f"actual returns disagree after alignment for model {name}")
    assert aligned is not None
    return aligned.sort_values([date_column, stock_column], kind="stable").reset_index(drop=True), audit


def build_daily_loss_panels(
    aligned: pd.DataFrame,
    models: list[str],
    *,
    task: str,
    date_column: str,
    threshold: float,
    huber_delta: float,
    min_stocks: int,
) -> dict[str, pd.DataFrame]:
    additive: dict[str, dict[str, pd.Series]] = {}
    for model in models:
        if task == "classification":
            losses = classification_losses(aligned["actual_return"], aligned[model], threshold=threshold)
        else:
            losses = regression_losses(aligned["actual_return"], aligned[model], huber_delta=huber_delta)
        for loss_name in losses:
            additive.setdefault(loss_name, {})[model] = daily_average_loss(
                aligned[date_column], losses[loss_name], name=model,
            )

    panels = {
        loss_name: pd.concat(series_by_model.values(), axis=1, join="inner").dropna()
        for loss_name, series_by_model in additive.items()
    }
    if task == "regression":
        rank_series: dict[str, pd.Series] = {}
        for model in models:
            values: dict[pd.Timestamp, float] = {}
            for date, group in aligned.groupby(date_column, sort=True, observed=True):
                if len(group) < min_stocks:
                    continue
                correlation = group["actual_return"].corr(group[model], method="spearman")
                if pd.notna(correlation):
                    values[pd.Timestamp(date)] = -float(correlation)
            rank_series[model] = pd.Series(values, name=model, dtype=float)
        panels["negative_rank_ic"] = pd.concat(rank_series.values(), axis=1, join="inner").dropna()
    return panels


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("classification", "regression"), required=True)
    parser.add_argument("--prediction", action="append", required=True, help="MODEL=PATH; repeat for each model")
    parser.add_argument("--panel", type=Path, default=None)
    parser.add_argument("--row-column", default="row_index")
    parser.add_argument("--stock-column", default="stock_id")
    parser.add_argument("--date-column", default="entry_date")
    parser.add_argument("--actual-column", default=None)
    parser.add_argument("--prediction-column", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--huber-delta", type=float, default=0.01)
    parser.add_argument("--min-stocks", type=int, default=5)
    parser.add_argument("--dm-lag", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--mcs-alpha", type=float, default=0.10)
    parser.add_argument("--mcs-reps", type=int, default=5_000)
    parser.add_argument("--block-lengths", default="5,10,20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    predictions = parse_named_paths(args.prediction)
    models = list(predictions)
    if args.task == "classification":
        if args.panel is None:
            raise ValueError("classification comparison requires --panel")
        actual_column = args.actual_column or "next_day_return"
        prediction_column = args.prediction_column or "probability"
        aligned, input_audit = load_classification_predictions(
            predictions, args.panel, row_column=args.row_column,
            stock_column=args.stock_column, date_column=args.date_column,
            actual_column=actual_column, prediction_column=prediction_column,
        )
    else:
        actual_column = args.actual_column or "actual_return"
        prediction_column = args.prediction_column or "prediction"
        aligned, input_audit = load_regression_predictions(
            predictions, stock_column=args.stock_column, date_column=args.date_column,
            actual_column=actual_column, prediction_column=prediction_column,
        )

    panels = build_daily_loss_panels(
        aligned, models, task=args.task, date_column=args.date_column,
        threshold=args.threshold, huber_delta=args.huber_delta,
        min_stocks=args.min_stocks,
    )
    block_lengths = sorted(set(int(value) for value in args.block_lengths.split(",") if value.strip()))
    if not block_lengths or any(value < 1 for value in block_lengths):
        raise ValueError("block lengths must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = args.output_dir / "aligned_stock_day_predictions.parquet"
    aligned.to_parquet(aligned_path, index=False)
    report: dict[str, Any] = {
        "format_version": "daily_forecast_comparison_v1",
        "task": args.task,
        "models": models,
        "input_audit": input_audit,
        "alignment": {
            "stock_days": int(len(aligned)),
            "dates": int(aligned[args.date_column].nunique()),
            "date_start": str(aligned[args.date_column].min().date()),
            "date_end": str(aligned[args.date_column].max().date()),
            "path": str(aligned_path),
        },
        "settings": {
            "dm_lag": args.dm_lag, "horizon": args.horizon,
            "mcs_alpha": args.mcs_alpha, "mcs_reps": args.mcs_reps,
            "block_lengths": block_lengths, "seed": args.seed,
            "threshold": args.threshold if args.task == "classification" else None,
            "huber_delta": args.huber_delta if args.task == "regression" else None,
        },
        "losses": {},
    }
    for loss_name, panel in panels.items():
        if len(panel) <= args.dm_lag:
            raise ValueError(f"{loss_name} has too few dates for DM lag {args.dm_lag}")
        loss_path = args.output_dir / f"daily_loss_{loss_name}.parquet"
        panel.rename_axis("date").reset_index().to_parquet(loss_path, index=False)
        dm = pairwise_dm_tests(panel, hac_lag=args.dm_lag, horizon=args.horizon)
        dm_path = args.output_dir / f"dm_{loss_name}.csv"
        dm.to_csv(dm_path, index=False)
        mcs_results: dict[str, Any] = {}
        for block_length in block_lengths:
            if block_length > len(panel):
                continue
            for method in ("range", "max"):
                key = f"{method}_block_{block_length}"
                mcs_results[key] = model_confidence_set(
                    panel, alpha=args.mcs_alpha, bootstrap_reps=args.mcs_reps,
                    block_length=block_length, seed=args.seed, method=method,
                )
        report["losses"][loss_name] = {
            "n_dates": int(len(panel)),
            "mean_losses": {name: float(panel[name].mean()) for name in models},
            "daily_loss_path": str(loss_path), "dm_path": str(dm_path),
            "mcs": mcs_results,
        }
    report_path = args.output_dir / "comparison.json"
    report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "models": models, "losses": list(panels)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
