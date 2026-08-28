"""Evaluate stock-day predictions with the alpha_team2 ``simple_states`` protocol.

The model runners emit a long table with one row per ``stock_id``/``entry_date``.
This adapter converts that table to the evaluator's authoritative date-by-ticker
factor grid, writes an auditable YAML configuration, and runs the cloned
``simple_states`` CLI.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


def _exchange_ticker(value: object) -> str | None:
    text = str(value or "").strip().upper()
    match = re.search(r"(\d{6})", text)
    if not match:
        return None
    code = match.group(1)
    exchange = "SH" if code.startswith(("5", "6")) else "SZ"
    return f"{code}.{exchange}"


def build_factor_table(
    predictions: pd.DataFrame,
    *,
    prediction_column: str,
    panel: pd.DataFrame | None = None,
    duplicate_policy: str = "error",
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"entry_date", prediction_column}
    if "stock_id" not in predictions.columns:
        required.add("row_index")
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {', '.join(sorted(missing))}")
    frame = predictions.copy()
    if "stock_id" not in frame.columns:
        if panel is None or "row_index" not in panel.columns or "stock_id" not in panel.columns:
            raise ValueError(
                "predictions without stock_id require a panel containing row_index and stock_id"
            )
        lookup = panel[["row_index", "stock_id"]].drop_duplicates("row_index")
        frame = frame.merge(lookup, on="row_index", how="left", validate="many_to_one")
    frame = frame[["stock_id", "entry_date", prediction_column]].copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["stock_id"].map(_exchange_ticker)
    frame["factor_value"] = pd.to_numeric(frame[prediction_column], errors="coerce")
    frame = frame.dropna(subset=["entry_date", "ticker", "factor_value"])
    duplicate_mask = frame.duplicated(["entry_date", "ticker"], keep=False)
    if duplicate_mask.any():
        if duplicate_policy == "error":
            examples = frame.loc[duplicate_mask].head(5).to_dict(orient="records")
            raise ValueError(
                "predictions must contain at most one row per entry_date/ticker; "
                f"duplicate examples: {examples}"
            )
        if duplicate_policy != "mean":
            raise ValueError(f"unsupported duplicate policy: {duplicate_policy}")
        frame = (
            frame.groupby(["entry_date", "ticker"], as_index=False, sort=False)["factor_value"]
            .mean()
        )
    factor = frame.pivot(index="entry_date", columns="ticker", values="factor_value")
    factor.index.name = "date"
    factor.columns.name = None
    metadata = {
        "input_rows": int(len(predictions)),
        "usable_rows": int(len(frame)),
        "factor_dates": int(len(factor.index)),
        "factor_tickers": int(len(factor.columns)),
        "prediction_column": prediction_column,
        "duplicate_policy": duplicate_policy,
        "date_start": str(factor.index.min().date()) if len(factor) else None,
        "date_end": str(factor.index.max().date()) if len(factor) else None,
    }
    if factor.empty:
        raise ValueError("predictions contain no usable dated numeric values")
    return factor.sort_index(), metadata


def align_factor_to_universe(
    factor: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    prediction_date_role: str = "execution",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Place sparse predictions on the evaluator's observation-date axes.

    The regression runner records ``entry_date`` as the open at which the
    position is entered.  ``simple_states`` expects a factor on observation
    date t and applies the execution constraints/return from t+1 itself.
    Execution-dated predictions must therefore move back one UNIVERSE trading
    day before evaluation.
    """
    if prediction_date_role not in {"observation", "execution"}:
        raise ValueError("prediction_date_role must be observation or execution")
    calendar = pd.DatetimeIndex(pd.to_datetime(universe.index)).normalize()
    if calendar.has_duplicates:
        raise ValueError("UNIVERSE contains duplicated dates")
    columns = pd.Index(universe.columns.astype(str))
    if columns.has_duplicates:
        raise ValueError("UNIVERSE contains duplicated ticker columns")
    original_factor_dates = factor.index
    if prediction_date_role == "execution":
        positions = calendar.get_indexer(original_factor_dates)
        valid = positions > 0
        factor = factor.loc[valid].copy()
        factor.index = calendar[positions[valid] - 1]
        if factor.index.has_duplicates:
            raise ValueError(
                "execution-to-observation mapping produced duplicate factor dates"
            )
    overlapping_dates = factor.index.intersection(calendar)
    overlapping_columns = factor.columns.intersection(columns)
    if overlapping_dates.empty or overlapping_columns.empty:
        raise ValueError("factor and UNIVERSE have no overlapping date/ticker cells")
    evaluation_calendar = calendar[
        (calendar >= overlapping_dates.min()) & (calendar <= overlapping_dates.max())
    ]
    aligned = factor.loc[overlapping_dates, overlapping_columns].reindex(
        index=evaluation_calendar,
        columns=columns,
    )
    metadata = {
        "universe_dates": int(len(calendar)),
        "universe_tickers": int(len(columns)),
        "prediction_date_role": prediction_date_role,
        "factor_date_shift_trading_days": -1 if prediction_date_role == "execution" else 0,
        "input_factor_dates": int(len(original_factor_dates)),
        "dropped_factor_dates": int(len(original_factor_dates) - len(overlapping_dates)),
        "dropped_factor_tickers": int(len(factor.columns.difference(columns))),
        "aligned_factor_dates": int(len(aligned.index)),
        "aligned_factor_tickers": int(len(aligned.columns)),
        "aligned_date_start": str(aligned.index.min().date()),
        "aligned_date_end": str(aligned.index.max().date()),
    }
    return aligned, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--factor-id", required=True)
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument(
        "--prediction-date-role",
        choices=("execution", "observation"),
        default="execution",
        help=(
            "Role of entry_date in the prediction file. Regression outputs are "
            "execution-dated and are shifted back one UNIVERSE trading day."
        ),
    )
    parser.add_argument("--duplicate-policy", choices=("error", "mean"), default="error")
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/processed/cninfo_full_classification_panel_2010_2026.parquet"),
        help="row_index-to-stock lookup for classification prediction files",
    )
    parser.add_argument("--direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--label-lag", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument(
        "--top-fraction", type=float,
        help="select this fraction of valid stocks per day (e.g. 0.2 for Top20%%); overrides top-n",
    )
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--min-amount", type=float, default=20_000_000)
    parser.add_argument("--max-missing-weight", type=float, default=0.05)
    parser.add_argument("--annualization", type=int, default=250)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/alpha_team2/shares/simple_states_data"),
    )
    parser.add_argument(
        "--simple-states-root",
        type=Path,
        default=Path("/mnt/lustre3/home/team/alpha_team2_zengl"),
    )
    parser.add_argument("--without-amount", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()
    if args.label_lag < 0:
        raise ValueError("label-lag must be non-negative")

    predictions = pd.read_parquet(args.predictions)
    panel = None
    if "stock_id" not in predictions.columns:
        panel = pd.read_parquet(args.panel, columns=["row_index", "stock_id"])
    factor, metadata = build_factor_table(
        predictions,
        prediction_column=args.prediction_column,
        panel=panel,
        duplicate_policy=args.duplicate_policy,
    )
    root = args.data_root.resolve()
    universe_path = root / "UNIVERSE.parquet"
    universe = pd.read_parquet(universe_path)
    factor, alignment = align_factor_to_universe(
        factor, universe, prediction_date_role=args.prediction_date_role
    )
    metadata.update(alignment)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_path = output_dir / f"{args.factor_id}.factor.parquet"
    factor.to_parquet(factor_path, index=True)
    stats_path = output_dir / f"{args.factor_id}.stats.csv"
    config = {
        "factor": str(factor_path),
        "return": str(root / "ret_vv.parquet"),
        "riseboard": str(root / "riseboard.parquet"),
        "jumpboard": str(root / "jumpboard.parquet"),
        "universe": str(universe_path),
        "factor_id": args.factor_id,
        "direction": args.direction,
        "label_lag": args.label_lag,
        "top_n": args.top_n,
        "top_fraction": args.top_fraction,
        "min_count": args.min_count,
        "min_amount": args.min_amount,
        "max_missing_weight": args.max_missing_weight,
        "annualization": args.annualization,
        "output": str(stats_path),
    }
    if not args.without_amount:
        config["amount"] = str(root / "amount.parquet")
    # Restrict a prediction file to its own observed window by default.  This
    # prevents the evaluator's full UNIVERSE calendar from turning absent
    # prediction dates into artificial zero-exposure days.
    config["start"] = args.start or metadata["aligned_date_start"]
    config["end"] = args.end or metadata["aligned_date_end"]
    config_path = output_dir / f"{args.factor_id}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    metadata.update({
        "factor": str(factor_path),
        "config": str(config_path),
        "stats": str(stats_path),
    })
    (output_dir / f"{args.factor_id}.input.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.skip_run:
        print(json.dumps(metadata, ensure_ascii=False))
        return

    command = [
        sys.executable,
        "-m",
        "simple_states",
        "--yaml",
        str(config_path),
    ]
    result = subprocess.run(command, cwd=args.simple_states_root, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
