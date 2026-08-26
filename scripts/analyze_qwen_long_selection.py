"""Decompose Qwen long selection into stock picking and coverage effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.portfolio.performance import portfolio_metrics


def select_count(rule: str, size: int) -> int:
    if rule == "top20pct":
        return max(1, int(np.ceil(size * 0.20))) if size >= 5 else 0
    count = int(rule.removeprefix("top"))
    return count if size >= 2 * count else 0


def daily_decomposition(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    *,
    prediction_column: str,
    return_column: str,
    rule: str,
) -> pd.DataFrame:
    required_predictions = {"entry_date", "stock_id", prediction_column}
    required_market = {"entry_date", "stock_id", return_column, "eligible_signal"}
    if missing := required_predictions.difference(predictions.columns):
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    if missing := required_market.difference(market.columns):
        raise ValueError(f"market missing columns: {sorted(missing)}")

    scores = predictions[["entry_date", "stock_id", prediction_column]].copy()
    scores["entry_date"] = pd.to_datetime(scores["entry_date"]).dt.normalize()
    scores["stock_id"] = scores["stock_id"].astype(str).str.extract(r"(\d{6})", expand=False)
    scores[prediction_column] = pd.to_numeric(scores[prediction_column], errors="coerce")
    if scores.duplicated(["entry_date", "stock_id"]).any():
        raise ValueError("predictions must contain one row per stock-day")

    returns = market[["entry_date", "stock_id", return_column, "eligible_signal"]].copy()
    returns["entry_date"] = pd.to_datetime(returns["entry_date"]).dt.normalize()
    returns["stock_id"] = returns["stock_id"].astype(str).str.extract(r"(\d{6})", expand=False)
    returns[return_column] = pd.to_numeric(returns[return_column], errors="coerce")
    returns = returns[returns["eligible_signal"].fillna(False) & returns[return_column].notna()]
    if returns.duplicated(["entry_date", "stock_id"]).any():
        raise ValueError("market must contain one row per stock-day")

    universe = returns.groupby("entry_date")[return_column].agg(
        universe_return="mean", universe_count="size"
    )
    candidates = scores.merge(
        returns[["entry_date", "stock_id", return_column]],
        on=["entry_date", "stock_id"], how="inner", validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for date, group in candidates.groupby("entry_date", sort=True):
        group = group.dropna(subset=[prediction_column, return_column]).sort_values(
            [prediction_column, "stock_id"], kind="mergesort"
        )
        count = select_count(rule, len(group))
        if count == 0 or date not in universe.index:
            continue
        selected = group.tail(count)
        bottom = group.head(count)
        top_return = float(selected[return_column].mean())
        pool_return = float(group[return_column].mean())
        universe_return = float(universe.at[date, "universe_return"])
        rows.append({
            "entry_date": date,
            "candidate_count": len(group),
            "selected_count": count,
            "universe_count": int(universe.at[date, "universe_count"]),
            "selected_return": top_return,
            "bottom_return": float(bottom[return_column].mean()),
            "coverage_return": pool_return,
            "universe_return": universe_return,
            "within_pool_excess": top_return - pool_return,
            "coverage_gap": pool_return - universe_return,
            "full_universe_excess": top_return - universe_return,
            "long_short_spread": top_return - float(bottom[return_column].mean()),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(f"no eligible dates for {rule}")
    identity_error = (
        result["within_pool_excess"] + result["coverage_gap"]
        - result["full_universe_excess"]
    ).abs().max()
    if identity_error > 1e-12:
        raise AssertionError(f"return decomposition identity failed: {identity_error}")
    return result


def block_bootstrap_mean(values: pd.Series, *, repetitions: int = 2000, block: int = 20) -> dict[str, float]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(array):
        return {"ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(42)
    starts = rng.integers(0, len(array), size=(repetitions, int(np.ceil(len(array) / block))))
    offsets = np.arange(block)
    positions = ((starts[..., None] + offsets) % len(array)).reshape(repetitions, -1)[:, :len(array)]
    means = array[positions].mean(axis=1)
    return {"ci_low": float(np.quantile(means, 0.025)), "ci_high": float(np.quantile(means, 0.975))}


def summarize(frame: pd.DataFrame, *, dataset: str, model: str, rule: str, return_column: str) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset, "model": model, "rule": rule, "return_column": return_column,
        "start": frame.entry_date.min(), "end": frame.entry_date.max(),
        "days": len(frame), "mean_candidates": frame.candidate_count.mean(),
        "mean_selected": frame.selected_count.mean(),
    }
    for column in (
        "selected_return", "coverage_return", "universe_return", "within_pool_excess",
        "coverage_gap", "full_universe_excess", "long_short_spread",
    ):
        metrics = portfolio_metrics(frame, return_column=column)
        row[f"{column}_mean_daily_bps"] = metrics["mean"] * 10_000
        row[f"{column}_annualized"] = metrics["annualized_return"]
        row[f"{column}_sharpe"] = metrics["sharpe"]
        row[f"{column}_positive_days"] = metrics["positive_day_rate"]
        ci = block_bootstrap_mean(frame[column])
        row[f"{column}_annualized_ci_low"] = ci["ci_low"] * 252
        row[f"{column}_annualized_ci_high"] = ci["ci_high"] * 252
        row[f"{column}_mean_daily_bps_ci_low"] = ci["ci_low"] * 10_000
        row[f"{column}_mean_daily_bps_ci_high"] = ci["ci_high"] * 10_000
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, sep="\t")
    required = {"dataset", "model", "predictions", "prediction_column"}
    if missing := required.difference(manifest.columns):
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    market = pd.read_parquet(args.market)
    args.output_root.mkdir(parents=True, exist_ok=True)
    daily_parts, summary_rows, yearly_rows = [], [], []
    for spec in manifest.to_dict(orient="records"):
        predictions = pd.read_parquet(spec["predictions"])
        for return_column in ("open_to_open_return", "tradable_open_to_open_return"):
            for rule in ("top20pct", "top5", "top10"):
                daily = daily_decomposition(
                    predictions, market, prediction_column=spec["prediction_column"],
                    return_column=return_column, rule=rule,
                )
                daily.insert(0, "return_column", return_column)
                daily.insert(0, "rule", rule)
                daily.insert(0, "model", spec["model"])
                daily.insert(0, "dataset", spec["dataset"])
                daily_parts.append(daily)
                summary_rows.append(summarize(
                    daily, dataset=spec["dataset"], model=spec["model"],
                    rule=rule, return_column=return_column,
                ))
                for year, part in daily.groupby(daily.entry_date.dt.year):
                    yearly_rows.append({
                        **summarize(part, dataset=spec["dataset"], model=spec["model"],
                                    rule=rule, return_column=return_column),
                        "test_year": int(year),
                    })
    all_daily = pd.concat(daily_parts, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    all_daily.to_parquet(args.output_root / "daily_decomposition.parquet", index=False)
    summary.to_csv(args.output_root / "overall_metrics.csv", index=False)
    yearly.to_csv(args.output_root / "yearly_metrics.csv", index=False)
    audit = {
        "models": len(manifest), "daily_rows": len(all_daily),
        "overall_rows": len(summary), "yearly_rows": len(yearly),
        "identity_max_error": float((all_daily.within_pool_excess + all_daily.coverage_gap
                                     - all_daily.full_universe_excess).abs().max()),
    }
    (args.output_root / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
