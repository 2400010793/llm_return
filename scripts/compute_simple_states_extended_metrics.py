"""Add transparent absolute and same-pool metrics to a simple_states result.

The platform RetL/RetS definitions are left untouched.  This companion
report computes absolute returns for the news pool and its prediction Top-N
from the same out-of-sample stock-day predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("predictions", type=Path)
    p.add_argument("stats", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prediction-column", default="prediction")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--annualization", type=int, default=250)
    args = p.parse_args()
    if args.top_n < 1:
        raise ValueError("top-n must be positive")
    d = pd.read_parquet(args.predictions)
    required = {"entry_date", "actual_return", args.prediction_column}
    missing = required.difference(d.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    d = d.copy()
    d["entry_date"] = pd.to_datetime(d["entry_date"], errors="coerce").dt.normalize()
    d["actual_return"] = pd.to_numeric(d["actual_return"], errors="coerce")
    d[args.prediction_column] = pd.to_numeric(d[args.prediction_column], errors="coerce")
    d = d.dropna(subset=["entry_date", "actual_return", args.prediction_column])
    rows: list[dict[str, object]] = []
    for date, g in d.groupby("entry_date", sort=True):
        if len(g) < args.min_count:
            continue
        n = min(args.top_n, len(g))
        top = g.nlargest(n, args.prediction_column)
        pool = float(g["actual_return"].mean())
        top_ret = float(top["actual_return"].mean())
        rows.append({
            "entry_date": date,
            "pool_abs_return": pool,
            "top_n_abs_return": top_ret,
            "top_n_excess_vs_pool": top_ret - pool,
            "pool_count": int(len(g)),
            "top_n_count": int(n),
        })
    yearly = pd.DataFrame(rows)
    if yearly.empty:
        raise ValueError("no usable prediction rows")
    yearly["year"] = yearly["entry_date"].dt.year
    annual = yearly.groupby("year", as_index=False).agg(
        signal_days=("entry_date", "size"),
        pool_abs_return=("pool_abs_return", "mean"),
        top_n_abs_return=("top_n_abs_return", "mean"),
        top_n_excess_vs_pool=("top_n_excess_vs_pool", "mean"),
    )
    annual["pool_abs_annualized"] = annual.pool_abs_return * args.annualization
    annual["top_n_abs_annualized"] = annual.top_n_abs_return * args.annualization
    annual["top_n_excess_annualized"] = annual.top_n_excess_vs_pool * args.annualization
    summary = {
        "signal_days": int(len(yearly)),
        "pool_abs_return_annualized": float(yearly.pool_abs_return.mean() * args.annualization),
        "top_n_abs_return_annualized": float(yearly.top_n_abs_return.mean() * args.annualization),
        "top_n_excess_vs_pool_annualized": float(yearly.top_n_excess_vs_pool.mean() * args.annualization),
        "positive_excess_years": int((annual.top_n_excess_vs_pool > 0).sum()),
        "years": int(len(annual)),
        "top_n": args.top_n,
        "annualization": args.annualization,
    }
    stats = pd.read_csv(args.stats)
    for key, value in summary.items():
        stats[key] = value
    args.output_dir.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(args.output_dir / "same_pool_yearly.csv", index=False)
    annual.to_csv(args.output_dir / "same_pool_annual.csv", index=False)
    stats.to_csv(args.output_dir / "extended_stats.csv", index=False)
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
