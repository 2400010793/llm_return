"""Compute direct CNINFO stock-pool and ex-post Top20% return benchmarks."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def summarize(panel: pd.DataFrame, target: str, start: int, end: int) -> pd.DataFrame:
    frame = panel[["stock_id", "entry_date", target]].copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce").dt.normalize()
    frame["year"] = frame.entry_date.dt.year
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame[frame.year.between(start, end)].dropna(subset=["stock_id", "entry_date", target])
    stock_day = frame.groupby(["stock_id", "entry_date"], as_index=False)[target].mean()
    stock_day["year"] = stock_day.entry_date.dt.year
    rows = []
    for year, group in stock_day.groupby("year", sort=True):
        daily = []
        for date, day in group.groupby("entry_date"):
            n = max(1, int(np.ceil(len(day) * 0.20)))
            ordered = day.sort_values(target)
            daily.append({
                "entry_date": date,
                "pool_mean": day[target].mean(),
                "expost_top20": ordered.tail(n)[target].mean(),
                "expost_bottom20": ordered.head(n)[target].mean(),
                "stocks": len(day),
            })
        daily = pd.DataFrame(daily)
        rows.append({
            "target": target,
            "year": int(year),
            "stock_days": int(len(group)),
            "trading_days": int(len(daily)),
            "mean_stocks_per_day": float(daily.stocks.mean()),
            "pool_mean_bp": float(daily.pool_mean.mean() * 1e4),
            "expost_top20_bp": float(daily.expost_top20.mean() * 1e4),
            "expost_bottom20_bp": float(daily.expost_bottom20.mean() * 1e4),
            "expost_long_short_bp": float((daily.expost_top20 - daily.expost_bottom20).mean() * 1e4),
        })
    return pd.DataFrame(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, default=Path("data/processed/cninfo_full_classification_panel_2010_2026.parquet"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start-year", type=int, default=2018)
    p.add_argument("--end-year", type=int, default=2026)
    args = p.parse_args()
    panel = pd.read_parquet(args.panel, columns=["stock_id", "entry_date", "next_day_return", "event_return_3d"])
    result = pd.concat([
        summarize(panel, "next_day_return", args.start_year, args.end_year),
        summarize(panel, "event_return_3d", args.start_year, args.end_year),
    ], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"saved {len(result)} rows to {args.output}")

if __name__ == "__main__":
    main()
