"""Summarize token-embedding EWCT runs and their calendar-year stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sharpe(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 2 or values.std(ddof=1) == 0:
        return float("nan")
    return float(np.sqrt(252) * values.mean() / values.std(ddof=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    for summary_path in sorted(args.root.glob("*/*/*/summary.json")):
        candidate, cost, mode = summary_path.parts[-4:-1]
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for run_name, run in payload["runs"].items():
            metrics = run["metrics"]
            gross = metrics["gross"]
            net = metrics["net"]
            execution = metrics["execution"]
            gamma = float(run["config"]["ewct_gamma"])
            rows.append({
                "candidate": candidate,
                "cost": cost,
                "mode": mode,
                "gamma": gamma,
                "n_days": run["n_days"],
                "gross_sharpe": gross["sharpe"],
                "net_sharpe": net["sharpe"],
                "gross_mean_bps": gross["mean"] * 10_000,
                "net_mean_bps": net["mean"] * 10_000,
                "net_annual_return": net["annualized_return"],
                "net_geometric_annual_return": net["geometric_annualized_return"],
                "net_cumulative_return": net["cumulative_return"],
                "net_max_drawdown": net["max_drawdown"],
                "turnover": execution["mean_daily_turnover"],
                "cost_bps_per_day": execution["mean_daily_transaction_cost"] * 10_000,
                "source": str(summary_path),
            })

            daily_path = Path(run["artifacts"]["daily"])
            daily = pd.read_parquet(daily_path)
            date_column = "entry_date" if "entry_date" in daily else "date"
            daily["year"] = pd.to_datetime(daily[date_column]).dt.year
            for year, frame in daily.groupby("year", sort=True):
                yearly_rows.append({
                    "candidate": candidate,
                    "cost": cost,
                    "mode": mode,
                    "gamma": gamma,
                    "year": int(year),
                    "n_days": len(frame),
                    "gross_sharpe": _sharpe(frame["gross_return"]),
                    "net_sharpe": _sharpe(frame["net_return"]),
                    "gross_mean_bps": frame["gross_return"].mean() * 10_000,
                    "net_mean_bps": frame["net_return"].mean() * 10_000,
                })

    if not rows:
        raise ValueError(f"no completed parent summaries under {args.root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_runs = pd.DataFrame(rows).sort_values(["mode", "cost", "candidate", "gamma"])
    yearly = pd.DataFrame(yearly_rows).sort_values(
        ["mode", "cost", "candidate", "gamma", "year"]
    )
    fixed = all_runs[np.isclose(all_runs["gamma"], 0.1)].copy()
    best = (
        all_runs.sort_values("net_sharpe", ascending=False)
        .groupby(["candidate", "cost", "mode"], as_index=False)
        .first()
        .sort_values(["mode", "cost", "net_sharpe"], ascending=[True, True, False])
    )
    all_runs.to_csv(args.output_dir / "all_gamma_results.csv", index=False)
    fixed.to_csv(args.output_dir / "fixed_gamma_0p10.csv", index=False)
    best.to_csv(args.output_dir / "ex_post_best_gamma.csv", index=False)
    yearly.to_csv(args.output_dir / "yearly_results.csv", index=False)
    print(json.dumps({
        "all_runs": len(all_runs),
        "fixed": len(fixed),
        "best": len(best),
        "yearly": len(yearly),
        "output_dir": str(args.output_dir),
    }))


if __name__ == "__main__":
    main()
