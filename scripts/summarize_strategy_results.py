"""Combine strategy summary JSON files into an execution-comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenario = path.parent.name
    rows = []
    for label, run in payload.get("runs", {}).items():
        metrics = run["metrics"]
        gross, net, execution = metrics["gross"], metrics["net"], metrics["execution"]
        rows.append({
            "scenario": scenario,
            "gamma": run["config"]["ewct_gamma"],
            "n_days": run["n_days"],
            "gross_mean_bps": gross["mean"] * 10_000,
            "gross_sharpe": gross["sharpe"],
            "net_mean_bps": net["mean"] * 10_000,
            "net_sharpe": net["sharpe"],
            "net_mean_t_stat": net["mean_t_stat"],
            "net_geometric_annual_return": net["geometric_annualized_return"],
            "net_sortino": net["sortino"],
            "net_cumulative_return": net["cumulative_return"],
            "net_max_drawdown": net["max_drawdown"],
            "positive_day_rate": net["positive_day_rate"],
            "turnover_pct": execution["mean_daily_turnover"] * 100,
            "cost_bps": (
                execution["mean_daily_transaction_cost"]
                + execution["mean_daily_borrow_cost"]
            ) * 10_000,
            "blocked_notional_pct": execution["mean_blocked_notional"] * 100,
            "source": str(path),
            "run": label,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for path in args.summaries for row in _rows(path)]
    if not rows:
        raise ValueError("summary files contain no strategy runs")
    table = pd.DataFrame(rows).sort_values(["scenario", "gamma"]).reset_index(drop=True)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    table.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(table.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "rows": len(table)}))


if __name__ == "__main__":
    main()
