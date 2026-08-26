"""Build O2O residual and cross-sectionally normalized training targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.o2o_residual_targets import (
    CS_ZSCORE_COLUMN,
    MARKET_RESIDUAL_COLUMN,
    MARKET_RETURN_COLUMN,
    WINSOR_RESIDUAL_COLUMN,
    attach_o2o_residual_targets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--market-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--winsor-lower", type=float, default=0.01)
    parser.add_argument("--winsor-upper", type=float, default=0.99)
    parser.add_argument("--min-stocks-per-day", type=int, default=5)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    market = pd.read_parquet(args.market_data)
    result, audit = attach_o2o_residual_targets(
        panel, market, winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
        min_stocks_per_day=args.min_stocks_per_day,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    dates = pd.to_datetime(result["entry_date"], errors="coerce")
    targets = [
        "next_day_open_to_open_return", MARKET_RESIDUAL_COLUMN,
        CS_ZSCORE_COLUMN, WINSOR_RESIDUAL_COLUMN,
    ]
    summary = {
        "format_version": "o2o_residual_panel_v1",
        "input_panel": str(args.panel),
        "market_data": str(args.market_data),
        "output": str(args.output),
        "market_return_column": MARKET_RETURN_COLUMN,
        "target_columns": targets,
        "audit": audit,
        "yearly_finite": {
            str(int(year)): {
                target: int(group[target].notna().sum()) for target in targets
            }
            for year, group in result.assign(_year=dates.dt.year).groupby("_year")
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
