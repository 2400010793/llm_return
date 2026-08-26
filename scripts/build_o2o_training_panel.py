"""Attach strict next-open-to-next-open targets to an announcement panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.o2o_targets import (
    DEFAULT_LABEL_COLUMN,
    DEFAULT_TARGET_COLUMN,
    attach_o2o_targets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--market-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stock-column", default="stock_id")
    parser.add_argument("--date-column", default="entry_date")
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN)
    parser.add_argument(
        "--market-return-column", default="tradable_open_to_open_return"
    )
    parser.add_argument("--market-eligible-column", default="eligible_signal")
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    market = pd.read_parquet(args.market_data)
    result = attach_o2o_targets(
        panel,
        market,
        stock_column=args.stock_column,
        date_column=args.date_column,
        market_return_column=args.market_return_column,
        market_eligible_column=args.market_eligible_column,
        target_column=args.target_column,
        label_column=args.label_column,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)

    dates = pd.to_datetime(result[args.date_column], errors="coerce")
    yearly = result.assign(_year=dates.dt.year).groupby("_year", dropna=True).agg(
        rows=(args.stock_column, "size"),
        eligible=("o2o_target_eligible", "sum"),
        stocks=(args.stock_column, "nunique"),
    )
    finite = pd.to_numeric(result[args.target_column], errors="coerce").notna()
    summary = {
        "format_version": "o2o_training_panel_v1",
        "input_panel": str(args.panel),
        "market_data": str(args.market_data),
        "output": str(args.output),
        "rows": int(len(result)),
        "stocks": int(result[args.stock_column].nunique()),
        "target_column": args.target_column,
        "label_column": args.label_column,
        "target_definition": (
            "adjusted observed open(entry_date + 1 exchange day) / "
            "adjusted observed open(entry_date) - 1"
        ),
        "finite_targets": int(finite.sum()),
        "coverage": float(finite.mean()),
        "positive_rate": float(result.loc[finite, args.label_column].mean()),
        "yearly": {
            str(int(year)): {
                "rows": int(values["rows"]),
                "eligible": int(values["eligible"]),
                "coverage": float(values["eligible"] / values["rows"]),
                "stocks": int(values["stocks"]),
            }
            for year, values in yearly.iterrows()
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
