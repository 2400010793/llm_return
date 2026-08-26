"""Run the paper-style net-of-cost news strategy on stock-day predictions.

Examples
--------
Paper one-day strategy using the large-stock 10 bps fallback::

    python scripts/run_portfolio_strategy.py predictions.parquet \
      --output-dir reports/strategy/example --gammas 1.0 --cost-model paper

EWCT requires a dense stock-day return panel for old positions::

    python scripts/run_portfolio_strategy.py predictions.parquet \
      --market-data daily_market.parquet --market-return-column open_to_open_return \
      --market-cap-column market_cap --gammas 0.1,0.2,0.3,0.4,0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.portfolio import (
    StrategyConfig,
    TransactionCostModel,
    backtest_ranked_strategy,
    strategy_metrics,
)


def _cost_model(args: argparse.Namespace) -> TransactionCostModel:
    if args.cost_model == "none":
        return TransactionCostModel(name="none")
    if args.cost_model == "paper":
        return TransactionCostModel.paper(
            market_cap_column=args.market_cap_column,
            small_stock_column=args.small_stock_column,
            strict_size=args.strict_paper_size,
            short_borrow_bps_annual=args.short_borrow_bps_annual,
        )
    return TransactionCostModel.china_a_share(
        commission_bps=args.commission_bps,
        stamp_duty_bps=args.stamp_duty_bps,
        slippage_bps=args.slippage_bps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--market-data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--return-column", default="actual_return")
    parser.add_argument("--market-return-column")
    parser.add_argument("--date-column", default="entry_date")
    parser.add_argument("--asset-column", default="stock_id")
    parser.add_argument("--market-cap-column")
    parser.add_argument("--small-stock-column")
    parser.add_argument("--can-buy-column")
    parser.add_argument("--can-sell-column")
    parser.add_argument("--shortable-column")
    parser.add_argument("--eligible-column")
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--min-stocks-per-day", type=int, default=5)
    parser.add_argument("--weighting", choices=("equal", "value"), default="equal")
    parser.add_argument("--gammas", default="1.0")
    side = parser.add_mutually_exclusive_group()
    side.add_argument("--long-only", action="store_true")
    side.add_argument("--short-only", action="store_true")
    parser.add_argument("--missing-return", choices=("raise", "zero"), default="raise")
    parser.add_argument(
        "--cost-model", choices=("paper", "china-a", "none"), default="paper"
    )
    parser.add_argument("--strict-paper-size", action="store_true")
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--stamp-duty-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--short-borrow-bps-annual", type=float, default=0.0)
    args = parser.parse_args()

    gammas = [float(value) for value in args.gammas.split(",") if value.strip()]
    if not gammas or any(not 0 < value <= 1 for value in gammas):
        raise ValueError("gammas must contain values in (0, 1]")
    if len(set(gammas)) != len(gammas):
        raise ValueError("gammas must not contain duplicates")
    predictions = pd.read_parquet(args.predictions)
    market = pd.read_parquet(args.market_data) if args.market_data else None
    cost_model = _cost_model(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "inputs": {
            "predictions": str(args.predictions),
            "market_data": str(args.market_data) if args.market_data else None,
            "prediction_rows": len(predictions),
            "market_rows": len(market) if market is not None else None,
        },
        "columns": {
            "prediction": args.prediction_column,
            "return": args.return_column,
            "market_return": args.market_return_column,
            "date": args.date_column,
            "asset": args.asset_column,
            "market_cap": args.market_cap_column,
            "small_stock": args.small_stock_column,
            "can_buy": args.can_buy_column,
            "can_sell": args.can_sell_column,
            "shortable": args.shortable_column,
            "eligible": args.eligible_column,
        },
        "cost_model": asdict(cost_model),
        "runs": {},
    }
    if args.cost_model == "paper" and not (
        args.market_cap_column or args.small_stock_column
    ):
        report["cost_size_note"] = (
            "No size field was supplied; the paper model's 10 bps large-stock "
            "rate was applied to every trade. Supply market-cap or is-small data "
            "for the paper's 10/20 bps split."
        )

    for gamma in gammas:
        config = StrategyConfig(
            quantiles=args.quantiles,
            weighting=args.weighting,
            ewct_gamma=gamma,
            long_only=args.long_only,
            short_only=args.short_only,
            min_stocks_per_day=args.min_stocks_per_day,
            missing_return=args.missing_return,
        )
        result = backtest_ranked_strategy(
            predictions,
            prediction=args.prediction_column,
            realized=args.return_column,
            date=args.date_column,
            asset=args.asset_column,
            market_data=market,
            market_return=args.market_return_column,
            market_cap=args.market_cap_column,
            can_buy=args.can_buy_column,
            can_sell=args.can_sell_column,
            shortable=args.shortable_column,
            eligible=args.eligible_column,
            config=config,
            costs=cost_model,
        )
        label = f"gamma_{gamma:.2f}".replace(".", "p")
        run_dir = args.output_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        result.daily.to_parquet(run_dir / "daily.parquet", index=False)
        result.holdings.to_parquet(run_dir / "holdings.parquet", index=False)
        result.trades.to_parquet(run_dir / "trades.parquet", index=False)
        metrics = strategy_metrics(result.daily) if not result.daily.empty else {}
        run_summary = {
            "config": asdict(config),
            "n_days": len(result.daily),
            "n_holding_rows": len(result.holdings),
            "n_trade_rows": len(result.trades),
            "metrics": metrics,
            "artifacts": {
                "daily": str(run_dir / "daily.parquet"),
                "holdings": str(run_dir / "holdings.parquet"),
                "trades": str(run_dir / "trades.parquet"),
            },
        }
        (run_dir / "summary.json").write_text(
            json.dumps(run_summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        report["runs"][label] = run_summary

    report_path = args.output_dir / "summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(report_path), "runs": list(report["runs"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
