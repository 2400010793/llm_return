"""Run a small, explicit paper-style time split on a returns parquet file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.paper_pipeline import fit_predict_return_ridge, fit_predict_sentiment, portfolio_metrics, quantile_portfolio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("panel")
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--target", default="return_1d")
    parser.add_argument("--output-dir", default="reports/paper_pipeline")
    parser.add_argument("--relation", default="direct")
    args = parser.parse_args()
    frame = pd.read_parquet(args.panel)
    required = {"text", "entry_date", args.target}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame = frame.loc[frame.get("stock_relation", args.relation).eq(args.relation)].dropna(subset=["text", "entry_date", args.target]).copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    train = frame[frame["entry_date"] < args.test_start]
    test = frame[frame["entry_date"] >= args.test_start]
    if train.empty or test.empty:
        raise ValueError("both train and test windows must contain rows")
    result = fit_predict_return_ridge(train.text, train[args.target], test.text, test[args.target])
    test = test.copy()
    test["prediction"] = result.predictions
    test["realized_return"] = test[args.target]
    portfolios = quantile_portfolio(test)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    test.to_parquet(output / "predictions.parquet", index=False)
    portfolios.to_parquet(output / "portfolios.parquet", index=False)
    summary = {"train_rows": len(train), "test_rows": len(test), "return_metrics": result.metrics, "portfolio_metrics": portfolio_metrics(portfolios)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
