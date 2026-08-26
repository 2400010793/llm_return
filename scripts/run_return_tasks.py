"""Run sentiment and horizon-specific return models on a prepared research panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.return_tasks import fit_return_model, fit_sentiment_model, sentiment_metrics
from src.portfolio import portfolio_metrics, quantile_portfolio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("panel")
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--predict-start", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--date-column", default="entry_date")
    parser.add_argument("--sentiment-target", default="return_3d_event")
    parser.add_argument("--horizons", default="1,5,20")
    parser.add_argument("--models", default="ols,ridge,lasso,random_forest,nn")
    parser.add_argument("--output-dir", default="reports/return_tasks")
    parser.add_argument("--portfolio-quantiles", type=int, default=5)
    args = parser.parse_args()
    frame = pd.read_parquet(args.panel).copy()
    required = {args.text_column, args.date_column, args.sentiment_target}
    horizons = [int(x) for x in args.horizons.split(",")]
    required.update(f"return_{h}" for h in horizons)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    frame[args.date_column] = pd.to_datetime(frame[args.date_column], errors="coerce")
    frame[args.text_column] = frame[args.text_column].fillna("").astype(str)
    frame = frame.dropna(subset=[args.date_column])
    train = frame[frame[args.date_column] < args.train_end].copy()
    predict = frame[frame[args.date_column] >= args.predict_start].copy()
    if train.empty or predict.empty:
        raise ValueError("training and prediction windows must both be non-empty")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    summary = {"train_rows": len(train), "predict_rows": len(predict), "models": {}}
    try:
        sentiment = fit_sentiment_model(train[args.text_column], train[args.sentiment_target], predict[args.text_column])
        summary["sentiment"] = {"n_train_labeled": len(sentiment.labels), "positive_rate": float(sentiment.labels.mean())}
        predict["sentiment_probability"] = sentiment.probabilities
        if predict[args.sentiment_target].notna().any():
            summary["sentiment"]["prediction_metrics"] = sentiment_metrics(predict[args.sentiment_target], sentiment.probabilities)
    except ValueError as exc:
        summary["sentiment_error"] = str(exc)
    for horizon in horizons:
        target = f"return_{horizon}"
        summary["models"][target] = {}
        for model_name in args.models.split(","):
            result = fit_return_model(train[args.text_column], train[target], predict[args.text_column], predict[target], model_name=model_name.strip())
            column = f"prediction_{target}_{model_name.strip()}"
            predict[column] = result.predictions
            summary["models"][target][model_name.strip()] = result.metrics
            portfolio = quantile_portfolio(
                predict[[args.date_column, column, target]].rename(
                    columns={args.date_column: "entry_date", column: "prediction", target: "realized_return"}
                ),
                quantiles=args.portfolio_quantiles,
            )
            portfolio_path = output / f"portfolio_{target}_{model_name.strip()}.parquet"
            portfolio.to_parquet(portfolio_path, index=False)
            summary["models"][target][model_name.strip()]["portfolio"] = portfolio_metrics(portfolio)
    predict.to_parquet(output / "predictions.parquet", index=False)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
