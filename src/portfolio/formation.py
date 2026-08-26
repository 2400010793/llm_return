"""Portfolio formation from cross-sectional return predictions."""

from __future__ import annotations

import pandas as pd


def quantile_portfolio(
    frame: pd.DataFrame,
    *,
    prediction: str = "prediction",
    realized: str = "realized_return",
    date: str = "entry_date",
    quantiles: int = 5,
) -> pd.DataFrame:
    """Form daily equal-weighted quantiles and a high-minus-low portfolio."""
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    required = {prediction, realized, date}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing portfolio columns: {', '.join(sorted(missing))}")
    rows = []
    for day, group in frame.dropna(subset=list(required)).groupby(date):
        if len(group) < quantiles:
            continue
        ranks = group[prediction].rank(method="first")
        buckets = pd.qcut(ranks, q=quantiles, labels=False, duplicates="drop")
        returns = {
            f"q{int(bucket) + 1}": float(group.loc[buckets == bucket, realized].mean())
            for bucket in sorted(buckets.dropna().unique())
        }
        if len(returns) != quantiles:
            continue
        low, high = returns["q1"], returns[f"q{quantiles}"]
        rows.append({
            "date": day, **returns, "low": low, "high": high,
            "long_short": high - low, "n": len(group),
        })
    columns = [
        "date", *(f"q{index}" for index in range(1, quantiles + 1)),
        "low", "high", "long_short", "n",
    ]
    return pd.DataFrame(rows, columns=columns)
