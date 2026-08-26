"""Summarize CNINFO stock-pool daily returns from HDBSCAN predictions."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def daily_metrics(path: Path) -> dict[str, object]:
    parts = path.parts
    # .../cninfo/model/prompt/variant/feature/target/mode/year/model_predictions.parquet
    root = parts.index("cninfo")
    model, prompt, variant, feature, target, mode, year = parts[root + 1:root + 8]
    model_mode = path.stem.replace("_predictions", "")
    frame = pd.read_parquet(path)
    actual_col = target if target in frame else "actual"
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce").dt.normalize()
    frame["actual"] = pd.to_numeric(frame[actual_col], errors="coerce")
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    frame = frame.dropna(subset=["stock_id", "entry_date", "actual", "prediction"])
    day = frame.groupby(["stock_id", "entry_date"], as_index=False).agg(actual=("actual", "mean"), prediction=("prediction", "mean"))
    rows = []
    for date, group in day.groupby("entry_date"):
        if len(group) < 10:
            continue
        n = max(1, int(np.ceil(len(group) * .2)))
        ordered = group.sort_values("prediction")
        long_ret = ordered.tail(n).actual.mean()
        short_ret = ordered.head(n).actual.mean()
        rows.append({"entry_date": date, "pool_mean": group.actual.mean(), "long_top20": long_ret, "short_bottom20": short_ret, "long_short": long_ret - short_ret, "stocks": len(group)})
    daily = pd.DataFrame(rows)
    if daily.empty:
        return {"dataset": "cninfo", "model": model, "prompt": prompt, "variant": variant, "feature": feature, "target": target, "mode": mode, "model_mode": model_mode, "test_year": int(year), "days": 0}
    return {"dataset": "cninfo", "model": model, "prompt": prompt, "variant": variant, "feature": feature, "target": target, "mode": mode, "model_mode": model_mode, "test_year": int(year), "days": len(daily), "pool_mean_bp": daily.pool_mean.mean() * 1e4, "long_top20_bp": daily.long_top20.mean() * 1e4, "short_bottom20_bp": daily.short_bottom20.mean() * 1e4, "long_short_bp": daily.long_short.mean() * 1e4, "positive_ls_days": int((daily.long_short > 0).sum()), "mean_stocks": daily.stocks.mean()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026/single_stock_cninfo_v1/prompt_positive_v1/results_all_embedding_cluster_regression/cninfo"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = [daily_metrics(path) for path in sorted(args.root.rglob("*_predictions.parquet"))]
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"summarized {len(rows)} prediction files -> {args.output}")

if __name__ == "__main__":
    main()
