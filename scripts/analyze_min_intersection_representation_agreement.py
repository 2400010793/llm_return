"""Compare existing representation predictions on the exact minimum intersection.

This is a prediction-level comparison: ``mean embedding`` and ``token
embedding`` are represented by their existing sample-out-of-sample prediction
files.  It does not compare raw vector cosine values, which are not directly
comparable across embedding models.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def load(path: Path, name: str) -> pd.DataFrame:
    try:
        d = pd.read_parquet(path, columns=["stock_id", "entry_date", "actual_return", "prediction"])
    except Exception as exc:
        # Some Qwen runs use a JSON manifest with a historical .parquet name.
        # Resolve its explicit stock-day artifact rather than guessing a path.
        import json
        try:
            manifest = json.loads(path.read_text())
            artifact = Path(manifest["stock_day_predictions"])
            d = pd.read_parquet(artifact, columns=["stock_id", "entry_date", "actual_return", "prediction"])
        except Exception:
            raise exc
    d["entry_date"] = pd.to_datetime(d["entry_date"], errors="coerce").dt.normalize()
    d = d.dropna(subset=["stock_id", "entry_date", "actual_return", "prediction"])
    return d.drop_duplicates(["stock_id", "entry_date"]).rename(columns={"prediction": name})


def daily_rank_ic(d: pd.DataFrame, name: str) -> float:
    values = []
    for _, g in d.groupby("entry_date"):
        if len(g) >= 3 and g[name].nunique() > 1 and g.actual_return.nunique() > 1:
            values.append(g[name].corr(g.actual_return, method="spearman"))
    return float(np.nanmean(values)) if values else np.nan


def daily_bp(d: pd.DataFrame, name: str) -> dict[str, float]:
    rows = []
    for _, g in d.groupby("entry_date"):
        if len(g) < 10:
            continue
        n = max(1, int(np.ceil(len(g) * 0.2)))
        s = g.sort_values([name, "stock_id"], kind="mergesort")
        long = s.tail(n).actual_return.mean()
        short_leg = s.head(n).actual_return.mean()
        rows.append((long, short_leg, long - short_leg))
    if not rows:
        return {"days": 0, "long_bp": np.nan, "short_leg_bp": np.nan, "ls_bp": np.nan}
    a = np.asarray(rows)
    return {"days": len(a), "long_bp": a[:, 0].mean() * 1e4, "short_leg_bp": a[:, 1].mean() * 1e4, "ls_bp": a[:, 2].mean() * 1e4}


def mean_daily_spearman(d: pd.DataFrame, left: str, right: str) -> tuple[float, int]:
    values = []
    for _, group in d.groupby("entry_date"):
        if len(group) >= 3 and group[left].nunique() > 1 and group[right].nunique() > 1:
            values.append(spearmanr(group[left], group[right]).statistic)
    return (float(np.nanmean(values)), len(values)) if values else (np.nan, 0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--factor", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    args = p.parse_args()
    factors = [(name, Path(path)) for name, path in args.factor]
    merged = None
    for name, path in factors:
        data = load(path, name)
        merged = data if merged is None else merged.merge(data, on=["stock_id", "entry_date", "actual_return"], how="inner", validate="one_to_one")
    assert merged is not None
    names = [name for name, _ in factors]
    summary = []
    for name in names:
        summary.append({"factor": name, "rows": len(merged), "days": merged.entry_date.nunique(), "rank_ic": daily_rank_ic(merged, name), **daily_bp(merged, name)})
    pairs = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            x = merged[left].to_numpy(float)
            y = merged[right].to_numpy(float)
            daily_s, daily_days = mean_daily_spearman(merged, left, right)
            pairs.append({"left": left, "right": right, "rows": len(merged), "calendar_days": merged.entry_date.nunique(), "daily_spearman": daily_s, "daily_spearman_days": daily_days, "spearman": spearmanr(x, y).statistic, "pearson": pearsonr(x, y)[0]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(args.output.with_name(args.output.stem + "_factors.csv"), index=False)
    pd.DataFrame(pairs).to_csv(args.output.with_name(args.output.stem + "_pairs.csv"), index=False)
    pd.DataFrame([{"rows": len(merged), "calendar_days": merged.entry_date.nunique(), "eligible_top20_days": daily_bp(merged, names[0])["days"], "min_date": merged.entry_date.min(), "max_date": merged.entry_date.max(), "factors": len(names)}]).to_csv(args.output, index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(pairs).to_string(index=False))


if __name__ == "__main__":
    main()