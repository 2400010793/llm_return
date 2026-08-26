#!/usr/bin/env python3
"""Join existing prompt predictions to shared 1-minute metrics.

No embeddings or clustering are fitted here.  The script selects already
completed rolling predictions using validation IR, then measures their
pre-signal and next-day association with realized volatility and bid/ask spread.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr


PRED_ROOT = Path(
    "/mnt/lustre3/home/gaozh/sina_all_news_records_2010_2026"
    "/single_stock_cninfo_v1"
)
SHARE_ROOT = Path("/data/alpha_team2/shares/260825")
OUT = Path("/mnt/lustre3/home/gaozh/llm_return/reports/intraday_token_correspondence")


def selected_specs() -> list[dict]:
    comparison = json.loads(Path("reports/aligned_v2_representation_comparison.json").read_text())
    rows = [r for r in comparison["selected_results"] if not r.get("unstable", False)]
    frame = pd.DataFrame(rows)
    # Select only by validation IR within an axis/target/representation. This
    # preserves the frozen selection rule and does not inspect test metrics.
    frame = frame.sort_values("validation_ir", ascending=False).drop_duplicates(
        ["axis", "target", "representation"], keep="first"
    )
    specs = []
    for row in frame.to_dict(orient="records"):
        rep_dir = "regression_token_v2" if row["representation"] == "target_span_mean" else "regression_body_v2"
        stem = f"{row['axis']}_{row['factor']}_{row['regressor']}_{row['target']}"
        path = PRED_ROOT / "aligned_masked_short_v1" / rep_dir / stem / "result.stock_day_predictions.parquet"
        if path.is_file():
            specs.append({**row, "prediction_path": str(path)})
    return specs


def prediction_tickers(specs: list[dict]) -> tuple[dict[str, str], dict[str, set[str]]]:
    base_to_ticker: dict[str, str] = {}
    duplicate: dict[str, set[str]] = {}
    available = set(pq.ParquetFile(SHARE_ROOT / "realized_volatility_cumulative.parquet").schema_arrow.names)
    available.discard("datetime")
    for code in available:
        base = str(code).split(".", 1)[0].zfill(6)
        if base in base_to_ticker and base_to_ticker[base] != code:
            duplicate.setdefault(base, set()).update({base_to_ticker[base], code})
        else:
            base_to_ticker[base] = code
    return base_to_ticker, duplicate


def stream_daily_metrics(path: Path, tickers: list[str]) -> pd.DataFrame:
    """Read selected wide columns in batches and derive daily close/mean values."""
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    selected = [x for x in tickers if x in names]
    if not selected:
        return pd.DataFrame(columns=["date", "ticker", "value"])
    last: dict[tuple[pd.Timestamp, str], float] = {}
    sums: dict[tuple[pd.Timestamp, str], float] = {}
    counts: dict[tuple[pd.Timestamp, str], int] = {}
    for batch in parquet.iter_batches(batch_size=8192, columns=["datetime", *selected], use_threads=False):
        frame = batch.to_pandas()
        if "datetime" not in frame.columns and frame.index.name == "datetime":
            frame = frame.reset_index()
        frame["date"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.normalize()
        for ticker in selected:
            values = pd.to_numeric(frame[ticker], errors="coerce")
            valid = values.notna() & frame["date"].notna()
            if not valid.any():
                continue
            small = pd.DataFrame({"date": frame.loc[valid, "date"].to_numpy(), "value": values.loc[valid].to_numpy()})
            grouped = small.groupby("date")["value"]
            for date, value in grouped.last().items():
                last[(date, ticker)] = float(value)
            for date, value in grouped.sum().items():
                key = (date, ticker)
                sums[key] = sums.get(key, 0.0) + float(value)
            for date, value in grouped.count().items():
                key = (date, ticker)
                counts[key] = counts.get(key, 0) + int(value)
    rows = []
    for key, value in last.items():
        date, ticker = key
        rows.append({
            "date": date,
            "ticker": ticker,
            "close": value,
            "mean": sums[key] / counts[key] if counts.get(key, 0) else np.nan,
        })
    return pd.DataFrame(rows)


def metric_panel(tickers: list[str]) -> pd.DataFrame:
    vol = stream_daily_metrics(SHARE_ROOT / "realized_volatility_cumulative.parquet", tickers).rename(
        columns={"close": "rvol_close", "mean": "rvol_mean"}
    )
    spread = stream_daily_metrics(SHARE_ROOT / "bid_ask_spread_bps.parquet", tickers).rename(
        columns={"close": "spread_bps_close", "mean": "spread_bps_mean"}
    )
    out = vol.merge(spread, on=["date", "ticker"], how="outer")
    valuation_files = {
        "pe": "DZ_DInd_pe.parquet",
        "pb": "DZ_DInd_pb.parquet",
        "ps": "DZ_DInd_ps.parquet",
        "evtoebitda": "DZ_DInd_evtoebitda.parquet",
    }
    for name, filename in valuation_files.items():
        value = stream_daily_metrics(SHARE_ROOT / filename, tickers)
        value = value.rename(columns={"close": name, "mean": f"{name}_mean"})
        out = out.merge(value[["date", "ticker", name]], on=["date", "ticker"], how="outer")
    out = out.sort_values(["ticker", "date"])
    for column in [
        "rvol_close", "rvol_mean", "spread_bps_close", "spread_bps_mean",
        "pe", "pb", "ps", "evtoebitda",
    ]:
        out[f"prev_{column}"] = out.groupby("ticker")[column].shift(1)
        out[f"next_{column}"] = out.groupby("ticker")[column].shift(-1)
    return out


def daily_corr(frame: pd.DataFrame, prediction: str, metric: str) -> dict[str, float]:
    rows = []
    for _, group in frame.groupby("entry_date"):
        x = group[[prediction, metric]].dropna()
        if len(x) < 10 or x[prediction].nunique() < 2 or x[metric].nunique() < 2:
            continue
        value = spearmanr(x[prediction], x[metric]).statistic
        if np.isfinite(value):
            rank = x[prediction].rank(method="first")
            n = max(1, int(np.ceil(len(x) * 0.2)))
            order = x.assign(rank=rank).sort_values("rank")
            rows.append({"date": group["entry_date"].iloc[0], "ic": float(value), "spread": float(order.tail(n)[metric].mean() - order.head(n)[metric].mean())})
    result = pd.DataFrame(rows)
    if result.empty:
        return {"days": 0, "mean_ic": np.nan, "positive_ic_days": 0, "mean_top_bottom_metric": np.nan, "positive_spread_days": 0}
    return {"days": int(len(result)), "mean_ic": float(result.ic.mean()), "positive_ic_days": int((result.ic > 0).sum()), "mean_top_bottom_metric": float(result.spread.mean()), "positive_spread_days": int((result.spread > 0).sum())}


def main() -> None:
    specs = selected_specs()
    mapping, duplicates = prediction_tickers(specs)
    all_predictions = []
    for spec in specs:
        frame = pd.read_parquet(spec["prediction_path"])
        frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce").dt.normalize()
        frame["base_code"] = frame["stock_id"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
        frame["ticker"] = frame["base_code"].map(mapping)
        frame["axis"] = spec["axis"]
        frame["target"] = spec["target"]
        frame["representation"] = spec["representation"]
        frame["factor"] = spec["factor"]
        frame["regressor"] = spec["regressor"]
        all_predictions.append(frame)
    predictions = pd.concat(all_predictions, ignore_index=True)
    tickers = sorted(predictions["ticker"].dropna().unique())
    metrics = metric_panel(tickers)
    joined = predictions.merge(metrics, left_on=["entry_date", "ticker"], right_on=["date", "ticker"], how="left")
    metric_names = [
        "prev_rvol_close", "prev_spread_bps_mean", "prev_spread_bps_close",
        "next_rvol_close", "next_spread_bps_mean",
        "prev_pe", "prev_pb", "prev_ps", "prev_evtoebitda",
        "next_pe", "next_pb", "next_ps", "next_evtoebitda",
    ]
    rows = []
    for keys, group in joined.groupby(["axis", "target", "representation", "factor", "regressor"]):
        axis, target, representation, factor, regressor = keys
        for metric in metric_names:
            result = daily_corr(group, "prediction", metric)
            rows.append({"axis": axis, "target": target, "representation": representation, "factor": factor, "regressor": regressor, "metric": metric, **result, "prediction_rows": int(len(group)), "matched_metric_rows": int(group[metric].notna().sum())})
    result = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "intraday_prediction_correlations.csv", index=False)
    joined.to_parquet(OUT / "joined_existing_predictions.parquet", index=False)
    report = {
        "prediction_specs": specs,
        "prediction_count": len(specs),
        "ticker_count": len(tickers),
        "duplicate_base_codes": {k: sorted(v) for k, v in duplicates.items()},
        "intraday_files": [str(SHARE_ROOT / "realized_volatility_cumulative.parquet"), str(SHARE_ROOT / "bid_ask_spread_bps.parquet")],
        "valuation_files": [str(SHARE_ROOT / f) for f in ["DZ_DInd_pe.parquet", "DZ_DInd_pb.parquet", "DZ_DInd_ps.parquet", "DZ_DInd_evtoebitda.parquet"]],
        "valuation_access": "readable",
    }
    (OUT / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 已有 Prompt 预测与 1 分钟指标对应分析",
        "",
        "本分析不重新训练 embedding、不重新聚类，只读取已经完成的 `regression_token_v2` / `regression_body_v2` 预测，并按股票和 entry_date 对齐前一交易日和后一交易日的累计实现波动率、买卖价差。",
        "",
        f"已读取 {len(specs)} 个按验证 IR 选择的既有预测配置，覆盖 {len(tickers)} 个可匹配股票代码。",
        "",
        "## 指标定义",
        "",
        "- `prev_rvol_close`：entry_date 前一交易日最后一个有效分钟的累计实现波动率；",
        "- `prev_spread_bps_mean`：前一交易日分钟 bps 价差均值；",
        "- `prev_spread_bps_close`：前一交易日最后一个有效分钟的 bps 价差；",
        "- `next_*`：后一交易日对应指标，用于检验预测后的风险/流动性状态。",
        "",
        "结果文件：`intraday_prediction_correlations.csv`；完整 join：`joined_existing_predictions.parquet`。",
        "",
        "估值文件已成功读取；PE/PB/PS/EV/EBITDA 均按前一交易日值加入对应分析。",
    ]
    (OUT / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
