#!/usr/bin/env python3
"""Attach next-day intraday and log-relative valuation labels to the existing panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SHARE = Path("/data/alpha_team2/shares/260825")


def code_mapping(panel_codes: list[str], path: Path) -> dict[str, str]:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    names.discard("datetime")
    result: dict[str, str] = {}
    for code in panel_codes:
        base = str(code).split(".", 1)[0].zfill(6)
        matches = [name for name in names if str(name).split(".", 1)[0].zfill(6) == base]
        if matches:
            result[str(code)] = sorted(matches)[0]
    return result


def daily_last_mean(path: Path, tickers: list[str]) -> pd.DataFrame:
    """Stream a wide minute/daily parquet and return date/ticker last and mean."""
    parquet = pq.ParquetFile(path)
    selected = [ticker for ticker in tickers if ticker in parquet.schema_arrow.names]
    if not selected:
        return pd.DataFrame(columns=["date", "ticker", "value", "mean"])
    chunks: list[pd.DataFrame] = []
    for batch in parquet.iter_batches(batch_size=8192, columns=["datetime", *selected], use_threads=False):
        frame = batch.to_pandas()
        if "datetime" not in frame.columns and frame.index.name == "datetime":
            frame = frame.reset_index()
        frame["date"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.normalize()
        long = frame.melt(id_vars=["date"], value_vars=selected, var_name="ticker", value_name="value")
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long = long.dropna(subset=["date", "value"])
        if not long.empty:
            chunks.append(long)
    if not chunks:
        return pd.DataFrame(columns=["date", "ticker", "value", "mean"])
    long = pd.concat(chunks, ignore_index=True)
    return long.groupby(["date", "ticker"], as_index=False).agg(value=("value", "last"), mean=("value", "mean"))


def daily_value(path: Path, tickers: list[str]) -> pd.DataFrame:
    """Read a wide daily valuation file and retain the last value per date."""
    frame = pd.read_parquet(path, columns=tickers)
    frame.index = pd.to_datetime(frame.index, errors="coerce").normalize()
    frame.index.name = "date"
    long = frame.reset_index().melt(id_vars=["date"], var_name="ticker", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.dropna(subset=["date", "value"])


def add_next_previous(frame: pd.DataFrame, column: str, group: str = "ticker") -> None:
    frame.sort_values([group, "date"], inplace=True)
    frame[f"prev_{column}"] = frame.groupby(group)[column].shift(1)
    frame[f"next_{column}"] = frame.groupby(group)[column].shift(-1)


def valuation_labels(panel: pd.DataFrame, ticker_by_stock: dict[str, str]) -> pd.DataFrame:
    codes = set(panel["stock_id"].astype(str))
    rows = []
    files = {"pe": "DZ_DInd_pe.parquet", "pb": "DZ_DInd_pb.parquet", "ps": "DZ_DInd_ps.parquet", "evtoebitda": "DZ_DInd_evtoebda.parquet"}
    # Keep the filename typo-proof while preserving the public target name.
    files["evtoebitda"] = "DZ_DInd_evtoebitda.parquet"
    tickers = sorted(set(ticker_by_stock.values()))
    for name, filename in files.items():
        values = daily_value(SHARE / filename, tickers)
        values["value"] = values["value"].where(values["value"] > 0)
        values = values.dropna(subset=["value"]).sort_values(["ticker", "date"])
        values["log_value"] = np.log(values["value"])
        values["trailing_log_median"] = values.groupby("ticker")["log_value"].transform(
            lambda series: series.shift(1).rolling(252, min_periods=60).median()
        )
        values["log_dev"] = values["log_value"] - values["trailing_log_median"]
        values[f"next_{name}_log_dev"] = values.groupby("ticker")["log_dev"].shift(-1)
        values[f"prev_{name}_log_dev"] = values.groupby("ticker")["log_dev"].shift(1)
        values["stock_id"] = values["ticker"].map({ticker: stock for stock, ticker in ticker_by_stock.items()})
        rows.append(values[["date", "stock_id", f"next_{name}_log_dev", f"prev_{name}_log_dev"]])
    merged = rows[0]
    for row in rows[1:]:
        merged = merged.merge(row, on=["date", "stock_id"], how="outer")
    return panel.merge(merged, left_on=["entry_date", "stock_id"], right_on=["date", "stock_id"], how="left").drop(columns=["date"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = pd.read_parquet(args.panel)
    panel["entry_date"] = pd.to_datetime(panel["entry_date"], errors="coerce").dt.normalize()
    panel["stock_id"] = panel["stock_id"].astype(str)
    code_map = code_mapping(sorted(panel["stock_id"].unique()), SHARE / "realized_volatility_cumulative.parquet")
    panel["ticker"] = panel["stock_id"].map(code_map)

    rvol = daily_last_mean(SHARE / "realized_volatility_cumulative.parquet", sorted(set(code_map.values())))
    spread = daily_last_mean(SHARE / "bid_ask_spread_bps.parquet", sorted(set(code_map.values())))
    rvol = rvol.rename(columns={"value": "rvol_close", "mean": "rvol_mean"})
    spread = spread.rename(columns={"value": "spread_bps_close", "mean": "spread_bps_mean"})
    metrics = rvol.merge(spread, on=["date", "ticker"], how="outer").sort_values(["ticker", "date"])
    for column in ["rvol_close", "rvol_mean", "spread_bps_close", "spread_bps_mean"]:
        add_next_previous(metrics, column)
    metrics["next_intraday_rvol_close"] = metrics["next_rvol_close"]
    metrics["next_intraday_rvol_log_change"] = np.log1p(metrics["next_rvol_close"].clip(lower=0)) - np.log1p(metrics["prev_rvol_close"].clip(lower=0))
    metrics["next_intraday_spread_bps_mean"] = metrics["next_spread_bps_mean"]
    metrics["next_intraday_spread_log_change"] = np.log1p(metrics["next_spread_bps_mean"].clip(lower=0)) - np.log1p(metrics["prev_spread_bps_mean"].clip(lower=0))
    keep = ["date", "ticker", "next_intraday_rvol_close", "next_intraday_rvol_log_change", "next_intraday_spread_bps_mean", "next_intraday_spread_log_change"]
    panel = panel.merge(metrics[keep], left_on=["entry_date", "ticker"], right_on=["date", "ticker"], how="left")
    if "date" in panel.columns:
        panel.drop(columns=["date"], inplace=True)
    panel = valuation_labels(panel, code_map)
    panel.drop(columns=["ticker"], inplace=True)
    for column in panel.columns:
        if column.startswith(("next_", "prev_")) or column.endswith("log_dev"):
            panel[column] = pd.to_numeric(panel[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output, index=False)
    targets = [
        "next_intraday_rvol_close", "next_intraday_rvol_log_change",
        "next_intraday_spread_bps_mean", "next_intraday_spread_log_change",
        "next_pe_log_dev", "next_pb_log_dev", "next_ps_log_dev", "next_evtoebitda_log_dev",
        "volatility_jump_log_v2",
    ]
    summary = {
        "rows": int(len(panel)),
        "stocks": int(panel["stock_id"].nunique()),
        "date_min": str(panel["entry_date"].min()),
        "date_max": str(panel["entry_date"].max()),
        "target_finite_counts": {target: int(panel[target].notna().sum()) for target in targets},
        "target_year_counts": {
            target: {str(year): int(panel.loc[panel[target].notna() & panel.entry_date.dt.year.eq(year)].shape[0]) for year in sorted(panel.entry_date.dt.year.dropna().unique())}
            for target in targets
        },
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
