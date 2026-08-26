"""Build a point-in-time aligned label panel for the masked-short prompt axes.

The news panel supplies the event row and ``entry_date``.  The market panel is
joined by ``stock_id`` and trading date, then forward returns and volatility
labels are calculated only from observations after the entry date.  Optional
valuation data must contain an explicit availability timestamp; silently using
the latest/current valuation would introduce look-ahead bias.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MARKET_COLUMNS = [
    "entry_date", "stock_id", "open", "close", "high", "low", "volume", "amount",
    "observed_market", "next_observed_market", "listing_trading_day", "ipo_initial_window",
    "open_gap_from_previous_close", "price_limit_fraction", "can_buy", "can_sell",
    "eligible_signal", "open_to_open_return", "close_to_close_return", "vwap_proxy_return",
    "open_to_close_return",
]
VALUATION_FIELDS = ["market_cap", "pe_ttm", "pe_forward", "pb", "ps", "ev_ebitda", "dividend_yield"]


def _stock_key(values: pd.Series) -> pd.Series:
    """Normalize mixed integer/string stock identifiers without losing leading zeros."""
    return values.astype("string").str.extract(r"(\d{6})", expand=False).fillna(values.astype("string"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _forward_price_return(frame: pd.DataFrame, column: str, horizon: int) -> pd.Series:
    return frame.groupby("stock_id", sort=False)[column].shift(-horizon).div(frame[column]).sub(1.0)


def _group_window_stat(
    frame: pd.DataFrame,
    value_column: str,
    window: int,
    *,
    direction: str,
    statistic: str,
) -> pd.Series:
    """Compute a prior or forward group window statistic without cross-stock bleed."""
    values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float)
    output = np.full(len(frame), np.nan, dtype=np.float64)
    groups = frame.groupby("stock_id", sort=False).indices
    for positions in groups.values():
        positions = np.asarray(positions, dtype=np.int64)
        local = pd.Series(values[positions])
        if direction == "prior":
            local = local.shift(1)
            rolled = getattr(local.rolling(window, min_periods=window), statistic)()
        elif direction == "forward":
            reversed_local = local.iloc[::-1]
            rolled = getattr(reversed_local.rolling(window, min_periods=window), statistic)().iloc[::-1]
        else:
            raise ValueError(f"unknown direction: {direction}")
        output[positions] = rolled.to_numpy(dtype=float)
    return pd.Series(output, index=frame.index)


def _forward_product(frame: pd.DataFrame, value_column: str, horizon: int) -> pd.Series:
    values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float)
    output = np.full(len(frame), np.nan, dtype=np.float64)
    groups = frame.groupby("stock_id", sort=False).indices
    for positions in groups.values():
        positions = np.asarray(positions, dtype=np.int64)
        local = pd.Series(values[positions])
        # A forward h-day compounded return requires every daily observation.
        rolled = (1.0 + local.iloc[::-1]).rolling(horizon, min_periods=horizon).apply(
            np.prod, raw=True
        ).iloc[::-1] - 1.0
        output[positions] = rolled.to_numpy(dtype=float)
    return pd.Series(output, index=frame.index)


def _daily_market_forward(market: pd.DataFrame, horizon: int) -> pd.DataFrame:
    daily = (
        market.groupby("entry_date", as_index=False, sort=True)["open_to_open_return"]
        .mean()
        .rename(columns={"open_to_open_return": "market_return_1d"})
    )
    values = daily["market_return_1d"].to_numpy(dtype=float)
    daily[f"market_forward_return_{horizon}d"] = (
        (1.0 + pd.Series(values[::-1])).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True)
        .iloc[::-1].to_numpy() - 1.0
    )
    return daily


def _attach_optional_valuation(panel: pd.DataFrame, path: Path | None) -> tuple[pd.DataFrame, dict]:
    for field in VALUATION_FIELDS:
        panel[field] = np.nan
    panel["valuation_available"] = False
    if path is None:
        return panel, {"provided": False, "status": "missing_source", "fields": VALUATION_FIELDS}

    valuation = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {"stock_id", "available_at"}
    missing = required.difference(valuation.columns)
    if missing:
        raise ValueError(f"valuation source missing required columns: {sorted(missing)}")
    valuation = valuation.copy()
    valuation["stock_id"] = _stock_key(valuation["stock_id"])
    valuation["available_at"] = pd.to_datetime(valuation["available_at"], errors="coerce")
    valuation = valuation.dropna(subset=["stock_id", "available_at"])
    valuation = valuation.sort_values(["stock_id", "available_at"], kind="stable")
    source_fields = [field for field in VALUATION_FIELDS if field in valuation.columns]
    if not source_fields:
        raise ValueError(f"valuation source has none of {VALUATION_FIELDS}")
    left = panel.copy()
    left["_event_time"] = pd.to_datetime(left["published_at"], errors="coerce")
    # merge_asof requires the time key itself to be globally sorted; ``by``
    # still keeps matches within each stock.
    left = left.sort_values(["_event_time", "stock_id"], kind="stable")
    right = valuation[["stock_id", "available_at", *source_fields]].rename(columns={"available_at": "_valuation_available_at"})
    merged = pd.merge_asof(
        left,
        right.sort_values(["_valuation_available_at", "stock_id"], kind="stable"),
        left_on="_event_time",
        right_on="_valuation_available_at",
        by="stock_id",
        direction="backward",
        allow_exact_matches=True,
    )
    for field in source_fields:
        merged[field] = pd.to_numeric(merged[field], errors="coerce")
        merged[f"{field}_percentile"] = merged.groupby("entry_date")[field].rank(pct=True)
    merged["valuation_available"] = merged["_valuation_available_at"].notna()
    merged = merged.drop(columns=["_event_time", "_valuation_available_at"], errors="ignore")
    merged = merged.sort_values("row_index", kind="stable")
    return merged, {
        "provided": True,
        "status": "aligned_point_in_time",
        "source": str(path),
        "source_sha256": _file_sha256(path),
        "fields": source_fields,
        "available_rows": int(merged["valuation_available"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--valuation", type=Path, default=None)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    required_panel = {"row_index", "stock_id", "published_at", "entry_date"}
    missing = required_panel.difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")
    panel = panel.copy()
    panel["stock_id"] = _stock_key(panel["stock_id"])
    panel["entry_date"] = pd.to_datetime(panel["entry_date"], errors="coerce").dt.normalize()
    panel["published_at"] = pd.to_datetime(panel["published_at"], errors="coerce")
    panel = panel.drop_duplicates("row_index", keep="first").sort_values("row_index", kind="stable")

    market = pd.read_parquet(args.market, columns=MARKET_COLUMNS)
    market["stock_id"] = _stock_key(market["stock_id"])
    market["entry_date"] = pd.to_datetime(market["entry_date"], errors="coerce").dt.normalize()
    market = market.dropna(subset=["stock_id", "entry_date"]).sort_values(["stock_id", "entry_date"], kind="stable")
    market = market.drop_duplicates(["stock_id", "entry_date"], keep="last")
    for column in ["open", "close", "high", "low", "volume", "amount", "open_to_open_return", "close_to_close_return"]:
        market[column] = pd.to_numeric(market[column], errors="coerce")

    # Join the contemporaneous tradability and pre-event state.
    join_columns = [
        "stock_id", "entry_date", "open", "close", "high", "low", "volume", "amount",
        "observed_market", "eligible_signal", "can_buy", "can_sell", "listing_trading_day",
        "open_gap_from_previous_close", "price_limit_fraction",
    ]
    point = market[join_columns].rename(columns={column: f"entry_{column}" for column in join_columns if column not in {"stock_id", "entry_date"}})
    panel = panel.merge(point, on=["stock_id", "entry_date"], how="left", validate="many_to_one")

    # Forward returns and post-event risk labels are computed on the full market calendar.
    for horizon in (1, 3, 5, 10, 20):
        market[f"forward_open_to_open_return_{horizon}d"] = _forward_price_return(market, "open", horizon)
        market[f"forward_close_to_close_return_{horizon}d"] = _forward_price_return(market, "close", horizon)
        market[f"forward_compounded_return_{horizon}d"] = _forward_product(market, "open_to_open_return", horizon)
        market[f"post_realized_volatility_{horizon}d"] = _group_window_stat(
            market, "close_to_close_return", horizon, direction="forward", statistic="std"
        )
        market[f"post_abs_return_mean_{horizon}d"] = _group_window_stat(
            market.assign(_abs_return=market["close_to_close_return"].abs()),
            "_abs_return", horizon, direction="forward", statistic="mean",
        )
    for window in (5, 20):
        market[f"pre_realized_volatility_{window}d"] = _group_window_stat(
            market, "close_to_close_return", window, direction="prior", statistic="std"
        )
        market[f"pre_volume_median_{window}d"] = _group_window_stat(
            market, "volume", window, direction="prior", statistic="median"
        )
        market[f"pre_range_mean_{window}d"] = _group_window_stat(
            market.assign(_range=(market["high"] - market["low"]) / market["close"].replace(0, np.nan)),
            "_range", window, direction="prior", statistic="mean",
        )
    market["volume_shock"] = np.log1p(market["volume"].clip(lower=0)) - np.log1p(market["pre_volume_median_20d"].clip(lower=0))
    market["range_shock"] = ((market["high"] - market["low"]) / market["close"].replace(0, np.nan)) / market["pre_range_mean_20d"]
    market["volatility_jump"] = market["post_realized_volatility_5d"] / market["pre_realized_volatility_20d"]

    derived_columns = [
        "stock_id", "entry_date", "volume_shock", "range_shock", "volatility_jump",
        *[column for column in market.columns if column.startswith(("forward_", "post_", "pre_"))],
    ]
    panel = panel.merge(market[derived_columns], on=["stock_id", "entry_date"], how="left", validate="many_to_one")

    # Equal-weight market baseline for market-adjusted forward-return labels.
    for horizon in (1, 3, 5, 10, 20):
        daily = _daily_market_forward(market, horizon)
        daily = daily[["entry_date", f"market_forward_return_{horizon}d"]]
        panel = panel.merge(daily, on="entry_date", how="left", validate="many_to_one")
        lhs = f"forward_compounded_return_{horizon}d"
        rhs = f"market_forward_return_{horizon}d"
        panel[f"market_adjusted_forward_return_{horizon}d"] = panel[lhs] - panel[rhs]

    if "event_return_3d" in panel:
        panel["event_abs_return_3d"] = pd.to_numeric(panel["event_return_3d"], errors="coerce").abs()
        panel["abnormal_event_return_3d"] = panel["event_return_3d"] - panel["market_forward_return_3d"]
    tail_cutoff = panel["forward_compounded_return_5d"].abs().quantile(0.95)
    panel["tail_event_5d"] = pd.Series(
        np.where(
            panel["forward_compounded_return_5d"].notna(),
            (panel["forward_compounded_return_5d"].abs() >= tail_cutoff).astype("int8"),
            -1,
        ),
        index=panel.index,
    ).replace(-1, pd.NA).astype("Int8")
    panel["label_market_data_available"] = panel["forward_compounded_return_5d"].notna() & panel["post_realized_volatility_5d"].notna()
    panel, valuation_audit = _attach_optional_valuation(panel, args.valuation)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output, index=False)
    numeric_columns = [column for column in panel.columns if column.startswith(("forward_", "market_adjusted_", "post_", "pre_", "volume_", "range_", "volatility_", "event_", "abnormal_", "valuation_"))]
    summary = {
        "format_version": "prompt_factor_label_panel_v1",
        "panel": str(args.panel),
        "panel_sha256": _file_sha256(args.panel),
        "market": str(args.market),
        "market_sha256": _file_sha256(args.market),
        "output": str(args.output),
        "rows": int(len(panel)),
        "columns": int(len(panel.columns)),
        "row_index_unique": bool(panel["row_index"].is_unique),
        "date_min": str(panel["entry_date"].min()),
        "date_max": str(panel["entry_date"].max()),
        "market_data_available_rows": int(panel["label_market_data_available"].sum()),
        "market_data_available_fraction": float(panel["label_market_data_available"].mean()),
        "finite_counts": {column: int(pd.to_numeric(panel[column], errors="coerce").notna().sum()) for column in numeric_columns},
        "valuation": valuation_audit,
        "valuation_warning": "No valuation source was supplied; valuation columns remain missing and must not enter the main evaluation.",
        "definitions": {
            "forward_return": "open price at entry_date to open price h trading sessions later",
            "post_realized_volatility": "sample std of close-to-close returns over the next h trading sessions",
            "pre_realized_volatility": "sample std of the preceding h close-to-close returns",
            "market_return": "equal-weight mean stock open-to-open return compounded over h sessions",
            "tail_event_5d": "absolute 5-day compounded return at or above the panel 95th percentile",
            "point_in_time_rule": "valuation uses the latest available_at <= published_at when a source is supplied",
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(panel), "output": str(args.output), "valuation": valuation_audit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
