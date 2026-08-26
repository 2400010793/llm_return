"""Leakage-safe cross-sectional targets derived from observed O2O returns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


MARKET_RETURN_COLUMN = "o2o_market_return_ew"
MARKET_RESIDUAL_COLUMN = "next_day_open_to_open_market_residual"
CS_ZSCORE_COLUMN = "next_day_open_to_open_cs_zscore"
WINSOR_RESIDUAL_COLUMN = "next_day_open_to_open_winsor_residual"


def _stock_ids(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def attach_o2o_residual_targets(
    panel: pd.DataFrame,
    market: pd.DataFrame,
    *,
    stock_column: str = "stock_id",
    date_column: str = "entry_date",
    source_target_column: str = "next_day_open_to_open_return",
    market_return_column: str = "open_to_open_return",
    market_eligible_column: str = "eligible_signal",
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
    min_stocks_per_day: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach market-demeaned, standardized, and winsorized O2O labels.

    Daily market moments are calculated from the full eligible market panel,
    never from the subset of stocks that happened to have announcements.  Each
    transformation uses returns from only its own target date, so no future
    date can affect an earlier fold.  The standardized target is intended for
    cross-sectional ranking rather than return-magnitude calibration.
    """
    if not 0.0 <= winsor_lower < winsor_upper <= 1.0:
        raise ValueError("winsor quantiles must satisfy 0 <= lower < upper <= 1")
    if min_stocks_per_day < 2:
        raise ValueError("min_stocks_per_day must be at least 2")
    panel_required = {stock_column, date_column, source_target_column}
    market_required = {
        stock_column, date_column, market_return_column, market_eligible_column,
    }
    missing_panel = panel_required.difference(panel.columns)
    missing_market = market_required.difference(market.columns)
    if missing_panel:
        raise ValueError(f"panel missing columns: {', '.join(sorted(missing_panel))}")
    if missing_market:
        raise ValueError(f"market missing columns: {', '.join(sorted(missing_market))}")

    result = panel.copy()
    result[stock_column] = _stock_ids(result[stock_column])
    result[date_column] = pd.to_datetime(result[date_column], errors="coerce")
    result[source_target_column] = pd.to_numeric(
        result[source_target_column], errors="coerce"
    )

    market_values = market.copy()
    market_values[stock_column] = _stock_ids(market_values[stock_column])
    market_values[date_column] = pd.to_datetime(
        market_values[date_column], errors="coerce"
    )
    if market_values.duplicated([stock_column, date_column]).any():
        raise ValueError("market contains duplicate stock-date keys")
    market_values[market_return_column] = pd.to_numeric(
        market_values[market_return_column], errors="coerce"
    )
    eligible = market_values[market_eligible_column].fillna(False).astype(bool)
    if "can_buy" in market_values:
        eligible &= market_values["can_buy"].fillna(False).astype(bool)
    eligible &= np.isfinite(market_values[market_return_column].to_numpy(dtype=float))
    eligible &= market_values[date_column].notna()
    eligible_market = market_values.loc[
        eligible, [stock_column, date_column, market_return_column]
    ].copy()
    if eligible_market.empty:
        raise ValueError("market contains no eligible finite O2O returns")

    groups = eligible_market.groupby(date_column, sort=True, observed=True)[
        market_return_column
    ]
    daily = groups.agg([("market_n", "size"), (MARKET_RETURN_COLUMN, "mean"), ("market_std", "std")])
    daily["winsor_lower"] = groups.quantile(winsor_lower)
    daily["winsor_upper"] = groups.quantile(winsor_upper)
    valid_day = daily["market_n"].ge(min_stocks_per_day)
    daily.loc[~valid_day, [MARKET_RETURN_COLUMN, "market_std", "winsor_lower", "winsor_upper"]] = np.nan

    eligible_market = eligible_market.join(daily, on=date_column)
    eligible_market["_winsor_return"] = eligible_market[market_return_column].clip(
        lower=eligible_market["winsor_lower"], upper=eligible_market["winsor_upper"]
    )
    winsor_mean = eligible_market.groupby(date_column, sort=True, observed=True)[
        "_winsor_return"
    ].mean().rename("_winsor_market_mean")
    eligible_market = eligible_market.join(winsor_mean, on=date_column)

    lookup = eligible_market[[
        stock_column, date_column, market_return_column, "_winsor_return",
        "_winsor_market_mean",
    ]]
    result["_source_order"] = np.arange(len(result), dtype=np.int64)
    result = result.merge(
        lookup, on=[stock_column, date_column], how="left", sort=False,
        validate="many_to_one", suffixes=("", "_market"),
    )
    result = result.merge(
        daily[[MARKET_RETURN_COLUMN, "market_std", "market_n"]],
        left_on=date_column, right_index=True, how="left", sort=False,
        validate="many_to_one",
    ).sort_values("_source_order", kind="stable").drop(columns="_source_order")
    result.index = panel.index

    source = result[source_target_column]
    observed_market = result[market_return_column]
    comparable = source.notna() & observed_market.notna()
    max_source_difference = (
        float((source[comparable] - observed_market[comparable]).abs().max())
        if comparable.any() else float("nan")
    )
    if comparable.any() and max_source_difference > 1e-10:
        raise ValueError(
            "panel O2O targets disagree with market returns; "
            f"max absolute difference={max_source_difference}"
        )

    result[MARKET_RESIDUAL_COLUMN] = source - result[MARKET_RETURN_COLUMN]
    valid_std = result["market_std"].where(result["market_std"].gt(0))
    result[CS_ZSCORE_COLUMN] = result[MARKET_RESIDUAL_COLUMN] / valid_std
    result[WINSOR_RESIDUAL_COLUMN] = (
        result["_winsor_return"] - result["_winsor_market_mean"]
    ).where(source.notna())
    result = result.drop(columns=[
        market_return_column, "_winsor_return", "_winsor_market_mean",
    ])

    audit = {
        "rows": int(len(result)),
        "eligible_market_rows": int(len(eligible_market)),
        "market_days": int(daily[MARKET_RETURN_COLUMN].notna().sum()),
        "min_stocks_per_day": int(min_stocks_per_day),
        "winsor_quantiles": [float(winsor_lower), float(winsor_upper)],
        "source_market_comparisons": int(comparable.sum()),
        "source_market_max_abs_difference": max_source_difference,
        "finite_targets": {
            column: int(pd.to_numeric(result[column], errors="coerce").notna().sum())
            for column in (
                source_target_column, MARKET_RESIDUAL_COLUMN, CS_ZSCORE_COLUMN,
                WINSOR_RESIDUAL_COLUMN,
            )
        },
    }
    return result, audit


__all__ = [
    "CS_ZSCORE_COLUMN", "MARKET_RESIDUAL_COLUMN", "MARKET_RETURN_COLUMN",
    "WINSOR_RESIDUAL_COLUMN", "attach_o2o_residual_targets",
]
