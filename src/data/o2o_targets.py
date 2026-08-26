"""Attach strictly aligned next-period O2O targets to announcement panels."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_TARGET_COLUMN = "next_day_open_to_open_return"
DEFAULT_LABEL_COLUMN = "next_day_open_to_open_label"


def _normalize_stock_ids(values: pd.Series) -> pd.Series:
    extracted = values.astype(str).str.extract(r"(\d{1,6})", expand=False)
    return extracted.str.zfill(6)


def _safe_bool(values: pd.Series) -> pd.Series:
    return values.astype("boolean").fillna(False).astype(bool)


def attach_o2o_targets(
    announcements: pd.DataFrame,
    market: pd.DataFrame,
    *,
    stock_column: str = "stock_id",
    date_column: str = "entry_date",
    market_return_column: str = "tradable_open_to_open_return",
    market_eligible_column: str = "eligible_signal",
    target_column: str = DEFAULT_TARGET_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> pd.DataFrame:
    """Merge a leakage-safe O2O target onto every announcement.

    ``date_column`` is the open at which the paper-style position is entered.
    The target therefore runs from that open to the next exchange-date open.
    A target is retained only when the entry is outside the configured IPO
    exclusion window and both opens were actually observed.
    """
    announcement_required = {stock_column, date_column}
    market_required = {
        stock_column,
        date_column,
        market_return_column,
        market_eligible_column,
        "observed_market",
        "next_observed_market",
    }
    missing_announcements = announcement_required.difference(announcements.columns)
    missing_market = market_required.difference(market.columns)
    if missing_announcements:
        raise ValueError(
            "announcement panel missing columns: "
            + ", ".join(sorted(missing_announcements))
        )
    if missing_market:
        raise ValueError(
            "market panel missing columns: " + ", ".join(sorted(missing_market))
        )

    frame = announcements.copy()
    frame[stock_column] = _normalize_stock_ids(frame[stock_column])
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()

    market_values = market.copy()
    market_values[stock_column] = _normalize_stock_ids(market_values[stock_column])
    market_values[date_column] = pd.to_datetime(
        market_values[date_column], errors="coerce"
    ).dt.normalize()
    keys = [stock_column, date_column]
    if market_values.duplicated(keys).any():
        examples = market_values.loc[market_values.duplicated(keys, keep=False), keys]
        raise ValueError(
            "market panel contains duplicate stock-date rows: "
            + str(examples.head(5).to_dict("records"))
        )

    audit_columns = [
        stock_column,
        date_column,
        market_return_column,
        market_eligible_column,
        "observed_market",
        "next_observed_market",
    ]
    if "can_buy" in market_values:
        audit_columns.append("can_buy")
    market_values = market_values[audit_columns].rename(columns={
        market_return_column: "_o2o_market_return",
        market_eligible_column: "_o2o_market_eligible",
        "observed_market": "o2o_entry_observed",
        "next_observed_market": "o2o_exit_observed",
        "can_buy": "o2o_can_buy",
    })
    replaceable = {
        target_column,
        label_column,
        "o2o_target_eligible",
        "o2o_entry_observed",
        "o2o_exit_observed",
        "o2o_can_buy",
        "_o2o_market_return",
        "_o2o_market_eligible",
    }
    frame = frame.drop(columns=[c for c in replaceable if c in frame], errors="ignore")
    result = frame.merge(market_values, on=keys, how="left", validate="many_to_one")

    numeric_return = pd.to_numeric(result["_o2o_market_return"], errors="coerce")
    eligible = (
        _safe_bool(result["_o2o_market_eligible"])
        & _safe_bool(result["o2o_entry_observed"])
        & _safe_bool(result["o2o_exit_observed"])
        & np.isfinite(numeric_return)
    )
    result["o2o_target_eligible"] = eligible
    result[target_column] = numeric_return.where(eligible)
    result[label_column] = pd.Series(pd.NA, index=result.index, dtype="Int8")
    result.loc[eligible, label_column] = (result.loc[eligible, target_column] > 0).astype(
        "int8"
    )
    return result.drop(columns=["_o2o_market_return", "_o2o_market_eligible"])


__all__ = [
    "DEFAULT_LABEL_COLUMN",
    "DEFAULT_TARGET_COLUMN",
    "attach_o2o_targets",
]
