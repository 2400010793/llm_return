"""Attach simple_states-aligned one-day return targets to a text panel."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _ticker(value: object) -> str | None:
    match = re.search(r"(\d{6})", str(value or ""))
    if not match:
        return None
    code = match.group(1)
    return f"{code}.{'SH' if code.startswith(('5', '6')) else 'SZ'}"


def _universe_bool(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.dtypes.eq(object).any() or str(frame.dtypes.iloc[0]) == "category":
        text = frame.astype("string")
        return text.notna() & ~text.isin(["ST", "NEW"])
    return frame.fillna(False).astype(bool)


def attach_simple_states_target(
    panel: pd.DataFrame,
    *,
    stock_return: pd.DataFrame,
    universe: pd.DataFrame,
    amount: pd.DataFrame,
    riseboard: pd.DataFrame,
    jumpboard: pd.DataFrame,
    min_amount: float = 20_000_000,
    amount_window: int = 20,
    target_column: str = "simple_states_ret_vv_return",
    eligible_column: str = "simple_states_target_eligible",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Map entry-date returns and the platform's tradable universe to rows."""
    required = {"stock_id", "entry_date"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {', '.join(sorted(missing))}")
    if amount_window < 1:
        raise ValueError("amount_window must be positive")

    calendar = pd.DatetimeIndex(pd.to_datetime(universe.index)).normalize()
    columns = pd.Index(universe.columns.astype(str))
    if calendar.has_duplicates or columns.has_duplicates:
        raise ValueError("UNIVERSE axes must be unique")
    result = panel.drop(columns=[target_column, eligible_column], errors="ignore").copy()
    entry_dates = pd.DatetimeIndex(
        pd.to_datetime(result["entry_date"], errors="coerce").dt.normalize()
    )
    tickers = pd.Index(result["stock_id"].map(_ticker))
    entry_positions = calendar.get_indexer(entry_dates)
    ticker_positions = columns.get_indexer(tickers)
    observation_positions = entry_positions - 1
    valid_axis = (entry_positions >= 1) & (ticker_positions >= 0)

    universe_values = _universe_bool(
        universe.reindex(index=calendar, columns=columns)
    ).to_numpy(dtype=bool)
    amount_aligned = amount.reindex(index=calendar, columns=columns)
    amount_values = amount_aligned.to_numpy(dtype=float)
    amount_ma = amount_aligned.rolling(amount_window, min_periods=amount_window).mean()
    amount_ma_values = amount_ma.to_numpy(dtype=float)
    return_values = stock_return.reindex(index=calendar, columns=columns).to_numpy(
        dtype=float
    )
    rise_values = riseboard.reindex(index=calendar, columns=columns).to_numpy(
        dtype=float, na_value=np.nan
    )
    jump_values = jumpboard.reindex(index=calendar, columns=columns).to_numpy(
        dtype=float, na_value=np.nan
    )

    target = np.full(len(result), np.nan, dtype=float)
    in_universe = np.zeros(len(result), dtype=bool)
    liquid = np.zeros(len(result), dtype=bool)
    board_known = np.zeros(len(result), dtype=bool)
    return_known = np.zeros(len(result), dtype=bool)
    rows = np.flatnonzero(valid_axis)
    obs = observation_positions[rows]
    ent = entry_positions[rows]
    cols = ticker_positions[rows]
    in_universe[rows] = universe_values[obs, cols]
    liquid[rows] = (
        np.isfinite(amount_values[obs, cols])
        & (amount_ma_values[obs, cols] >= float(min_amount))
    )
    board_known[rows] = ~(
        np.isnan(rise_values[ent, cols]) & np.isnan(jump_values[ent, cols])
    )
    candidate_returns = return_values[ent, cols]
    return_known[rows] = np.isfinite(candidate_returns)
    eligible = valid_axis & in_universe & liquid & board_known & return_known
    target[eligible] = return_values[
        entry_positions[eligible], ticker_positions[eligible]
    ]
    result[target_column] = target
    result[eligible_column] = eligible

    years = entry_dates.year
    yearly = {}
    for year in sorted(set(int(value) for value in years[~pd.isna(years)])):
        mask = np.asarray(years == year)
        yearly[str(year)] = {
            "rows": int(mask.sum()),
            "eligible": int((eligible & mask).sum()),
            "coverage": float((eligible & mask).sum() / mask.sum()),
        }
    audit = {
        "rows": int(len(result)),
        "target_column": target_column,
        "eligible_column": eligible_column,
        "target_definition": (
            "ret_vv at entry_date; factor observation date is the previous "
            "UNIVERSE trading day and simple_states label_lag is 1"
        ),
        "valid_axes": int(valid_axis.sum()),
        "in_universe": int((valid_axis & in_universe).sum()),
        "liquid": int((valid_axis & liquid).sum()),
        "board_known": int((valid_axis & board_known).sum()),
        "return_known": int((valid_axis & return_known).sum()),
        "eligible": int(eligible.sum()),
        "coverage": float(eligible.mean()),
        "min_amount": float(min_amount),
        "amount_window": int(amount_window),
        "yearly": yearly,
    }
    return result, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/alpha_team2/shares/simple_states_data"),
    )
    parser.add_argument("--min-amount", type=float, default=20_000_000)
    parser.add_argument("--amount-window", type=int, default=20)
    parser.add_argument("--target-column", default="simple_states_ret_vv_return")
    parser.add_argument("--eligible-column", default="simple_states_target_eligible")
    args = parser.parse_args()

    root = args.data_root
    result, audit = attach_simple_states_target(
        pd.read_parquet(args.panel),
        stock_return=pd.read_parquet(root / "ret_vv.parquet"),
        universe=pd.read_parquet(root / "UNIVERSE.parquet"),
        amount=pd.read_parquet(root / "amount.parquet"),
        riseboard=pd.read_parquet(root / "riseboard.parquet"),
        jumpboard=pd.read_parquet(root / "jumpboard.parquet"),
        min_amount=args.min_amount,
        amount_window=args.amount_window,
        target_column=args.target_column,
        eligible_column=args.eligible_column,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    summary = {
        "format_version": "simple_states_training_panel_v1",
        "input": str(args.panel),
        "data_root": str(root),
        "output": str(args.output),
        **audit,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
