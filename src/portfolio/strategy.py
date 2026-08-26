"""Holdings-level news strategy backtests with costs and trading constraints.

The implementation supports the paper's exponentially-weighted calendar-time
(EWCT) portfolio as well as an implementable, long-only China A-share variant.
Signals and returns use the same date: a signal assigned to ``entry_date`` is
traded at that date's open and earns that date's open-to-open return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


Weighting = Literal["equal", "value"]
MissingReturnPolicy = Literal["raise", "zero"]
NoSignalAction = Literal["hold", "close"]
LongShortNormalization = Literal["spread", "gross"]


@dataclass(frozen=True)
class TransactionCostModel:
    """Per-trade cost assumptions expressed in basis points.

    ``paper`` applies a symmetric 10/20 bps rate to large/small stocks.  Small
    stocks can be supplied explicitly or approximated by the bottom market-cap
    quantile of the daily universe.  With neither column, all stocks use the
    large-stock rate unless ``strict_size`` is enabled.

    ``custom`` supports asymmetric buy/sell costs.  Stamp duty is sell-only,
    while commission and slippage apply in both directions.
    """

    name: Literal["none", "paper", "custom"] = "paper"
    large_stock_bps: float = 10.0
    small_stock_bps: float = 20.0
    commission_bps: float = 0.0
    stamp_duty_bps: float = 0.0
    slippage_bps: float = 0.0
    short_borrow_bps_annual: float = 0.0
    market_cap_column: str | None = None
    small_stock_column: str | None = None
    small_cap_quantile: float = 0.20
    strict_size: bool = False

    def __post_init__(self) -> None:
        rates = (
            self.large_stock_bps,
            self.small_stock_bps,
            self.commission_bps,
            self.stamp_duty_bps,
            self.slippage_bps,
            self.short_borrow_bps_annual,
        )
        if any(not np.isfinite(value) or value < 0 for value in rates):
            raise ValueError("transaction-cost rates must be finite and non-negative")
        if not 0 < self.small_cap_quantile < 1:
            raise ValueError("small_cap_quantile must lie strictly between zero and one")

    @classmethod
    def paper(
        cls,
        *,
        market_cap_column: str | None = None,
        small_stock_column: str | None = None,
        strict_size: bool = False,
        short_borrow_bps_annual: float = 0.0,
    ) -> "TransactionCostModel":
        """Paper cost model: 10 bps large stocks and 20 bps small stocks."""
        return cls(
            name="paper",
            market_cap_column=market_cap_column,
            small_stock_column=small_stock_column,
            strict_size=strict_size,
            short_borrow_bps_annual=short_borrow_bps_annual,
        )

    @classmethod
    def china_a_share(
        cls,
        *,
        commission_bps: float = 3.0,
        stamp_duty_bps: float = 5.0,
        slippage_bps: float = 0.0,
    ) -> "TransactionCostModel":
        """Configurable A-share costs; stamp duty is charged on sells only."""
        return cls(
            name="custom",
            commission_bps=commission_bps,
            stamp_duty_bps=stamp_duty_bps,
            slippage_bps=slippage_bps,
        )


@dataclass(frozen=True)
class StrategyConfig:
    """Portfolio formation and execution assumptions."""

    quantiles: int = 5
    weighting: Weighting = "equal"
    ewct_gamma: float = 1.0
    long_only: bool = False
    short_only: bool = False
    min_stocks_per_day: int = 5
    missing_return: MissingReturnPolicy = "raise"
    t_plus_one: bool = True
    no_signal_action: NoSignalAction = "hold"
    long_short_normalization: LongShortNormalization = "spread"

    def __post_init__(self) -> None:
        if self.quantiles < 2:
            raise ValueError("quantiles must be at least 2")
        if self.weighting not in {"equal", "value"}:
            raise ValueError("weighting must be 'equal' or 'value'")
        if not 0 < self.ewct_gamma <= 1:
            raise ValueError("ewct_gamma must lie in (0, 1]")
        if self.long_only and self.short_only:
            raise ValueError("long_only and short_only are mutually exclusive")
        if self.min_stocks_per_day < self.quantiles:
            raise ValueError("min_stocks_per_day cannot be smaller than quantiles")
        if self.missing_return not in {"raise", "zero"}:
            raise ValueError("missing_return must be 'raise' or 'zero'")
        if self.no_signal_action not in {"hold", "close"}:
            raise ValueError("no_signal_action must be 'hold' or 'close'")
        if self.long_short_normalization not in {"spread", "gross"}:
            raise ValueError(
                "long_short_normalization must be 'spread' or 'gross'"
            )


@dataclass
class StrategyResult:
    """Auditable daily results, holdings, and trades from one backtest."""

    daily: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame


def _required(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {', '.join(sorted(missing))}")


def _as_bool(value: object, default: bool = True) -> bool:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _normalize_sleeve(
    group: pd.DataFrame,
    assets: pd.Index,
    *,
    weighting: Weighting,
    market_cap_column: str | None,
) -> dict[object, float]:
    if len(assets) == 0:
        return {}
    if weighting == "equal":
        values = np.ones(len(assets), dtype=float)
    else:
        if market_cap_column is None:
            raise ValueError("value weighting requires market_cap_column")
        values = pd.to_numeric(
            group.set_index("__asset").loc[assets, market_cap_column], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("value weighting requires positive finite market caps")
    values /= values.sum()
    return dict(zip(assets.tolist(), values.tolist()))


def _signal_weights(
    group: pd.DataFrame,
    *,
    config: StrategyConfig,
    market_cap_column: str | None,
) -> dict[object, float]:
    eligible = group[np.isfinite(pd.to_numeric(group["__prediction"], errors="coerce"))]
    if len(eligible) < config.min_stocks_per_day:
        return {}
    ranks = eligible["__prediction"].rank(method="first")
    buckets = pd.qcut(ranks, q=config.quantiles, labels=False, duplicates="drop")
    if buckets.nunique() != config.quantiles:
        return {}
    low_assets = pd.Index(eligible.loc[buckets.eq(0), "__asset"])
    high_assets = pd.Index(eligible.loc[buckets.eq(config.quantiles - 1), "__asset"])
    weights: dict[object, float] = {}
    if not config.short_only:
        weights = _normalize_sleeve(
            eligible, high_assets, weighting=config.weighting,
            market_cap_column=market_cap_column,
        )
    if not config.long_only:
        short_weights = _normalize_sleeve(
            eligible, low_assets, weighting=config.weighting,
            market_cap_column=market_cap_column,
        )
        weights.update({asset: -weight for asset, weight in short_weights.items()})
    return weights


def _small_stock_map(
    market: pd.DataFrame,
    assets: set[object],
    costs: TransactionCostModel,
) -> dict[object, bool]:
    if costs.name != "paper":
        return {asset: False for asset in assets}
    indexed = market.set_index("__asset", drop=False)
    if costs.small_stock_column:
        return {
            asset: _as_bool(
                indexed.at[asset, costs.small_stock_column]
                if asset in indexed.index else None,
                default=False,
            )
            for asset in assets
        }
    if costs.market_cap_column:
        caps = pd.to_numeric(market[costs.market_cap_column], errors="coerce")
        finite = caps[np.isfinite(caps) & caps.gt(0)]
        if finite.empty:
            raise ValueError("paper cost model found no positive finite market caps")
        breakpoint = float(finite.quantile(costs.small_cap_quantile))
        cap_map = dict(zip(market["__asset"], caps))
        return {
            asset: bool(np.isfinite(cap_map.get(asset, np.nan)) and cap_map[asset] < breakpoint)
            for asset in assets
        }
    if costs.strict_size:
        raise ValueError(
            "paper cost model requires small_stock_column or market_cap_column "
            "when strict_size=True"
        )
    return {asset: False for asset in assets}


def _trade_rates(
    costs: TransactionCostModel,
    *,
    is_small: bool,
) -> tuple[float, float]:
    if costs.name == "none":
        return 0.0, 0.0
    if costs.name == "paper":
        rate = costs.small_stock_bps if is_small else costs.large_stock_bps
        return rate / 10_000.0, rate / 10_000.0
    buy = costs.commission_bps + costs.slippage_bps
    sell = costs.commission_bps + costs.slippage_bps + costs.stamp_duty_bps
    return buy / 10_000.0, sell / 10_000.0


def backtest_ranked_strategy(
    signals: pd.DataFrame,
    *,
    prediction: str = "prediction",
    realized: str = "actual_return",
    date: str = "entry_date",
    asset: str = "stock_id",
    market_data: pd.DataFrame | None = None,
    market_return: str | None = None,
    market_cap: str | None = None,
    can_buy: str | None = None,
    can_sell: str | None = None,
    shortable: str | None = None,
    eligible: str | None = None,
    config: StrategyConfig | None = None,
    costs: TransactionCostModel | None = None,
) -> StrategyResult:
    """Backtest a daily ranked strategy with holdings, costs, and constraints.

    When ``ewct_gamma < 1``, a dense ``market_data`` panel is normally needed
    because positions remain alive after their news date.  Missing held-position
    returns raise by default rather than being silently treated as zero.

    ``eligible`` filters the cross-section *before* ranking.  ``can_buy`` and
    ``can_sell`` then describe feasibility at the rebalance
    open.  They can encode suspension and limit-up/down rules upstream.  The
    ``shortable`` column controls only the opening or expansion of short stock.
    With one rebalance per trading date, T+1 is naturally respected: positions
    bought at one open cannot be sold until a later date in this daily engine.
    """
    config = config or StrategyConfig()
    costs = costs or TransactionCostModel.paper(market_cap_column=market_cap)
    _required(signals, {prediction, date, asset}, "signals")
    if market_data is None:
        _required(signals, {realized}, "signals")
        market = signals.copy()
        return_column = realized
    else:
        return_column = market_return or realized
        _required(market_data, {date, asset, return_column}, "market_data")
        market = market_data.copy()
    optional_columns = {
        value for value in (
            market_cap, can_buy, can_sell, shortable, eligible,
            costs.market_cap_column, costs.small_stock_column,
        ) if value
    }
    _required(market, optional_columns, "market_data")
    if config.weighting == "value":
        if market_cap is None:
            raise ValueError("value weighting requires market_cap")
        _required(signals if market_data is None else market, {market_cap}, "market data")

    signal = signals.copy()
    signal[date] = pd.to_datetime(signal[date], errors="coerce")
    market[date] = pd.to_datetime(market[date], errors="coerce")
    signal = signal.dropna(subset=[date, asset])
    market = market.dropna(subset=[date, asset])
    if signal.duplicated([date, asset]).any():
        raise ValueError("signals must contain at most one row per date and asset")
    if market.duplicated([date, asset]).any():
        raise ValueError("market_data must contain at most one row per date and asset")
    if signal.empty:
        return StrategyResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    signal = signal.rename(columns={asset: "__asset", date: "__date", prediction: "__prediction"})
    market = market.rename(columns={asset: "__asset", date: "__date", return_column: "__return"})
    start, end = signal["__date"].min(), signal["__date"].max()
    if market_data is not None and config.no_signal_action == "close":
        later_dates = market.loc[market["__date"].gt(end), "__date"]
        if not later_dates.empty:
            # A one-period signal earns the return beginning on its signal date
            # and pays its closing trade at the next market open.
            end = later_dates.min()
    market = market[market["__date"].between(start, end)].copy()
    dates = sorted(set(signal["__date"]) | set(market["__date"]))
    signal_groups = {day: group for day, group in signal.groupby("__date", sort=False)}
    market_groups = {day: group for day, group in market.groupby("__date", sort=False)}

    desired_previous: dict[object, float] = {}
    executed_previous: dict[object, float] = {}
    previous_returns: dict[object, float] = {}
    first_signal_portfolio = True
    daily_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    tolerance = 1e-14

    for day in dates:
        signal_day = signal_groups.get(day, signal.iloc[0:0])
        n_signal_raw = len(signal_day)
        market_day = market_groups.get(day, market.iloc[0:0])
        if market_day.empty and not signal_day.empty and market_data is None:
            market_day = signal_day
        if market_data is not None and not signal_day.empty:
            signal_day = signal_day.merge(
                market_day.drop(columns=["__date"], errors="ignore"),
                on="__asset", how="left", suffixes=("", "__market"),
            )
        if eligible and not signal_day.empty:
            signal_day = signal_day[
                signal_day[eligible].map(lambda value: _as_bool(value, default=False))
            ]
        signal_weights = _signal_weights(
            signal_day,
            config=config,
            market_cap_column=market_cap,
        )
        if signal_weights:
            if first_signal_portfolio:
                desired = dict(signal_weights)
                first_signal_portfolio = False
            else:
                assets_for_target = set(desired_previous) | set(signal_weights)
                desired = {
                    item: (
                        (1.0 - config.ewct_gamma) * desired_previous.get(item, 0.0)
                        + config.ewct_gamma * signal_weights.get(item, 0.0)
                    )
                    for item in assets_for_target
                }
        else:
            desired = (
                {} if config.no_signal_action == "close"
                else dict(desired_previous)
            )
        desired = {item: value for item, value in desired.items() if abs(value) > tolerance}
        if not desired and not executed_previous:
            continue

        # The paper's turnover definition compares new weights with prior
        # weights drifted by the return earned over the previous holding day.
        pretrade = {
            item: weight * (1.0 + previous_returns.get(item, 0.0))
            for item, weight in executed_previous.items()
        }
        indexed = market_day.set_index("__asset", drop=False)
        all_assets = set(desired) | set(pretrade)
        is_small = _small_stock_map(market_day, all_assets, costs)
        executed: dict[object, float] = {}
        requested_deltas: dict[object, float] = {}
        executed_deltas: dict[object, float] = {}
        blocked_deltas: dict[object, float] = {}
        trade_cost_by_asset: dict[object, float] = {}

        for item in all_assets:
            current = pretrade.get(item, 0.0)
            requested = desired.get(item, 0.0)
            metadata = indexed.loc[item] if item in indexed.index else None
            if isinstance(metadata, pd.DataFrame):
                metadata = metadata.iloc[0]
            buy_allowed = _as_bool(
                metadata.get(can_buy) if metadata is not None and can_buy else None,
                default=not bool(can_buy),
            )
            sell_allowed = _as_bool(
                metadata.get(can_sell) if metadata is not None and can_sell else None,
                default=not bool(can_sell),
            )
            short_allowed = _as_bool(
                metadata.get(shortable) if metadata is not None and shortable else None,
                default=not bool(shortable),
            )
            candidate = requested
            if not short_allowed and candidate < min(current, 0.0):
                candidate = min(current, 0.0)
            delta = candidate - current
            if delta > tolerance and not buy_allowed:
                candidate = current
            elif delta < -tolerance and not sell_allowed:
                candidate = current
            executed_delta = candidate - current
            if abs(candidate) > tolerance:
                executed[item] = candidate
            requested_delta = requested - current
            requested_deltas[item] = requested_delta
            executed_deltas[item] = executed_delta
            blocked_deltas[item] = requested_delta - executed_delta
            buy_rate, sell_rate = _trade_rates(costs, is_small=is_small[item])
            trade_cost_by_asset[item] = (
                max(executed_delta, 0.0) * buy_rate
                + max(-executed_delta, 0.0) * sell_rate
            )

        # The academic long-short return invests one unit in each sleeve.  Cost
        # and turnover are normalized by the number of active sleeves, matching
        # the paper's 2 * turnover * one-way-cost convention.
        active_sleeves = 1 if config.long_only or config.short_only else 2
        buy_turnover = sum(max(value, 0.0) for value in executed_deltas.values()) / active_sleeves
        sell_turnover = sum(max(-value, 0.0) for value in executed_deltas.values()) / active_sleeves
        turnover = 0.5 * (buy_turnover + sell_turnover)
        requested_turnover = (
            0.5 * sum(abs(value) for value in requested_deltas.values()) / active_sleeves
        )
        transaction_cost = sum(trade_cost_by_asset.values()) / active_sleeves

        returns: dict[object, float] = {}
        missing_held: list[object] = []
        for item, weight in executed.items():
            raw = indexed.at[item, "__return"] if item in indexed.index else np.nan
            if isinstance(raw, pd.Series):
                raw = raw.iloc[0]
            value = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
            if not np.isfinite(value):
                missing_held.append(item)
                value = 0.0
            returns[item] = float(value)
        if missing_held and config.missing_return == "raise":
            preview = ", ".join(map(str, missing_held[:5]))
            raise ValueError(
                f"missing return for {len(missing_held)} held assets on {day.date()}: {preview}; "
                "provide dense market_data or use missing_return='zero' explicitly"
            )

        long_pnl = sum(weight * returns[item] for item, weight in executed.items() if weight > 0)
        short_pnl = sum(weight * returns[item] for item, weight in executed.items() if weight < 0)
        pnl_divisor = (
            active_sleeves
            if not (config.long_only or config.short_only)
            and config.long_short_normalization == "gross"
            else 1
        )
        long_pnl /= pnl_divisor
        short_pnl /= pnl_divisor
        gross_return = long_pnl + short_pnl
        short_exposure = sum(-weight for weight in executed.values() if weight < 0)
        borrow_cost = (
            short_exposure * costs.short_borrow_bps_annual / 10_000.0 / 252.0
            / active_sleeves
        )
        net_return = gross_return - transaction_cost - borrow_cost
        blocked_notional = sum(abs(value) for value in blocked_deltas.values()) / active_sleeves

        for item in sorted(all_assets, key=str):
            weight = executed.get(item, 0.0)
            contribution = weight * returns.get(item, 0.0)
            holding_rows.append({
                "date": day,
                "asset": item,
                "signal_weight": signal_weights.get(item, 0.0),
                "target_weight": desired.get(item, 0.0),
                "pretrade_weight": pretrade.get(item, 0.0),
                "weight": weight,
                "realized_return": returns.get(item, np.nan),
                "gross_contribution": contribution,
                "is_small_stock": is_small[item],
                "missing_return_filled": item in missing_held,
            })
            if abs(requested_deltas[item]) > tolerance or abs(executed_deltas[item]) > tolerance:
                delta = executed_deltas[item]
                trade_rows.append({
                    "date": day,
                    "asset": item,
                    "requested_delta": requested_deltas[item],
                    "executed_delta": delta,
                    "blocked_delta": blocked_deltas[item],
                    "side": "buy" if delta > tolerance else "sell" if delta < -tolerance else "blocked",
                    "trade_cost": trade_cost_by_asset[item] / active_sleeves,
                    "is_small_stock": is_small[item],
                })

        daily_rows.append({
            "date": day,
            "gross_long_return": long_pnl,
            "gross_short_return": short_pnl,
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "borrow_cost": borrow_cost,
            "net_return": net_return,
            "turnover": turnover,
            "requested_turnover": requested_turnover,
            "buy_turnover": buy_turnover,
            "sell_turnover": sell_turnover,
            "blocked_notional": blocked_notional,
            "gross_exposure": sum(abs(value) for value in executed.values()),
            "net_exposure": sum(executed.values()),
            "n_long": sum(value > tolerance for value in executed.values()),
            "n_short": sum(value < -tolerance for value in executed.values()),
            "n_signal": len(signal_day),
            "n_signal_raw": n_signal_raw,
            "n_missing_returns_filled": len(missing_held),
        })
        desired_previous = desired
        executed_previous = executed
        previous_returns = returns

    daily = pd.DataFrame(daily_rows)
    if not daily.empty:
        daily["cumulative_gross_return"] = (1.0 + daily["gross_return"]).cumprod() - 1.0
        daily["cumulative_net_return"] = (1.0 + daily["net_return"]).cumprod() - 1.0
    return StrategyResult(
        daily=daily,
        holdings=pd.DataFrame(holding_rows),
        trades=pd.DataFrame(trade_rows),
    )


__all__ = [
    "StrategyConfig",
    "StrategyResult",
    "TransactionCostModel",
    "backtest_ranked_strategy",
]
