import numpy as np
import pandas as pd
import pytest

from src.portfolio import (
    StrategyConfig,
    TransactionCostModel,
    backtest_ranked_strategy,
    strategy_metrics,
    portfolio_metrics,
)


def _day(date: str, assets: range, predictions: list[float], returns=None) -> list[dict]:
    returns = returns if returns is not None else [0.0] * len(predictions)
    return [
        {
            "entry_date": date,
            "stock_id": f"s{item}",
            "prediction": prediction,
            "actual_return": realized,
        }
        for item, prediction, realized in zip(assets, predictions, returns)
    ]


def test_paper_costs_and_net_return_are_applied_to_actual_trades() -> None:
    frame = pd.DataFrame(
        _day("2024-01-02", range(5), [1, 2, 3, 4, 5], [-0.01, 0, 0, 0, 0.03])
        + _day("2024-01-03", range(5), [5, 4, 3, 2, 1], [0.01, 0, 0, 0, -0.03])
    )
    result = backtest_ranked_strategy(frame, costs=TransactionCostModel.paper())

    # Day one opens one unit on each side.  The paper normalization charges
    # 10 bps in total; gross high-minus-low return is 4%.
    assert result.daily.loc[0, "gross_return"] == pytest.approx(0.04)
    assert result.daily.loc[0, "transaction_cost"] == pytest.approx(0.001)
    assert result.daily.loc[0, "net_return"] == pytest.approx(0.039)
    # Prior weights drift before the second rebalance, exactly as in the
    # turnover definition reported by the paper.
    assert result.daily.loc[1, "turnover"] == pytest.approx(1.005)
    assert result.daily.loc[1, "transaction_cost"] == pytest.approx(0.00201)
    metrics = strategy_metrics(result.daily)
    assert metrics["execution"]["total_cost_drag"] == pytest.approx(0.00301)


def test_ewct_gamma_controls_reallocation_and_weight_decay() -> None:
    signals = pd.DataFrame(
        _day("2024-01-02", range(5), [1, 2, 3, 4, 5])
        + _day("2024-01-03", range(5, 10), [1, 2, 3, 4, 5])
    )
    market = pd.DataFrame(
        [
            {"entry_date": date, "stock_id": f"s{item}", "ret": 0.0}
            for date in ("2024-01-02", "2024-01-03")
            for item in range(10)
        ]
    )
    result = backtest_ranked_strategy(
        signals,
        market_data=market,
        market_return="ret",
        config=StrategyConfig(ewct_gamma=0.4),
        costs=TransactionCostModel(name="none"),
    )

    day_two = result.holdings[result.holdings["date"].eq(pd.Timestamp("2024-01-03"))]
    weights = day_two.set_index("asset")["weight"]
    assert weights["s4"] == pytest.approx(0.6)
    assert weights["s0"] == pytest.approx(-0.6)
    assert weights["s9"] == pytest.approx(0.4)
    assert weights["s5"] == pytest.approx(-0.4)
    assert result.daily.loc[1, "turnover"] == pytest.approx(0.4)


def test_china_cost_model_charges_stamp_duty_only_on_sells() -> None:
    frame = pd.DataFrame(
        _day("2024-01-02", range(5), [1, 2, 3, 4, 5])
        + _day("2024-01-03", range(5), [5, 4, 3, 2, 1])
    )
    result = backtest_ranked_strategy(
        frame,
        config=StrategyConfig(long_only=True),
        costs=TransactionCostModel.china_a_share(
            commission_bps=3, stamp_duty_bps=5, slippage_bps=0
        ),
    )

    assert result.daily.loc[0, "transaction_cost"] == pytest.approx(0.0003)
    assert result.daily.loc[1, "transaction_cost"] == pytest.approx(0.0011)
    trades = result.trades[result.trades["date"].eq(pd.Timestamp("2024-01-03"))]
    assert trades.loc[trades["side"].eq("sell"), "trade_cost"].iloc[0] == pytest.approx(0.0008)
    assert trades.loc[trades["side"].eq("buy"), "trade_cost"].iloc[0] == pytest.approx(0.0003)


def test_short_only_uses_low_score_sleeve_and_single_sleeve_cost_basis() -> None:
    frame = pd.DataFrame(
        _day("2024-01-02", range(5), [1, 2, 3, 4, 5], [-0.03, 0, 0, 0, 0.01])
    )
    result = backtest_ranked_strategy(
        frame,
        config=StrategyConfig(short_only=True),
        costs=TransactionCostModel.paper(),
    )

    assert result.daily.loc[0, "n_long"] == 0
    assert result.daily.loc[0, "n_short"] == 1
    assert result.daily.loc[0, "gross_return"] == pytest.approx(0.03)
    assert result.daily.loc[0, "transaction_cost"] == pytest.approx(0.001)
    assert result.daily.loc[0, "net_return"] == pytest.approx(0.029)


def test_strategy_rejects_simultaneous_long_only_and_short_only() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        StrategyConfig(long_only=True, short_only=True)


def test_buy_sell_constraints_keep_the_existing_long_position() -> None:
    frame = pd.DataFrame(
        _day("2024-01-02", range(5), [1, 2, 3, 4, 5])
        + _day("2024-01-03", range(5), [5, 4, 3, 2, 1])
    )
    frame["can_buy"] = True
    frame["can_sell"] = True
    frame.loc[frame["entry_date"].eq("2024-01-03") & frame["stock_id"].eq("s0"), "can_buy"] = False
    frame.loc[frame["entry_date"].eq("2024-01-03") & frame["stock_id"].eq("s4"), "can_sell"] = False
    result = backtest_ranked_strategy(
        frame,
        can_buy="can_buy",
        can_sell="can_sell",
        config=StrategyConfig(long_only=True),
        costs=TransactionCostModel(name="none"),
    )

    day_two = result.holdings[result.holdings["date"].eq(pd.Timestamp("2024-01-03"))]
    weights = day_two.set_index("asset")["weight"]
    assert weights["s4"] == pytest.approx(1.0)
    assert weights.get("s0", 0.0) == pytest.approx(0.0)
    assert result.daily.loc[1, "turnover"] == pytest.approx(0.0)
    assert result.daily.loc[1, "blocked_notional"] == pytest.approx(2.0)


def test_shortability_blocks_new_short_but_not_the_long_sleeve() -> None:
    frame = pd.DataFrame(_day("2024-01-02", range(5), [1, 2, 3, 4, 5]))
    frame["shortable"] = True
    frame.loc[frame["stock_id"].eq("s0"), "shortable"] = False
    result = backtest_ranked_strategy(
        frame,
        shortable="shortable",
        costs=TransactionCostModel(name="none"),
    )

    assert result.daily.loc[0, "n_long"] == 1
    assert result.daily.loc[0, "n_short"] == 0
    assert result.daily.loc[0, "blocked_notional"] == pytest.approx(0.5)


def test_ewct_requires_dense_returns_for_positions_kept_after_news() -> None:
    signals = pd.DataFrame(
        _day("2024-01-02", range(5), [1, 2, 3, 4, 5])
        + _day("2024-01-03", range(5, 10), [1, 2, 3, 4, 5])
    )
    with pytest.raises(ValueError, match="provide dense market_data"):
        backtest_ranked_strategy(
            signals,
            config=StrategyConfig(ewct_gamma=0.5),
            costs=TransactionCostModel(name="none"),
        )


def test_paper_small_stock_cost_uses_daily_market_cap_breakpoint() -> None:
    frame = pd.DataFrame(_day("2024-01-02", range(5), [1, 2, 3, 4, 5]))
    frame["market_cap"] = [1, 2, 3, 4, 5]
    result = backtest_ranked_strategy(
        frame,
        market_cap="market_cap",
        costs=TransactionCostModel.paper(market_cap_column="market_cap"),
    )
    # Short s0 is below the 20% breakpoint (20 bps), long s4 is large (10 bps).
    assert result.daily.loc[0, "transaction_cost"] == pytest.approx(0.0015)
    assert set(result.trades.loc[result.trades["is_small_stock"], "asset"]) == {"s0"}


def test_result_has_finite_cumulative_net_return() -> None:
    frame = pd.DataFrame(_day("2024-01-02", range(5), [1, 2, 3, 4, 5]))
    result = backtest_ranked_strategy(frame)
    assert np.isfinite(result.daily.loc[0, "cumulative_net_return"])


def test_eligibility_is_applied_before_cross_sectional_ranking() -> None:
    frame = pd.DataFrame(_day("2024-01-02", range(6), [1, 2, 3, 4, 5, 100]))
    frame["eligible"] = True
    frame.loc[frame["stock_id"].eq("s5"), "eligible"] = False
    result = backtest_ranked_strategy(
        frame,
        eligible="eligible",
        config=StrategyConfig(long_only=True),
        costs=TransactionCostModel(name="none"),
    )
    held = result.holdings.loc[result.holdings["weight"].gt(0), "asset"].tolist()
    assert held == ["s4"]
    assert result.daily.loc[0, "n_signal_raw"] == 6
    assert result.daily.loc[0, "n_signal"] == 5


def test_performance_metrics_include_initial_wealth_in_drawdown() -> None:
    portfolio = pd.DataFrame({"net_return": [-0.10, 0.05]})
    metrics = portfolio_metrics(portfolio, return_column="net_return")
    assert metrics["max_drawdown"] == pytest.approx(-0.10)
    assert metrics["worst_day"] == pytest.approx(-0.10)
    assert np.isfinite(metrics["mean_t_stat"])


def test_one_period_signal_closes_on_next_market_date() -> None:
    signals = pd.DataFrame(
        _day("2024-01-02", range(5), [1, 2, 3, 4, 5])
    )
    market = pd.DataFrame([
        {
            "entry_date": date,
            "stock_id": f"s{item}",
            "ret": 0.02 if item == 4 else -0.01 if item == 0 else 0.0,
        }
        for date in ("2024-01-02", "2024-01-03")
        for item in range(5)
    ])
    result = backtest_ranked_strategy(
        signals,
        market_data=market,
        market_return="ret",
        config=StrategyConfig(no_signal_action="close"),
        costs=TransactionCostModel(name="custom", commission_bps=5.0),
    )

    assert result.daily["date"].tolist() == [
        pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")
    ]
    assert result.daily.loc[0, "n_long"] == 1
    assert result.daily.loc[0, "n_short"] == 1
    assert result.daily.loc[1, "n_long"] == 0
    assert result.daily.loc[1, "n_short"] == 0
    assert result.daily.loc[1, "transaction_cost"] == pytest.approx(0.0005025)


def test_long_short_gross_normalization_matches_cost_capital_basis() -> None:
    frame = pd.DataFrame(
        _day("2024-01-02", range(5), [1, 2, 3, 4, 5], [-0.01, 0, 0, 0, 0.03])
    )
    result = backtest_ranked_strategy(
        frame,
        config=StrategyConfig(long_short_normalization="gross"),
        costs=TransactionCostModel(name="custom", commission_bps=5.0),
    )

    assert result.daily.loc[0, "gross_return"] == pytest.approx(0.02)
    assert result.daily.loc[0, "transaction_cost"] == pytest.approx(0.0005)


def test_eligibility_with_dense_market_allows_dates_without_signals() -> None:
    signals = pd.DataFrame(
        _day("2024-01-02", range(5), [1, 2, 3, 4, 5])
        + _day("2024-01-04", range(5), [1, 2, 3, 4, 5])
    )
    market = pd.DataFrame([
        {
            "entry_date": date,
            "stock_id": f"s{item}",
            "ret": 0.0,
            "eligible": True,
        }
        for date in ("2024-01-02", "2024-01-03", "2024-01-04")
        for item in range(5)
    ])

    result = backtest_ranked_strategy(
        signals,
        market_data=market,
        market_return="ret",
        eligible="eligible",
        config=StrategyConfig(ewct_gamma=0.1),
        costs=TransactionCostModel(name="none"),
    )

    assert result.daily["date"].tolist() == [
        pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
    ]
    assert result.daily.loc[1, "n_signal"] == 0
