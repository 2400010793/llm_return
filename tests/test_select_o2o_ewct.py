from scripts.select_o2o_ewct import selection_key


def test_selection_key_prioritizes_net_sharpe_then_mean_then_turnover() -> None:
    low_sharpe = {"net_sharpe": 1.0, "net_mean": 0.02, "mean_daily_turnover": 0.1}
    high_sharpe = {"net_sharpe": 1.1, "net_mean": 0.01, "mean_daily_turnover": 0.9}
    assert selection_key(high_sharpe) > selection_key(low_sharpe)

    high_mean = {"net_sharpe": 1.1, "net_mean": 0.02, "mean_daily_turnover": 0.9}
    assert selection_key(high_mean) > selection_key(high_sharpe)

    low_turnover = {"net_sharpe": 1.1, "net_mean": 0.02, "mean_daily_turnover": 0.2}
    assert selection_key(low_turnover) > selection_key(high_mean)
