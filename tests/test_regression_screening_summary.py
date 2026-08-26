from scripts.summarize_regression_screening import selection_key


def test_regression_screen_selection_prioritizes_rank_ic() -> None:
    high_ic = {
        "rank_ic_mean": 0.05,
        "oos_r2_vs_historical_mean": -1.0,
        "mse": 2.0,
    }
    high_r2 = {
        "rank_ic_mean": 0.04,
        "oos_r2_vs_historical_mean": 0.2,
        "mse": 0.1,
    }
    assert selection_key(high_ic) > selection_key(high_r2)
