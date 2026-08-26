import numpy as np
import pandas as pd
import pytest

from scripts.compare_prompt_cluster_logistic_backtests import (
    block_bootstrap_difference,
    build_common_signals,
)


def test_common_signals_use_strict_executable_intersection() -> None:
    cluster = pd.DataFrame({
        "stock_id": ["000001", "000002", "000003"],
        "entry_date": pd.to_datetime(["2024-01-02"] * 3),
        "cluster_score": [-0.1, 0.0, 0.1],
        "label_return": [-0.02, 0.01, 0.03],
    })
    logistic = pd.DataFrame({
        "stock_id": ["000001", "000002", "000003"],
        "entry_date": pd.to_datetime(["2024-01-02"] * 3),
        "logistic_plain": [0.2, 0.5, 0.8],
    })
    market = pd.DataFrame({
        "stock_id": ["000001", "000002", "000003"],
        "entry_date": pd.to_datetime(["2024-01-02"] * 3),
        "open_to_open_return": [-0.01, np.nan, 0.02],
        "can_buy": [True] * 3,
        "can_sell": [True] * 3,
        "eligible_signal": [True, True, False],
    })

    common, audit = build_common_signals(cluster, [("plain", logistic)], market)

    assert common["stock_id"].tolist() == ["000001"]
    assert audit["prediction_common_stock_days"] == 3
    assert audit["finite_open_to_open_stock_days"] == 2
    assert audit["eligible_executable_stock_days"] == 1


def test_block_bootstrap_reports_positive_paired_advantage() -> None:
    cluster = np.full(100, 0.002)
    baseline = np.full(100, 0.001)

    result = block_bootstrap_difference(
        cluster, baseline, samples=100, block_length=10, seed=7,
    )

    assert result["annualized_mean_difference"] == pytest.approx(0.252)
    assert result["ci_low_annualized"] == pytest.approx(0.252)
    assert result["bootstrap_probability_positive"] == pytest.approx(1.0)
