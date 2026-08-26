import pandas as pd

from scripts.analyze_qwen_long_selection import daily_decomposition, select_count


def test_select_count_requires_breadth():
    assert select_count("top20pct", 5) == 1
    assert select_count("top5", 9) == 0
    assert select_count("top5", 10) == 5


def test_daily_decomposition_identity_and_selection():
    date = pd.Timestamp("2026-01-02")
    predictions = pd.DataFrame({
        "entry_date": [date] * 5,
        "stock_id": [f"00000{i}" for i in range(1, 6)],
        "prediction": [1, 2, 3, 4, 5],
    })
    market = pd.DataFrame({
        "entry_date": [date] * 6,
        "stock_id": [f"00000{i}" for i in range(1, 7)],
        "open_to_open_return": [-0.03, -0.02, -0.01, 0.01, 0.05, 0.06],
        "eligible_signal": [True] * 6,
    })
    result = daily_decomposition(
        predictions, market, prediction_column="prediction",
        return_column="open_to_open_return", rule="top20pct",
    ).iloc[0]
    assert result.selected_count == 1
    assert result.selected_return == 0.05
    assert abs(result.within_pool_excess + result.coverage_gap - result.full_universe_excess) < 1e-12
