import numpy as np
import pandas as pd

from src.data.rolling_semantic_dedup import rolling_prior_cosine


def test_rolling_prior_cosine_is_stock_local_time_ordered_and_windowed() -> None:
    frame = pd.DataFrame({
        "row_index": [1, 2, 3, 4, 5],
        "stock_id": ["A", "A", "A", "B", "A"],
        "entry_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-11", "2024-01-03", "2024-01-03"]),
        "published_at": pd.to_datetime(["2024-01-01 18:00", "2024-01-02 18:00", "2024-01-10 18:00", "2024-01-02 17:00", "2024-01-02 19:00"]),
    })
    matrix = np.array([
        [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.8, 0.6],
    ], dtype=np.float32)
    calendar = pd.date_range("2024-01-02", "2024-01-11", freq="B")
    maximum, source = rolling_prior_cosine(frame, matrix, trading_dates=calendar, lookback_trading_days=5)
    assert np.isnan(maximum[0])
    assert maximum[1] == 1.0 and source[1] == 1
    # Row 5 is later on the same mapped day and compares with rows 1 and 2.
    assert maximum[4] == np.float32(0.8) and source[4] in {1, 2}
    # More than five trading days later, row 3 has no eligible prior document.
    assert np.isnan(maximum[2]) and source[2] == -1
    # Cross-stock row 4 cannot match stock A.
    assert np.isnan(maximum[3])


def test_rolling_prior_cosine_excludes_ineligible_rows_from_history() -> None:
    frame = pd.DataFrame({
        "row_index": [1, 2], "stock_id": ["A", "A"],
        "entry_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "published_at": pd.to_datetime(["2024-01-01", "2024-01-02"]),
    })
    maximum, source = rolling_prior_cosine(
        frame, np.ones((2, 2), dtype=np.float32),
        trading_dates=pd.date_range("2024-01-02", "2024-01-03", freq="B"),
        eligible=np.array([False, True]),
    )
    assert np.isnan(maximum).all()
    assert source.tolist() == [-1, -1]
