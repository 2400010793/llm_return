from __future__ import annotations

import pandas as pd
import pytest

from src.data.o2o_targets import attach_o2o_targets


def test_attach_o2o_targets_preserves_rows_and_masks_unobserved_exits() -> None:
    announcements = pd.DataFrame({
        "row_index": [1, 2, 3],
        "stock_id": ["1", "000001", "000002"],
        "entry_date": ["2024-01-02", "2024-01-02", "2024-01-02"],
    })
    market = pd.DataFrame({
        "stock_id": ["000001", "000002"],
        "entry_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
        "tradable_open_to_open_return": [0.03, float("nan")],
        "eligible_signal": [True, True],
        "observed_market": [True, True],
        "next_observed_market": [True, False],
        "can_buy": [True, True],
    })

    result = attach_o2o_targets(announcements, market)

    assert result["row_index"].tolist() == [1, 2, 3]
    assert result.loc[:1, "next_day_open_to_open_return"].tolist() == [0.03, 0.03]
    assert result.loc[:1, "next_day_open_to_open_label"].tolist() == [1, 1]
    assert result.loc[:1, "o2o_target_eligible"].all()
    assert pd.isna(result.loc[2, "next_day_open_to_open_return"])
    assert pd.isna(result.loc[2, "next_day_open_to_open_label"])
    assert not bool(result.loc[2, "o2o_target_eligible"])


def test_attach_o2o_targets_rejects_duplicate_market_keys() -> None:
    announcements = pd.DataFrame({
        "stock_id": ["000001"], "entry_date": ["2024-01-02"]
    })
    market = pd.DataFrame({
        "stock_id": ["000001", "000001"],
        "entry_date": ["2024-01-02", "2024-01-02"],
        "tradable_open_to_open_return": [0.01, 0.02],
        "eligible_signal": [True, True],
        "observed_market": [True, True],
        "next_observed_market": [True, True],
    })
    with pytest.raises(ValueError, match="duplicate stock-date"):
        attach_o2o_targets(announcements, market)
