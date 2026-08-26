from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.o2o_residual_targets import (
    CS_ZSCORE_COLUMN,
    MARKET_RESIDUAL_COLUMN,
    MARKET_RETURN_COLUMN,
    WINSOR_RESIDUAL_COLUMN,
    attach_o2o_residual_targets,
)


def _market() -> pd.DataFrame:
    return pd.DataFrame({
        "stock_id": ["1", "2", "3"],
        "entry_date": ["2024-01-02"] * 3,
        "open_to_open_return": [-0.1, 0.0, 0.2],
        "eligible_signal": [True, True, True],
        "can_buy": [True, True, True],
    })


def test_attach_o2o_residual_targets_uses_full_market_and_preserves_rows() -> None:
    panel = pd.DataFrame({
        "row_index": [8, 9, 10],
        "stock_id": ["000001", "000001", "000003"],
        "entry_date": ["2024-01-02"] * 3,
        "next_day_open_to_open_return": [-0.1, -0.1, 0.2],
    })
    result, audit = attach_o2o_residual_targets(
        panel, _market(), min_stocks_per_day=2, winsor_lower=0.0,
        winsor_upper=1.0,
    )

    assert result["row_index"].tolist() == [8, 9, 10]
    assert result["stock_id"].tolist() == ["000001", "000001", "000003"]
    np.testing.assert_allclose(result[MARKET_RETURN_COLUMN], [1 / 30, 1 / 30, 1 / 30])
    np.testing.assert_allclose(
        result[MARKET_RESIDUAL_COLUMN], [-2 / 15, -2 / 15, 1 / 6]
    )
    assert np.isfinite(result[CS_ZSCORE_COLUMN]).all()
    np.testing.assert_allclose(
        result[WINSOR_RESIDUAL_COLUMN], result[MARKET_RESIDUAL_COLUMN]
    )
    assert audit["source_market_comparisons"] == 3
    assert audit["market_days"] == 1


def test_attach_o2o_residual_targets_masks_ineligible_market_rows() -> None:
    market = _market()
    market.loc[2, "can_buy"] = False
    panel = pd.DataFrame({
        "stock_id": ["000001", "000003"],
        "entry_date": ["2024-01-02", "2024-01-02"],
        "next_day_open_to_open_return": [-0.1, np.nan],
    })
    result, _ = attach_o2o_residual_targets(
        panel, market, min_stocks_per_day=2, winsor_lower=0.0,
        winsor_upper=1.0,
    )
    np.testing.assert_allclose(result.loc[0, MARKET_RETURN_COLUMN], -0.05)
    assert pd.isna(result.loc[1, MARKET_RESIDUAL_COLUMN])


def test_attach_o2o_residual_targets_rejects_inconsistent_source() -> None:
    panel = pd.DataFrame({
        "stock_id": ["000001"], "entry_date": ["2024-01-02"],
        "next_day_open_to_open_return": [0.5],
    })
    with pytest.raises(ValueError, match="disagree"):
        attach_o2o_residual_targets(
            panel, _market(), min_stocks_per_day=2,
        )


def test_attach_o2o_residual_targets_rejects_duplicate_market_keys() -> None:
    market = pd.concat([_market(), _market().iloc[[0]]], ignore_index=True)
    panel = pd.DataFrame({
        "stock_id": ["000001"], "entry_date": ["2024-01-02"],
        "next_day_open_to_open_return": [-0.1],
    })
    with pytest.raises(ValueError, match="duplicate stock-date"):
        attach_o2o_residual_targets(panel, market, min_stocks_per_day=2)
