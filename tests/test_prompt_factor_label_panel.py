import numpy as np
import pandas as pd

from scripts.build_prompt_factor_label_panel import _forward_product, _group_window_stat


def _market():
    return pd.DataFrame(
        {
            "stock_id": ["A"] * 5 + ["B"] * 5,
            "entry_date": list(pd.date_range("2024-01-01", periods=5)) * 2,
            "close_to_close_return": [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
            "open_to_open_return": [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
        }
    )


def test_forward_product_does_not_cross_stock_boundaries():
    frame = _market()
    result = _forward_product(frame, "open_to_open_return", 2)
    np.testing.assert_allclose(result.iloc[0], (1.01 * 1.02) - 1.0)
    np.testing.assert_allclose(result.iloc[4], np.nan, equal_nan=True)
    np.testing.assert_allclose(result.iloc[5], (1.10 * 1.20) - 1.0)


def test_prior_window_stat_uses_only_information_before_entry():
    frame = _market()
    result = _group_window_stat(frame, "close_to_close_return", 2, direction="prior", statistic="mean")
    assert np.isnan(result.iloc[0])
    np.testing.assert_allclose(result.iloc[2], (0.01 + 0.02) / 2.0)
    np.testing.assert_allclose(result.iloc[7], (0.10 + 0.20) / 2.0)
