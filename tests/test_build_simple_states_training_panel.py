import numpy as np
import pandas as pd

from scripts.build_simple_states_training_panel import attach_simple_states_target


def test_attach_simple_states_target_uses_entry_return_and_previous_day_filters():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    columns = ["000001.SZ", "600000.SH"]
    panel = pd.DataFrame({
        "stock_id": ["000001", "600000", "000001"],
        "entry_date": ["2024-01-03", "2024-01-03", "2024-01-04"],
    })
    universe = pd.DataFrame(True, index=dates, columns=columns)
    stock_return = pd.DataFrame(
        [[0.0, 0.0], [0.11, -0.12], [0.21, -0.22]], index=dates, columns=columns
    )
    amount = pd.DataFrame(30.0, index=dates, columns=columns)
    amount.loc[dates[0], "600000.SH"] = 5.0
    riseboard = pd.DataFrame(0.0, index=dates, columns=columns)
    jumpboard = pd.DataFrame(0.0, index=dates, columns=columns)
    riseboard.loc[dates[2], "000001.SZ"] = np.nan
    jumpboard.loc[dates[2], "000001.SZ"] = np.nan

    result, audit = attach_simple_states_target(
        panel,
        stock_return=stock_return,
        universe=universe,
        amount=amount,
        riseboard=riseboard,
        jumpboard=jumpboard,
        min_amount=20.0,
        amount_window=1,
    )
    assert result["simple_states_target_eligible"].tolist() == [True, False, False]
    assert result.loc[0, "simple_states_ret_vv_return"] == 0.11
    assert result.loc[1:, "simple_states_ret_vv_return"].isna().all()
    assert audit["eligible"] == 1
