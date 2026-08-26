import pandas as pd
import pytest

from scripts.evaluate_predictions_simple_states import (
    align_factor_to_universe,
    build_factor_table,
)


def test_build_factor_table_maps_stock_ids_and_rejects_duplicates():
    predictions = pd.DataFrame({
        "stock_id": ["000001", "600000"],
        "entry_date": ["2024-01-02", "2024-01-02"],
        "prediction": [0.1, -0.2],
    })
    factor, metadata = build_factor_table(
        predictions,
        prediction_column="prediction",
    )
    assert list(factor.columns) == ["000001.SZ", "600000.SH"]
    assert metadata["usable_rows"] == 2

    duplicated = pd.concat([predictions.iloc[[0]], predictions.iloc[[0]]])
    with pytest.raises(ValueError, match="at most one row"):
        build_factor_table(duplicated, prediction_column="prediction")


def test_align_observation_factor_drops_extra_dates_and_fills_full_axis():
    factor = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        index=pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-05"]),
        columns=["000001.SZ", "999999.SZ"],
    )
    universe = pd.DataFrame(
        [[True, True], [True, False], [True, True]],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        columns=["000001.SZ", "600000.SH"],
    )
    aligned, metadata = align_factor_to_universe(
        factor, universe, prediction_date_role="observation"
    )
    assert aligned.index.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
    ]
    assert aligned.columns.tolist() == ["000001.SZ", "600000.SH"]
    assert aligned.loc[pd.Timestamp("2024-01-02"), "000001.SZ"] == 0.1
    assert pd.isna(aligned.loc[pd.Timestamp("2024-01-02"), "600000.SH"])
    assert aligned.loc[pd.Timestamp("2024-01-03")].isna().all()
    assert metadata["dropped_factor_dates"] == 1
    assert metadata["dropped_factor_tickers"] == 1


def test_align_execution_factor_moves_back_one_universe_trading_day():
    factor = pd.DataFrame(
        [[0.1], [0.2]],
        index=pd.to_datetime(["2024-01-03", "2024-01-05"]),
        columns=["000001.SZ"],
    )
    universe = pd.DataFrame(
        True,
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        columns=["000001.SZ"],
    )
    aligned, metadata = align_factor_to_universe(factor, universe)
    assert aligned.index.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
    ]
    assert aligned.loc[pd.Timestamp("2024-01-02"), "000001.SZ"] == 0.1
    assert aligned.loc[pd.Timestamp("2024-01-03")].isna().all()
    assert aligned.loc[pd.Timestamp("2024-01-04"), "000001.SZ"] == 0.2
    assert metadata["prediction_date_role"] == "execution"
    assert metadata["factor_date_shift_trading_days"] == -1
