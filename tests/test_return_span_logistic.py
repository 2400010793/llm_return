import numpy as np
import pandas as pd

from scripts.run_return_span_logistic_fold import _metrics, _select_accuracy_threshold


def test_majority_accuracy_uses_training_probability() -> None:
    stock_days = pd.DataFrame({
        "actual_return": [0.02, 0.01, -0.01],
        "probability": [0.6, 0.4, 0.4],
        "entry_date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03"]),
    })

    metrics = _metrics(stock_days, majority_probability=0.4)

    assert np.isclose(metrics["accuracy"], 2 / 3)
    assert np.isclose(metrics["majority_accuracy"], 1 / 3)
    assert metrics["majority_probability"] == 0.4


def test_threshold_is_selected_only_from_supplied_validation_rows() -> None:
    stock_days = pd.DataFrame({
        "actual_return": [-0.01, -0.02, 0.01, 0.02],
        "probability": [0.40, 0.45, 0.55, 0.60],
    })

    threshold, accuracy = _select_accuracy_threshold(stock_days)

    assert threshold == 0.5
    assert accuracy == 1.0
