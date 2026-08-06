import numpy as np

import pandas as pd
import pytest

from src.models.return_tasks import fit_return_model, fit_sentiment_model, make_forward_return, make_sentiment_labels


def test_three_day_labels_are_strictly_positive_only() -> None:
    assert make_sentiment_labels([-0.1, 0.0, 0.2]).tolist() == [0, 0, 1]


def test_sentiment_is_training_only() -> None:
    result = fit_sentiment_model(["盈利增长", "亏损扩大", "盈利提升", "亏损增加"], [-0.1, 0.2, 0.1, -0.2], ["盈利增长"])
    assert result.probabilities.shape == (1,)
    assert result.labels.tolist() == [0, 1, 1, 0]


def test_each_return_horizon_can_use_independent_model() -> None:
    result = fit_return_model(["盈利增长", "亏损扩大", "盈利提升", "亏损增加"], np.array([0.1, -0.1, 0.2, -0.2]), ["盈利增长"], [0.1], model_name="ridge", alpha=1.0)
    assert result.predictions.shape == (1,)
    assert result.metrics["n"] == 1.0


def test_forward_return_is_horizon_specific() -> None:
    prices = pd.DataFrame({"stock_id": ["A"] * 4, "date": pd.date_range("2024-01-01", periods=4), "close": [100, 110, 121, 133.1]})
    result = make_forward_return(prices, 2)
    assert result.iloc[0] == pytest.approx(0.21)
