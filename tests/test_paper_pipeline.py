import pandas as pd

from src.models.paper_pipeline import portfolio_metrics, quantile_portfolio


def test_quantile_portfolio() -> None:
    frame = pd.DataFrame({
        "entry_date": pd.to_datetime(["2024-01-01"] * 5),
        "prediction": [1, 2, 3, 4, 5],
        "realized_return": [0.01, 0.02, 0.03, 0.04, 0.05],
    })
    result = quantile_portfolio(frame)
    assert result.loc[0, "long_short"] == 0.04
    assert portfolio_metrics(result)["n_days"] == 1
