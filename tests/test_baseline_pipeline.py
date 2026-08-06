import numpy as np

from src.evaluation.prediction_metrics import regression_metrics
from src.models.baselines import fit_predict_tfidf_ridge


def test_tfidf_ridge_and_metrics() -> None:
    train_texts = ["公司业绩增长", "公司业绩下降", "订单增长", "订单下降"]
    train_returns = np.array([0.04, -0.03, 0.02, -0.01])
    result = fit_predict_tfidf_ridge(
        train_texts, train_returns, ["公司业绩增长", "订单下降"], alpha=1.0
    )
    metrics = regression_metrics(np.array([0.03, -0.02]), result.predictions)
    assert result.predictions.shape == (2,)
    assert metrics["n"] == 2.0
