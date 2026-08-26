import numpy as np
import pandas as pd

from scripts.run_positive_token_soft_cluster_fold import (
    fit_projection, project, response_targets, soft_predict, validation_metrics,
)
from scripts.build_qwen_sina_soft_cluster_manifest import REPRESENTATIONS, VARIANTS


def test_response_targets_do_not_overweight_duplicate_news() -> None:
    frame = pd.DataFrame({
        "stock_id": ["A", "A", "B", "A", "B"],
        "entry_date": ["2024-01-02"] * 3 + ["2024-01-03"] * 2,
        "next_day_return": [0.03, 0.03, -0.01, 0.02, 0.00],
    })
    targets = response_targets(frame, "next_day_return")
    np.testing.assert_allclose(targets["demeaned"][:3], [0.02, 0.02, -0.02])
    np.testing.assert_allclose(targets["rank"][:3], [0.5, 0.5, 0.0])


def test_soft_predict_hard_and_soft_assignments() -> None:
    distances = np.array([[0.1, 2.0], [2.0, 0.1]])
    means = np.array([-0.02, 0.03])
    np.testing.assert_allclose(soft_predict(distances, means, 0.0), means)
    soft = soft_predict(distances, means, 1.0)
    assert -0.02 < soft[0] < 0.03
    assert -0.02 < soft[1] < 0.03
    assert soft[0] < soft[1]


def test_qwen_soft_cluster_matrix_has_six_independent_configs() -> None:
    assert len(REPRESENTATIONS) * len(VARIANTS) == 6
    assert set(REPRESENTATIONS) == {"article_mean", "prompt_mean", "return_token"}


def test_validation_metrics_include_long_side_objective() -> None:
    rows = []
    for day in ("2024-01-02", "2024-01-03"):
        for index in range(10):
            rows.append({
                "entry_date": day, "prediction": float(index),
                "actual_return": float(index - 4.5) / 100,
            })
    metrics = validation_metrics(pd.DataFrame(rows))
    assert metrics["long_excess_mean"] > 0
    assert metrics["long_excess_positive_days"] == 1.0


def test_raw_projection_preserves_dimension() -> None:
    matrix = np.arange(80, dtype=np.float32).reshape(10, 8)
    positions = np.arange(8)
    reducer, scaler = fit_projection(matrix, positions, sample_rows=8, projection_mode="raw")
    transformed = project(matrix, np.arange(10), reducer, scaler, components=4)
    assert reducer is None
    assert transformed.shape == (10, 8)
