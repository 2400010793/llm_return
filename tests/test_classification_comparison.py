import numpy as np

from src.evaluation.classification import holm_adjust, paired_classification_comparison


def test_holm_adjust_preserves_order_and_monotonicity():
    adjusted = holm_adjust([0.04, 0.01, 0.03])
    np.testing.assert_allclose(adjusted, [0.06, 0.03, 0.06])


def test_clustered_comparison_uses_common_pairs_and_reports_mcnemar():
    actual = [-1, 1, -1, 1, -1, 1]
    weak = [0.6, 0.4, 0.6, 0.4, 0.6, 0.4]
    strong = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
    dates = ["d1", "d1", "d2", "d2", "d3", "d3"]
    result = paired_classification_comparison(
        actual, weak, strong, dates, n_bootstrap=50, seed=7
    )
    assert result["direction"] == "b_minus_a"
    assert result["n"] == 6
    assert result["n_clusters"] == 3
    assert result["delta"]["accuracy"] == 1.0
    assert result["clustered_95_ci"]["accuracy"] == [1.0, 1.0]
    assert result["mcnemar_accuracy"]["a_wrong_b_correct"] == 6
    assert result["mcnemar_accuracy"]["a_correct_b_wrong"] == 0