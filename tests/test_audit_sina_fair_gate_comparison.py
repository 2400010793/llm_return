from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.audit_sina_fair_gate_comparison import (
    _date_cluster_bootstrap_mean,
    _exact_mcnemar_pvalue,
    paired_classification_comparison,
)


def test_mcnemar_exact_is_one_when_no_discordant_pairs() -> None:
    assert _exact_mcnemar_pvalue(0, 0) == 1.0


def test_date_bootstrap_resamples_whole_dates() -> None:
    frame = pd.DataFrame({
        "entry_date": pd.to_datetime([
            "2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02",
        ]),
        "value": [1.0, 1.0, -1.0, -1.0],
    })

    interval = _date_cluster_bootstrap_mean(
        frame,
        value_column="value",
        rng=np.random.default_rng(42),
        n_bootstrap=100,
    )

    assert len(interval) == 2
    assert interval[0] <= 0.0 <= interval[1]


def test_paired_classification_comparison_aligns_articles() -> None:
    dates = pd.to_datetime(["2026-01-01"] * 4 + ["2026-01-02"] * 4)
    actual = [0.1, -0.1, 0.2, -0.2] * 2
    left = pd.DataFrame({
        "article_id": [str(index) for index in range(8)],
        "entry_date": dates,
        "actual_return": actual,
        "score": [0.9, 0.9, 0.1, 0.1] * 2,
    })
    right = left.copy()
    right["score"] = [0.9, 0.1, 0.9, 0.1] * 2

    result = paired_classification_comparison(
        left,
        right,
        rng=np.random.default_rng(42),
        n_bootstrap=100,
    )

    assert result["n_articles"] == 8
    assert result["n_dates"] == 2
    assert np.isclose(result["left_accuracy"], 0.5)
    assert np.isclose(result["right_accuracy"], 1.0)
    assert np.isclose(result["accuracy_difference_right_minus_left"], 0.5)
    assert result["mcnemar"]["left_wrong_right_correct"] == 4