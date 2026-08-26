from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_forecast_comparison import (
    build_daily_loss_panels,
    load_classification_predictions,
)
from src.evaluation.forecast_comparison import (
    benjamini_hochberg_adjust,
    diebold_mariano_test,
    holm_adjust,
    model_confidence_set,
    pairwise_dm_tests,
)


def test_dm_detects_lower_aligned_daily_loss() -> None:
    weak = np.array([0.8, 1.0, 0.7, 1.1, 0.9] * 20)
    strong = weak - np.tile([0.20, 0.10, 0.25, 0.15, 0.18], 20)
    result = diebold_mariano_test(weak, strong, hac_lag=2)
    assert result["mean_loss_difference_a_minus_b"] > 0
    assert result["favored"] == "b"
    assert result["p_value"] < 0.01


def test_pairwise_dm_adjustments_are_bounded_and_named() -> None:
    losses = pd.DataFrame({
        "best": np.linspace(0.1, 0.2, 30),
        "middle": np.linspace(0.2, 0.35, 30),
        "worst": np.linspace(0.4, 0.7, 30),
    })
    result = pairwise_dm_tests(losses, hac_lag=1)
    assert len(result) == 3
    assert {"p_value_holm", "p_value_bh", "favored_model"}.issubset(result.columns)
    assert result["p_value_holm"].between(0.0, 1.0).all()
    assert result["p_value_bh"].between(0.0, 1.0).all()
    assert np.all(holm_adjust([0.01, 0.04, 0.5]) >= np.array([0.01, 0.04, 0.5]))
    assert np.all(benjamini_hochberg_adjust([0.01, 0.04, 0.5]) >= np.array([0.01, 0.04, 0.5]))


def test_mcs_eliminates_clearly_inferior_models_reproducibly() -> None:
    rng = np.random.default_rng(7)
    common = rng.normal(0.0, 0.02, size=80)
    losses = pd.DataFrame({
        "good": 0.20 + common,
        "near_good": 0.205 + common + rng.normal(0.0, 0.01, size=80),
        "bad": 0.60 + common,
    })
    first = model_confidence_set(
        losses, alpha=0.10, bootstrap_reps=300, block_length=5,
        seed=11, method="range",
    )
    second = model_confidence_set(
        losses, alpha=0.10, bootstrap_reps=300, block_length=5,
        seed=11, method="range",
    )
    assert first == second
    assert "bad" in first["excluded_models"]
    assert "good" in first["included_models"]


def test_classification_predictions_are_aggregated_to_stock_day(tmp_path: Path) -> None:
    panel = pd.DataFrame({
        "row_index": [1, 2, 3, 4],
        "stock_id": ["A", "A", "B", "A"],
    })
    common = pd.DataFrame({
        "row_index": [1, 2, 3, 4],
        "entry_date": pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-02", "2026-01-05"]),
        "next_day_return": [0.01, 0.01, -0.02, -0.03],
    })
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    paths = {}
    for name, probabilities in {"a": [0.6, 0.8, 0.2, 0.4], "b": [0.7, 0.9, 0.3, 0.2]}.items():
        path = tmp_path / f"{name}.parquet"
        common.assign(probability=probabilities).to_parquet(path, index=False)
        paths[name] = path
    aligned, audit = load_classification_predictions(
        paths, panel_path, row_column="row_index", stock_column="stock_id",
        date_column="entry_date", actual_column="next_day_return",
        prediction_column="probability",
    )
    assert len(aligned) == 3
    assert aligned.loc[(aligned["stock_id"] == "A") & (aligned["entry_date"] == pd.Timestamp("2026-01-02")), "a"].iloc[0] == 0.7
    assert audit["a"]["matched_rows"] == 4
    panels = build_daily_loss_panels(
        aligned, ["a", "b"], task="classification", date_column="entry_date",
        threshold=0.5, huber_delta=0.01, min_stocks=1,
    )
    assert set(panels) == {"log_loss", "brier", "zero_one"}
    assert len(panels["log_loss"]) == 2
