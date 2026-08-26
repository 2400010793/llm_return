from pathlib import Path

import pandas as pd
import pytest

from scripts.ensemble_regression_predictions import ensemble_predictions


def _write(path: Path, predictions: list[float], *, shift_return: float = 0.0) -> None:
    pd.DataFrame({
        "stock_id": ["000001", "600000"],
        "entry_date": pd.to_datetime(["2024-01-03", "2024-01-03"]),
        "test_year": [2024, 2024],
        "actual_return": [0.01 + shift_return, -0.02],
        "prediction": predictions,
    }).to_parquet(path, index=False)


def test_ensemble_regression_predictions_aligns_and_averages(tmp_path: Path):
    first, second = tmp_path / "a.parquet", tmp_path / "b.parquet"
    _write(first, [0.1, -0.1])
    _write(second, [0.3, -0.3])
    output, summary = ensemble_predictions([first, second])
    assert output["prediction"].tolist() == pytest.approx([0.2, -0.2])
    assert output["prediction_seed_std"].tolist() == pytest.approx([0.1, 0.1])
    assert summary["ensemble_size"] == 2


def test_ensemble_regression_predictions_rejects_label_mismatch(tmp_path: Path):
    first, second = tmp_path / "a.parquet", tmp_path / "b.parquet"
    _write(first, [0.1, -0.1])
    _write(second, [0.3, -0.3], shift_return=0.001)
    with pytest.raises(ValueError, match="actual returns differ"):
        ensemble_predictions([first, second])
