import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import MiniBatchKMeans

from scripts.build_qwen_linear_manifests import main as build_main
from scripts.run_qwen_cluster_linear import cluster_features
from scripts.summarize_qwen_linear_study import block_bootstrap_delta


def test_qwen_screen_manifest_contains_only_linear_models(tmp_path, monkeypatch) -> None:
    output = tmp_path / "screen.tsv"
    monkeypatch.setattr("sys.argv", [
        "build_qwen_linear_manifests.py", "--output", str(output),
        "--result-root", str(tmp_path / "results"),
    ])
    build_main()
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 63
    assert {row["regressor"] for row in rows} == {"ridge", "huber_sgd"}
    assert sum(row["model"] == "qwen3_embedding_8b" for row in rows) == 27
    assert {row["variant"] for row in rows if row["model"] == "qwen3_embedding_8b"} == {
        "plain", "short", "masked_short",
    }


def test_cluster_probability_features_are_finite_and_normalized() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(size=(100, 4)).astype(np.float32)
    model = MiniBatchKMeans(n_clusters=4, random_state=42, n_init=1).fit(values)
    augmented = cluster_features(values, model, "probability")
    assert augmented.shape == (100, 8)
    assert np.isfinite(augmented).all()
    np.testing.assert_allclose(augmented[:, 4:].sum(axis=1), 1.0, atol=1e-6)


def test_block_bootstrap_delta_preserves_comparison_direction() -> None:
    dates = pd.date_range("2026-01-01", periods=60)
    left = pd.DataFrame({"entry_date": dates, "rank_ic": np.zeros(60)})
    right = pd.DataFrame({"entry_date": dates, "rank_ic": np.full(60, 0.02)})
    result = block_bootstrap_delta(left, right, samples=200, block=20)
    assert result["mean_delta"] == pytest.approx(0.02)
    assert result["ci_low"] > 0
