import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_minimal_prompt_knn_fold import (
    _kneighbors,
    _repair_invalid_neighbors,
    _select_knn,
    run,
)
from scripts.run_soft_return_family_fold import _shrunk_means, _soft_predict
from scripts.summarize_minimal_prompt_knn_v2 import summarize
from src.analysis.prompt_token_mechanisms import YearlyTokenMoments, rolling_windows


def test_knn_validation_selects_a_predictive_configuration() -> None:
    rng = np.random.default_rng(42)
    fit = rng.normal(size=(240, 6)).astype(np.float32)
    validation = rng.normal(size=(80, 6)).astype(np.float32)
    y_fit = (fit[:, 0] + 0.25 * fit[:, 1] > 0).astype(np.int8)
    y_validation = (validation[:, 0] + 0.25 * validation[:, 1] > 0).astype(np.int8)
    winner, rows = _select_knn(fit, y_fit, validation, y_validation, jobs=1)
    assert len(rows) == 12
    assert winner["neighbors"] in (11, 31, 101)
    assert winner["validation_accuracy"] > 0.7


def test_hnsw_backend_preserves_nearest_neighbor_order_and_distance_scale() -> None:
    train = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]])
    query = np.asarray([[1.0, 0.0]])
    for metric in ("euclidean", "cosine"):
        exact_distance, exact_index, _ = _kneighbors(
            train, query, metric=metric, neighbors=3, jobs=1,
            approximate_min_rows=10,
        )
        hnsw_distance, hnsw_index, backend = _kneighbors(
            train, query, metric=metric, neighbors=3, jobs=1,
            approximate_min_rows=0,
        )
        assert backend == "catboost_hnsw"
        assert hnsw_index.tolist() == exact_index.tolist()
        np.testing.assert_allclose(hnsw_distance, exact_distance, atol=1e-6)


def test_invalid_hnsw_rows_are_repaired_with_exact_neighbors() -> None:
    train = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [5.0, 5.0]], dtype=np.float32,
    )
    query = np.asarray([[0.9, 0.0], [0.0, 1.9]], dtype=np.float32)
    distances = np.asarray([[0.1, 0.9], [np.nan, 0.1]])
    indexes = np.asarray([[1, 0], [4294967295, 2]], dtype=np.int64)

    repaired_distances, repaired_indexes, repaired_rows = _repair_invalid_neighbors(
        train, query, distances, indexes,
        metric="euclidean", neighbors=2, jobs=1,
    )

    assert repaired_rows == 1
    np.testing.assert_array_equal(repaired_indexes[0], indexes[0])
    np.testing.assert_allclose(repaired_distances[0], distances[0])
    np.testing.assert_array_equal(repaired_indexes[1], [2, 0])
    np.testing.assert_allclose(repaired_distances[1], [0.1, 1.9], atol=1e-6)


def test_soft_family_prediction_shrinks_and_orders_cluster_returns() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    returns = np.array([-0.04, -0.02, 0.02, 0.04])
    means = _shrunk_means(labels, returns, 2, shrinkage=2.0)
    assert means[0] < 0 < means[1]
    assert abs(means[0]) < 0.03 and abs(means[1]) < 0.03
    distances = np.array([[0.0, 2.0], [2.0, 0.0]])
    hard = _soft_predict(distances, means, temperature=0.0)
    soft = _soft_predict(distances, means, temperature=1.0)
    assert hard[0] < hard[1]
    assert soft[0] < soft[1]


def test_short_fold_runs_existing_cluster_knn_and_logistic_pipeline(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    years = np.repeat(np.arange(2010, 2019), 16)
    rows = len(years)
    labels = np.tile(np.r_[np.zeros(8), np.ones(8)], 9)
    returns = np.where(labels > 0, 0.01, -0.01)
    frame = pd.DataFrame({
        "row_index": np.arange(1, rows + 1),
        "entry_date": pd.to_datetime([f"{year}-06-01" for year in years]),
        "stock_id": [f"{index % 16:06d}" for index in range(rows)],
        "event_return_3d": returns,
        "next_day_return": returns,
    })
    cache = tmp_path / "cache" / "roberta" / "short"
    cache.mkdir(parents=True)
    frame.to_parquet(cache / "rows.parquet", index=False)
    groups = rng.normal(size=(rows, 3, 6)).astype(np.float32)
    groups[:, 2, 0] += np.where(labels > 0, 2.0, -2.0)
    np.save(cache / "short_group_embeddings.npy", groups)
    np.save(cache / "masked_short_group_embeddings.npy", groups * 0.95)
    token_values = groups.copy()
    moments = YearlyTokenMoments(range(2010, 2019), 3, 6)
    moments.update(token_values, years, {
        "event_return_3d": returns, "next_day_return": returns,
    })
    moments.save(cache / "short_token_moments.npz")
    moments.save(cache / "masked_short_token_moments.npz")
    np.savez_compressed(
        cache / "mask_delta_moments.npz",
        years=np.arange(2010, 2019),
        delta_sum=np.zeros((9, 3, 6)),
        delta_sq=np.zeros((9, 3, 6)),
        delta_count=np.full(9, 16),
    )
    window = next(row for row in rolling_windows(range(2010, 2019)) if row["test_year"] == 2018)
    (cache / "manifest.json").write_text(json.dumps({
        "windows": [window],
        "variants": ["short", "masked_short"],
        "prompt_tokens": ["分析", "股票", "未来收益"],
        "position_groups": ["analysis", "target_stock", "future_return"],
        "groups": ["analysis", "target_stock", "future_return"],
        "group_positions": {
            "analysis": [0], "target_stock": [1], "future_return": [2],
        },
    }, default=list), encoding="utf-8")
    baseline = tmp_path / "baseline" / "roberta"
    baseline.mkdir(parents=True)
    np.savez_compressed(baseline / "short.npz", prompt_only=np.zeros((3, 6)))
    result = run(Namespace(
        cache_root=tmp_path / "cache", baseline_root=tmp_path / "baseline",
        output_root=tmp_path / "folds", model="roberta", prompt_length="short",
        variant="short", target="next_day_return", test_year=2018,
        group_components=2, pca_sample_vectors=1000, silhouette_sample=16, jobs=1,
    ))
    assert result["semantic_groups"] == ["analysis", "target_stock", "future_return"]
    assert len(result["classification"]) == 6
    predictions = pd.read_parquet(Path(result["output"]) / "predictions.parquet")
    assert {
        "group_pca_knn", "cluster_augmented_knn",
        "group_pca_logistic", "cluster_augmented_logistic",
    }.issubset(predictions.columns)
    summary = summarize(Namespace(
        fold_root=tmp_path / "folds", output_root=tmp_path / "summary",
        strict=True, expected_folds=1,
    ))
    assert summary["complete"] is True
    manifest = pd.read_csv(
        tmp_path / "summary" / "simple_states_manifest.tsv", sep="\t",
    )
    assert len(manifest) == 4
    assert set(manifest["prediction_column"]) == {
        "group_pca_knn", "cluster_augmented_knn",
        "group_pca_logistic", "cluster_augmented_logistic",
    }


def test_return_span_uses_existing_cluster_augmented_pipeline(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    years = np.repeat(np.arange(2010, 2019), 16)
    rows = len(years)
    labels = np.tile(np.r_[np.zeros(8), np.ones(8)], 9)
    returns = np.where(labels > 0, 0.01, -0.01)
    frame = pd.DataFrame({
        "row_index": np.arange(1, rows + 1),
        "entry_date": pd.to_datetime([f"{year}-06-01" for year in years]),
        "stock_id": [f"{index % 16:06d}" for index in range(rows)],
        "event_return_3d": returns,
        "next_day_return": returns,
    })
    cache = tmp_path / "cache" / "roberta" / "short"
    cache.mkdir(parents=True)
    frame.to_parquet(cache / "rows.parquet", index=False)
    window = next(
        row for row in rolling_windows(range(2010, 2019))
        if row["test_year"] == 2018
    )
    (cache / "manifest.json").write_text(json.dumps({
        "windows": [window], "variants": ["short", "masked_short"],
    }, default=list), encoding="utf-8")

    span = tmp_path / "span" / "roberta" / "short"
    span.mkdir(parents=True)
    values = rng.normal(size=(rows, 6)).astype(np.float32)
    values[:, 0] += np.where(labels > 0, 2.0, -2.0)
    np.save(span / "short_return_span.npy", values)
    np.save(span / "masked_short_return_span.npy", values * 0.95)
    (span / "manifest.json").write_text(json.dumps({
        "rows": rows, "variants": ["short", "masked_short"],
        "phrase": "收益", "tokens": ["收", "益"],
    }), encoding="utf-8")

    result = run(Namespace(
        cache_root=tmp_path / "cache", baseline_root=None,
        span_cache_root=tmp_path / "span", output_root=tmp_path / "span_folds",
        model="roberta", metadata_model="roberta", prompt_length="short", variant="short",
        target="next_day_return", test_year=2018, group_components=2,
        pca_sample_vectors=1000, silhouette_sample=16, jobs=1,
    ))
    assert result["representation_scope"] == "return_span"
    assert result["semantic_groups"] == ["return_span"]
    predictions = pd.read_parquet(Path(result["output"]) / "predictions.parquet")
    assert {
        "group_pca_logistic", "cluster_augmented_logistic",
        "group_pca_knn", "cluster_augmented_knn",
    }.issubset(predictions.columns)
    assert not (Path(result["output"]) / "token_metrics.parquet").exists()

    summary = summarize(Namespace(
        fold_root=tmp_path / "span_folds", output_root=tmp_path / "span_summary",
        strict=True, expected_folds=1, model=None,
    ))
    assert summary["complete"] is True
    assert (tmp_path / "span_summary" / "cluster_incremental_effect.csv").is_file()
