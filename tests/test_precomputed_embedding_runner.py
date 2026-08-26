import json
import sys

import joblib
import numpy as np
import pandas as pd

from scripts.run_precomputed_embedding_classification import (
    align_precomputed_embedding,
    main,
)


def test_align_precomputed_embedding_applies_explicit_row_offset(tmp_path):
    panel = pd.DataFrame({"row_index": [1, 2, 3]})
    matrix = np.array([[10, 11], [20, 21], [30, 31]], dtype=np.float32)
    matrix_path = tmp_path / "matrix.npy"
    metadata_path = tmp_path / "metadata.parquet"
    np.save(matrix_path, matrix)
    pd.DataFrame({"row_index": [0, 1, 2]}).to_parquet(metadata_path, index=False)
    frame, aligned, audit = align_precomputed_embedding(
        panel, matrix_path, metadata_path,
        panel_row_index_column="row_index",
        metadata_row_index_column="row_index",
        metadata_row_index_offset=1,
        expected_rows=3,
    )
    assert frame["row_index"].tolist() == [1, 2, 3]
    np.testing.assert_array_equal(aligned, matrix)
    assert audit["metadata_row_index_offset"] == 1
    assert audit["positionally_identical_after_offset"] is True


def test_align_precomputed_embedding_accepts_exact_string_key(tmp_path):
    panel = pd.DataFrame({"article_id": ["article-b", "article-a"]})
    matrix = np.array([[10, 11], [20, 21]], dtype=np.float32)
    matrix_path = tmp_path / "matrix.npy"
    metadata_path = tmp_path / "metadata.parquet"
    np.save(matrix_path, matrix)
    pd.DataFrame({"article_id": ["article-a", "article-b"]}).to_parquet(
        metadata_path, index=False
    )

    _, aligned, audit = align_precomputed_embedding(
        panel,
        matrix_path,
        metadata_path,
        panel_row_index_column="article_id",
        metadata_row_index_column="article_id",
        metadata_row_index_offset=0,
        expected_rows=2,
    )

    np.testing.assert_array_equal(aligned, matrix[[1, 0]])
    assert audit["key_mode"] == "exact_string"


def test_small_precomputed_run_saves_models_and_resumes(tmp_path, monkeypatch):
    rows = 18
    base = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
    matrix_path = tmp_path / "matrix.npy"
    metadata_path = tmp_path / "metadata.parquet"
    panel_path = tmp_path / "panel.parquet"
    output = tmp_path / "report.json"
    np.save(matrix_path, base)
    pd.DataFrame({"row_index": np.arange(rows)}).to_parquet(metadata_path, index=False)
    dates = [f"{year}-01-{day:02d}" for year in range(2018, 2027) for day in (2, 3)]
    signs = np.tile(np.array([-0.01, 0.01]), 9)
    pd.DataFrame({
        "row_index": np.arange(1, rows + 1),
        "entry_date": dates,
        "event_return_3d": signs,
        "next_day_return": signs,
    }).to_parquet(panel_path, index=False)
    monkeypatch.setattr(sys, "argv", [
        "run_precomputed_embedding_classification.py",
        str(panel_path),
        "--matrix", str(matrix_path),
        "--metadata", str(metadata_path),
        "--embedding-model", "bge_m3",
        "--input-name", "text_plain",
        "--classifier", "logistic",
        "--scaler", "none",
        "--metadata-row-index-offset", "1",
        "--expected-rows", str(rows),
        "--output", str(output),
    ])
    main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["design"]["train_target"] == "next_day_return"
    assert report["design"]["evaluation_target"] == "next_day_return"
    assert report["input"]["alignment"]["matched_rows"] == rows
    assert report["input"]["alignment"]["metadata_key_raw_min"] == 0
    assert report["results"][0]["n_test_rows"] == 2
    predictions = pd.read_parquet(output.with_suffix(".predictions.parquet"))
    assert predictions["row_index"].tolist() == [17, 18]
    bundle = output.with_suffix(".artifacts")
    model_path = bundle / "test_year_2026" / "final_model.joblib"
    assert model_path.is_file()
    model = joblib.load(model_path)
    np.testing.assert_allclose(
        model.predict_proba(base[16:18])[:, 1],
        predictions["probability"].to_numpy(),
        atol=1e-7,
    )
    mtime = model_path.stat().st_mtime_ns
    main()
    assert model_path.stat().st_mtime_ns == mtime
