from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_precomputed_model_manifest import INPUTS, MODELS, build_rows


def test_model_only_manifest_builds_two_models_and_five_inputs(tmp_path: Path):
    for model, filename in MODELS.items():
        for input_name in INPUTS:
            directory = tmp_path / "data/processed/embeddings/all_inputs" / model / input_name
            directory.mkdir(parents=True)
            np.save(directory / filename, np.ones((2, 3), dtype=np.float32))
            pd.DataFrame({"row_index": [0, 1]}).to_parquet(
                directory / "metadata.parquet", index=False
            )
    rows = build_rows(tmp_path, "global_logistic", 42)
    assert len(rows) == 10
    assert {row["embedding_model"] for row in rows} == set(MODELS)
    assert {row["input_name"] for row in rows} == set(INPUTS)
    assert {row["classifier"] for row in rows} == {"logistic"}
    assert {row["reducer"] for row in rows} == {"none"}


def test_global_nonlinear_and_pca_manifests_are_bounded(tmp_path: Path):
    for model, filename in MODELS.items():
        for input_name in INPUTS:
            directory = tmp_path / "data/processed/embeddings/all_inputs" / model / input_name
            directory.mkdir(parents=True)
            np.save(directory / filename, np.ones((2, 3), dtype=np.float32))
            pd.DataFrame({"row_index": [0, 1]}).to_parquet(
                directory / "metadata.parquet", index=False
            )
    nonlinear = build_rows(tmp_path, "global_nonlinear", 42)
    assert len(nonlinear) == 20
    assert {row["classifier"] for row in nonlinear} == {
        "simple_mlp", "hist_gradient_boosting"
    }
    pca = build_rows(tmp_path, "global_pca", 42)
    assert len(pca) == 10
    assert {row["reducer"] for row in pca} == {"pca"}
    assert {row["components"] for row in pca} == {128}
    for rows in (nonlinear, pca):
        assert {row["train_target"] for row in rows} == {"next_day_return"}
        assert {row["evaluation_target"] for row in rows} == {"next_day_return"}


def test_event3_target_is_limited_to_explicit_label_comparison(tmp_path: Path):
    for model, filename in MODELS.items():
        for input_name in INPUTS:
            directory = tmp_path / "data/processed/embeddings/all_inputs" / model / input_name
            directory.mkdir(parents=True)
            np.save(directory / filename, np.ones((2, 3), dtype=np.float32))
            pd.DataFrame({"row_index": [0, 1]}).to_parquet(
                directory / "metadata.parquet", index=False
            )
    comparison = build_rows(tmp_path, "label_comparison", 42)
    assert {row["train_target"] for row in comparison} == {
        "event_return_3d", "next_day_return",
    }


def test_next1_global_manifests_exclude_unproductive_cells(tmp_path: Path):
    for model, filename in MODELS.items():
        for input_name in INPUTS:
            directory = tmp_path / "data/processed/embeddings/all_inputs" / model / input_name
            directory.mkdir(parents=True)
            np.save(directory / filename, np.ones((2, 3), dtype=np.float32))
            pd.DataFrame({"row_index": [0, 1]}).to_parquet(
                directory / "metadata.parquet", index=False
            )
    core = build_rows(tmp_path, "next1_core", 42)
    assert len(core) == 12
    assert {row["classifier"] for row in core} == {"logistic", "simple_mlp"}
    assert {row["input_name"] for row in core} == {
        "text_plain", "text_input_2_short", "text_input_5_masked_short",
    }
    assert {row["reducer"] for row in core} == {"none"}
    assert {row["train_target"] for row in core} == {"next_day_return"}
    assert {row["evaluation_target"] for row in core} == {"next_day_return"}
    sentinel = build_rows(tmp_path, "next1_sgd_sentinel", 42)
    assert len(sentinel) == 2
    assert {row["classifier"] for row in sentinel} == {"sgd"}
    assert {row["input_name"] for row in sentinel} == {"text_input_5_masked_short"}
