import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import joblib
import pytest
import scripts.run_pooled_embedding_classification as pooled_runner

from sklearn.model_selection import ParameterGrid

from scripts.run_pooled_embedding_classification import (
    dense_grid,
    fit_mlp_classifier_with_validation,
    final_epoch_budget,
    learning_rate_at_epoch,
    main,
    preprocess_windows,
    validation_selection_score,
)


def test_strong_classifier_grids_are_available_and_bounded():
    expected_limits = {
        "extra_trees": 24,
        "hist_gradient_boosting": 36,
        "xgboost": 48,
        "lightgbm": 48,
        "catboost": 36,
    }
    for classifier, maximum in expected_limits.items():
        coarse = list(ParameterGrid(dense_grid(classifier, "coarse")))
        fine = list(ParameterGrid(dense_grid(classifier, "fine")))
        assert coarse
        assert len(coarse) <= len(fine) <= maximum


def test_mlp_early_stopping_uses_chronological_validation():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(80, 4)).astype(np.float32)
    labels = (x[:, 0] > 0).astype(np.int8)
    validation = pd.Series(np.where(labels[60:] > 0, 0.01, -0.01))
    model, probabilities, score, selected_epoch, history = (
        fit_mlp_classifier_with_validation(
            x[:60], labels[:60], x[60:], validation,
            candidate={
                "hidden_layer_sizes": (8,), "alpha": 1e-3,
                "max_iter": 20, "batch_size": 16,
                "learning_rate_init": 1e-3, "n_iter_no_change": 3,
            }, seed=42, selection_metric="balanced_accuracy",
        )
    )
    assert 1 <= selected_epoch <= len(history) <= 20
    assert probabilities.shape == (20,)
    assert np.isfinite(probabilities).all()
    assert score == pytest.approx(
        max(row["validation_balanced_accuracy"] for row in history)
    )
    assert model.selected_epoch_ == selected_epoch
    assert all(np.isfinite(row["train_loss"]) for row in history)
    assert all(np.isfinite(row["learning_rate"]) for row in history)
    assert all("validation_log_loss" in row for row in history)


def test_validation_selection_metric_does_not_reward_majority_shortcut() -> None:
    metrics = {"accuracy": 0.57, "balanced_accuracy": 0.50, "auc": 0.54}
    assert validation_selection_score(metrics, "accuracy") == pytest.approx(0.57)
    assert validation_selection_score(metrics, "balanced_accuracy") == pytest.approx(0.50)
    assert validation_selection_score(metrics, "auc") == pytest.approx(0.54)
    with pytest.raises(ValueError, match="unsupported validation selection metric"):
        validation_selection_score(metrics, "f1")


def test_mlp_burn_in_excludes_early_epochs_from_selection() -> None:
    rng = np.random.default_rng(47)
    x = rng.normal(size=(100, 4)).astype(np.float32)
    labels = (x[:, 0] > 0).astype(np.int8)
    validation = pd.Series(np.where(labels[70:] > 0, 0.01, -0.01))
    _, _, _, selected_epoch, history = fit_mlp_classifier_with_validation(
        x[:70], labels[:70], x[70:], validation,
        candidate={
            "hidden_layer_sizes": (8,), "alpha": 1e-3,
            "max_iter": 12, "batch_size": 16,
            "learning_rate_init": 1e-3, "n_iter_no_change": 3,
        },
        seed=42, selection_metric="balanced_accuracy",
        selection_min_epoch=5,
    )
    assert selected_epoch >= 5
    assert len(history) >= 5
    assert not any(row["selection_eligible"] for row in history[:4])
    assert all(row["selection_eligible"] for row in history[4:])


def test_mlp_fine_grid_tunes_batch_and_learning_rate() -> None:
    from sklearn.model_selection import ParameterGrid

    for classifier in ("mlp", "simple_mlp"):
        candidates = list(ParameterGrid(dense_grid(classifier, "fine")))
        assert {candidate["batch_size"] for candidate in candidates} == {128, 256, 512}
        assert {candidate["learning_rate_init"] for candidate in candidates} == {3e-4, 1e-3}
        assert len({candidate["alpha"] for candidate in candidates}) == 2


def test_stable_simple_mlp_grid_is_single_conservative_candidate() -> None:
    candidates = list(ParameterGrid(dense_grid("simple_mlp", "stable")))
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["hidden_layer_sizes"] == (32,)
    assert candidate["learning_rate_init"] == pytest.approx(3e-4)
    assert candidate["n_iter_no_change"] == 20


def test_learning_rate_annealing_is_monotone_and_bounded() -> None:
    cosine = [learning_rate_at_epoch(1e-3, epoch, 20, "cosine") for epoch in range(1, 21)]
    inverse = [learning_rate_at_epoch(1e-3, epoch, 20, "inverse_time") for epoch in range(1, 21)]
    assert cosine[0] == pytest.approx(1e-3)
    assert inverse[0] == pytest.approx(1e-3)
    assert all(left >= right for left, right in zip(cosine, cosine[1:]))
    assert all(left >= right for left, right in zip(inverse, inverse[1:]))
    assert cosine[-1] == pytest.approx(1e-4)
    assert inverse[-1] >= cosine[-1]


def test_final_epoch_budget_is_ceil_scaled_validation_epoch() -> None:
    assert final_epoch_budget(7) == 7
    assert final_epoch_budget(7, 1.25) == 9
    assert final_epoch_budget(7, 1.5) == 11


def test_final_mlp_is_not_refit_after_fixed_epoch_budget(monkeypatch) -> None:
    rng = np.random.default_rng(31)
    x = rng.normal(size=(120, 5)).astype(np.float32)
    returns = pd.Series(np.where(x[:, 0] + 0.2 * x[:, 1] > 0, 0.01, -0.01))
    monkeypatch.setattr(
        pooled_runner,
        "dense_grid",
        lambda classifier, stage: {
            "hidden_layer_sizes": [(8,)], "alpha": [1e-3],
            "max_iter": [6], "batch_size": [16],
            "learning_rate_init": [1e-3], "n_iter_no_change": [3],
        },
    )
    args = SimpleNamespace(
        classifier="simple_mlp", search_stage="coarse", seed=42,
        learning_rate_schedule="constant",
        early_stopping_metric="balanced_accuracy",
        max_iter_override=6, patience_override=3,
        final_epoch_multiplier=1.0,
    )
    result = pooled_runner.fit_one(
        x[:60], returns.iloc[:60], x[60:90], returns.iloc[60:90],
        x[:90], returns.iloc[:90], x[90:], args,
    )
    params, final_model = result[1], result[5]
    assert final_model.n_iter_ == params["final_fit_epoch"]
    assert len(final_model.loss_curve_) == params["final_fit_epoch"]


def test_token_gate_pca_fits_separate_fit_and_all_train_gates():
    # Two prompt positions with two hidden coordinates. The fit subset makes
    # token 0 discriminative; extra all-train rows make token 1 dominant.
    x_fit = np.array([
        [-4, -4, 0, 0], [-3, -3, 0, 0],
        [4, 4, 0, 0], [3, 3, 0, 0],
    ], dtype=np.float32)
    y_fit = np.array([-1, -1, 1, 1], dtype=float)
    x_extra = np.array([
        [0, 0, -20, -20], [0, 0, -18, -18],
        [0, 0, 20, 20], [0, 0, 18, 18],
    ], dtype=np.float32)
    x_all = np.concatenate([x_fit, x_extra])
    y_all = np.concatenate([y_fit, [-1, -1, 1, 1]])
    args = SimpleNamespace(
        reducer="token_gate_pca", reducer_components=1, seed=42,
        token_gate_keep=1, token_gate_method="fisher",
        scaler="none", classifier="logistic",
    )
    transformed = preprocess_windows(
        x_fit, x_fit[:2], x_all, x_extra[:2], args,
        y_fit=y_fit, y_all=y_all, prompt_token_shape=(2, 2),
    )
    x_fit_out, x_val_out, x_all_out, x_test_out = transformed[:4]
    gate_audits = transformed[-1]
    assert x_fit_out.shape == (4, 1)
    assert x_val_out.shape == (2, 1)
    assert x_all_out.shape == (8, 1)
    assert x_test_out.shape == (2, 1)
    assert gate_audits["fit"]["selected_positions_zero_based"] == [0]
    assert gate_audits["all_train"]["selected_positions_zero_based"] == [1]


def test_small_rolling_run_persists_alignment_runtime_and_labels(tmp_path, monkeypatch):
    rows = 18
    embedding_root = tmp_path / "embeddings"
    part = embedding_root / "shard-0" / "roberta" / "short"
    part.mkdir(parents=True)
    base = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
    np.savez_compressed(
        part / "short_pooling.npz",
        prompt_mean=base,
        title_mean=base,
        body_mean=base + 1,
        title_body_mean=base + 2,
        full_mean=base + 3,
    )
    (part / "metadata.jsonl").write_text(
        "".join(json.dumps({"row_index": index}) + "\n" for index in range(1, rows + 1)),
        encoding="utf-8",
    )
    (part / "summary.json").write_text(
        json.dumps({
            "model": "roberta",
            "variant": "short",
            "rows": rows,
            "outputs": {
                "prompt_mean": [rows, 3],
                "title_mean": [rows, 3],
                "body_mean": [rows, 3],
                "title_body_mean": [rows, 3],
                "full_mean": [rows, 3],
            },
        }),
        encoding="utf-8",
    )
    dates = [f"{year}-01-{day:02d}" for year in range(2018, 2027) for day in (2, 3)]
    signs = np.tile(np.array([-0.01, 0.01]), 9)
    panel = pd.DataFrame({
        "row_index": np.arange(1, rows + 1),
        "entry_date": dates,
        "event_return_3d": signs,
        "next_day_return": signs,
    })
    panel_path = tmp_path / "panel.parquet"
    output = tmp_path / "report.json"
    panel.to_parquet(panel_path, index=False)
    monkeypatch.setattr(sys, "argv", [
        "run_pooled_embedding_classification.py",
        str(panel_path),
        "--embedding-root", str(embedding_root),
        "--model", "roberta",
        "--variant", "short",
        "--feature", "full_mean",
        "--classifier", "logistic",
        "--reducer", "none",
        "--scaler", "none",
        "--expected-embedding-rows", str(rows),
        "--output", str(output),
    ])
    main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["input"]["embedding_alignment"]["matched_rows"] == rows
    assert report["input"]["embedding_alignment"]["unmatched_panel_rows"] == 0
    assert report["results"][0]["n_fit"] == 12
    assert report["results"][0]["n_validation"] == 4
    assert report["results"][0]["n_test_rows"] == 2
    assert report["runtime"]["total_seconds"] > 0
    predictions = pd.read_parquet(output.with_suffix(".predictions.parquet"))
    assert {"row_index", "next_day_return", "actual_label", "probability"}.issubset(predictions.columns)
    assert predictions["row_index"].tolist() == [17, 18]
    bundle = output.with_suffix(".artifacts")
    assert (bundle / "COMPLETED").is_file()
    assert (bundle / "spec.json").is_file()
    assert report["artifact_bundle"] == str(bundle)
    fold_bundle = bundle / "test_year_2026"
    final_model = joblib.load(fold_bundle / "final_model.joblib")
    preprocessor = joblib.load(fold_bundle / "all_train_preprocessor.joblib")
    x_test = base[16:18] + 3
    if preprocessor["reducer"] is not None:
        x_test = preprocessor["reducer"].transform(x_test)
    if preprocessor["scaler"] is not None:
        x_test = preprocessor["scaler"].transform(x_test)
    restored = final_model.predict_proba(x_test)[:, 1]
    np.testing.assert_allclose(restored, predictions["probability"].to_numpy(), atol=1e-7)

    model_mtime = (fold_bundle / "final_model.joblib").stat().st_mtime_ns
    main()
    assert (fold_bundle / "final_model.joblib").stat().st_mtime_ns == model_mtime

    screen_output = tmp_path / "screen.json"
    monkeypatch.setattr(sys, "argv", [
        "run_pooled_embedding_classification.py",
        str(panel_path),
        "--embedding-root", str(embedding_root),
        "--model", "roberta", "--variant", "short",
        "--feature", "full_mean", "--classifier", "logistic",
        "--reducer", "none", "--scaler", "none",
        "--run-mode", "screen",
        "--expected-embedding-rows", str(rows),
        "--output", str(screen_output),
    ])
    main()
    screen = json.loads(screen_output.read_text(encoding="utf-8"))
    assert screen["design"]["run_mode"] == "screen"
    assert screen["predictions"] is None
    assert screen["results"][0]["test_year"] is None
    assert screen["results"][0]["n_validation"] == 4
    assert not screen_output.with_suffix(".predictions.parquet").exists()
    assert not (
        screen_output.with_suffix(".artifacts") / "screen" / "final_model.joblib"
    ).exists()


def test_runner_accepts_ckip_plain_identity(tmp_path, monkeypatch):
    rows = 18
    embedding_root = tmp_path / "embeddings"
    part = embedding_root / "shard-0" / "ckip_bert" / "plain"
    part.mkdir(parents=True)
    matrix = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
    np.savez_compressed(part / "short_pooling.npz", full_mean=matrix)
    (part / "metadata.jsonl").write_text(
        "".join(json.dumps({"row_index": index}) + "\n" for index in range(1, rows + 1)),
        encoding="utf-8",
    )
    (part / "summary.json").write_text(
        json.dumps({
            "model": "ckip_bert",
            "variant": "plain",
            "rows": rows,
            "outputs": {"full_mean": [rows, 3]},
        }),
        encoding="utf-8",
    )
    dates = [f"{year}-01-{day:02d}" for year in range(2018, 2027) for day in (2, 3)]
    signs = np.tile(np.array([-0.01, 0.01]), 9)
    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame({
        "row_index": np.arange(1, rows + 1),
        "entry_date": dates,
        "next_day_return": signs,
    }).to_parquet(panel_path, index=False)
    output = tmp_path / "ckip.json"
    monkeypatch.setattr(sys, "argv", [
        "run_pooled_embedding_classification.py",
        str(panel_path),
        "--embedding-root", str(embedding_root),
        "--model", "ckip_bert",
        "--variant", "plain",
        "--feature", "full_mean",
        "--classifier", "logistic",
        "--reducer", "none",
        "--scaler", "none",
        "--train-target-column", "next_day_return",
        "--evaluation-target-column", "next_day_return",
        "--expected-embedding-rows", str(rows),
        "--output", str(output),
    ])

    main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["experiment"]["model"] == "ckip_bert"
    assert report["experiment"]["variant"] == "plain"
