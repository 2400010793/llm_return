from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import build_sina_cninfo_method_dynamic_classifier_manifest as dynamic_manifest
from scripts.run_pooled_embedding_classification import main, preprocess_windows
from src.models.dynamic_token_classifier import DynamicTokenGateTransformer


def test_dynamic_classifier_manifest_excludes_plain_and_covers_requested_classifiers(tmp_path: Path, monkeypatch) -> None:
    embedding_root = tmp_path / "embeddings"
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(dynamic_manifest, "MODELS", ("roberta",))
    monkeypatch.setattr(dynamic_manifest, "VARIANTS", ("short", "masked_short"))
    for variant in dynamic_manifest.VARIANTS:
        for shard in range(2):
            directory = embedding_root / f"shard-{shard}" / "roberta" / variant
            directory.mkdir(parents=True)
            (directory / "summary.json").write_text(json.dumps({"rows": 2}), encoding="utf-8")
            for name in (
                "metadata.jsonl", "short_pooling.npz", "prompt_token_embeddings.npy",
                "prompt_input_ids.npy", "prompt_tokens.json",
            ):
                (directory / name).write_bytes(b"x")

    rows = dynamic_manifest.build_rows(
        embedding_root, output_root, expected_shards=2, expected_rows=4,
        classifiers=("logistic", "knn"),
    )
    assert len(rows) == 4
    assert {row["variant"] for row in rows} == {"short", "masked_short"}
    assert {row["classifier"] for row in rows} == {"logistic", "knn"}
    assert {row["reducer"] for row in rows} == {"dynamic_token_gate"}


def test_dynamic_token_gate_exports_finite_representations_and_weights() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(12, 3 * 4)).astype(np.float32)
    targets = np.array([-1.0, 1.0] * 6)
    gate = DynamicTokenGateTransformer(
        3, 4, gate_hidden_size=5, representation_size=6,
        epochs=2, batch_size=4, device="cpu", random_state=11,
    ).fit(matrix, targets)

    representation, weights = gate.transform(matrix, return_weights=True)
    assert representation.shape == (12, 6)
    assert weights.shape == (12, 3)
    assert np.isfinite(representation).all()
    assert np.isfinite(weights).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    assert gate.audit(matrix, scope="fit")["training_rows"] == 12


def test_dynamic_gate_fit_ignores_nonfinite_targets_and_transform_has_no_label_argument() -> None:
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(6, 2 * 3)).astype(np.float32)
    targets = np.array([-1.0, 1.0, np.nan, -1.0, 1.0, np.nan])
    gate = DynamicTokenGateTransformer(
        2, 3, gate_hidden_size=4, representation_size=4,
        epochs=1, batch_size=2, device="cpu", random_state=13,
    ).fit(matrix, targets)
    assert gate.n_rows_fit_ == 4
    before = gate.transform(matrix)
    after = gate.transform(matrix)
    np.testing.assert_array_equal(before, after)


def test_runner_dynamic_gate_refits_fit_and_all_train_separately() -> None:
    x_fit = np.array([
        [-4, -4, 0, 0], [-3, -3, 0, 0],
        [4, 4, 0, 0], [3, 3, 0, 0],
    ], dtype=np.float32)
    y_fit = np.array([-1, -1, 1, 1], dtype=float)
    x_extra = np.array([
        [0, 0, -5, -5], [0, 0, -4, -4],
        [0, 0, 5, 5], [0, 0, 4, 4],
    ], dtype=np.float32)
    x_all = np.concatenate([x_fit, x_extra])
    y_all = np.concatenate([y_fit, [-1, -1, 1, 1]])
    args = SimpleNamespace(
        reducer="dynamic_token_gate", seed=42, scaler="none", classifier="logistic",
        dynamic_gate_mode="dynamic", dynamic_gate_hidden_size=4,
        dynamic_gate_representation_size=5, dynamic_gate_dropout=0.0,
        dynamic_gate_learning_rate=1e-3, dynamic_gate_weight_decay=0.0,
        dynamic_gate_epochs=1, dynamic_gate_batch_size=2,
        dynamic_gate_gradient_clip_norm=1.0, dynamic_gate_device="cpu",
    )

    transformed = preprocess_windows(
        x_fit, x_fit[:2], x_all, x_extra[:2], args,
        y_fit=y_fit, y_all=y_all, prompt_token_shape=(2, 2),
    )
    x_fit_out, x_val_out, x_all_out, x_test_out = transformed[:4]
    preprocessors = transformed[7]
    audits = transformed[-1]
    assert x_fit_out.shape == (4, 5)
    assert x_val_out.shape == (2, 5)
    assert x_all_out.shape == (8, 5)
    assert x_test_out.shape == (2, 5)
    assert preprocessors["fit"]["dynamic_token_gate"].n_rows_fit_ == 4
    assert preprocessors["all_train"]["dynamic_token_gate"].n_rows_fit_ == 8
    assert audits["fit"]["scope"] == "fit"
    assert audits["all_train"]["scope"] == "all_train"


def test_dynamic_gate_screen_run_persists_frozen_preprocessor(tmp_path: Path, monkeypatch) -> None:
    rows = 18
    embedding_root = tmp_path / "embeddings"
    part = embedding_root / "shard-0" / "roberta" / "short"
    part.mkdir(parents=True)
    tokens = np.arange(rows * 2 * 3, dtype=np.float32).reshape(rows, 2, 3)
    np.save(part / "prompt_token_embeddings.npy", tokens)
    np.savez_compressed(part / "short_pooling.npz")
    (part / "metadata.jsonl").write_text(
        "".join(json.dumps({"row_index": index}) + "\n" for index in range(1, rows + 1)),
        encoding="utf-8",
    )
    (part / "summary.json").write_text(json.dumps({
        "model": "roberta", "variant": "short", "rows": rows,
        "outputs": {"prompt_token_embeddings": [rows, 2, 3]},
    }), encoding="utf-8")
    dates = [
        f"{year}-01-{day:02d}"
        for year in range(2018, 2027)
        for day in (2, 3)
    ]
    signs = np.tile(np.array([-0.01, 0.01]), 9)
    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame({
        "row_index": np.arange(1, rows + 1),
        "entry_date": dates,
        "event_return_3d": signs,
        "next_day_return": signs,
    }).to_parquet(panel_path, index=False)
    output = tmp_path / "dynamic.json"
    monkeypatch.setattr(__import__("sys").modules["sys"], "argv", [
        "run_pooled_embedding_classification.py", str(panel_path),
        "--embedding-root", str(embedding_root), "--model", "roberta",
        "--variant", "short", "--feature", "prompt_tokens_flat",
        "--classifier", "logistic", "--reducer", "dynamic_token_gate",
        "--scaler", "none", "--dynamic-gate-device", "cpu",
        "--dynamic-gate-epochs", "1", "--dynamic-gate-batch-size", "4",
        "--dynamic-gate-hidden-size", "4", "--dynamic-gate-representation-size", "5",
        "--expected-embedding-rows", str(rows), "--run-mode", "screen",
        "--output", str(output),
    ])

    main()

    report = json.loads(output.read_text(encoding="utf-8"))
    result = report["results"][0]
    assert result["reducer"] == "dynamic_token_gate"
    assert result["dynamic_token_gate"]["fit"]["training_rows"] == 12
    assert result["dynamic_token_gate"]["all_train"]["training_rows"] == 12
    assert (output.with_suffix(".artifacts") / "screen" / "fit_preprocessor.joblib").is_file()
