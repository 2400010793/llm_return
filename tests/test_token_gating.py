import numpy as np
import pytest
from types import SimpleNamespace

from scripts.run_pooled_embedding_classification import preprocess_windows
from src.models.token_gating import fit_streaming_token_gate, fit_token_gate


def test_fisher_gate_selects_discriminative_token_and_preserves_full_vector():
    # Three token positions, two hidden coordinates. Only token position 1
    # changes systematically with the class label.
    matrix = np.array([
        [1, 2, -5, -4, 7, 8],
        [2, 1, -4, -5, 8, 7],
        [1, 2, 5, 4, 7, 8],
        [2, 1, 4, 5, 8, 7],
    ], dtype=np.float32)
    targets = np.array([-1.0, -2.0, 1.0, 2.0])
    gate = fit_token_gate(
        matrix, targets, token_count=3, keep_tokens=1, method="fisher", batch_size=2
    )
    assert gate.selected_positions.tolist() == [1]
    transformed = gate.transform(matrix)
    assert transformed.shape == (4, 2)
    np.testing.assert_array_equal(transformed, matrix[:, 2:4])


def test_gate_selection_uses_only_supplied_training_rows():
    train = np.array([
        [-4, -4, 1, 1],
        [-3, -3, 2, 2],
        [4, 4, 1, 1],
        [3, 3, 2, 2],
    ], dtype=np.float32)
    targets = np.array([-1.0, -1.0, 1.0, 1.0])
    gate = fit_token_gate(train, targets, token_count=2, keep_tokens=1)
    assert gate.selected_positions.tolist() == [0]


def test_gate_rejects_invalid_keep_count():
    with pytest.raises(ValueError, match="keep_tokens"):
        fit_token_gate(
            np.ones((4, 6), dtype=np.float32), np.array([-1, -1, 1, 1]),
            token_count=3, keep_tokens=4,
        )


def test_logistic_l1_gate_learns_discriminative_group():
    rng = np.random.default_rng(42)
    labels = np.tile(np.array([-1.0, 1.0]), 100)
    matrix = rng.normal(0, 0.1, size=(200, 12)).astype(np.float32)
    matrix[:, 4:8] += labels[:, None] * 2.0
    gate = fit_token_gate(
        matrix, labels, token_count=3, keep_tokens=1,
        method="logistic_l1", batch_size=31,
    )
    assert gate.selected_positions.tolist() == [1]
    np.testing.assert_array_equal(gate.transform(matrix), matrix[:, 4:8])


def test_group_gate_is_refitted_for_fit_and_all_train_windows():
    fit = np.array([
        [-3, -2, 0, 0, 1, 1], [-2, -3, 0, 0, 1, 1],
        [3, 2, 0, 0, 1, 1], [2, 3, 0, 0, 1, 1],
    ], dtype=np.float32)
    val = fit.copy()
    all_train = np.column_stack((
        np.zeros((4, 2), dtype=np.float32), fit[:, :2], np.ones((4, 2), dtype=np.float32)
    ))
    test = all_train.copy()
    targets = np.array([-1.0, -1.0, 1.0, 1.0])
    args = SimpleNamespace(
        reducer="group_gate", group_gate_method="fisher", group_gate_keep=1,
        token_gate_method="fisher", token_gate_keep=1,
        scaler="none", classifier="logistic", reducer_components=2, seed=42,
    )
    transformed = preprocess_windows(
        fit, val, all_train, test, args,
        y_fit=targets, y_all=targets, group_shape=(3, 2),
    )
    x_fit, _, x_all, _, _, _, _, preprocessors, _ = transformed
    assert x_fit.shape == (4, 2)
    assert x_all.shape == (4, 2)
    assert preprocessors["fit"]["token_gate"].selected_positions.tolist() == [0]
    assert preprocessors["all_train"]["token_gate"].selected_positions.tolist() == [1]


def test_streaming_fisher_gate_matches_in_memory_gate():
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(20, 12)).astype(np.float32)
    targets = np.array([-1.0] * 10 + [1.0] * 10)
    matrix[targets > 0, 4:8] += 2.0
    expected = fit_token_gate(
        matrix, targets, token_count=3, keep_tokens=2, method="fisher",
    )
    actual = fit_streaming_token_gate(
        ((matrix[:7], targets[:7]), (matrix[7:15], targets[7:15]),
         (matrix[15:], targets[15:])),
        token_count=3, keep_tokens=2, method="fisher",
    )
    np.testing.assert_allclose(actual.scores, expected.scores, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(actual.selected_positions, expected.selected_positions)


def test_streaming_variance_gate_matches_in_memory_gate():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(18, 15)).astype(np.float32)
    targets = np.linspace(-1.0, 1.0, len(matrix))
    expected = fit_token_gate(
        matrix, targets, token_count=3, keep_tokens=1, method="variance",
    )
    actual = fit_streaming_token_gate(
        ((matrix[:9], targets[:9]), (matrix[9:], targets[9:])),
        token_count=3, keep_tokens=1, method="variance",
    )
    np.testing.assert_allclose(actual.scores, expected.scores, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(actual.selected_positions, expected.selected_positions)