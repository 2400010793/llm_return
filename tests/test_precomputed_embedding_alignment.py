import numpy as np
import pandas as pd
import pytest

from scripts.run_paper_rolling_classification import align_precomputed_matrix


def panel(keys, key="document_id"):
    return pd.DataFrame({key: keys, "__panel_position": np.arange(len(keys))})


def test_metadata_alignment_reorders_matrix_and_supports_panel_subset():
    matrix = np.array([[30.0], [10.0], [20.0], [99.0]])
    metadata = pd.DataFrame({"document_id": ["c", "a", "b", "unused"]})
    aligned, audit = align_precomputed_matrix(
        panel(["a", "b", "c"]), matrix, metadata, alignment_key="document_id"
    )
    np.testing.assert_array_equal(aligned[:, 0], [10.0, 20.0, 30.0])
    assert audit["mode"] == "metadata_key"
    assert audit["key"] == "document_id"


def test_positional_alignment_requires_equal_complete_rows():
    frame = panel(["a", "b"])
    with pytest.raises(ValueError, match="different row counts"):
        align_precomputed_matrix(frame, np.zeros((3, 2)), None)


def test_positional_alignment_tracks_pre_sort_source_positions():
    frame = panel(["a", "b", "c"]).iloc[[2, 0, 1]].reset_index(drop=True)
    matrix = np.array([[10.0], [20.0], [30.0]])
    aligned, audit = align_precomputed_matrix(frame, matrix, None)
    np.testing.assert_array_equal(aligned[:, 0], [30.0, 10.0, 20.0])
    assert audit["mode"] == "exact_positional"


@pytest.mark.parametrize("side", ["panel", "metadata"])
def test_duplicate_alignment_keys_fail(side):
    frame = panel(["a", "b"] if side == "metadata" else ["a", "a"])
    metadata = pd.DataFrame({"document_id": ["a", "a"] if side == "metadata" else ["a", "b"]})
    with pytest.raises(ValueError, match=f"{side} alignment key"):
        align_precomputed_matrix(frame, np.zeros((2, 1)), metadata, alignment_key="document_id")


def test_missing_panel_key_fails_closed():
    metadata = pd.DataFrame({"document_id": ["a", "c"]})
    with pytest.raises(ValueError, match="does not cover all panel keys"):
        align_precomputed_matrix(
            panel(["a", "b"]), np.zeros((2, 1)), metadata, alignment_key="document_id"
        )


def test_metadata_matrix_length_mismatch_fails():
    metadata = pd.DataFrame({"document_id": ["a"]})
    with pytest.raises(ValueError, match="metadata and matrix row counts differ"):
        align_precomputed_matrix(
            panel(["a"]), np.zeros((2, 1)), metadata, alignment_key="document_id"
        )


def test_zero_based_metadata_requires_explicit_offset():
    frame = panel([1, 2, 3], key="row_index")
    metadata = pd.DataFrame({"row_index": [0, 1, 2]})
    matrix = np.array([[10.0], [20.0], [30.0]])
    with pytest.raises(ValueError, match="does not cover all panel keys"):
        align_precomputed_matrix(frame, matrix, metadata, alignment_key="row_index")
    aligned, audit = align_precomputed_matrix(
        frame,
        matrix,
        metadata,
        alignment_key="row_index",
        metadata_row_index_offset=1,
    )
    np.testing.assert_array_equal(aligned[:, 0], [10.0, 20.0, 30.0])
    assert audit["metadata_row_index_offset"] == 1
