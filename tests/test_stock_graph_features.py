import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from scripts.build_stock_graph_features import (
    aggregate_stock_days,
    graph_group_codes,
    shuffled_group_labels,
    write_neighbor_means,
)
from scripts.run_stock_graph_rolling_classification import make_features


def test_aggregate_stock_days_averages_announcements_and_preserves_counts():
    frame = pd.DataFrame({
        "row_index": [1, 2, 3, 4],
        "stock_id": ["000002", "000001", "000001", "000001"],
        "entry_date": ["2020-01-02", "2020-01-01", "2020-01-01", "2020-01-02"],
        "next_day_return": [0.02, -0.01, -0.01, 0.03],
    })
    matrix = np.array([[20, 22], [10, 12], [14, 16], [30, 32]], dtype=np.float32)

    nodes, values = aggregate_stock_days(frame, matrix)

    assert nodes[["entry_date", "stock_id"]].astype(str).values.tolist() == [
        ["2020-01-01", "000001"],
        ["2020-01-02", "000001"],
        ["2020-01-02", "000002"],
    ]
    assert nodes["announcement_count"].tolist() == [2, 1, 1]
    np.testing.assert_array_equal(
        values,
        np.array([[12, 14], [30, 32], [20, 22]], dtype=np.float32),
    )


def test_neighbor_means_exclude_self_and_zero_singletons(tmp_path):
    nodes = pd.DataFrame({
        "entry_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01", "2020-01-02"]),
        "stock_id": ["000001", "000002", "000003", "000001"],
    })
    groups = {"000001": "A", "000002": "A", "000003": "B"}
    codes, neighbor_counts = graph_group_codes(nodes, groups)
    features = np.array([[1, 2], [3, 4], [10, 20], [5, 6]], dtype=np.float32)
    output = tmp_path / "neighbor.npy"

    written_counts = write_neighbor_means(features, codes, output, batch_size=2)

    np.testing.assert_array_equal(neighbor_counts, [1, 1, 0, 0])
    np.testing.assert_array_equal(written_counts, neighbor_counts)
    np.testing.assert_array_equal(
        np.load(output),
        np.array([[3, 4], [1, 2], [0, 0], [0, 0]], dtype=np.float32),
    )


def test_random_graph_preserves_group_sizes():
    mapping = {
        "000001": "large",
        "000002": "large",
        "000003": "large",
        "000004": "small",
    }
    shuffled = shuffled_group_labels(mapping, seed=42)

    assert set(shuffled) == set(mapping)
    assert sorted(shuffled.values()) == sorted(mapping.values())
    assert shuffled_group_labels(mapping, seed=42) == shuffled


def test_feature_scaling_preserves_absent_neighbor_zero():
    own = np.array([[10, 20], [12, 24]], dtype=np.float32)
    neighbor = np.array([[0, 0], [12, 21]], dtype=np.float32)
    values = make_features(
        own,
        neighbor,
        np.array([0, 1]),
        StandardScaler(),
        fit=True,
        neighbor_present=np.array([False, True]),
    )

    np.testing.assert_array_equal(values[0, 2:], [0, 0])
    assert not np.array_equal(values[1, 2:], [0, 0])
