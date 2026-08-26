import pandas as pd
import numpy as np

from scripts.ensemble_title_body_mlp_predictions import load_aligned


def test_load_aligned_stacks_seed_probabilities(tmp_path):
    base = pd.DataFrame({
        "row_index": [2, 1],
        "entry_date": pd.to_datetime(["2024-01-03", "2024-01-02"]),
        "test_year": [2024, 2024],
        "actual_label": [1.0, 0.0],
        "probability": [0.8, 0.2],
    })
    paths = []
    for seed, offset in ((42, 0.0), (43, 0.1)):
        path = tmp_path / f"seed{seed}.parquet"
        frame = base.copy()
        frame["probability"] += offset
        frame.to_parquet(path, index=False)
        paths.append(path)
    reference, probabilities = load_aligned(paths)
    assert reference["row_index"].tolist() == [1, 2]
    np.testing.assert_allclose(probabilities, [[0.2, 0.3], [0.8, 0.9]])
