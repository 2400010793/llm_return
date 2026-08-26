import json
import sys

import numpy as np
import pandas as pd

from scripts.compose_prompt_pca_segment_features import main
from tests.test_pooled_embeddings import write_part


def test_composes_training_only_prompt_pca_before_title_and_body(tmp_path, monkeypatch):
    panel = tmp_path / "panel.parquet"
    pd.DataFrame({"row_index": [2, 1]}).to_parquet(panel, index=False)
    pooled_root = tmp_path / "pooled"
    write_part(
        pooled_root,
        0,
        [1, 2],
        np.array([[10, 11], [20, 21]], dtype=np.float32),
    )
    pca_dir = tmp_path / "pca"
    pca_dir.mkdir()
    np.save(pca_dir / "pca_1.npy", np.array([[100], [200]], dtype=np.float32))
    (pca_dir / "summary.json").write_text(
        json.dumps({
            "model": "roberta",
            "variant": "short",
            "rows": 2,
            "components": 1,
            "input_dimension": 4,
            "explained_variance": 0.9,
            "pca_fit_years": [2010, 2011, 2012, 2013, 2014, 2015],
            "preprocessing_protocol": (
                "PCA and token gate fit once on the earliest six-year fit population, "
                "then frozen for every leakage-safe rolling classifier fold"
            ),
        }),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "compose_prompt_pca_segment_features.py",
        str(panel),
        "--embedding-root", str(pooled_root),
        "--prompt-pca-dir", str(pca_dir),
        "--output-dir", str(output),
        "--model", "roberta",
        "--variant", "short",
        "--components", "1",
        "--expected-rows", "2",
        "--batch-size", "1",
    ])

    main()

    np.testing.assert_array_equal(
        np.load(output / "prompt_pca128_title.npy"),
        np.array([[100, 11, 12], [200, 21, 22]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.load(output / "prompt_pca128_body.npy"),
        np.array([[100, 12, 13], [200, 22, 23]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.load(output / "prompt_pca128_title_body.npy"),
        np.array(
            [[100, 11, 12, 12, 13], [200, 21, 22, 22, 23]],
            dtype=np.float32,
        ),
    )
    assert pd.read_parquet(output / "metadata.parquet")["row_index"].tolist() == [1, 2]
    assert (output / "COMPLETED").is_file()
