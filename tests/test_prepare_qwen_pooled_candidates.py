import json
import sys

import numpy as np
import pandas as pd

from scripts.prepare_qwen_pooled_candidates import main
from src.data.pooled_embeddings import load_pooled_embeddings


def test_prepare_qwen_candidates_is_aligned_and_resumable(tmp_path, monkeypatch):
    source = tmp_path / "source"
    rows = 0
    for part_id, count in enumerate((3, 2)):
        part = source / f"part-{part_id:05d}"
        part.mkdir(parents=True)
        indexes = np.arange(rows, rows + count, dtype=np.int64)
        pd.DataFrame({"row_index": indexes, "stock_id": "A"}).to_parquet(
            part / "metadata.parquet", index=False
        )
        for view_number, view in enumerate(("short", "masked_short", "plain")):
            matrix = np.full((count, 4), part_id + view_number, dtype=np.float32)
            np.save(part / f"{view}.npy", matrix)
        rows += count
    output = tmp_path / "pooled"
    argv = [
        "prepare_qwen_pooled_candidates.py", str(source), str(output),
        "--expected-parts", "2", "--expected-rows", "5",
        "--expected-dimension", "4",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    prepared = load_pooled_embeddings(
        output, model="qwen3_embedding_8b", variant="short",
        feature="full_mean", require_complete_rows=5,
    )
    assert prepared.matrix.shape == (5, 4)
    assert prepared.metadata["row_index"].tolist() == [1, 2, 3, 4, 5]
    main()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["converted_cells"] == 0
    assert summary["resumed_cells"] == 6
