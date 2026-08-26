import json
import sys

import numpy as np
import pandas as pd

from scripts.run_dynamic_prompt_token_gating import fit_hard_prompt_gate
from scripts.run_prompt_token_pca_v5 import main
from src.data.prompt_token_embeddings import PromptTokenEmbeddingStore


def test_v5_prompt_gate_preserves_selected_vectors_before_pca(tmp_path, monkeypatch):
    rows = 36
    panel = pd.DataFrame({
        "row_index": np.arange(1, rows + 1),
        "entry_date": [f"{year}-01-{day:02d}" for year in range(2018, 2027) for day in range(1, 5)],
        "event_return_3d": np.tile(np.array([-1.0, 1.0, -2.0, 2.0]), 9),
    })
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    input_root = tmp_path / "tokens"
    for shard in range(32):
        row_indexes = np.arange(shard + 1, rows + 1, 32, dtype=np.int64)
        directory = input_root / f"shard-{shard}" / "roberta" / "short"
        directory.mkdir(parents=True)
        values = np.zeros((len(row_indexes), 3, 2), dtype=np.float32)
        targets = panel.set_index("row_index").loc[row_indexes, "event_return_3d"].to_numpy()
        values[:, 0, :] = np.array([1.0, 2.0])
        values[:, 1, 0] = targets * 5
        values[:, 1, 1] = targets * 4
        values[:, 2, :] = np.array([3.0, 4.0])
        np.save(directory / "prompt_token_embeddings.npy", values)
        (directory / "metadata.jsonl").write_text(
            "".join(json.dumps({"row_index": int(row)}) + "\n" for row in row_indexes),
            encoding="utf-8",
        )
    output_dir = tmp_path / "reduced"
    monkeypatch.setattr(sys, "argv", [
        "run_prompt_token_pca_v5.py", str(panel_path),
        "--input-root", str(input_root),
        "--output-dir", str(output_dir),
        "--model", "roberta", "--variant", "short",
        "--components", "1", "--fit-sample-rows", "24",
        "--transform-batch-size", "2",
        "--token-gate-method", "fisher", "--token-gate-keep", "1",
    ])
    main()
    matrix = np.load(output_dir / "fisher_gate1_pca_1.npy")
    summary = json.loads(
        (output_dir / "fisher_gate1_pca_1.summary.json").read_text(encoding="utf-8")
    )
    assert matrix.shape == (rows, 1)
    assert summary["token_gate"]["selected_positions_zero_based"] == [1]
    assert summary["token_gate"]["gated_dimension"] == 2
    assert summary["token_gate"]["aggregation"].startswith("none")


def test_fixed_and_random_prompt_controls_are_deterministic(tmp_path):
    root = tmp_path / "tokens"
    directory = root / "shard-0" / "roberta" / "short"
    directory.mkdir(parents=True)
    values = np.arange(4 * 6 * 2, dtype=np.float32).reshape(4, 6, 2)
    np.save(directory / "prompt_token_embeddings.npy", values)
    (directory / "metadata.jsonl").write_text(
        "".join(json.dumps({"row_index": i}) + "\n" for i in range(1, 5)),
        encoding="utf-8",
    )
    (directory / "prompt_tokens.json").write_text(
        json.dumps({"tokens": list("abcdef"), "text": "abcdef"}), encoding="utf-8",
    )
    (directory / "summary.json").write_text("{}", encoding="utf-8")
    store = PromptTokenEmbeddingStore(
        root, model="roberta", variant="short", expected_shards=1,
        expected_rows=4, expected_token_count=6, expected_hidden_size=2,
    )
    selected = np.ones(5, dtype=bool)
    targets = np.arange(5, dtype=float)
    first = fit_hard_prompt_gate(
        store, selected, targets, method="first", keep_tokens=4,
        batch_size=2, seed=42,
    )
    random_a = fit_hard_prompt_gate(
        store, selected, targets, method="random", keep_tokens=4,
        batch_size=2, seed=13,
    )
    random_b = fit_hard_prompt_gate(
        store, selected, targets, method="random", keep_tokens=4,
        batch_size=2, seed=13,
    )
    assert first.selected_positions.tolist() == [0, 1, 2, 3]
    np.testing.assert_array_equal(random_a.selected_positions, random_b.selected_positions)
    assert len(random_a.selected_positions) == 4
