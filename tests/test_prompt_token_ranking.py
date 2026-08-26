import json
import sys

import numpy as np
import pandas as pd

from scripts.rank_prompt_tokens_v5 import main


def test_rank_prompt_tokens_streams_shards_and_maps_positions(tmp_path, monkeypatch):
    records = []
    matrices = []
    row_index = 1
    for year in range(2018, 2027):
        for target in (-1.0, 1.0):
            records.append({
                "row_index": row_index,
                "entry_date": f"{year}-06-01",
                "next_day_return": target,
            })
            matrices.append([
                [0.0, 0.0],
                [target * 3.0, target * 2.0],
                [1.0, 1.0],
            ])
            row_index += 1
    panel = tmp_path / "panel.parquet"
    pd.DataFrame(records).to_parquet(panel, index=False)
    directory = tmp_path / "tokens" / "shard-0" / "roberta" / "masked_short"
    directory.mkdir(parents=True)
    np.save(directory / "prompt_token_embeddings.npy", np.asarray(matrices, dtype=np.float32))
    (directory / "metadata.jsonl").write_text(
        "".join(json.dumps({"row_index": index}) + "\n" for index in range(1, row_index)),
        encoding="utf-8",
    )
    (directory / "prompt_tokens.json").write_text(json.dumps({
        "text": "甲乙丙", "tokens": ["甲", "乙", "丙"],
    }, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "ranking.json"
    monkeypatch.setattr(sys, "argv", [
        "rank_prompt_tokens_v5.py", str(panel), "--input-root", str(tmp_path / "tokens"),
        "--model", "roberta", "--variant", "masked_short", "--method", "fisher",
        "--keep-tokens", "1", "--batch-size", "3", "--expected-shards", "1",
        "--output", str(output),
    ])
    main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["fit"]["years"] == list(range(2018, 2024))
    assert report["all_train"]["years"] == list(range(2018, 2026))
    assert report["fit"]["ranking"][0]["token"] == "乙"
    assert report["all_train"]["selected_positions_one_based"] == [2]

    initial = output.stat().st_mtime_ns
    main()
    assert output.stat().st_mtime_ns == initial
