import json
import sys

import numpy as np
import pandas as pd

from scripts.run_reduced_prompt_classification import main


def test_reduced_prompt_defaults_to_one_day_target(tmp_path, monkeypatch):
    rows = []
    for year in range(2018, 2027):
        for stock_number in range(2):
            rows.append({
                "row_index": len(rows) + 1,
                "entry_date": f"{year}-01-0{stock_number + 2}",
                "next_day_return": -0.01 if stock_number == 0 else 0.01,
            })
    panel = tmp_path / "panel.parquet"
    matrix = tmp_path / "reduced.npy"
    output = tmp_path / "report.json"
    pd.DataFrame(rows).to_parquet(panel, index=False)
    np.save(matrix, np.asarray([
        [float(index), float(index % 3), 1.0]
        for index in range(len(rows))
    ], dtype=np.float32))

    monkeypatch.setattr(sys, "argv", [
        "run_reduced_prompt_classification.py", str(panel),
        "--matrix", str(matrix), "--model", "roberta", "--variant", "short",
        "--classifier", "logistic", "--output", str(output),
    ])
    main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["train_target_column"] == "next_day_return"
    assert report["evaluation_target_column"] == "next_day_return"
    assert report["results"][0]["train_target"] == "next_day_return"
    assert report["results"][0]["evaluation_target"] == "next_day_return"