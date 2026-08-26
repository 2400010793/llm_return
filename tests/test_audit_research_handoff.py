import json
from pathlib import Path

import pandas as pd

from scripts.audit_research_handoff import build_status


def test_build_status_reports_panel_and_missing_shards(tmp_path: Path):
    data_root = tmp_path / "data-root"
    neutral = data_root / "neutral_masked_short_v1"
    for dataset, rows in (("sina", 3), ("cninfo", 4)):
        input_root = neutral / "inputs" / dataset
        input_root.mkdir(parents=True)
        (input_root / "manifest.json").write_text(
            json.dumps({"dataset": dataset, "rows": rows, "shards": 256}),
            encoding="utf-8",
        )
        for model in ("roberta", "bge_m3"):
            marker = neutral / "embeddings" / dataset / model / "shard-0"
            marker.mkdir(parents=True)
            (marker / "COMPLETED").write_text("ok\n", encoding="utf-8")

    sina_panel = data_root / "classification" / "sina_full_classification_panel.parquet"
    sina_panel.parent.mkdir(parents=True)
    pd.DataFrame({"row_index": [1, 2, 3]}).to_parquet(sina_panel, index=False)
    cninfo_panel = tmp_path / "cninfo.parquet"
    pd.DataFrame({"row_index": [1, 2, 3, 4]}).to_parquet(cninfo_panel, index=False)
    report = tmp_path / "report.md"
    report.write_text("# report\n\nresult\n", encoding="utf-8")

    status = build_status(
        data_root=data_root,
        cninfo_panel=cninfo_panel,
        report=report,
        include_slurm=False,
        slurm_user="unused",
    )

    assert status["panels"]["sina"]["rows"] == 3
    assert status["panels"]["cninfo"]["rows"] == 4
    family = status["neutral_masked_short_embeddings"]["sina"]["roberta"]
    assert family["completed_shards"] == 1
    assert family["missing_shards"][:2] == [1, 2]
    assert not family["complete"]
    assert status["master_report"]["lines"] == 3
