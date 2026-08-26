import json
import sys
from pathlib import Path

import pandas as pd

from scripts.build_cninfo_unified_classification_input import main


def _record(stock_id: str, published_at: str, document_id: str) -> dict[str, str]:
    return {
        "stock_id": stock_id,
        "content_type": "announcement",
        "stock_relation": "direct",
        "published_at": published_at,
        "announcement_date": published_at[:10],
        "text_model": f"公告正文 {document_id}",
        "document_id": document_id,
    }


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_builder_streams_current_and_sorts_historical_with_global_rows(tmp_path, monkeypatch):
    current = tmp_path / "current.jsonl"
    historical = tmp_path / "historical.jsonl"
    panel = tmp_path / "panel.parquet"
    output = tmp_path / "unified.jsonl"
    historical_output = tmp_path / "historical_increment.jsonl"
    summary = tmp_path / "summary.json"

    current_records = [
        _record("000001", "2018-01-02T09:00:00", "current-1"),
        _record("000002", "2018-01-03T09:00:00", "current-2"),
    ]
    historical_records = [
        _record("000002", "2010-01-04T09:00:00", "historical-2"),
        _record("000001", "2010-01-03T09:00:00", "historical-1"),
    ]
    _write_jsonl(current, current_records)
    _write_jsonl(historical, historical_records)
    pd.DataFrame(
        {
            "row_index": [1, 2],
            "stock_id": ["000001", "000002"],
            "published_at": [
                "2018-01-02T09:00:00+00:00",
                "2018-01-03T09:00:00+00:00",
            ],
        }
    ).to_parquet(panel, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_cninfo_unified_classification_input.py",
            "--current", str(current),
            "--historical", str(historical),
            "--existing-panel", str(panel),
            "--stock-ids-from", str(panel),
            "--output", str(output),
            "--historical-output", str(historical_output),
            "--summary", str(summary),
            "--sort-chunk-rows", "1",
        ],
    )
    main()

    unified = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    increment = [
        json.loads(line)
        for line in historical_output.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["document_id"] for row in unified] == [
        "current-1", "current-2", "historical-1", "historical-2"
    ]
    assert [row["row_index"] for row in unified] == [1, 2, 3, 4]
    assert [row["row_index"] for row in increment] == [3, 4]
    assert json.loads(summary.read_text(encoding="utf-8"))["historical_sort"] == (
        "disk-backed bounded external merge sort"
    )


def test_builder_preserves_empty_text_row_in_frozen_current_input(tmp_path, monkeypatch):
    current = tmp_path / "current.jsonl"
    historical = tmp_path / "historical.jsonl"
    panel = tmp_path / "panel.parquet"
    output = tmp_path / "unified.jsonl"
    summary = tmp_path / "summary.json"

    current_records = [
        _record("000001", "2018-01-02T09:00:00", "current-1"),
        _record("000001", "2018-01-03T09:00:00", "current-empty"),
        _record("000002", "2018-01-04T09:00:00", "current-3"),
    ]
    current_records[1]["text_model"] = ""
    _write_jsonl(current, current_records)
    _write_jsonl(
        historical,
        [_record("000001", "2010-01-02T09:00:00", "historical-1")],
    )
    pd.DataFrame(
        {
            "row_index": [1, 2, 3],
            "stock_id": ["000001", "000001", "000002"],
            "published_at": [
                "2018-01-02T09:00:00+00:00",
                "2018-01-03T09:00:00+00:00",
                "2018-01-04T09:00:00+00:00",
            ],
        }
    ).to_parquet(panel, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_cninfo_unified_classification_input.py",
            "--current", str(current),
            "--historical", str(historical),
            "--existing-panel", str(panel),
            "--stock-ids-from", str(panel),
            "--output", str(output),
            "--summary", str(summary),
            "--sort-chunk-rows", "1",
        ],
    )
    main()

    unified = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["document_id"] for row in unified] == [
        "current-1", "current-empty", "current-3", "historical-1"
    ]
    assert [row["row_index"] for row in unified] == [1, 2, 3, 4]
    assert unified[1]["text_model"] == ""
    report = json.loads(summary.read_text(encoding="utf-8"))
    assert report["current_rows"] == 3
    assert report["rejected_records"]["2018_2026:empty_text_model_preserved"] == 1
