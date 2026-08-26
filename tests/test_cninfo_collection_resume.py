import json
from argparse import Namespace

from scripts.collect_cninfo_announcements import build_output_payload, write_json_atomic
from scripts.run_cninfo_paper_collection import reusable_complete_output


def test_reusable_complete_output_accepts_zero_record_exact_query(tmp_path):
    output = tmp_path / "stock.json"
    output.write_text(json.dumps({
        "status": "complete",
        "query": {
            "start_date": "2018-01-01", "end_date": "2018-12-31",
            "index_only": False, "focus_only": False,
        },
        "records": [],
    }), encoding="utf-8")

    reusable, rows = reusable_complete_output(
        output, start_date="2018-01-01", end_date="2018-12-31",
        index_only=False, focus_only=False,
    )

    assert reusable is True
    assert rows == 0


def test_reusable_complete_output_rejects_query_mismatch_and_errors(tmp_path):
    output = tmp_path / "stock.json"
    output.write_text(json.dumps({
        "status": "complete",
        "query": {
            "start_date": "2018-01-01", "end_date": "2018-12-31",
            "index_only": False, "focus_only": False,
        },
        "records": [{"content_type": "collection_error"}],
    }), encoding="utf-8")

    with_error, _ = reusable_complete_output(
        output, start_date="2018-01-01", end_date="2018-12-31",
        index_only=False, focus_only=False,
    )
    wrong_date, _ = reusable_complete_output(
        output, start_date="2019-01-01", end_date="2019-12-31",
        index_only=False, focus_only=False,
    )

    assert with_error is False
    assert wrong_date is False


def test_collector_payload_records_query_and_completion_status(tmp_path):
    args = Namespace(
        start_date="2023-01-01", end_date="2023-12-31",
        index_only=False, focus_only=False,
    )
    complete = build_output_payload(args, [{"content_type": "announcement"}], "now")
    partial = build_output_payload(
        args, [{"content_type": "collection_error"}], "now"
    )

    assert complete["status"] == "complete"
    assert complete["query"] == {
        "start_date": "2023-01-01", "end_date": "2023-12-31",
        "index_only": False, "focus_only": False,
    }
    assert partial["status"] == "partial"

    output = tmp_path / "nested" / "result.json"
    write_json_atomic(output, complete)
    assert json.loads(output.read_text(encoding="utf-8")) == complete
    assert not list(output.parent.glob("*.tmp"))
