import pandas as pd

from scripts.build_sina_cninfo_method_inputs import (
    LONG_PROMPT,
    SHORT_PROMPT,
    build_records,
)


def test_sina_cninfo_method_inputs_use_paired_fixed_prompts():
    clean = pd.DataFrame({
        "article_id": ["a", "b"],
        "stock_id": ["000001", "000002"],
        "stock_name": ["平安银行", "万科A"],
        "title_clean": ["平安银行2025年业绩", "万科A公告"],
        "body_clean": ["平安银行上涨", "万科A下跌"],
        "text": ["标题正文一", "标题正文二"],
    })
    panel = pd.DataFrame({"article_id": ["a", "b"], "row_index": [1, 2]})

    rows = build_records(clean, panel)

    assert [row["row_index"] for row in rows] == [1, 2]
    assert all(row["short_prompt"] == row["masked_short_prompt"] == SHORT_PROMPT for row in rows)
    assert all(row["long_prompt"] == row["masked_long_prompt"] == LONG_PROMPT for row in rows)
    assert rows[0]["short_body"] != rows[0]["masked_short_body"]
    assert "000001" not in rows[0]["masked_short_body"]


def test_sina_cninfo_method_inputs_reject_non_contiguous_row_index():
    clean = pd.DataFrame({
        "article_id": ["a"], "stock_id": ["1"], "stock_name": ["公司"],
        "title_clean": ["标题"], "body_clean": ["正文"], "text": ["全文"],
    })
    panel = pd.DataFrame({"article_id": ["a"], "row_index": [0]})

    try:
        build_records(clean, panel)
    except ValueError as error:
        assert "1..N" in str(error)
    else:
        raise AssertionError("non-contiguous row_index should fail")
