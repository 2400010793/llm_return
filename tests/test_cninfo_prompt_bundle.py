from scripts.build_cninfo_prompt_bundle import (
    LONG_PROMPT,
    PROMPT_VERSION,
    SHORT_PROMPT,
    build_prompt_record,
)
from scripts.audit_cninfo_prompt_bundle import _residual_leakage
from scripts.build_cninfo_prompt_inputs import (
    COMPANY_NAME,
    DATE_PATTERNS,
    mask_identity_and_time,
)


def test_prompt_bundle_matches_frozen_prompt_and_mask_contract():
    source = {
        "document_id": "doc-1",
        "stock_id": "000001",
        "stock_name": "平安银行",
        "announcement_date": "2026-08-01",
        "title_clean_final": "平安银行2026年董事会公告",
        "text_model": "平安银行股份有限公司于2026年8月1日发布公告。",
    }

    record = build_prompt_record(source, 1, layout="long")

    assert record["prompt_version"] == PROMPT_VERSION
    assert record["short_prompt"] == record["masked_short_prompt"] == SHORT_PROMPT
    assert record["long_prompt"] == record["masked_long_prompt"] == LONG_PROMPT
    assert record["text_input_2_short"].startswith(SHORT_PROMPT)
    assert record["text_input_5_masked_short"].startswith(SHORT_PROMPT)
    assert record["text_input_7_fixed_long"].startswith(LONG_PROMPT)
    assert record["text_input_8_fixed_masked_long"].startswith(LONG_PROMPT)
    assert "平安银行" not in record["masked_short_body"]
    assert "000001" not in record["masked_short_body"]
    assert "2026年8月1日" not in record["masked_short_body"]
    assert "某公司" in record["masked_short_body"]
    assert "某时间" in record["masked_short_body"]

    short_record = build_prompt_record(source, 1, layout="short")
    assert "text_input_7_fixed_long" not in short_record
    assert "long_prompt" not in short_record
    assert short_record["text_input_3_long"] == record["text_input_3_long"]


def test_prompt_bundle_keeps_identity_outside_model_text():
    source = {
        "document_id": "doc-2",
        "stock_id": "000002",
        "stock_name": "万科A",
        "announcement_date": "2026-08-02",
        "title_clean_final": "万科A公告",
        "text_model": "万科A股份有限公司公告。",
    }

    record = build_prompt_record(source, 7)

    assert record["stock_id_alignment"] == "000002"
    assert record["document_id"] == "doc-2"
    assert record["stock_id_alignment"] not in record["masked_long_body"]


def test_masking_repeats_until_concatenated_patterns_are_exhausted():
    value = (
        "母公司股东权益冲减子公司深圳房地产有限责任公司"
        "2014.12.312013.12.312012.12.31000001"
    )

    masked = mask_identity_and_time(value, "平安银行", "000001")

    assert not COMPANY_NAME.search(masked)
    assert not any(pattern.search(masked) for pattern in DATE_PATTERNS)
    assert "000001" not in masked
    assert mask_identity_and_time(masked, "平安银行", "000001") == masked


def test_residual_audit_does_not_create_cross_field_dates():
    record = {
        "stock_name": "长虹华意",
        "stock_id": "000404",
        "title_clean_final": "原标题",
        "text_model": "原正文",
        "masked_title": "某公司：某时间年度股东大会决议公告2021-42",
        "masked_body": "-042证券代码:某公司",
    }

    leakage = _residual_leakage(record)

    assert leakage["calendar_date_remaining"] is False


def test_residual_audit_checks_each_masked_field():
    record = {
        "stock_name": "平安银行",
        "stock_id": "000001",
        "title_clean_final": "原标题",
        "text_model": "原正文",
        "masked_title": "某公司公告",
        "masked_body": "公告日期为2026年8月1日",
    }

    assert _residual_leakage(record)["calendar_date_remaining"] is True
