from __future__ import annotations

import pandas as pd

from scripts.build_sina_prompt_variants import VARIANT_COLUMNS, build_variants


def test_build_variants_masks_identity_and_keeps_alignment() -> None:
    source = pd.DataFrame({
        "article_id": ["a"],
        "stock_id": ["1"],
        "stock_name": ["平安银行"],
        "published_at": ["2026-08-01T10:00:00+08:00"],
        "publication_date": ["2026-08-01"],
        "title_clean": ["平安银行2026年业绩改善"],
        "body_clean": ["平安银行（000001）于2026年8月1日发布公告。"],
        "text": ["平安银行2026年业绩改善\n平安银行（000001）于2026年8月1日发布公告。"],
        "text_hash": ["h"],
    })

    result = build_variants(source)

    assert result.loc[0, "article_id"] == "a"
    assert result.loc[0, "stock_id"] == "000001"
    assert "平安银行" in result.loc[0, "text_prompt_long"]
    assert "平安银行" not in result.loc[0, "text_masked_long"]
    assert "000001" not in result.loc[0, "text_masked_long"]
    assert "2026年8月1日" not in result.loc[0, "text_masked_long"]
    assert "某公司" in result.loc[0, "text_masked_long"]
    assert all(result.loc[0, name] for name in VARIANT_COLUMNS)
