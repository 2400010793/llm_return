from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from scripts.clean_sina_uploaded_archive import (
    page_rejection_reason,
    resolve_stock_evidence,
)


def test_page_rejection_reason_filters_dynamic_and_topic_pages() -> None:
    assert (
        page_rejection_reason(
            "平安银行(000001)股票股价,行情,新闻,财报数据_新浪财经_新浪网",
            "https://finance.sina.com.cn/realstock/company/sz000001/nc.shtml",
        )
        == "stock_quote_page"
    )
    assert (
        page_rejection_reason(
            "年度财经论坛",
            "https://finance.sina.com.cn/zt_d/subject-123/",
        )
        == "topic_page"
    )
    assert (
        page_rejection_reason(
            "浦发银行发布年度报告",
            "https://finance.sina.com.cn/stock/2026-08-01/doc-example.shtml",
        )
        is None
    )


def test_resolve_stock_evidence_requires_text_evidence_and_disambiguates_short_names() -> None:
    names = {"000001": "平安银行", "300024": "机器人", "600000": "浦发银行"}
    unrelated = {
        "stock_matches": [
            {"stock_id": "000001", "stock_name": "", "stock_match_method": "sina_stock_link"}
        ]
    }
    evidence, reason = resolve_stock_evidence(
        unrelated,
        title="保险行业发布新规",
        body="多家保险公司介绍了业务发展情况。",
        stock_names=names,
    )
    assert evidence is None
    assert reason == "no_reliable_stock_evidence"

    ambiguous = {
        "stock_matches": [
            {"stock_id": "300024", "stock_name": "机器人", "stock_match_method": "company_name"}
        ]
    }
    evidence, reason = resolve_stock_evidence(
        ambiguous,
        title="机器人产业加快发展",
        body="工业机器人产量持续提高。",
        stock_names=names,
    )
    assert evidence is None
    assert reason == "no_reliable_stock_evidence"

    evidence, reason = resolve_stock_evidence(
        ambiguous,
        title="机器人发布公告",
        body="机器人（300024）披露了最新经营情况。",
        stock_names=names,
    )
    assert reason is None
    assert evidence is not None
    assert evidence["stock_evidence"] == "code_in_body"


def _record(
    article_id: str,
    *,
    stock_id: str,
    stock_name: str,
    match_method: str,
    title: str,
    body: str,
    published_at: str,
    url: str,
    truncated: bool = False,
) -> dict[str, object]:
    return {
        "article_id": article_id,
        "stock_matches": [
            {
                "stock_id": stock_id,
                "stock_name": stock_name,
                "stock_match_method": match_method,
            }
        ],
        "title": title,
        "body": body,
        "published_at": published_at,
        "url": url,
        "source": "sina_finance",
        "content_type": "financial_news",
        "body_truncated": truncated,
        "collected_at": "2026-08-18T01:00:00+00:00",
        "depth": 2,
    }


def test_clean_uploaded_archive_cli_writes_strict_outputs(tmp_path: Path) -> None:
    filler = "公司披露了经营进展、财务数据和后续计划。" * 12
    rows = [
        _record(
            "valid-long-name",
            stock_id="600000",
            stock_name="浦发银行",
            match_method="company_name",
            title="浦发银行发布年度经营报告",
            body=f"浦发银行发布最新经营数据。{filler}",
            published_at="2018年06月01日 09:30",
            url="https://finance.sina.com.cn/stock/2018-06-01/doc-valid.shtml",
        ),
        _record(
            "duplicate-clean-text",
            stock_id="600000",
            stock_name="浦发银行",
            match_method="company_name",
            title="浦发银行发布年度经营报告",
            body=f"浦发银行发布最新经营数据。{filler}",
            published_at="2018/6/1",
            url="https://finance.sina.com.cn/stock/2018-06-01/doc-duplicate.shtml",
        ),
        _record(
            "unrelated-link",
            stock_id="000001",
            stock_name="",
            match_method="sina_stock_link",
            title="保险行业发布新规",
            body=f"多家保险公司介绍了经营情况。{filler}",
            published_at="2024-11-13T16:59:57+08:00",
            url="https://finance.sina.com.cn/money/insurance/2024-11-13/doc-unrelated.shtml",
        ),
        _record(
            "ambiguous-name",
            stock_id="300024",
            stock_name="机器人",
            match_method="company_name",
            title="机器人产业加快发展",
            body=f"工业机器人产量持续提高。{filler}",
            published_at="2025年9月25",
            url="https://finance.sina.com.cn/stock/2025-09-25/doc-ambiguous.shtml",
        ),
        _record(
            "valid-short-name-code",
            stock_id="300024",
            stock_name="机器人",
            match_method="explicit_code",
            title="机器人公司披露经营情况",
            body=f"上市公司机器人（300024）披露经营情况。{filler}",
            published_at="2025年9月25",
            url="https://finance.sina.com.cn/stock/2025-09-25/doc-code.shtml",
        ),
        _record(
            "future-quote-page",
            stock_id="600663",
            stock_name="陆家嘴",
            match_method="sina_stock_link",
            title="陆家嘴(600663)股票股价,行情,新闻,财报数据_新浪财经_新浪网",
            body=f"陆家嘴行情数据。{filler}",
            published_at="2026年9月1",
            url="https://finance.sina.com.cn/realstock/company/sh600663/nc.shtml",
        ),
    ]
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    universe = tmp_path / "stocks.csv"
    with universe.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stock_id", "stock_name"])
        writer.writerows(
            [
                ["000001", "平安银行"],
                ["300024", "机器人"],
                ["600000", "浦发银行"],
                ["600663", "陆家嘴"],
            ]
        )
    output = tmp_path / "cleaned"

    subprocess.run(
        [
            sys.executable,
            "scripts/clean_sina_uploaded_archive.py",
            str(source),
            "--output-dir",
            str(output),
            "--stock-universe",
            str(universe),
            "--cutoff-date",
            "2026-08-18",
            "--min-body-chars",
            "120",
            "--parquet-batch-size",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    cleaned = pd.read_parquet(output / "sina_single_stock_clean_new.parquet")
    assert cleaned["article_id"].tolist() == ["valid-long-name", "valid-short-name-code"]
    assert cleaned["published_at"].tolist()[0] == "2018-06-01T09:30:00+08:00"
    assert cleaned["stock_name"].tolist() == ["浦发银行", "机器人"]

    summary = json.loads((output / "sina_single_stock_clean.summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["input_rows"] == 6
    assert summary["counts"]["output_rows"] == 2
    assert summary["counts"]["rejected_duplicate_clean_text_stock"] == 1
    assert summary["counts"]["rejected_no_reliable_stock_evidence"] == 2
    assert summary["counts"]["rejected_stock_quote_page"] == 1
