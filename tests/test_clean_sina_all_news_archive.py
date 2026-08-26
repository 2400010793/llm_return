from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from scripts.clean_sina_all_news_archive import resolve_stocks_from_clean_text


def test_resolver_ignores_sidebar_links_and_expands_verified_stocks() -> None:
    record = {
        "stock_matches": [
            {"stock_id": "000001", "stock_name": "", "stock_match_method": "sina_stock_link"},
            {"stock_id": "600000", "stock_name": "浦发银行", "stock_match_method": "company_name"},
            {"stock_id": "600036", "stock_name": "招商银行", "stock_match_method": "company_name"},
        ]
    }
    resolved, counts = resolve_stocks_from_clean_text(
        record,
        title="两家银行披露经营数据",
        body="浦发银行和招商银行均披露了最新经营情况。",
        stock_names={"000001": "平安银行", "600000": "浦发银行", "600036": "招商银行"},
        return_codes={"000001", "600000", "600036"},
    )
    assert [item["stock_id"] for item in resolved] == ["600000", "600036"]
    assert counts["candidate_without_clean_text_evidence"] == 1


def test_resolver_discovers_explicit_code_without_raw_match() -> None:
    resolved, _ = resolve_stocks_from_clean_text(
        {"stock_matches": []},
        title="公司披露年度报告",
        body="股票代码600000对应公司披露了经营情况。",
        stock_names={"600000": "浦发银行"},
        return_codes={"600000"},
    )
    assert len(resolved) == 1
    assert resolved[0]["stock_evidence"] == "code_in_body"
    assert resolved[0]["stock_confidence"] == "gold"


def _row(article_id: str, title: str, body: str, stocks: list[dict], *, truncated: bool = False) -> dict:
    return {
        "source": "sina_finance",
        "content_type": "financial_news",
        "article_id": article_id,
        "url": f"https://finance.sina.com.cn/stock/2024-01-02/doc-{article_id}.shtml",
        "depth": 3,
        "title": title,
        "published_at": "2024-01-02T09:30:00+08:00",
        "stock_matches": stocks,
        "body": body,
        "body_truncated": truncated,
        "collected_at": "2026-08-18T01:00:00+00:00",
    }


def test_cli_reattributes_and_expands_multi_stock_rows(tmp_path: Path) -> None:
    filler = "公司披露了营业收入、净利润、现金流和后续经营计划。" * 10
    rows = [
        _row(
            "multi",
            "浦发银行和招商银行发布经营数据",
            f"浦发银行和招商银行均发布经营数据。{filler}",
            [
                {"stock_id": "000001", "stock_name": "", "stock_match_method": "sina_stock_link"},
                {"stock_id": "600000", "stock_name": "浦发银行", "stock_match_method": "company_name"},
                {"stock_id": "600036", "stock_name": "招商银行", "stock_match_method": "company_name"},
            ],
            truncated=True,
        ),
        _row(
            "sidebar-only",
            "保险行业发布新规",
            f"保险行业发布了新的经营规则。{filler}",
            [{"stock_id": "000001", "stock_name": "", "stock_match_method": "sina_stock_link"}],
        ),
        _row(
            "code-only",
            "上市公司披露经营进展",
            f"上市公司（600000）披露经营进展。{filler}",
            [],
        ),
    ]
    source = tmp_path / "raw.jsonl"
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    universe = tmp_path / "stocks.csv"
    with universe.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stock_id", "stock_name"])
        writer.writerows([["000001", "平安银行"], ["600000", "浦发银行"], ["600036", "招商银行"]])
    output = tmp_path / "output"

    subprocess.run(
        [
            sys.executable,
            "scripts/clean_sina_all_news_archive.py",
            str(source),
            "--output-dir",
            str(output),
            "--stock-universe",
            str(universe),
            "--cutoff-date",
            "2026-08-18",
            "--parquet-batch-size",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    cleaned = pd.read_parquet(output / "sina_all_news_clean_expanded.parquet")
    assert cleaned["article_id"].is_unique
    assert set(cleaned["article_id"]) == {"multi:600000", "multi:600036", "code-only"}
    assert cleaned.loc[cleaned["source_article_id"].eq("multi"), "is_multi_stock_article"].all()
    assert cleaned.loc[cleaned["source_article_id"].eq("multi"), "body_truncated"].all()
    assert set(cleaned["stock_id"]) == {"600000", "600036"}

    summary = json.loads((output / "sina_all_news_clean.summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["input_rows"] == 3
    assert summary["counts"]["accepted_multi_stock_articles"] == 1
    assert summary["counts"]["rejected_no_verified_stock"] == 1
    assert summary["counts"]["output_rows"] == 3
