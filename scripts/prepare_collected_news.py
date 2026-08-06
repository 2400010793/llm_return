"""Prepare browser-collected records for text-model validation.

This creates a transparent clean-text sample; it does not invent return labels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dedup_news import deduplicate_news
from src.text.lexicon_features import lexicon_score
from src.text.preprocess_zh import combine_news_text, clean_text, text_hash

_NAV = {"财经", "焦点", "股票", "新股", "期指", "期权", "行情", "数据", "全球", "美股", "港股", "期货", "外汇", "银行", "基金", "理财", "债券", "直播", "股吧", "基金吧", "博客", "财富号", "搜索", "举报", "登录 | 注册", "网友评论"}
_AD_MARKERS = ("在东方财富看资讯行情", "炒股第一步", "全新妙想投研助理", "扫一扫下载APP")


def clean_detail_body(raw: str) -> str:
    lines = [clean_text(x) for x in str(raw or "").splitlines() if clean_text(x)]
    source_pos = next((i for i, line in enumerate(lines) if " 来源：" in line), -1)
    if source_pos >= 0:
        source_line = lines[source_pos]
        after_source = re.split(r" 来源：[^ ]+", source_line, maxsplit=1)[-1].strip()
        lines = ([after_source] if after_source else []) + lines[source_pos + 1 :]
    kept: list[str] = []
    for line in lines:
        if line in _NAV:
            continue
        for marker in _AD_MARKERS:
            line = line.replace(marker, " ")
        if line.startswith(("文章来源：", "责任编辑：", "原标题：", "郑重声明：", "网友评论")):
            break
        if re.fullmatch(r"[0-9]+", line) or "行情" == line:
            continue
        kept.append(line)
    return " ".join(kept)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="data/processed/news_text_probe.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = []
    for record in payload["records"]:
        if record.get("content_type") != "news":
            continue
        body = clean_detail_body(record.get("body", ""))
        text = combine_news_text(record.get("title"), body)
        features = lexicon_score(text)
        rows.append({**record, "body_clean": body, "text": text, "text_hash": text_hash(text), **features})
    unique, duplicate = deduplicate_news(__import__("pandas").DataFrame(rows))
    result = {"input": args.input, "n_raw": len(rows), "n_unique": len(unique), "n_duplicate": len(duplicate), "records": unique.to_dict(orient="records")}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
