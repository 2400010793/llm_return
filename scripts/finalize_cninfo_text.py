"""Finalize deduplicated CNINFO text for modeling.

This pass is intentionally conservative: remove PDF/page boilerplate and
layout whitespace, while excluding only high-confidence non-financial greeting
announcements. Records remain keyed by stock and are never merged across stocks.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

PAGE_MARK = re.compile(r"第\s*\d+\s*页\s*(?:共\s*\d+\s*页)?|第\s*\d+\s*/\s*\d+\s*页")
BOILERPLATE = re.compile(
    r"当前浏览器不支持在线预览PDF文件|请升级IE版本后刷新阅读|移步下载|公告下载|浏览收藏"
)
HTML_TAG = re.compile(r"</?(?:p|br|div|span|table|tr|td|th|html|body|font|strong|b|i|u)(?:\s[^>]*)?>", re.I)
URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
# High-confidence greeting/holiday material, selected by announcement title.
NOISE_TITLE = re.compile(
    r"新年祝福|新春祝福|春节祝福|元旦祝福|中秋祝福|端午祝福|节日祝福|圣诞祝福|新年致辞|新春致辞|节日问候|贺岁",
    re.I,
)
CJK_OR_DIGIT = r"\u4e00-\u9fff\d%％（）()《》【】"


def remove_layout_noise(value: object) -> str:
    text = str(value or "")
    text = PAGE_MARK.sub(" ", text)
    text = BOILERPLATE.sub(" ", text)
    text = HTML_TAG.sub(" ", text)
    text = URL.sub(" ", text)
    text = text.replace("\u3000", " ").replace("\ufeff", " ").replace("\ufffd", " ")
    # Remove whitespace introduced between Chinese characters, numbers and
    # units, while preserving spaces inside ordinary English phrases.
    text = re.sub(r"\s+", "", text)
    return text.strip()


def noise_reason(record: dict) -> str | None:
    title = str(record.get("title", ""))
    if NOISE_TITLE.search(title):
        return "high_confidence_greeting_title"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--excluded", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.excluded.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    out_tmp = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    excluded_tmp = args.excluded.with_suffix(args.excluded.suffix + f".tmp.{os.getpid()}")
    counts: Counter[str] = Counter()
    input_records = output_records = excluded_records = 0
    chars_before = chars_after = 0
    with args.input.open(encoding="utf-8") as source, out_tmp.open("w", encoding="utf-8") as output, excluded_tmp.open("w", encoding="utf-8") as excluded:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            input_records += 1
            reason = noise_reason(record)
            if reason:
                excluded.write(json.dumps({"reason": reason, "record": record}, ensure_ascii=False) + "\n")
                excluded_records += 1
                counts[reason] += 1
                continue
            before = str(record.get("text", "") or "")
            title = remove_layout_noise(record.get("title", ""))
            text = remove_layout_noise(before)
            record["title_clean_final"] = title
            record["text_model"] = text
            record["text_model_chars"] = len(text)
            record["final_cleaning_version"] = "cninfo_v2_no_layout_space"
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_records += 1
            chars_before += len(before)
            chars_after += len(text)
    os.replace(out_tmp, args.output)
    os.replace(excluded_tmp, args.excluded)
    summary = {
        "input": str(args.input),
        "input_records": input_records,
        "output_records": output_records,
        "excluded_records": excluded_records,
        "excluded_by_reason": dict(counts),
        "chars_before": chars_before,
        "chars_after": chars_after,
        "char_reduction_rate": 1 - chars_after / chars_before if chars_before else 0.0,
        "noise_policy": "exclude only high-confidence greeting/holiday titles; preserve same text for different stocks",
        "output": str(args.output),
        "excluded": str(args.excluded),
    }
    summary_tmp = args.summary.with_suffix(args.summary.suffix + f".tmp.{os.getpid()}")
    summary_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(summary_tmp, args.summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
