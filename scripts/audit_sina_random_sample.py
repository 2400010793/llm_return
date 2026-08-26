"""Create and validate a deterministic random sample of cleaned Sina news."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd


BAD_CHARACTER_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ue000-\uf8ff\ufffd]")
HTML_TAG_RE = re.compile(r"<\s*/?\s*[A-Za-z!][^>]*>")
HTML_ENTITY_RE = re.compile(r"&(?:nbsp|amp|quot|apos|lt|gt|#\d+|#x[0-9a-f]+);", re.IGNORECASE)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
SUSPICIOUS_TEMPLATE_MARKERS = (
    "海量资讯、精准解读，尽在新浪财经APP",
    "进入 【新浪财经股吧】",
    "进入【新浪财经股吧】",
    "相关新闻 加载中 点击加载更多",
    "当前浏览器不支持在线预览PDF文件",
    "特别声明：以上文章内容仅代表作者本人观点",
    "新浪财经意见反馈留言板",
    "新浪科技意见反馈留言板",
    "新浪意见反馈留言板",
    "新浪财经APP 缩小字体 放大字体 收藏",
    "分享到:",
    "分享到：",
    "下一条快讯将在",
    "@@title@@",
    "@@teacher_name@@",
    "炒股开户享福利",
    "炒股就看 金麒麟分析师研报",
    "股市跌了别害怕！",
    "反弹行情下的专属投资礼包！",
    "股市回暖，抄底炒股先开户！",
    "查看更多董秘问答>>",
    "文章关键词：",
    "我要反馈 新浪直播 百位牛人在线解读股市热点",
    "登录新浪财经APP 搜索【信披】查看更多考评等级",
    "热点栏目 资金流向 千股千评 个股诊断",
    "热点栏目 自选股 数据中心 行情中心",
)
HARD_CHECKS = (
    "required_fields",
    "article_id_unique",
    "text_hash",
    "text_structure",
    "body_length_metadata",
    "minimum_body_length",
    "timestamp_alignment",
    "not_truncated",
    "no_bad_characters",
    "no_html_tags",
    "no_html_entities",
    "no_template_residue",
)


def _read_and_sample(path: Path, size: int, seed: int) -> tuple[list[dict[str, Any]], int]:
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    total = 0
    with path.open(encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            record["source_line"] = source_line
            total += 1
            if len(sample) < size:
                sample.append(record)
                continue
            replace_at = rng.randrange(total)
            if replace_at < size:
                sample[replace_at] = record
    if total < size:
        raise ValueError(f"requested {size} rows but input only contains {total}")
    return sample, total


def _load_stock_names(path: Path | None) -> list[str]:
    if path is None:
        return []
    frame = pd.read_csv(path, dtype={"stock_id": str})
    if "stock_name" not in frame:
        raise ValueError(f"stock universe has no stock_name column: {path}")
    return sorted({str(value).strip() for value in frame["stock_name"] if str(value).strip()}, key=len, reverse=True)


def _inspect_record(record: dict[str, Any], stock_names: list[str]) -> dict[str, Any]:
    required = (
        "article_id",
        "stock_id",
        "stock_name",
        "published_at",
        "publication_date",
        "title_clean",
        "body_clean",
        "text",
        "text_hash",
        "body_clean_chars",
    )
    title = str(record.get("title_clean") or "")
    body = str(record.get("body_clean") or "")
    text = str(record.get("text") or "")
    stock_id = str(record.get("stock_id") or "")
    stock_name = str(record.get("stock_name") or "")
    expected_text = f"标题：{title}。正文：{body}" if title and body else title or body
    expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    timestamp_ok = False
    try:
        published = datetime.fromisoformat(str(record.get("published_at") or ""))
        timestamp_ok = (
            published.tzinfo is not None
            and published.date().isoformat() == str(record.get("publication_date") or "")
            and published.year == int(record.get("coverage_year"))
        )
    except (TypeError, ValueError):
        pass

    template_hits = [marker for marker in SUSPICIOUS_TEMPLATE_MARKERS if marker in text]
    mentioned_names = [name for name in stock_names if name in text]
    other_names = [name for name in mentioned_names if name != stock_name]
    checks = {
        "required_fields": all(record.get(field) not in (None, "") for field in required),
        "article_id_unique": True,
        "text_hash": record.get("text_hash") == expected_hash,
        "text_structure": text == expected_text,
        "body_length_metadata": record.get("body_clean_chars") == len(body),
        "minimum_body_length": len(body) >= 50,
        "timestamp_alignment": timestamp_ok,
        "not_truncated": record.get("body_truncated") is False,
        "no_bad_characters": BAD_CHARACTER_RE.search(text) is None,
        "no_html_tags": HTML_TAG_RE.search(text) is None,
        "no_html_entities": HTML_ENTITY_RE.search(text) is None,
        "no_template_residue": not template_hits,
    }
    warnings = {
        "target_name_not_in_text": bool(stock_name and stock_name not in text),
        "target_code_not_in_text": bool(stock_id and not re.search(rf"(?<!\d){re.escape(stock_id)}(?!\d)", text)),
        "target_name_absent_but_code_present": bool(
            stock_name
            and stock_name not in text
            and stock_id
            and re.search(rf"(?<!\d){re.escape(stock_id)}(?!\d)", text)
        ),
        "other_universe_stock_names": other_names,
        "url_in_text": bool(URL_RE.search(text)),
        "does_not_end_with_punctuation": bool(body and body[-1] not in "。！？!?；;）)]】》”’…"),
        "title_repeated_at_body_start": bool(title and body[: max(100, len(title) * 2)].count(title)),
    }
    return {
        "source_line": int(record["source_line"]),
        "article_id": record.get("article_id"),
        "stock_id": stock_id,
        "stock_name": stock_name,
        "publication_date": record.get("publication_date"),
        "title_clean": title,
        "body_chars": len(body),
        "text_chars": len(text),
        "hard_checks": checks,
        "hard_failures": [name for name, passed in checks.items() if not passed],
        "warnings": warnings,
        "template_hits": template_hits,
        "body_head": body[:800],
        "body_tail": body[-300:] if len(body) > 300 else body,
    }


def _write_review(path: Path, inspected: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        output.write("# 新浪财经单股票文本随机审计（100篇）\n\n")
        output.write("完整文本见同目录 `random_100.jsonl`；本文件用于快速人工浏览首尾。\n\n")
        for number, item in enumerate(inspected, 1):
            warning_names = [
                key for key, value in item["warnings"].items()
                if value not in (False, [], None, "")
            ]
            output.write(f"## {number:03d}. {item['title_clean']}\n\n")
            output.write(
                f"- 股票：{item['stock_name']}（{item['stock_id']}）\n"
                f"- 日期：{item['publication_date']}\n"
                f"- article_id：`{item['article_id']}`；源行：{item['source_line']}\n"
                f"- 长度：正文 {item['body_chars']} 字；全文 {item['text_chars']} 字\n"
                f"- 硬错误：{', '.join(item['hard_failures']) or '无'}\n"
                f"- 复核提示：{', '.join(warning_names) or '无'}\n\n"
                f"正文开头：\n\n{item['body_head']}\n\n"
                f"正文结尾：\n\n{item['body_tail']}\n\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stock-universe", type=Path)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    if args.size <= 0:
        raise ValueError("sample size must be positive")

    sample, total = _read_and_sample(args.input, args.size, args.seed)
    stock_names = _load_stock_names(args.stock_universe)
    article_counts = Counter(str(record.get("article_id") or "") for record in sample)
    inspected = [_inspect_record(record, stock_names) for record in sample]
    for item in inspected:
        if article_counts[str(item["article_id"])] != 1:
            item["hard_checks"]["article_id_unique"] = False
            if "article_id_unique" not in item["hard_failures"]:
                item["hard_failures"].append("article_id_unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "random_100.jsonl"
    parquet_path = args.output_dir / "random_100.parquet"
    with jsonl_path.open("w", encoding="utf-8") as output:
        for record in sample:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    pd.DataFrame(sample).to_parquet(parquet_path, index=False)

    hard_failure_counts = Counter(
        failure for item in inspected for failure in item["hard_failures"]
    )
    warning_counts: Counter[str] = Counter()
    for item in inspected:
        for name, value in item["warnings"].items():
            if value not in (False, [], None, ""):
                warning_counts[name] += 1
    body_lengths = sorted(item["body_chars"] for item in inspected)
    summary = {
        "input": str(args.input.resolve()),
        "input_records": total,
        "sample_size": len(sample),
        "sampling": "reservoir sampling without replacement",
        "seed": args.seed,
        "hard_checks": list(HARD_CHECKS),
        "hard_failure_records": sum(bool(item["hard_failures"]) for item in inspected),
        "hard_failure_counts": dict(sorted(hard_failure_counts.items())),
        "machine_validation_passed": not hard_failure_counts,
        "warning_counts": dict(sorted(warning_counts.items())),
        "body_chars": {
            "min": body_lengths[0],
            "median": body_lengths[len(body_lengths) // 2],
            "max": body_lengths[-1],
        },
        "sample_article_ids": [item["article_id"] for item in inspected],
        "artifacts": {
            "full_jsonl": str(jsonl_path.resolve()),
            "embedding_input_parquet": str(parquet_path.resolve()),
            "record_audit_json": str((args.output_dir / "random_100_audit_records.json").resolve()),
            "human_review_markdown": str((args.output_dir / "random_100_review.md").resolve()),
        },
    }
    (args.output_dir / "random_100_audit_records.json").write_text(
        json.dumps(inspected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "random_100_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_review(args.output_dir / "random_100_review.md", inspected)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
