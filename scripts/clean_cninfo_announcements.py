"""Clean CNINFO announcement records with PDF-extracted text.

The cleaner is conservative: it removes PDF layout noise and joins wrapped
lines, but keeps Chinese text, numbers, dates, units, and table content.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text.preprocess_zh import text_hash


SPACE = re.compile(r"[ \t\r\f\v]+")
MANY_NEWLINES = re.compile(r"\n{3,}")
PAGE_NUMBER = re.compile(r"^\s*(?:第\s*)?\d+\s*(?:页|/\s*\d+)?\s*$")
HTML_TAG = re.compile(r"</?(?:p|br|div|span|table|tr|td|th|html|body|font|strong|b|i|u)(?:\s[^>]*)?>", re.I)
URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
REPLACEMENT = "\ufffd"
BAD_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ue000-\uf8ff]")
CJK = re.compile(r"[\u4e00-\u9fff]")
ASCII_ALNUM = re.compile(r"[A-Za-z0-9]")
PRIVATE_USE = re.compile(r"[\ue000-\uf8ff]")


def _source_files(inputs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        else:
            files.append(path)
    return sorted(set(files))


def _remove_bad_characters(text: str) -> str:
    """Remove encoding/control noise but retain meaningful financial symbols."""
    text = text.replace(REPLACEMENT, " ")
    text = URL.sub(" ", text)
    text = HTML_TAG.sub(" ", text)
    return BAD_CONTROL.sub("", text)


def clean_pdf_text(value: object) -> str:
    """Remove layout artifacts while preserving meaningful announcement text."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = _remove_bad_characters(text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    raw_lines = [SPACE.sub(" ", line).strip() for line in text.splitlines()]
    raw_lines = [line for line in raw_lines if line and not PAGE_NUMBER.fullmatch(line)]

    paragraphs: list[str] = []
    current = ""
    for line in raw_lines:
        if not current:
            current = line
            continue
        # PDF extraction wraps sentences and Chinese words arbitrarily. Keep
        # explicit section/table boundaries, otherwise join with no separator
        # for Chinese and one space for adjacent Latin/numeric tokens.
        boundary = bool(re.match(r"^(第[一二三四五六七八九十百]+节|[一二三四五六七八九十百]+、|\(?[一二三四五六七八九十百]+\)|重要提示|目录|附件|特此公告)", line))
        if boundary or current.endswith(("。", "；", "：", ";", ":")):
            paragraphs.append(current)
            current = line
        elif re.search(r"[A-Za-z0-9]$", current) and re.match(r"^[A-Za-z0-9]", line):
            current += " " + line
        else:
            current += line
    if current:
        paragraphs.append(current)
    return MANY_NEWLINES.sub("\n\n", "\n".join(paragraphs)).strip()


def quality_flags(text: str) -> dict[str, object]:
    """Return auditable quality indicators before any model truncation."""
    chars = len(text)
    if not chars:
        return {
            "text_chars": 0, "control_chars": 0, "replacement_chars": 0,
            "private_use_chars": 0, "cjk_chars": 0, "ascii_alnum_chars": 0,
            "low_quality_text": True,
        }
    control = len(BAD_CONTROL.findall(text))
    replacement = text.count(REPLACEMENT)
    private_use = len(PRIVATE_USE.findall(text))
    cjk = len(CJK.findall(text))
    ascii_alnum = len(ASCII_ALNUM.findall(text))
    non_readable = control + replacement + private_use
    return {
        "text_chars": chars,
        "control_chars": int(control),
        "replacement_chars": int(replacement),
        "private_use_chars": int(private_use),
        "cjk_chars": int(cjk),
        "ascii_alnum_chars": int(ascii_alnum),
        "low_quality_text": bool(non_readable / chars > 0.01 or (cjk + ascii_alnum) / chars < 0.10),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="collector JSON files or directories")
    parser.add_argument("--output", default="data/processed/cleaned/cninfo_announcements_cleaned.json")
    parser.add_argument("--summary", default="", help="Optional JSON summary output")
    parser.add_argument("--sample-output", default="", help="Optional fixed random sample audit file")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--max-chars", type=int, default=50000)
    parser.add_argument("--truncate-lengths", default="8000,12000,20000")
    args = parser.parse_args()
    records: list[dict] = []
    files = _source_files(args.inputs)
    invalid_files: list[str] = []
    for filename in files:
        try:
            payload = json.loads(filename.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            invalid_files.append(str(filename))
            continue
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            records.extend(payload["records"])
    cleaned: list[dict] = []
    for record in records:
        if record.get("content_type") != "announcement":
            continue
        pdf_text = clean_pdf_text(record.get("pdf_text", ""))
        fallback = clean_pdf_text(record.get("body", ""))
        text = pdf_text or fallback
        quality = quality_flags(text)
        item = {
            **record,
            "pdf_text_clean": pdf_text,
            "body_clean": fallback,
            "text": text,
            "text_hash": text_hash(text),
            "text_source": "pdf_text" if pdf_text else "detail_body",
            **quality,
            "raw_text_chars": len(str(record.get("pdf_text", "") or record.get("body", "") or "")),
        }
        cleaned.append(item)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"records": cleaned}, ensure_ascii=False, indent=2), encoding="utf-8")
    lengths = [int(item["text_chars"]) for item in cleaned]
    truncation = {}
    for value in [int(x) for x in args.truncate_lengths.split(",") if x.strip()]:
        truncation[str(value)] = {
            "truncated_records": sum(length > value for length in lengths),
            "truncated_rate": sum(length > value for length in lengths) / len(lengths) if lengths else 0.0,
            "retained_chars_total": sum(min(length, value) for length in lengths),
        }
    import random
    rng = random.Random(args.seed)
    sample = rng.sample(cleaned, min(args.sample_size, len(cleaned)))
    if args.sample_output:
        sample_path = Path(args.sample_output)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        with sample_path.open("w", encoding="utf-8") as handle:
            for index, item in enumerate(sample, 1):
                handle.write(f"===== SAMPLE {index} =====\n")
                handle.write(f"stock_id: {item.get('stock_id')}\n")
                handle.write(f"title: {item.get('title')}\n")
                handle.write(f"raw_text_chars: {item.get('raw_text_chars')}\n")
                handle.write(f"text_chars: {item.get('text_chars')}\n")
                handle.write(f"low_quality_text: {item.get('low_quality_text')}\n")
                handle.write(str(item.get("text", ""))[:3000])
                handle.write("\n\n")
    stats = {
        "input_files": len(files),
        "invalid_files": len(invalid_files),
        "invalid_file_paths": invalid_files,
        "input_records": len(records),
        "announcement_records": len(cleaned),
        "usable_text_records": sum(bool(item["text"]) for item in cleaned),
        "pdf_text_records": sum(item["text_source"] == "pdf_text" for item in cleaned),
        "empty_text_records": sum(not item["text"] for item in cleaned),
        "below_min_chars": sum(item["text_chars"] < args.min_chars for item in cleaned),
        "above_max_chars": sum(item["text_chars"] > args.max_chars for item in cleaned),
        "low_quality_records": sum(item["low_quality_text"] for item in cleaned),
        "min_chars": min(lengths, default=0),
        "median_chars": sorted(item["text_chars"] for item in cleaned)[len(cleaned) // 2] if cleaned else 0,
        "length_quantiles": {str(q): float(__import__("numpy").quantile(lengths, q)) for q in (0.5, 0.75, 0.95, 0.99, 1.0)} if lengths else {},
        "truncation": truncation,
        "output": str(destination),
    }
    summary = Path(args.summary) if args.summary else destination.with_suffix(".summary.json")
    summary.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
