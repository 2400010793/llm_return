"""Clean CNINFO announcement records with PDF-extracted text.

The cleaner is conservative: it removes PDF layout noise and joins wrapped
lines, but keeps Chinese text, numbers, dates, units, and table content.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text.preprocess_zh import text_hash


SPACE = re.compile(r"[ \t\r\f\v]+")
MANY_NEWLINES = re.compile(r"\n{3,}")
PAGE_NUMBER = re.compile(r"^\s*(?:第\s*)?\d+\s*(?:页|/\s*\d+)?\s*$")


def clean_pdf_text(value: object) -> str:
    """Remove layout artifacts while preserving meaningful announcement text."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\ufeff", "").replace("\x00", "")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="collector JSON files")
    parser.add_argument("--output", default="data/processed/cninfo_core_2018_clean.json")
    args = parser.parse_args()
    records: list[dict] = []
    for filename in args.inputs:
        payload = json.loads(Path(filename).read_text(encoding="utf-8"))
        records.extend(payload.get("records", []))
    cleaned: list[dict] = []
    for record in records:
        if record.get("content_type") != "announcement":
            continue
        pdf_text = clean_pdf_text(record.get("pdf_text", ""))
        fallback = clean_pdf_text(record.get("body", ""))
        text = pdf_text or fallback
        item = {
            **record,
            "pdf_text_clean": pdf_text,
            "body_clean": fallback,
            "text": text,
            "text_hash": text_hash(text),
            "text_source": "pdf_text" if pdf_text else "detail_body",
            "text_chars": len(text),
        }
        cleaned.append(item)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"records": cleaned}, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {
        "input_records": len(records),
        "announcement_records": len(cleaned),
        "usable_text_records": sum(bool(item["text"]) for item in cleaned),
        "pdf_text_records": sum(item["text_source"] == "pdf_text" for item in cleaned),
        "empty_text_records": sum(not item["text"] for item in cleaned),
        "min_chars": min((item["text_chars"] for item in cleaned), default=0),
        "median_chars": sorted(item["text_chars"] for item in cleaned)[len(cleaned) // 2] if cleaned else 0,
        "output": str(destination),
    }
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
