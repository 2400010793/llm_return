"""Clean one CNINFO collector JSON file and save its result immediately."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.clean_cninfo_announcements import clean_pdf_text, quality_flags, sanitize_json_value
from src.text.preprocess_zh import text_hash


def clean_record(record: dict) -> dict | None:
    record = sanitize_json_value(record)
    if not isinstance(record, dict):
        return None
    if record.get("content_type") != "announcement":
        return None
    pdf_text = clean_pdf_text(record.get("pdf_text", ""))
    fallback = clean_pdf_text(record.get("body", ""))
    text = pdf_text or fallback
    return {
        **record,
        "pdf_text_clean": pdf_text,
        "body_clean": fallback,
        "text": text,
        "text_hash": text_hash(text),
        "text_source": "pdf_text" if pdf_text else "detail_body",
        **quality_flags(text),
        "raw_text_chars": len(str(record.get("pdf_text", "") or record.get("body", "") or "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    source_records = payload.get("records", []) if isinstance(payload, dict) else []
    cleaned = [item for record in source_records if (item := clean_record(record)) is not None]
    target = args.output_dir / f"{args.input.stem}.jsonl"
    temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in cleaned:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(temporary, target)
    stats = {
        "input": str(args.input),
        "output": str(target),
        "input_records": len(source_records),
        "announcement_records": len(cleaned),
        "usable_text_records": sum(bool(item["text"]) for item in cleaned),
        "empty_text_records": sum(not item["text"] for item in cleaned),
        "low_quality_records": sum(bool(item["low_quality_text"]) for item in cleaned),
        "text_chars": sum(int(item["text_chars"]) for item in cleaned),
    }
    stats_target = args.output_dir / f"{args.input.stem}.summary.json"
    stats_temporary = stats_target.with_suffix(stats_target.suffix + f".tmp.{os.getpid()}")
    stats_temporary.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(stats_temporary, stats_target)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
